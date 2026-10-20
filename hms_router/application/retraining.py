"""Learning loop, last step: rebuild the /route-symptom model with what people approved, test it, and
put it in service only if it is not worse.

    1. corpus      repo datasets + CMS SymptomTrainingExamples (incl. corrections approved in the CMS
                   review queue), minus the held-out validation sentences
    2. train       a candidate ModelBundle
    3. test        candidate and the model in service on the same validation set (evaluation.evaluate)
    4. decide      promotion_verdict: top-1 must not fall, confidently-wrong and emergency flags must not rise
    5. promote     save the file atomically and swap the service in this process (requests in flight keep
                   the old one); otherwise keep the current model
    6. report      one "model version" record to the CMS either way, with metrics, the verdict and the
                   CMS example ids it learned from, so the dashboard can show what changed and why

Runs in a background thread; one at a time. Triggered from the CMS (POST /admin/rebuild) and, when the
CMS is configured, once after start-up so a fresh deployment picks up rows approved since the image was built.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..rag.corpus import load_cms_examples, load_examples, load_validation, load_validation_texts
from ..rag.evaluation import EvalResult, evaluate, promotion_verdict
from ..rag.model_bundle import ModelBundle
from ..telemetry import MODEL_VERSIONS, DecisionRecorder, now_iso

log = logging.getLogger("hms_router.retraining")


class Retrainer:
    def __init__(self, *, build_service: Callable[[ModelBundle, bool], object],
                 get_current: Callable[[], tuple[object | None, dict]],
                 promote: Callable[[ModelBundle], None], recorder: DecisionRecorder,
                 cms_url: str | None, cms_key: str | None,
                 load_cms: Callable[[str, str], list] = load_cms_examples):
        self.build_service = build_service      # (bundle, with_reasoner) -> RouteSymptomService
        self.get_current = get_current          # () -> (service or None, meta)
        self.promote = promote
        self.recorder = recorder
        self.cms_url, self.cms_key = cms_url, cms_key
        self.load_cms = load_cms
        self._lock = threading.Lock()
        self.status: dict = {"state": "idle"}

    @property
    def running(self) -> bool:
        return self.status.get("state") == "running"

    def start(self, reason: str) -> bool:
        """Start a rebuild in the background. False if one is already running."""
        with self._lock:
            if self.running:
                return False
            self.status = {"state": "running", "reason": reason, "startedAt": now_iso(), "step": "starting"}
        threading.Thread(target=self._run_safely, args=(reason,), name="route-rebuild", daemon=True).start()
        return True

    def _step(self, step: str) -> None:
        self.status["step"] = step
        log.info("rebuild: %s", step)

    def _run_safely(self, reason: str) -> None:
        try:
            self.status = {**self.status, **self.run(reason), "state": "done", "finishedAt": now_iso()}
        except Exception as exc:
            log.exception("rebuild failed")
            self.status = {**self.status, "state": "failed", "error": f"{type(exc).__name__}: {exc}"[:400],
                           "finishedAt": now_iso()}
            self.recorder.record(MODEL_VERSIONS, {"modelVersion": None, "builtAt": now_iso(), "trigger": reason,
                                                  "promoted": False, "verdict": self.status["error"]})

    def run(self, reason: str) -> dict:
        t0 = time.time()
        cms_rows, cms_error = [], None
        if self.cms_url and self.cms_key:
            self._step("fetching approved examples from the CMS")
            try:
                cms_rows = self.load_cms(self.cms_url, self.cms_key)
            except Exception as exc:              # still rebuild; the report says the CMS rows were missing
                cms_error = f"CMS training rows not loaded: {type(exc).__name__}: {exc}"[:300]
                log.warning(cms_error)

        rows = load_validation()
        if not rows:   # without a test set every score is 0 and any model would "pass"
            raise RuntimeError("validation set not found (data_pipeline/validation_set.csv); rebuild refused")

        self._step("training the candidate model")
        examples = load_examples(extra=cms_rows, holdout=load_validation_texts())
        bundle = ModelBundle.train(examples)
        bundle.meta["validationHeldOut"] = True

        self._step("testing candidate and current model on the validation set")
        candidate = evaluate(self.build_service(bundle, False), rows)
        current_svc, current_meta = self.get_current()
        current: EvalResult | None = None
        if current_svc is not None:
            # A model trained on the validation sentences scores in-sample; comparing against that would
            # block every honest model, so such a model is replaced by the first held-out build.
            if current_meta.get("validationHeldOut"):
                current = evaluate(self.build_service_from_current(current_svc), rows)
        promoted, verdict = promotion_verdict(candidate, current)
        if current_svc is not None and current is None:
            verdict = "replaced a model that had been trained on the test sentences (its score was not comparable); " + verdict
        if cms_error:
            verdict += f" [{cms_error}]"
        bundle.meta["validation"] = candidate.to_dict()

        if promoted:
            self._step("putting the new model in service")
            self.promote(bundle)

        report = {
            "modelVersion": bundle.meta.get("modelVersion"),
            "previousVersion": current_meta.get("modelVersion"),
            "builtAt": now_iso(),
            "trigger": reason,
            "promoted": promoted,
            "verdict": verdict,
            "trainingRows": bundle.meta.get("trainingRows"),
            "cmsRows": len(cms_rows),
            "sources": bundle.meta.get("sources"),
            "validation": candidate.to_dict(),
            "previousValidation": current.to_dict() if current else None,
            "cmsExampleIds": bundle.meta.get("cmsExampleIds") if promoted else [],
            "seconds": round(time.time() - t0, 1),
        }
        self.recorder.record(MODEL_VERSIONS, report)
        self.recorder.flush()
        log.info("rebuild done: %s promoted=%s %s", report["modelVersion"], promoted, verdict)
        return {k: report[k] for k in ("modelVersion", "promoted", "verdict", "validation", "previousValidation",
                                       "cmsRows", "trainingRows", "seconds")}

    @staticmethod
    def build_service_from_current(service):
        """The in-service RouteSymptomService may carry the LLM reasoner; evaluation must not call it."""
        from ..reasoning.protocols import NoReasoner
        clone = object.__new__(type(service))
        clone.__dict__.update(service.__dict__)
        if hasattr(clone, "reasoner"):
            clone.reasoner = NoReasoner()
        return clone
