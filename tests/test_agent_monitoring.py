"""AI Agents dashboard feed (telemetry) and the learning-loop rebuild (application/retraining.py)."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.application.retraining import Retrainer
from hms_router.rag.corpus import LabelledExample
from hms_router.rag.evaluation import EvalResult, promotion_verdict
from hms_router.telemetry import DECISIONS, FEEDBACK, MODEL_VERSIONS, DecisionRecorder, route_record
from hms_router.textnorm import preprocess


class ListSink:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, kind, items):
        if self.fail:
            raise RuntimeError("cms down")
        self.sent.append((kind, list(items)))


# ----------------------------------------------------------------------------- recorder
def test_recorder_off_without_sink():
    rec = DecisionRecorder(None)
    rec.record(DECISIONS, {"x": 1})
    assert not rec.enabled and rec.pending() == 0


def test_recorder_batches_and_requeues_on_failure():
    sink = ListSink(fail=True)
    rec = DecisionRecorder(sink, "k", max_batch=2, max_queue=10)
    for i in range(3):
        rec.record(DECISIONS, {"i": i})
    rec.flush()
    assert rec.pending() == 3 and rec.stats["failedBatches"] == 1      # nothing lost, retried later
    sink.fail = False
    rec.flush()
    assert [x["i"] for _, batch in sink.sent for x in batch] == [0, 1, 2] and rec.pending() == 0


def test_recorder_drops_oldest_when_full():
    rec = DecisionRecorder(ListSink(), "k", max_queue=2)
    for i in range(3):
        rec.record(DECISIONS, {"i": i})
    assert rec.pending() == 2 and rec.stats["dropped"] == 1


def test_query_hash_is_keyed_and_normalized():
    a, b = DecisionRecorder(ListSink(), "k1"), DecisionRecorder(ListSink(), "k2")
    assert a.query_hash("Pet Dard ") == a.query_hash("pet dard") != b.query_hash("pet dard")


# ----------------------------------------------------------------------------- record content
def _rec_for(service, text):
    rec = DecisionRecorder(ListSink(), "k")
    d = service.route(text)
    return route_record(d, recorder=rec, caller="doctordekho", latency_ms=12.3, request_id="r1",
                        normalized=preprocess(text), raw_text=text, decisive_min_conf=service.policy.decisive_min_conf)


def test_confident_answer_carries_no_patient_text(service):
    r = _rec_for(service, "dant mein bahut dard hai")
    assert r["decision"] == "answer" and r["topSpecialtyId"] == "dentistry" and r["reviewText"] is None
    assert len(r["queryHash"]) == 64 and r["caller"] == "doctordekho"


def test_abstain_and_emergency_are_kept_for_review(service):
    assert _rec_for(service, "asdkjh qwe")["reviewText"] == "asdkjh qwe"
    em = _rec_for(service, "seene mein dard ho raha hai")
    assert em["decision"] == "emergency" and em["isEmergency"] and em["reviewText"]


# ----------------------------------------------------------------------------- promotion gate
def _ev(top1, wrong, em=0.05):
    return EvalResult(100, top1, top1, 0.7, 0.9, wrong, 0.05, 0.25, em, 1.0)


def test_promotion_gate():
    assert promotion_verdict(_ev(0.80, 0.03), None)[0]
    assert promotion_verdict(_ev(0.82, 0.03), _ev(0.80, 0.03))[0]
    assert not promotion_verdict(_ev(0.78, 0.03), _ev(0.80, 0.03))[0]          # accuracy fell
    assert not promotion_verdict(_ev(0.80, 0.05), _ev(0.80, 0.03))[0]          # more confident mistakes
    assert not promotion_verdict(_ev(0.80, 0.03, em=0.10), _ev(0.80, 0.03))[0]  # emergency flags jumped


# ----------------------------------------------------------------------------- retrainer (fast fakes)
class FakeService:
    def __init__(self, tag):
        self.tag = tag


def test_retrainer_promotes_and_reports(monkeypatch):
    import hms_router.application.retraining as rt
    trained = {}

    class FakeBundle:
        def __init__(self, examples):
            self.meta = {"modelVersion": "v2", "trainingRows": len(examples),
                         "cmsExampleIds": [e.id[4:] for e in examples if e.source == "cms"]}

    monkeypatch.setattr(rt.ModelBundle, "train", classmethod(lambda cls, ex: trained.setdefault("b", FakeBundle(ex))))
    monkeypatch.setattr(rt, "load_examples", lambda extra, holdout: [LabelledExample("ds:1", "a", "a", "general", None, 1, "dataset")] + extra)
    monkeypatch.setattr(rt, "load_validation", lambda: [("x", "general")])
    monkeypatch.setattr(rt, "load_validation_texts", lambda: set())
    scores = {"new": _ev(0.85, 0.02), "old": _ev(0.80, 0.03)}
    monkeypatch.setattr(rt, "evaluate", lambda svc, rows: scores[svc.tag])
    monkeypatch.setattr(Retrainer, "build_service_from_current", staticmethod(lambda svc: svc))
    sink, promoted = ListSink(), []
    cms = [LabelledExample("cms:abc", "kaan dard", "kaan dard", "ent", None, 3.0, "cms")]
    r = Retrainer(build_service=lambda b, llm: FakeService("new"),
                  get_current=lambda: (FakeService("old"), {"modelVersion": "v1", "validationHeldOut": True}),
                  promote=promoted.append, recorder=DecisionRecorder(sink, "k"),
                  cms_url="http://cms", cms_key="key", load_cms=lambda u, k: cms)
    out = r.run("manual")
    assert out["promoted"] and promoted and out["cmsRows"] == 1
    kind, items = sink.sent[0]
    assert kind == MODEL_VERSIONS and items[0]["cmsExampleIds"] == ["abc"] and items[0]["previousVersion"] == "v1"


def test_retrainer_keeps_current_when_worse(monkeypatch):
    import hms_router.application.retraining as rt

    class FakeBundle:
        meta = {"modelVersion": "v2"}

    monkeypatch.setattr(rt.ModelBundle, "train", classmethod(lambda cls, ex: FakeBundle()))
    monkeypatch.setattr(rt, "load_examples", lambda extra, holdout: [])
    monkeypatch.setattr(rt, "load_validation", lambda: [("x", "general")])
    monkeypatch.setattr(rt, "load_validation_texts", lambda: set())
    scores = {"new": _ev(0.70, 0.05), "old": _ev(0.80, 0.03)}
    monkeypatch.setattr(rt, "evaluate", lambda svc, rows: scores[svc.tag])
    monkeypatch.setattr(Retrainer, "build_service_from_current", staticmethod(lambda svc: svc))
    promoted = []
    r = Retrainer(build_service=lambda b, llm: FakeService("new"),
                  get_current=lambda: (FakeService("old"), {"modelVersion": "v1", "validationHeldOut": True}),
                  promote=promoted.append, recorder=DecisionRecorder(ListSink(), "k"),
                  cms_url=None, cms_key=None)
    out = r.run("manual")
    assert not out["promoted"] and not promoted and "kept the current model" in out["verdict"]


def test_retrainer_refuses_without_validation_set(monkeypatch):
    """No test set (e.g. a Docker image without data_pipeline/validation_set.csv): every score would be 0
    and any model would pass the gate, so the rebuild must stop instead of promoting blindly."""
    import hms_router.application.retraining as rt

    monkeypatch.setattr(rt, "load_validation", lambda: [])
    promoted = []
    r = Retrainer(build_service=lambda b, llm: FakeService("new"),
                  get_current=lambda: (FakeService("old"), {"modelVersion": "v1"}),
                  promote=promoted.append, recorder=DecisionRecorder(ListSink(), "k"),
                  cms_url=None, cms_key=None)
    with pytest.raises(RuntimeError, match="validation set"):
        r.run("manual")
    assert not promoted


# ----------------------------------------------------------------------------- HTTP wiring
@pytest.fixture(scope="module")
def monitored(bundle):
    env = {"HMS_CMS_API_URL": "http://cms.invalid/api/v1", "HMS_CMS_SERVICE_KEY": "svc", "HMS_REBUILD_ON_START": "0"}
    with TestClient(create_app(ServiceSettings.from_env(env))) as c:
        c.app.state.telemetry.sink = ListSink()
        yield c


def test_route_and_chat_are_recorded(monitored):
    rec = monitored.app.state.telemetry
    monitored.post("/route-symptom", json={"query": "dant mein bahut dard hai"})
    monitored.post("/chat", json={"text": "mujhe bukhar hai"})
    monitored.post("/feedback", json={"kind": "wrong_specialist", "specialist": "ENT", "symptoms": ["kaan dard"],
                                      "suggested_specialist": "Neurologist"})
    rec.flush()
    kinds = {k: [x for _, b in rec.sink.sent if _ == k for x in b] for k in (DECISIONS, FEEDBACK)}
    agents = {x["agent"] for x in kinds[DECISIONS]}
    assert {"route", "chat"} <= agents and kinds[FEEDBACK][0]["text"] == "kaan dard"


def test_admin_endpoints(monitored):
    assert monitored.get("/admin/telemetry").json()["enabled"] is True
    assert monitored.get("/admin/rebuild").json()["state"] == "idle"
    info = monitored.get("/model-info").json()
    assert "telemetry" in info and "cmsExampleIds" not in info
