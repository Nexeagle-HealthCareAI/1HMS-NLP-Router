"""The trained artefacts behind /route-symptom, built and stored as one versioned file.

    python -m hms_router.rag.model_bundle build [--out PATH]

Contents: fitted vectorizer, specialty classifier, emergency classifier, the L2-normalized index of
every labelled example (for sentence retrieval) and the examples themselves (id, text, specialtyId).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import LabelledExample, load_examples, load_validation_texts
from .encoder import build_vectorizer, content_text, l2_normalize_rows

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "route_model.joblib"
BUNDLE_SCHEMA = 2


@dataclass
class ModelBundle:
    vectorizer: object
    classifier: object
    emergency_model: object
    index: object                                       # L2-normalized sparse rows, aligned with `examples`
    examples: list[tuple[str, str, str]]                # (id, text, specialtyId)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, examples: list[LabelledExample] | None = None) -> "ModelBundle":
        """Default corpus = the repo's datasets minus the held-out validation sentences."""
        from sklearn.linear_model import LogisticRegression

        holdout = load_validation_texts() if examples is None else set()
        held_out = examples is None and bool(holdout)      # False when the validation file is missing
        examples = examples if examples is not None else load_examples(holdout=holdout)
        if not examples:
            raise ValueError("no training examples")
        vectorizer = build_vectorizer()
        X = vectorizer.fit_transform([content_text(e.normalized) for e in examples])

        clf = LogisticRegression(C=10, max_iter=4000, class_weight="balanced")
        clf.fit(X, [e.specialtyId for e in examples], sample_weight=[e.weight for e in examples])

        triaged = [i for i, e in enumerate(examples) if e.emergency is not None]
        emerg = LogisticRegression(C=10, max_iter=4000, class_weight="balanced")
        emerg.fit(X[triaged], [bool(examples[i].emergency) for i in triaged])

        digest = hashlib.sha1("\n".join(sorted(f"{e.normalized}\t{e.specialtyId}" for e in examples))
                              .encode()).hexdigest()[:8]
        meta = {"schema": BUNDLE_SCHEMA, "modelVersion": time.strftime("%Y-%m-%d-%H%M%S") + "-" + digest,
                "trainedAt": time.time(), "trainingRows": len(examples), "emergencyRows": len(triaged),
                "sources": {s: n for s in ("dataset", "legacy", "seed", "cms")
                            if (n := sum(e.source == s for e in examples))},
                "cmsExampleIds": [e.id[4:] for e in examples if e.source == "cms"],
                "validationHeldOut": held_out}
        return cls(vectorizer, clf, emerg, l2_normalize_rows(X),
                   [(e.id, e.text, e.specialtyId) for e in examples], meta)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path | None = None) -> Path:
        import joblib
        path = Path(path or MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump(self.__dict__, tmp, compress=3)
        tmp.replace(path)                                # atomic: a crash never leaves half a model
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelBundle":
        import joblib
        data = joblib.load(Path(path or MODEL_PATH))
        if data.get("meta", {}).get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"model file schema {data.get('meta', {}).get('schema')} != {BUNDLE_SCHEMA}; rebuild it")
        return cls(**data)

    @classmethod
    def load_or_build(cls, path: str | Path | None = None) -> "ModelBundle":
        path = Path(path or MODEL_PATH)
        if path.exists():
            return cls.load(path)
        bundle = cls.train()
        bundle.save(path)
        return bundle


def _main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m hms_router.rag.model_bundle")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="train on the corpus and write the model file")
    b.add_argument("--out", default=str(MODEL_PATH))
    args = p.parse_args(argv)
    t = time.time()
    bundle = ModelBundle.train()
    path = bundle.save(args.out)
    print(json.dumps({**bundle.meta, "path": str(path), "seconds": round(time.time() - t, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
