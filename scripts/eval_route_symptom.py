"""Measure /route-symptom (RouteSymptomService). Run from the repo root:

    python scripts/eval_route_symptom.py [--dd-dir ../DoctorDekhoWeb/scripts/search-eval] [--sweep] [--skip-held-out]

1. Held-out: 80/20 split of Datasets/combined_symptom_dataset.csv (fixed seed). The model is trained
   on the 80% + English seeds only - the legacy V28 file holds spelling variants of many dataset
   sentences and would leak test rows. Reports specialty top-1/top-3 and emergency recall/false flags.
2. Doctor Dekho's hand-labelled queries (queries-set1.json for tuning, queries-set2.json kept unseen),
   scored with the production model. A hit = top specialty in the query's `ok` list; a non-medical query
   (ok == []) is answered correctly by abstaining. "Decisive precision" = how often an ANSWER (what
   Doctor Dekho trusts without Groq) is right.
Use --sweep on set 1 only to choose thresholds; then score set 2 once.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hms_router.application import build_route_symptom_service  # noqa: E402
from hms_router.domain import Decision, RoutingPolicy  # noqa: E402
from hms_router.rag import ModelBundle, load_examples  # noqa: E402


def held_out(seed: int = 7) -> None:
    examples = load_examples(legacy_path=None)
    data = [e for e in examples if e.source == "dataset"]
    seeds = [e for e in examples if e.source == "seed"]
    random.Random(seed).shuffle(data)
    cut = int(len(data) * 0.8)
    train, test = data[:cut] + seeds, data[cut:]
    svc = build_route_symptom_service(ModelBundle.train(train))
    top1 = top3 = 0
    em, rt = [], []
    for e in test:
        d = svc.route(e.text, top_k=3)
        ids = [c.specialtyId for c in d.candidates]
        top1 += bool(ids) and ids[0] == e.specialtyId
        top3 += e.specialtyId in ids
        (em if e.emergency else rt).append(d.emergency)
    n = len(test)
    print(f"held-out n={n}: top1={top1 / n:.1%} top3={top3 / n:.1%}")
    rules = lambda a: "rules" in (a.source or "")  # noqa: E731
    print(f"  emergency n={len(em)}: recall={sum(a.flag for a in em) / len(em):.0%} "
          f"(rules alone {sum(rules(a) for a in em) / len(em):.0%}); "
          f"false flags on n={len(rt)} others: {sum(a.flag for a in rt) / len(rt):.0%}")


def report(name: str, qs: list[dict], decisions) -> dict:
    med = [(q, d) for q, d in zip(qs, decisions) if q["ok"]]
    non = [(q, d) for q, d in zip(qs, decisions) if not q["ok"]]
    top = lambda d: d.candidates[0].specialtyId if d.candidates else None  # noqa: E731
    t1 = sum(top(d) in q["ok"] for q, d in med)
    t3 = sum(any(c.specialtyId in q["ok"] for c in d.candidates[:3]) for q, d in med)
    abst = sum(d.noMatch for _, d in non)
    ans = [(q, d) for q, d in zip(qs, decisions) if d.decision == Decision.ANSWER]
    ans_ok = sum(bool(q["ok"]) and d.specialtyIds[0] in q["ok"] for q, d in ans)
    clar = sum(d.decision == Decision.CLARIFY for d in decisions)
    r = {"n": len(qs), "answered_correctly": (t1 + abst) / len(qs), "top1": t1 / len(med), "top3": t3 / len(med),
         "nonmedical_abstained": f"{abst}/{len(non)}", "answers": len(ans), "answer_precision": ans_ok / max(1, len(ans)),
         "clarify": clar, "medical_wrongly_abstained": sum(d.noMatch for _, d in med)}
    print(f"{name}: " + "  ".join(f"{k}={v:.0%}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return r


def doctor_dekho(dd_dir: Path, sweep: bool) -> None:
    svc = build_route_symptom_service()
    for name in ("queries-set1.json", "queries-set2.json"):
        path = dd_dir / name
        if not path.exists():
            print(f"skip {path}")
            continue
        qs = json.loads(path.read_text(encoding="utf-8"))
        report(name, qs, [svc.route(q["q"]) for q in qs])
        if sweep and name == "queries-set1.json":
            base = svc.policy
            for dm in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
                svc.policy = replace(base, decisive_min_conf=dm)
                report(f"   decisive_min_conf={dm}", qs, [svc.route(q["q"]) for q in qs])
            for mm, lm, lc in ((0.15, 0.3, 0.3), (0.2, 0.33, 0.3), (0.25, 0.35, 0.3), (0.2, 0.4, 0.35)):
                svc.policy = replace(base, min_match=mm, low_match=lm, low_match_min_conf=lc)
                report(f"   gate {mm}/{lm}/{lc}", qs, [svc.route(q["q"]) for q in qs])
            svc.policy = base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd-dir", type=Path)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--skip-held-out", action="store_true")
    a = ap.parse_args()
    if not a.skip_held_out:
        held_out()
    if a.dd_dir:
        doctor_dekho(a.dd_dir, a.sweep)
    print("policy:", RoutingPolicy().as_dict())
