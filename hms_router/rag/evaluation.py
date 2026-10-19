"""Score a RouteSymptomService on the held-out validation set (data_pipeline/validation_set.csv).

The numbers the CMS "AI Agents" dashboard shows for each model version, and the gate a rebuilt model
must pass before it replaces the one in service:

    top1                share of sentences whose first specialty is the labelled one
    top3                labelled specialty among the first three
    answered            share the router answered without asking (decision = answer)
    answeredAccuracy    of those it answered, how many were right  (what patients actually experience)
    confidentlyWrong    answered AND wrong, as a share of all sentences - the costly mistake
    abstain / clarify   share with no match / a clarifying question
    emergencyFlagRate   share flagged as emergency (the validation set is mostly routine complaints, so
                        a jump here means the emergency model got trigger-happy)

The service should be built without the LLM reasoner so the score is deterministic and free.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .corpus import load_validation


@dataclass
class EvalResult:
    rows: int
    top1: float
    top3: float
    answered: float
    answeredAccuracy: float
    confidentlyWrong: float
    abstain: float
    clarify: float
    emergencyFlagRate: float
    seconds: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def evaluate(service, rows: list[tuple[str, str]] | None = None) -> EvalResult:
    rows = rows if rows is not None else load_validation()
    t0 = time.time()
    n = len(rows)
    top1 = top3 = answered = answered_right = abstain = clarify = emergency = 0
    for text, expected in rows:
        d = service.route(text, 5)
        ids = [c.specialtyId for c in d.candidates] if not d.noMatch else []
        hit1 = bool(ids) and ids[0] == expected
        top1 += hit1
        top3 += expected in ids[:3]
        emergency += bool(d.emergency.flag)
        dec = d.decision.value
        if dec == "answer":
            answered += 1
            answered_right += hit1
        elif dec == "abstain":
            abstain += 1
        elif dec == "clarify":
            clarify += 1
    div = max(n, 1)
    return EvalResult(rows=n, top1=top1 / div, top3=top3 / div, answered=answered / div,
                      answeredAccuracy=answered_right / max(answered, 1),
                      confidentlyWrong=(answered - answered_right) / div,
                      abstain=abstain / div, clarify=clarify / div, emergencyFlagRate=emergency / div,
                      seconds=round(time.time() - t0, 1))


# A rebuilt model replaces the one in service only if it is not worse on what matters. Tolerances absorb
# noise of a few sentences out of ~1.5k.
TOLERANCE = 0.005


def promotion_verdict(candidate: EvalResult, current: EvalResult | None) -> tuple[bool, str]:
    if current is None:
        return True, "no model in service to compare with"
    problems = []
    if candidate.top1 < current.top1 - TOLERANCE:
        problems.append(f"top-1 accuracy fell {current.top1:.1%} -> {candidate.top1:.1%}")
    if candidate.confidentlyWrong > current.confidentlyWrong + TOLERANCE:
        problems.append(f"confidently-wrong rose {current.confidentlyWrong:.1%} -> {candidate.confidentlyWrong:.1%}")
    if candidate.emergencyFlagRate > current.emergencyFlagRate + 0.02:
        problems.append(f"emergency flags rose {current.emergencyFlagRate:.1%} -> {candidate.emergencyFlagRate:.1%}")
    if problems:
        return False, "kept the current model: " + "; ".join(problems)
    better = candidate.top1 - current.top1
    return True, (f"top-1 {current.top1:.1%} -> {candidate.top1:.1%} ({better:+.1%}), "
                  f"confidently-wrong {current.confidentlyWrong:.1%} -> {candidate.confidentlyWrong:.1%}")
