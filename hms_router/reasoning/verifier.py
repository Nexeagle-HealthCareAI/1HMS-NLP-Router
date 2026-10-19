"""Mechanical checks on a ReasonerResult before anything it says is used. Closed set, cited evidence,
no diagnosis or dosing language. A rejected result is logged and the deterministic answer stands."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.models import Candidate, ReasonerResult

MAX_REASON_CHARS = 240
_FORBIDDEN = re.compile(
    r"\b(\d+\s?(mg|ml|mcg|tablets?|capsules?|units?)|dos(e|age)|prescri\w*|diagnos\w*|you (have|are suffering from)|"
    r"take (a |an |one |two )?\w*(tablet|pill|medicine|capsule))\b", re.I)


@dataclass
class Verdict:
    ok: bool
    errors: list[str] = field(default_factory=list)


class Verifier:
    def check(self, result: ReasonerResult, candidates: list[Candidate]) -> Verdict:
        errors: list[str] = []
        allowed = {c.specialtyId for c in candidates}
        evidence = {e.id: e for c in candidates for e in c.evidence}

        if not result.choices:
            errors.append("no choice made")
        bad = [s for s in result.choices if s not in allowed]
        if bad:
            errors.append(f"choices outside the retrieved candidates: {bad}")
        if len(set(result.choices)) != len(result.choices):
            errors.append("duplicate choices")

        if not result.citedEvidenceIds:
            errors.append("no evidence cited")
        unknown = [i for i in result.citedEvidenceIds if i not in evidence]
        if unknown:
            errors.append(f"cited evidence not in this request: {unknown}")
        if result.choices and evidence and not any(
                evidence[i].specialtyId == result.choices[0] for i in result.citedEvidenceIds if i in evidence):
            errors.append("no cited evidence supports the first choice")

        reason = result.reason or ""
        if not reason.strip():
            errors.append("empty reason")
        if len(reason) > MAX_REASON_CHARS:
            errors.append("reason too long")
        if _FORBIDDEN.search(reason):
            errors.append("reason contains diagnosis or dosing language")
        return Verdict(not errors, errors)
