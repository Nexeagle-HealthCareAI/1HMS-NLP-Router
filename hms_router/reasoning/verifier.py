"""Mechanical checks on a ReasonerResult before anything it says is used. Closed set, cited evidence,
no diagnosis or dosing language, and no symptom the patient did not mention. A rejected result is logged
and the deterministic answer stands."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable

from ..domain.models import Candidate, ReasonerResult
from ..textnorm import preprocess

# Words a reason may use without the patient having said them: the specialties being suggested and the
# generic words for care. Everything else the medical vocabulary recognises must come from the patient.
_ALWAYS_ALLOWED = set("""
doctor doctors specialist specialists physician general medicine care consult consultation checkup check
similar cases case symptoms symptom problem problems complaint health clinic hospital
fit fits fitting match matches matching suggest suggested
""".split())   # "fits" is also a seizure word, but in an explanation it is almost always "fits because"

MAX_REASON_CHARS = 240
_FORBIDDEN = re.compile(
    r"\b(\d+\s?(mg|ml|mcg|tablets?|capsules?|units?)|dos(e|age)|prescri\w*|diagnos\w*|you (have|are suffering from)|"
    r"take (a |an |one |two )?\w*(tablet|pill|medicine|capsule))\b", re.I)


@dataclass
class Verdict:
    ok: bool
    errors: list[str] = field(default_factory=list)


class Verifier:
    def __init__(self, medical_terms: Callable[[str], list[str]] | None = None):
        """medical_terms(normalized) -> the health words in a text (MedicalVocabulary.terms_in). Without it,
        the "only the patient's own symptoms" check is skipped and the other checks still run."""
        self.medical_terms = medical_terms

    def unsupported_terms(self, reason: str, patient_text: str, candidates: list[Candidate]) -> list[str]:
        """Health words in the reason that the patient never used (allowing small spelling differences
        and the names of the suggested specialties)."""
        if self.medical_terms is None or not reason:
            return []
        patient_words = set(preprocess(patient_text).split())
        allowed = set(_ALWAYS_ALLOWED)
        for c in candidates:
            allowed.update(preprocess(c.name).split())
            allowed.update(preprocess(c.specialtyId).split())
        out: list[str] = []
        for term in dict.fromkeys(self.medical_terms(preprocess(reason))):
            if term in allowed or term in patient_words:
                continue
            if difflib.get_close_matches(term, patient_words, n=1, cutoff=0.7):     # the patient's "pian" = "pain"
                continue
            out.append(term)
        return out

    def check(self, result: ReasonerResult, candidates: list[Candidate], patient_text: str | None = None) -> Verdict:
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
        if patient_text is not None:
            invented = self.unsupported_terms(reason, patient_text, candidates)
            if invented:
                errors.append(f"reason mentions symptoms the patient did not say: {invented}")
        return Verdict(not errors, errors)
