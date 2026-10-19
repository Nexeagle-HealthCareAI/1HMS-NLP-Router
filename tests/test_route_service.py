"""RouteSymptomService: the flow with fakes (fast, exact), then behaviour with the real model."""
import pytest

from hms_router.application.route_symptom import RouteSymptomService
from hms_router.domain import Decision, RoutingPolicy, SPECIALTY_NAMES
from hms_router.domain.models import ReasonerResult
from tests.conftest_route import FakeDetector, FakeRetriever, FakeScorer, ev


def make(evidence, scores, flag=False, reasoner=None, policy=None):
    det = FakeDetector(flag)
    svc = RouteSymptomService(retriever=FakeRetriever(evidence), scorer=FakeScorer(scores), detector=det,
                              reasoner=reasoner, policy=policy)
    return svc, det


# ----------------------------------------------------------------------------- flow (fakes)
def test_safety_runs_first_even_when_abstaining():
    svc, det = make([], {}, flag=True)
    d = svc.route("12345")
    assert det.calls == 1 and d.emergency.flag and d.decision == Decision.ABSTAIN and d.noMatch


def test_decisive_answer():
    svc, _ = make([ev("a", "dentistry", 0.9), ev("b", "dentistry", 0.8)], {"dentistry": 0.95, "ent": 0.05})
    d = svc.route("dant mein dard")
    assert d.decision == Decision.ANSWER and d.decisive and d.specialtyIds == ["dentistry"]
    assert d.method == "deterministic" and d.candidates[0].evidence[0].id == "a"


def test_low_similarity_abstains():
    svc, _ = make([ev("a", "dentistry", 0.1)], {"dentistry": 0.9, "ent": 0.1})
    d = svc.route("cricket score")
    assert d.decision == Decision.ABSTAIN and d.noMatch and d.specialtyIds == [] and d.candidates == []


def test_close_call_clarifies_with_hedge():
    svc, _ = make([ev("a", "cardiology", 0.6), ev("b", "gastroenterology", 0.6)],
                  {"cardiology": 0.4, "gastroenterology": 0.4, "general": 0.2})
    d = svc.route("seene ke neeche dard")
    assert d.decision == Decision.CLARIFY and not d.decisive
    assert d.clarify and d.clarify.kind == "specialty" and set(d.clarify.optionIds[:2]) == {"cardiology", "gastroenterology"}
    assert len(d.specialtyIds) == 3          # hedge = all candidates (fewer than hedge_size here)


class _Reasoner:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def decide(self, query, candidates):
        self.calls += 1
        return self.result


def _close_call(reasoner):
    return make([ev("a", "cardiology", 0.6), ev("b", "gastroenterology", 0.6)],
                {"cardiology": 0.45, "gastroenterology": 0.55}, reasoner=reasoner)


def test_reasoner_not_called_when_decisive():
    r = _Reasoner(None)
    svc, _ = make([ev("a", "dentistry", 0.9)], {"dentistry": 1.0}, reasoner=r)
    svc.route("dant dard")
    assert r.calls == 0


def test_verified_reasoner_answer_is_used():
    r = _Reasoner(ReasonerResult(["cardiology"], ["a"], "Patient said chest."))
    svc, _ = _close_call(r)
    d = svc.route("seene mein dard")
    assert d.decision == Decision.ANSWER and d.method == "deterministic+llm"
    assert d.candidates[0].specialtyId == "cardiology" and d.reason == "Patient said chest."


@pytest.mark.parametrize("bad", [
    ReasonerResult(["oncology"], ["a"], "invented specialty"),                 # outside candidates
    ReasonerResult(["cardiology"], ["zzz"], "fake citation"),                  # evidence not retrieved
    ReasonerResult(["cardiology"], ["b"], "cites the other specialty"),        # citation doesn't support choice
    ReasonerResult(["cardiology"], ["a"], "Take 500 mg paracetamol"),          # dosing language
    ReasonerResult(["cardiology"], ["a"], "You have angina."),                 # diagnosis language
])
def test_hallucinated_reasoner_output_is_rejected(bad):
    svc, _ = _close_call(_Reasoner(bad))
    d = svc.route("seene mein dard")
    assert d.method == "deterministic" and d.decision == Decision.CLARIFY
    assert any(n.startswith("llm_rejected") for n in d.notes)


def test_adult_context_removes_paediatrics():
    svc, _ = make([ev("a", "pediatrics", 0.7), ev("b", "gastroenterology", 0.65)],
                  {"pediatrics": 0.5, "gastroenterology": 0.5})
    d = svc.route("pet dard ulti", age_group="adult")
    assert all(c.specialtyId != "pediatrics" for c in d.candidates)


def test_child_context_puts_paediatrics_first():
    svc, _ = make([ev("a", "gastroenterology", 0.7), ev("b", "pediatrics", 0.65)],
                  {"gastroenterology": 0.6, "pediatrics": 0.4})
    d = svc.route("pet dard ulti", age_group="child")
    assert d.candidates[0].specialtyId == "pediatrics" and d.decision == Decision.ANSWER
    assert "policy: child -> paediatrics first" in d.notes


def test_bad_age_group_rejected():
    svc, _ = make([], {})
    with pytest.raises(ValueError):
        svc.route("x", age_group="teen")


# ----------------------------------------------------------------------------- behaviour (real model)
@pytest.mark.parametrize("text,expected", [
    ("dant mein bahut dard hai", "dentistry"),
    ("aankh laal hai aur khujli", "ophthalmology"),
    ("periods time pe nahi aa rahe", "gynecology"),
    ("baal jhad rahe hain", "dermatology"),
    ("neend nahi aati, tension rehta hai", "psychiatry"),
    ("toothache and swollen gums", "dentistry"),
])
def test_routes_common_complaints(service, text, expected):
    assert service.route(text).candidates[0].specialtyId == expected


@pytest.mark.parametrize("text", [
    "seene mein dard ho raha hai", "seene mein dard kam nahi ho raha", "saans nahi aa raha",
    "सीने में दर्द हो रहा है", "daura pad raha hai, jhatke aa rahe hain", "bachche ne zeher pi liya",
    "pregnancy mein khoon aa raha hai", "I want to kill myself",
])
def test_emergency_flag(service, text):
    e = service.route(text).emergency
    assert e.flag and e.reasons and e.recommendedSpecialtyIds[0] == "emergencymedicine"


def test_self_harm_recommends_psychiatry(service):
    e = service.route("mujhe khud ko maarne ka mann karta hai").emergency
    assert e.flag and "psychiatry" in e.recommendedSpecialtyIds


@pytest.mark.parametrize("text", ["dant mein bahut dard hai", "baal jhad rahe hain", "acne pimples treatment"])
def test_no_emergency_for_routine(service, text):
    assert service.route(text).emergency.flag is False


@pytest.mark.parametrize("text", ["asdkjh qwe", "hello", "...", "12345"])
def test_non_medical_abstains(service, text):
    d = service.route(text)
    assert d.decision == Decision.ABSTAIN and d.noMatch and d.specialtyIds == []


def test_output_invariants(service):
    for text in ("pet mein dard aur ulti", "ghutne mein dard", "pishab mein jalan", "dant dard"):
        d = service.route(text, top_k=5)
        confs = [c.confidence for c in d.candidates]
        assert confs == sorted(confs, reverse=True) or d.notes      # policy reorder is noted
        assert all(0 <= c <= 1 for c in confs) and all(c.specialtyId in SPECIALTY_NAMES for c in d.candidates)
        assert all(e.specialtyId == c.specialtyId for c in d.candidates for e in c.evidence)
        if d.decision == Decision.ANSWER:
            assert 1 <= len(d.specialtyIds) <= 3
        if d.decision == Decision.CLARIFY:
            assert d.clarify is not None and len(d.specialtyIds) == RoutingPolicy().hedge_size


def test_reason_quotes_only_patient_words(service):
    d = service.route("dant mein bahut dard hai")
    quoted = [w.strip("'") for w in d.reason.split() if w.startswith("'")]
    assert quoted and all(w.strip("',") in "dant mein bahut dard hai" for w in quoted)


def test_hinglish_reply_language(service):
    d = service.route("pet mein dard aur ulti")
    assert d.clarify and d.clarify.kind == "age" and d.clarify.question.startswith("Yeh bachche")
