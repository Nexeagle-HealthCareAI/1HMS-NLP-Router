"""Regression tests for the two 2026-10-19 fixes on /route-symptom:
1. explicit denials ("saans lene mein koi dikkat nahi") must not raise an emergency, real emergencies still must;
2. queries with no health words ("cricket score today") are abstained, low-match medical queries are not."""
import pytest

from hms_router.application.route_symptom import RouteSymptomService
from hms_router.domain import Decision, RoutingPolicy
from hms_router.domain.models import EmergencyAssessment, Query
from hms_router.rag.vocabulary import MedicalVocabulary
from hms_router.safety.denial import DenialAware, strip_denials
from hms_router.textnorm import preprocess
from tests.conftest_route import FakeDetector, FakeRetriever, FakeScorer, ev


# ------------------------------------------------------------------ 1. denials
@pytest.mark.parametrize("text", [
    "saans lene mein koi dikkat nahi",
    "saans lene mein koi dikkat nahi hai bas khansi hai",
    "सांस लेने में कोई दिक्कत नहीं",
    "I don't have chest pain, just cough",
    "i do not have any breathing problem",
    "seene mein koi dard nahi hai sirf khansi",
])
def test_denials_do_not_flag(service, text):
    assert service.route(text).emergency.flag is False


@pytest.mark.parametrize("text", [
    "saans nahi aa raha",                                    # bare nahi = cannot breathe
    "saans lene mein dikkat ho rahi hai",
    "seene mein dard hai, saans ki dikkat nahi",             # denial of one thing, chest pain still present
    "kal se koi dikkat nahi thi aaj seene mein tez dard",
    "never had chest pain before, now severe chest pain",
    "I don't get chest pain when walking but fainted today",
    "chest pain ho raha hai",
    "behosh ho gaya",
    "I fainted",
])
def test_real_emergencies_still_flag(service, text):
    assert service.route(text).emergency.flag is True


@pytest.mark.parametrize("text, kept", [
    ("saans lene mein koi dikkat nahi hai bas khansi", "bas khansi"),
    ("dard nahi ja raha", "dard nahi ja raha"),              # persistence, not denial
    ("i don t have any feeling in my left arm", "i don t have any feeling in my left arm"),  # loss, not denial
    ("saans nahi aa raha", "saans nahi aa raha"),
])
def test_strip_denials(text, kept):
    assert strip_denials(text)[0] == kept


def test_denial_aware_passes_untouched_query_through():
    inner = FakeDetector(flag=True)
    q = Query(raw="seene mein dard", normalized="seene mein dard")
    assert DenialAware(inner).detect(q).flag and inner.calls == 1


def test_denial_aware_fully_denied_text_does_not_call_inner():
    inner = FakeDetector(flag=True)
    q = Query(raw="koi dikkat nahi", normalized="koi dikkat nahi")
    assert DenialAware(inner).detect(q) == EmergencyAssessment(flag=False) and inner.calls == 0


# ------------------------------------------------------------------ 2. off-topic
@pytest.fixture(scope="module")
def vocab():
    return MedicalVocabulary.from_data()


@pytest.mark.parametrize("text", ["cricket score today", "weather in delhi tomorrow", "asdfgh qwerty", "xyz",
                                  "good morning", "book movie tickets", "stock market today"])
def test_no_health_words(vocab, text):
    assert not vocab.has_signal(preprocess(text))


@pytest.mark.parametrize("text", ["my tummy hurts after eating spicy food", "cut my finger deep and bleeding",
                                  "left ear itchy after swimming", "bache ka vaccine lagwana hai",
                                  "child not growing well", "my son is coughing", "stomache", "pet mein dard"])
def test_health_words_found(vocab, text):
    assert vocab.has_signal(preprocess(text))


@pytest.mark.parametrize("text", ["cricket score today", "weather in delhi tomorrow", "asdfgh qwerty", "xyz"])
def test_service_abstains_off_topic(service, text):
    d = service.route(text)
    assert d.decision == Decision.ABSTAIN and d.specialtyIds == [] and d.noMatch


@pytest.mark.parametrize("text", ["my tummy hurts after eating spicy food", "cut my finger deep and bleeding",
                                  "left ear itchy after swimming", "cant hear properly in left ear",
                                  "feeling anxious and cant sleep at night", "want to lose weight diet plan"])
def test_service_keeps_low_match_medical(service, text):
    assert service.route(text).decision != Decision.ABSTAIN


def _svc(signal, flag=False, sim=0.45):
    return RouteSymptomService(retriever=FakeRetriever([ev("1", "orthopedics", sim)]),
                               scorer=FakeScorer({"orthopedics": 0.9, "general": 0.1}),
                               detector=FakeDetector(flag), medical_signal=signal)


def test_gate_abstains_without_signal_and_notes_why():
    d = _svc(lambda _t: False).route("cricket score")
    assert d.decision == Decision.ABSTAIN and "no health terms in the query" in d.notes


def test_gate_never_hides_an_emergency():
    d = _svc(lambda _t: False, flag=True).route("cricket score")
    assert d.decision != Decision.ABSTAIN and d.emergency.flag


def test_gate_allows_a_strong_corpus_match():
    assert _svc(lambda _t: False, sim=0.9).route("x y z words").decision != Decision.ABSTAIN


def test_gate_failure_treats_query_as_medical():
    def boom(_t):
        raise RuntimeError("vocab broken")
    assert _svc(boom).route("knee").decision != Decision.ABSTAIN


def test_policy_off_topic_rule():
    p = RoutingPolicy()
    assert p.is_off_topic(False, 0.4, False)
    assert not p.is_off_topic(True, 0.4, False)
    assert not p.is_off_topic(False, 0.4, True)
    assert not p.is_off_topic(False, p.no_signal_min_match, False)
