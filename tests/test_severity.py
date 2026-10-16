import pytest

from hms_router.severity import (EMERGENCY_RULES, SeverityClassifier, band, build_features, duration_days, emergency_check,
                      FEATURE_NAMES)

clf = SeverityClassifier()


@pytest.mark.parametrize("score,level", [(0, "MILD"), (30, "MILD"), (31, "MODERATE"), (60, "MODERATE"),
                                         (61, "SEVERE"), (85, "SEVERE"), (86, "EMERGENCY"), (100, "EMERGENCY")])
def test_bands_match_doc(score, level):
    assert band(score)[0] == level


def test_mild_and_moderate_and_severe_cases():
    assert clf.classify({"symptoms": ["cough"], "intensity": "mild"}).level == "MILD"
    assert clf.classify({"symptoms": ["fever", "cough"]}).level == "MODERATE"
    r = clf.classify({"symptoms": ["stomach pain", "vomiting", "fever"]})
    assert r.level == "SEVERE" and not r.emergency and r.alert is None


def test_intensity_and_duration_raise_score():
    base = clf.classify({"symptoms": ["fever"]}).score
    assert clf.classify({"symptoms": ["fever"], "intensity": "severe"}).score > base
    assert clf.classify({"symptoms": ["fever"], "duration": "for 5 days"}).score > base


@pytest.mark.parametrize("text", [
    "seene mein dard ho raha hai", "seena me dard", "saans nahi aa raha", "behosh ho gaya",
    "zyada khoon aa raha hai", "laqwa ho gaya", "muh tedha ho gaya", "aankhon ke aage andhera",
    "haath pair sune ho gaye", "sar mein achanak tez dard", "ulti mein khoon", "i have chest pain",
])
def test_every_doc_emergency_phrase_fires_from_text_alone(text):
    r = clf.classify({"symptoms": []}, [text])
    assert r.emergency and r.level == "EMERGENCY" and r.triggered_by == "safety_net" and r.score >= 90
    assert r.alert and "112" in r.alert


def test_typo_still_caught():
    assert clf.classify({"symptoms": []}, ["seene mein dardd"]).emergency
    assert clf.classify({"symptoms": []}, ["behos ho gaya"]).emergency


def test_canonical_symptom_channel_fires_without_text():
    assert clf.classify({"symptoms": ["chest pain"]}).emergency


def test_explicit_denial_suppresses_text_hit():
    r = clf.classify({"symptoms": [], "negated_symptoms": ["chest pain"]}, ["seene mein dard nahi hai"])
    assert not r.emergency


def test_no_false_alarm_on_ordinary_text():
    assert not clf.classify({"symptoms": ["fever"]}, ["kal se bukhar hai aur sir dard"]).emergency
    assert not clf.classify({"symptoms": []}, ["mujhe seene me jalan nahi hai pet dard hai"]).emergency


def test_appendicitis_red_flag_is_severe_not_emergency():
    r = clf.classify({"symptoms": ["stomach pain", "fever"], "intensity": "moderate"})
    assert r.level == "SEVERE" and r.score >= 70 and r.red_flags and not r.emergency


class Fixed:
    def __init__(self, v): self.v = v
    def predict(self, X): return [self.v]


class Broken:
    def predict(self, X): raise RuntimeError("model down")


def test_ml_can_raise_but_never_lower():
    st = {"symptoms": ["fever", "cough"]}
    base = clf.classify(st).score
    assert SeverityClassifier(Fixed(5)).classify(st).score == base
    r = SeverityClassifier(Fixed(95)).classify(st)
    assert r.emergency and r.triggered_by == "model" and r.ml_score == 95


def test_safety_net_beats_a_low_ml_score_and_survives_a_broken_model():
    assert SeverityClassifier(Fixed(1)).classify({"symptoms": ["chest pain"]}).emergency
    r = SeverityClassifier(Broken()).classify({"symptoms": []}, ["behosh ho gaya"])
    assert r.emergency and any("unavailable" in f for f in r.factors)


def test_rule_score_alone_never_reaches_emergency():
    r = clf.classify({"symptoms": list(__import__("hms_router.severity", fromlist=["x"]).SYMPTOM_WEIGHTS), "intensity": "severe", "duration": "for 40 days"})
    assert r.level == "SEVERE" and not r.emergency


def test_helpers():
    assert duration_days("for 2 weeks") == 14 and duration_days(None) is None
    assert len(build_features({"symptoms": ["fever"]})) == len(FEATURE_NAMES)
    assert emergency_check({"symptoms": []}) == [] and len(EMERGENCY_RULES) >= 10
