import fakeredis
import pytest

from hms_router import HMSRouterPipeline
from hms_router.ner import parse_duration


@pytest.fixture
def pipe():
    return HMSRouterPipeline(fakeredis.FakeRedis(decode_responses=True))


def obj(pipe, text, sid=None):
    return pipe.process(text, sid).symptom_object


def test_doc_example_matches_schema(pipe):
    o = obj(pipe, "kal raat se bahut tez sir dard ho raha hai").to_dict()
    assert o["symptoms"] == ["headache"] and o["body_part"] == "head"
    assert o["duration"] == "since last night" and o["intensity"] == "severe"
    assert o["language_mix"] == "hinglish" and o["turn"] == 1 and o["session_id"].startswith("sess_")
    assert {"raw_input", "symptoms", "body_part", "duration", "intensity", "language_mix", "session_id", "turn"} <= o.keys()


@pytest.mark.parametrize("words,expected", [
    ("3 din se", "for 3 days"), ("1 hafte se", "for 1 week"), ("subah se", "since morning"),
    ("kal se", "since yesterday"), ("aaj subah se", "since this morning"),
    ("pichle 2 hafte se", "for the past 2 weeks"), ("since yesterday", "since yesterday"),
])
def test_duration(words, expected):
    assert parse_duration(words.split())[0] == expected


def test_number_words_and_generic_pain_with_body_part(pipe):
    o = obj(pipe, "teen din se pet me bahut dard")
    assert o.symptoms == ["stomach pain"] and o.duration == "for 3 days" and o.body_part == "stomach"


def test_negation_hindi_and_english(pipe):
    o = obj(pipe, "bukhar hai lekin ulti nahi")
    assert o.symptoms == ["fever"] and o.negated_symptoms == ["vomiting"]
    o = obj(pipe, "sir dard, no fever")
    assert o.symptoms == ["headache"] and o.negated_symptoms == ["fever"]


def test_nahi_inside_phrase_is_not_negation(pipe):
    assert obj(pipe, "saans nahi aa raha").symptoms == ["breathlessness"]


def test_intensity_levels(pipe):
    assert obj(pipe, "halka bukhar").intensity == "mild"
    assert obj(pipe, "kaafi bukhar").intensity == "moderate"
    assert obj(pipe, "bukhar").intensity is None


def test_turn_counter_carries_through(pipe):
    a = obj(pipe, "pet dard")
    b = obj(pipe, "aur ulti bhi", a.session_id)
    assert (a.turn, b.turn) == (1, 2) and b.symptoms == ["vomiting"]
