import fakeredis
import pytest

from hms_router.accumulator import SymptomAccumulator


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def acc(r):
    return SymptomAccumulator(r)


def test_doc_three_turn_flow(acc):
    s1 = acc.add("s", {"symptoms": ["stomach pain"], "body_parts": ["stomach"]})
    assert (s1.turn, s1.ready, s1.next_question) == (1, False, "Koi aur symptoms hain?")
    s2 = acc.add("s", {"symptoms": ["vomiting"]})
    assert s2.symptoms == ["stomach pain", "vomiting"] and s2.next_question == "Kitne time se?"
    s3 = acc.add("s", {"symptoms": ["fever"], "duration": "since morning"})
    assert s3.symptoms == ["stomach pain", "vomiting", "fever"]
    assert s3.ready and s3.ready_reason == "max_turns" and s3.next_question is None


def test_complete_first_message_needs_no_followups(acc):
    s = acc.add("s", {"symptoms": ["headache"], "duration": "since last night", "intensity": "severe"})
    assert s.ready and s.ready_reason == "complete"


def test_redis_layout_and_sliding_ttl(acc, r):
    acc.add("s", {"symptoms": ["fever"], "language_mix": "hinglish", "intensity": "mild"})
    assert r.llen("session:s:symptoms") == 1 and r.get("session:s:turns") == "1"
    ctx = r.hgetall("session:s:context")
    assert ctx["language"] == "hinglish" and ctx["intensity"] == "mild"
    for k in ("symptoms", "turns", "context"):
        assert 0 < r.ttl(f"session:s:{k}") <= 1800


def test_later_negation_removes_and_mention_readds(acc):
    acc.add("s", {"symptoms": ["fever", "cough"]})
    s = acc.add("s", {"negated_symptoms": ["fever"], "symptoms": []})
    assert s.symptoms == ["cough"] and s.negated_symptoms == ["fever"]
    s = acc.add("s", {"symptoms": ["fever"]})
    assert "fever" in s.symptoms and s.negated_symptoms == []


def test_intensity_is_max_and_duration_is_latest(acc):
    acc.add("s", {"symptoms": ["fever"], "intensity": "severe", "duration": "for 1 day"})
    s = acc.add("s", {"symptoms": ["cough"], "intensity": "mild", "duration": "for 3 days"})
    assert s.intensity == "severe" and s.duration == "for 3 days" and s.durations == ["for 1 day", "for 3 days"]


def test_urgent_symptom_skips_questions(acc):
    s = acc.add("s", {"symptoms": ["chest pain"]})
    assert s.ready and s.ready_reason == "urgent_symptom" and s.next_question is None


def test_user_says_no_more_symptoms(acc):
    acc.add("s", {"symptoms": ["fever"]}, user_text="bukhar hai")           # asks "Koi aur symptoms hain?"
    s = acc.add("s", {"symptoms": []}, user_text="nahi")
    assert s.ready and s.ready_reason == "user_done"


def test_bare_no_is_not_done_unless_answering_the_more_question(acc):
    s = acc.add("s", {"symptoms": []}, user_text="nahi")
    assert not s.ready


def test_questions_not_repeated_then_exhausted(acc):
    acc = SymptomAccumulator(fakeredis.FakeRedis(decode_responses=True), max_turns=10)
    qs = [acc.add("s", {"symptoms": ["fever"]}).next_question for _ in range(3)]
    assert qs == ["Koi aur symptoms hain?", "Kitne time se?", "Takleef kitni hai - halki, theek-thaak ya bahut tez?"]
    s = acc.add("s", {"symptoms": ["fever"]})
    assert s.ready and s.ready_reason == "followups_exhausted"


def test_no_symptoms_keeps_asking_then_flags(acc):
    for _ in range(2):
        assert acc.add("s", {"symptoms": []}).ready_reason == "no_symptoms"
    s = acc.add("s", {"symptoms": []})
    assert not s.ready and s.ready_reason == "no_symptoms_after_max_turns"


def test_external_turn_counter_not_double_counted(acc, r):
    r.set("session:s:turns", 2)                      # another component already counted this turn
    s = acc.add("s", {"symptoms": ["fever"]}, turn=2)
    assert s.turn == 2


def test_expired_session_is_empty(acc, r):
    acc.add("s", {"symptoms": ["fever"]})
    r.delete("session:s:symptoms", "session:s:turns", "session:s:context")
    assert acc.snapshot("s").symptoms == []


def test_buffer_is_capped_and_raw_text_not_stored(r):
    acc = SymptomAccumulator(r, max_buffer=3)
    for i in range(6):
        acc.add("s", {"symptoms": [f"s{i}"], "raw_input": "private text"})
    assert r.llen("session:s:symptoms") == 3
    assert "private text" not in "".join(r.lrange("session:s:symptoms", 0, -1))
