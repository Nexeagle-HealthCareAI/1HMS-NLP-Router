"""End-to-end: all layers through HMSRouterAgent. These pin the bugs found when the separately-built steps were joined."""
import fakeredis
import pytest

from hms_router import HMSRouterAgent


@pytest.fixture
def agent():
    return HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True))


def say(agent, *msgs):
    sid, rep = None, None
    for m in msgs:
        rep = agent.handle(m, sid)
        sid = rep.session_id
    return rep


# ----------------------------------------------------------------- the doc's own flows
def test_doc_three_turn_flow_asks_then_recommends(agent):
    r1 = say(agent, "pet dard ho raha hai")
    assert r1.kind == "follow_up" and r1.question == "Koi aur symptoms hain?"
    r2 = say(agent, "pet dard ho raha hai", "aur ulti bhi aa rahi hai")
    assert r2.kind == "recommendation" or r2.question == "Kitne time se?"


def test_doc_three_turn_flow_end_state(agent):
    r = say(agent, "pet dard ho raha hai", "aur ulti bhi aa rahi hai", "subah se, aur bukhaar bhi hai")
    assert r.kind == "recommendation"
    assert set(r.state.symptoms) == {"stomach pain", "vomiting", "fever"} and r.state.duration == "since morning"
    roles = [(o.role, o.specialist) for o in r.card.outputs]
    assert roles[0] == ("PRIMARY", "Gastroenterologist") and len(roles) == 3
    assert roles[2][0] == "ALTERNATIVE" and roles[2][1] in ("General Physician", "Telemedicine consult")
    assert len({s for _, s in roles}) == 3                                   # three DISTINCT outputs


def test_doc_unique_keyword_example(agent):
    r = say(agent, "subah uthke chakkar aaya, aur aankhon ke aage andhera chha gaya")
    lv = {k.canonical: k.level for k in r.keywords if k.canonical}
    assert lv["dizziness"] == "HIGH" and lv["vision loss"] == "CRITICAL" and lv["morning"] == "TEMPORAL"
    assert r.kind == "emergency"                                             # doc: "Vision loss ... EMERGENCY flag"


def test_doc_emergency_table_every_phrase(agent):
    for phrase in ["seene mein dard", "saans nahi aa raha", "behosh ho gaya", "zyada khoon aa raha",
                   "laqwa", "muh tedha", "aankhon ke aage andhera", "haath pair sune ho gaye",
                   "sar mein achanak tez dard", "ulti mein khoon"]:
        assert say(HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True)), phrase).kind == "emergency", phrase


def test_doc_appendicitis_row_is_severe_not_emergency(agent):
    r = say(agent, "pet mein kaafi dard aur bukhar bhi hai")
    assert r.severity.level == "SEVERE" and not r.severity.emergency
    assert any("Appendicitis" in f for f in r.severity.red_flags)


# ----------------------------------------------------------------- safety net: variants that used to slip through
@pytest.mark.parametrize("text", [
    "sine mein dard", "saans nai aa raha", "khoon ki ulti ho rahi hai", "I can't breathe",
    "achanak se sar mein bahut tez dard hua", "सीने में दर्द हो रहा है", "साँस नहीं आ रही",
    "mere papa behosh ho gaye", "seene me dard ho raha hai", "sans nahi aa rahi hai", "chest pain",
    "vomiting blood", "chhaati mein dard",
])
def test_emergency_variants_are_caught(text):
    r = say(HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True)), text)
    assert r.kind == "emergency" and r.severity.emergency, text


@pytest.mark.parametrize("text", ["halka sir dard hai", "bukhar nahi hai, khansi hai", "seene mein dard nahi hai",
                                  "no chest pain, only mild cough"])
def test_benign_or_denied_is_not_emergency(text):
    r = say(HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True)), text)
    assert r.kind != "emergency", text


def test_emergency_never_triggers_follow_up_question(agent):
    for text in ["sar mein achanak tez dard", "bahut zyada khoon beh raha hai", "ulti mein khoon"]:
        r = say(HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True)), text)
        assert r.kind == "emergency" and r.question is None and r.state.ready, text


def test_emergency_on_later_turn_after_benign_start(agent):
    assert say(agent, "bukhar hai").kind == "follow_up"
    assert say(agent, "bukhar hai", "ab saans nahi aa raha").kind == "emergency"


def test_denial_in_later_turn_removes_the_symptom(agent):
    r = say(agent, "seene mein dard nahi hai", "bas halka bukhar hai")
    assert r.kind != "emergency" and "chest pain" not in r.state.symptoms


def test_safety_net_survives_nlp_crash(agent, caplog):
    def boom(*a, **k):
        raise RuntimeError("model down")
    agent.pipeline.extractor.extract = boom
    r = say(agent, "mujhe seene mein dard ho raha hai")
    assert r.kind == "emergency" and r.degraded == ["nlp"]


def test_nlp_crash_keeps_symptoms_spelling_already_found(agent):
    def boom(*a, **k):
        raise RuntimeError
    agent.pipeline.extractor.extract = boom
    r = say(agent, "mujhe bukhar hai")
    assert r.degraded == ["nlp"] and "fever" in r.state.symptoms


def test_sudden_severe_headache_gets_its_canonical_symptom(agent):
    r = say(agent, "kal raat se bahut tez sir dard ho raha hai")           # the doc's Step 3 example: NOT sudden
    assert r.state.symptoms == ["headache"] and r.state.duration == "since last night" and r.state.intensity == "severe"
    r = say(HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True)), "sar mein achanak tez dard")
    assert r.state.symptoms == ["sudden severe headache"] and r.kind == "emergency"


# ----------------------------------------------------------------- misc wiring
def test_reply_language_follows_user(agent):
    assert "Symptom summary" in say(agent, "I have fever and cough for 3 days", "no other").message
    r = say(HMSRouterAgent(fakeredis.FakeRedis(decode_responses=True)), "seene mein dard")
    assert "Turant 108" in r.message


def test_unknown_token_queue_gets_original_word_not_our_guess(agent):
    say(agent, "mujhe ghabrahat ho rahi hai")
    import json
    queued = [json.loads(x)["token"] for x in agent.r.lrange("review_queue", 0, -1)]
    assert "ghabrahat" in queued and "ghabrana" not in queued


def test_noise_tokens_are_not_queued(agent):
    say(agent, "bas bukhar hai 2")
    assert agent.r.lrange("review_queue", 0, -1) == []
