"""Vita adapts to how each patient wants to talk (vita_graph/style.py), keeps earlier requests in long conversations,
and finds approved examples by meaning, not spelling (vita_graph/meaning.py)."""
import json

import fakeredis
import pytest

pytest.importorskip("langgraph")
from hms_router.vita_graph import style  # noqa: E402
from hms_router.vita_graph.learning import ExampleStore  # noqa: E402
from hms_router.vita_graph.meaning import concepts  # noqa: E402
from hms_router.vita_graph.memory import PatientMemory  # noqa: E402
from tests.test_vita_graph import client, llm, say, u, vita  # noqa: E402,F401


# ------------------------------------------------------------------------------------------- reading the style
@pytest.mark.parametrize("text,field,value", [
    ("short me batao", "length", "short"), ("seedha batao", "length", "short"), ("keep it short", "length", "short"),
    ("detail mein samjhao", "length", "detailed"), ("explain in detail", "length", "detailed"),
    ("sawal mat poocho", "questions", "avoid"), ("no more questions", "questions", "avoid"),
    ("bas doctor dikhao", "questions", "avoid"), ("just show me the doctors", "questions", "avoid"),
    ("aap please batayein ji", "tone", "formal"), ("yaar tum batao na bhai", "tone", "casual")])
def test_style_requests_are_read(text, field, value):
    assert style.read(text)[field] == value


def test_style_is_kept_until_changed():
    st = style.read("short me batao")
    st = style.read("bukhar hai 3 din se", st)
    assert st["length"] == "short"
    assert style.read("ab detail mein samjhao", st)["length"] == "detailed"


def test_how_to_reply_carries_the_choice_and_the_language():
    how = style.how_to_reply({"length": "short", "questions": "avoid", "tone": "casual"}, "hinglish")
    assert "one short sentence" in how["length"] and "do not ask" in how["questions"]
    assert "hinglish" in how["language"]


def test_digest_keeps_what_was_asked_before_the_window():
    msgs = []
    for i in range(10):
        msgs += [{"role": "patient", "text": f"request {i}"}, {"role": "vita", "text": "ok", "kind": "info"}]
    d = style.digest(msgs, 12)
    assert [x["patient"] for x in d] == [f"request {i}" for i in range(4)] and d[0]["vita_did"] == "info"
    assert style.digest(msgs[:6], 12) == []


# ------------------------------------------------------------------------------------------- in the conversation
def test_no_questions_goes_straight_to_advice(client, llm):
    model = llm(u("symptoms", "fever"), say("Bukhar kab se hai?"),
                u("symptoms", "fever", enough=False), say("Bukhar ke liye General Physician sahi rahenge."))
    first = vita(client, "bukhar hai")
    assert first["kind"] == "follow_up"
    out = vita(client, "sawal mat poocho, bas doctor batao", first["session_id"])
    assert out["kind"] == "recommendation" and out["doctors"]["items"]       # not another question
    told = json.loads(model.seen[3][1].content)
    assert "do not ask" in told["how_to_reply"]["questions"]


def test_short_reply_request_reaches_the_reply_prompt(client, llm):
    model = llm(u("other"), say("Ji."), u("symptoms", "toothache", enough=True), say("Dentist."))
    first = vita(client, "short me batao please")
    vita(client, "daant mein dard", first["session_id"])
    told = json.loads(model.seen[3][1].content)
    assert "one short sentence" in told["how_to_reply"]["length"] and told["how_to_reply"]["tone"].startswith("respect")


def test_language_choice_reaches_the_reply_prompt(client, llm):
    model = llm(u("symptoms", "fever"), say("Bukhar kab se hai?"))
    first = vita(client, "hindi mein baat karo")                              # a code step: no model
    vita(client, "I have fever", first["session_id"])
    told = json.loads(model.seen[1][1].content)
    assert "hinglish" in told["how_to_reply"]["language"] and told["reply_in_language"] == "hinglish"


def test_style_is_remembered_for_the_next_visit():
    mem = PatientMemory(fakeredis.FakeRedis(decode_responses=True))
    mem.update("9876543210", consent=False, style={"length": "short", "questions": "avoid"})
    assert mem.load("9876543210")["style"]["length"] == "short"


# ------------------------------------------------------------------------------------------- examples by meaning
POOL = ["acidity hoti hai khane ke baad", "mere bete ko bukhar hai", "skin doctor chahiye", "thank you so much",
        "sasta doctor dikhao", "kal ka appointment cancel karna hai", "main apni maa ke liye doctor dhoondh raha hoon",
        "chest pain while walking", "सिर में दर्द है", "bye, take care"]


@pytest.mark.parametrize("asked,found", [
    ("pet me jalan ho rahi hai", "acidity hoti hai khane ke baad"), ("chamdi ka doctor", "skin doctor chahiye"),
    ("shukriya", "thank you so much"), ("kam fees wala", "sasta doctor dikhao"),
    ("appointment radd karo", "kal ka appointment cancel karna hai"),
    ("mummy ke liye", "main apni maa ke liye doctor dhoondh raha hoon"),
    ("seene me dard chalne par", "chest pain while walking"), ("sar dard", "सिर में दर्द है"),
    ("alvida", "bye, take care"), ("heartburn after meals", "acidity hoti hai khane ke baad"),
    ("मेरे बेटे को बुखार है", "mere bete ko bukhar hai"), ("fevar hai bachhe ko", "mere bete ko bukhar hai"),
    ("beta 5 saal ka, tez bukhar", "mere bete ko bukhar hai")])
def test_examples_are_found_by_meaning(asked, found):
    store = ExampleStore(fakeredis.FakeRedis(decode_responses=True))
    pool = [{"id": str(i), "patient": p, "reply": "r"} for i, p in enumerate(POOL)]
    assert store.similar(asked, k=1, examples=pool)[0]["patient"] == found


def test_concepts_never_fail_and_name_the_meaning():
    assert concepts("") == frozenset()
    assert {"sym:acidity"} <= concepts("pet me jalan")
    assert {"spec:dermatology"} <= concepts("chamdi ka doctor")
    assert {"who:child", "sym:fever"} <= concepts("mere bete ko bukhar hai")
