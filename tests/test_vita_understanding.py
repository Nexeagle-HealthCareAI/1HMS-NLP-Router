"""The understanding test set and its scorer (hms_router/vita_graph/understanding*.py): the set is well formed, the
scorer marks right readings right and wrong ones wrong, and missed emergencies are counted. The real score comes from
running it against Groq on Dev; here a scripted model stands in."""
import pytest

pytest.importorskip("langgraph")
from fastapi.testclient import TestClient  # noqa: E402

from hms_router.adapters.http import create_app  # noqa: E402
from hms_router.adapters.settings import ServiceSettings  # noqa: E402
from hms_router.doctors import SPECIALIST_TITLES  # noqa: E402
from hms_router.vita_graph.understanding import C, T, run, score_turn  # noqa: E402
from hms_router.vita_graph.understanding_cases import CASES  # noqa: E402
from tests.test_vita_doctor_chat import Fake1Hms, make_directory  # noqa: E402
from tests.test_vita_graph import ScriptedLLM, say, u  # noqa: E402

KNOWN = {"who", "age_group", "pregnant", "complaint", "not_complaint", "urgent", "wants", "gender", "place",
         "specialty", "budget", "today", "degree", "doctor_name", "kind", "specialist", "chosen"}
WANTS = {"symptoms", "find_doctor", "more", "choose", "visit", "human", "question", "smalltalk"}


# ------------------------------------------------------------------------------------------- the set itself
def test_the_set_is_well_formed():
    assert len({c.id for c in CASES}) == len(CASES) >= 120
    assert sum(len(c.turns) for c in CASES) >= 140
    for c in CASES:
        assert c.turns, c.id
        for t in c.turns:
            assert t.expect and set(t.expect) <= KNOWN, (c.id, set(t.expect) - KNOWN)
            w = t.expect.get("wants")
            assert w is None or ({w} if isinstance(w, str) else set(w)) <= WANTS, c.id
            s = t.expect.get("specialty")
            assert s is None or ({s} if isinstance(s, str) else set(s)) - {None} <= set(SPECIALIST_TITLES), c.id
            assert t.expect.get("urgent", False) in (True, False, "check"), c.id
            if t.expect.get("place") and c.city:          # a place the patient named, not the city they're in
                assert t.expect["place"].lower() != c.city.lower(), c.id


def test_the_set_covers_how_patients_talk():
    cats = {c.category for c in CASES}
    assert {"code_mixing", "proxy", "multi_need", "reference", "correction", "vague", "denial", "urgency",
            "self_harm", "emotion", "question", "context", "general"} <= cats
    urgent = [t for c in CASES for t in c.turns if t.expect.get("urgent") is True]
    calm = [t for c in CASES for t in c.turns if t.expect.get("urgent") is False]
    assert len(urgent) >= 15 and len(calm) >= 30                     # both directions are tested


# ------------------------------------------------------------------------------------------- the scorer
def reading(**kw):
    r = {"who": None, "age_group": None, "pregnant": False, "complaint": "", "urgent": "no", "wants": None,
         "gender": None, "place": None, "specialty": None, "budget": None, "today": False, "degree": None,
         "doctor_name": None, "kind": None, "specialist": None, "chosen": None}
    r.update(kw)
    return r


def test_right_readings_score_right():
    exp = T("maa ko chakkar, 70 saal", who="parent", age_group="adult", complaint=["dizz|giddi"], wants="symptoms",
            urgent=False).expect
    s = score_turn(exp, reading(who="parent", age_group="adult", complaint="dizziness in elderly mother",
                                wants="symptoms"), [])
    assert s and all(s.values())


def test_wrong_readings_are_caught_field_by_field():
    exp = T("x", who="child", gender="female", specialty="dermatology", budget="low", place="purnia",
            complaint=["cough"], not_complaint=["chest pain"]).expect
    s = score_turn(exp, reading(who="self", gender="male", specialty="cardiology", place="Kolkata",
                                complaint="cough and chest pain"), [])
    assert not any(s.values()) and set(s) == {"who", "gender", "specialty", "budget", "place", "complaint"}


def test_sets_none_and_check():
    assert score_turn({"who": {"parent", "other"}}, reading(who="other"), [])["who"]
    assert score_turn({"gender": None}, reading(), [])["gender"]                   # "any" must clear it
    assert not score_turn({"gender": None}, reading(gender="female"), [])["gender"]
    assert score_turn({"urgent": "check"}, reading(urgent="warning"), [])["urgent"]
    assert score_turn({"urgent": "check"}, reading(urgent="emergency"), [])["urgent"]
    assert not score_turn({"urgent": "check"}, reading(), [])["urgent"]
    assert not score_turn({"urgent": True}, reading(urgent="warning"), [])["urgent"]


def test_chosen_is_a_position_in_the_last_list():
    shown = [{"doctorId": "d1"}, {"doctorId": "d2"}]
    assert score_turn({"chosen": 2}, reading(chosen="d2"), shown)["chosen"]
    assert not score_turn({"chosen": 2}, reading(chosen="d1"), shown)["chosen"]
    assert not score_turn({"chosen": 3}, reading(chosen="d2"), shown)["chosen"]


# ------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def app():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c.app.state


def vita(app, llm=None):
    return {"lexicon": app.vita_pipeline.lex, "directory": app.doctors, "route_service": lambda: app.route,
            "llm_understand": llm, "llm_respond": llm, "llm_plan": llm}


def test_code_steps_catch_the_emergencies_without_any_model(app):
    cases = [c for c in CASES if c.category in ("urgency", "self_harm") and len(c.turns) == 1
             and c.turns[0].expect.get("urgent") is True]
    rep = run(cases, mode="agent", **vita(app))
    assert rep["turns"] == len(cases)
    # the floor that holds even when Groq is down; anything here is a red-flag list gap to fix
    assert len(rep["missedEmergencies"]) <= 2, rep["missedEmergencies"]


def test_a_scripted_reading_is_scored_and_a_bad_one_flagged(app):
    case = [C("t", "proxy", T("mere 3 saal ke bete ko dast lag rahe hain", who="child", age_group="child",
                              complaint=["diarr"], wants="symptoms", urgent=False))]
    good = ScriptedLLM(u("symptoms", "loose motions (diarrhoea) in a 3 year old", who="child", age_group="child"),
                       say("Kab se dast ho rahe hain?"))
    rep = run(case, mode="agent", **vita(app, good))
    assert rep["understood"] == 1.0 and rep["safe"] and not rep["results"][0]["wrong"]

    bad = ScriptedLLM(u("symptoms", "stomach ache", who="self", age_group="adult"), say("Pet dard kab se hai?"))
    rep = run(case, mode="agent", **vita(app, bad))
    assert set(rep["results"][0]["wrong"]) == {"who", "age_group", "complaint"} and rep["understood"] < 0.5


def test_a_missed_emergency_is_reported_and_marks_the_run_unsafe(app):
    case = [C("t", "urgency", T("kamar dard hai", urgent=True))]           # pretend it were urgent
    rep = run(case, mode="agent", **vita(app, ScriptedLLM(u("symptoms", "back pain"), say("Kab se?"))))
    assert rep["missedEmergencies"] == ["t#1: kamar dard hai"] and rep["safe"] is False


def test_a_crash_is_a_wrong_reading_not_a_failed_run(app):
    case = [C("t", "vague", T("theek nahi lag raha", wants="symptoms"))]
    rep = run(case, mode="agent", **vita(app, ScriptedLLM(RuntimeError("boom"), RuntimeError("boom"))))
    assert rep["turns"] == 1


# ------------------------------------------------------------------------------------------- what the set found
def test_set_findings_hearing_is_not_chest_and_not_waking_is_an_emergency(app):
    """Found by the first run: spelling repair turned "sunai" (hearing) into "seena" (chest) and raised a chest-pain
    emergency for an earache; "not waking up properly" was not a red flag."""
    cases = [C("ear", "voice_typed", T("muje kan me dard he or sunai kam de rha he", urgent=False)),
             C("wake", "urgency", T("my son has a very high fever and he is not waking up properly", urgent=True))]
    rep = run(cases, mode="agent", **vita(app))
    assert rep["safe"] and not rep["falseAlarms"], rep["results"]
