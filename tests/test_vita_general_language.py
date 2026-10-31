"""General language: the everyday things patients type that are not a symptom or a doctor ask (greetings, chat
shorthand, thanks, goodbyes, "ruko", "sorry", their name, "test", flirting, off-topic, "hindi mein baat karo").

Vita must never read them as a symptom or an emergency, must answer them warmly even without the model, and must keep
to the language the patient asks for. Found by throwing ~190 such messages at Vita (31 Oct 2026)."""
import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from hms_router.spelling import SpellingCorrector
from hms_router.textnorm import preprocess
from tests.test_vita_doctor_chat import Fake1Hms, make_directory

EVERYDAY = """hi|hello ji|namaste|salaam|assalamualaikum|adaab|ram ram|jai shree ram|sat sri akal|good morning|kaise ho|
kya haal chaal|sab theek?|how r u|kemon acho|tum theek ho?|kaisa chal raha hai|kkrh|kya kr rhi ho|wyd|kya chal raha hai|
whats up|kahan ho|aur batao|or bta|thanks|thank u|shukriya|dhanyavaad|bahut shukriya|dhonnobad|meherbani|🙏|bye|
chalo bye|tc|take care|alvida|phir milte|see you|good night|tata|khuda hafiz|allah hafiz|baad mein baat karte|ok|hmm|
theek hai|achha ji|done|sahi|👍|nahi abhi nahi|abhi nahi chahiye|baad mein|kal dekhte hain|zarurat nahi|rehne do|
bas itna hi|kuch nahi bas aise hi|bas timepass|samjha nahi|kya?|???|repeat karo|bakwas|faltu ho|dimag kharab mat karo|
wah|superb|zabardast|love it|tum kaun ho|tum bot ho?|ladki ho ya ladka|tum doctor ho?|kya kar sakti ho|help|ruko|
ek minute|wait|hold on|1 min|thoda ruko|abhi aata hu|sorry|oops|galti se likha|sorry galat type hua|maaf karna|
mera naam Rahul hai|my name is Priya|main Anjali bol rahi hoon|test|testing|hello hello|check|koi hai|joke sunao|
cricket score|gana sunao|tum khana khaogi|i love you|shaadi karogi|tum cute ho|date pe chalogi|send pic"""
EVERYDAY = [t.strip() for t in EVERYDAY.replace("\n", "").split("|")]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


def chat(client, text, sid=None):
    r = client.post("/v2/agent/chat", json={"text": text, "session_id": sid, "city": "Kolkata"})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("text", EVERYDAY)
def test_everyday_talk_gets_an_answer_without_the_model(client, text):
    out = chat(client, text)
    assert out["kind"] not in ("emergency", "doctor_list", "recommendation"), (text, out["message"])
    assert not out.get("degraded"), (text, out["message"])                 # never "try again in a minute / 108"
    assert "didn't get that" not in out["message"] and "samajh nahi paayi. Apni takleef likhiye" not in out["message"]


@pytest.mark.parametrize("text,said", [("tata", ("Take care", "dhyan")), ("ruko", ("take your time", "aaram se")),
                                       ("dhanyavaad", ("welcome", "swagat")), ("test", ("I'm here", "main yahin")),
                                       ("i love you", ("AI assistant", "AI assistant")), ("sorry", ("No problem",
                                                                                                  "Koi baat nahi"))])
def test_the_answer_fits(client, text, said):
    msg = chat(client, text)["message"]
    assert any(s.lower() in msg.lower() for s in said), msg


def test_their_name_is_greeted_not_read_as_a_symptom(client):
    assert "Rahul!" in chat(client, "mera naam Rahul hai")["message"]
    assert "Priya!" in chat(client, "my name is Priya")["message"]


@pytest.mark.parametrize("text", ["done", "hold on", "chalo bye", "kya chal raha hai", "baad mein baat karte",
                                  "sorry galat type hua", "tum khana khaogi", "date pe chalogi", "kya haal chaal"])
def test_everyday_words_are_never_repaired_into_symptoms(client, text):
    """The spelling repair turned "done"/"date" into "dane" (rash), "hold" into "cold", "chal" into "chhale" (mouth
    ulcers), "baad" into "daad" (ringworm), "type" into "tap" (fever), "khaogi" into "khansi" (cough)."""
    sc = SpellingCorrector(client.app.state.vita_pipeline.lex)
    assert sc.correct(preprocess(text)).symptoms == [], text


@pytest.mark.parametrize("text,symptom", [("chalne mein dikkat", "chalne mein dikkat"), ("khasi aa rhi", "cough"),
                                          ("bukhaar hai", "fever"), ("sir dard", "headache"),
                                          ("saas lene me dikkat", "breathlessness")])
def test_real_symptoms_are_still_read(client, text, symptom):
    sc = SpellingCorrector(client.app.state.vita_pipeline.lex)
    assert symptom in sc.correct(preprocess(text)).symptoms


def test_polite_refusal_is_not_an_emergency(client):
    for text in ("abhi nahi chahiye", "zarurat nahi", "no thanks, later"):
        assert chat(client, text)["kind"] != "emergency", text


# ------------------------------------------------------------------------------------------- the patient's language
def test_asking_for_hindi_keeps_hindi_even_when_they_type_english(client):
    first = chat(client, "hello")
    out = chat(client, "hindi mein baat karo", first["session_id"])
    assert out["language"] == "hinglish" and "Hindi mein baat karungi" in out["message"]
    out = chat(client, "I need a skin doctor", out["session_id"])
    assert out["language"] == "hinglish" and "skin ke doctor" in out["message"]


def test_asking_for_english_keeps_english(client):
    first = chat(client, "mujhe skin doctor chahiye")
    out = chat(client, "speak in english", first["session_id"])
    assert out["language"] == "english" and out["message"].startswith("Sure, I'll talk in English")
    assert chat(client, "dil ka doctor", out["session_id"])["language"] == "english"


def test_asking_in_hindi_script_gets_hindi_script(client):
    out = chat(client, "हिंदी में बात करो")
    assert out["language"] == "hindi" and "हिंदी में बात" in out["message"]


def test_bangla_request_says_bangla_is_not_ready(client):
    assert chat(client, "bangla bolo")["language"] == "bangla"


@pytest.mark.parametrize("text", ["english bolne wale doctor chahiye", "hindi bolne wala doctor"])
def test_a_doctor_who_speaks_a_language_is_neither_a_language_switch_nor_a_place_nor_bone(client, text):
    out = chat(client, text)
    assert "Sure, I'll talk" not in out["message"] and "baat karungi" not in out["message"]
    assert (out.get("doctors") or {}).get("filters", {}).get("city") is None
    assert "Orthopaedic" not in out["message"]


@pytest.mark.parametrize("text", ["ruko ek minute", "ek minute ruko", "abhi nahi chahiye", "nahi, aur kuch nahi"])
def test_a_model_only_flag_on_everyday_talk_is_not_an_emergency(client, text):
    """The emergency model alone flagged these (0.62-0.73) without naming any sign."""
    assert chat(client, text)["kind"] != "emergency"


@pytest.mark.parametrize("text", ["hosh nahi hai", "accident ho gaya", "emergency", "saans nahi aa rahi", "behosh"])
def test_real_emergencies_still_are(client, text):
    assert chat(client, text)["kind"] == "emergency", text


@pytest.mark.parametrize("text", ["bachao", "help please", "jaldi aao", "koi madad karo jaldi", "please help"])
def test_a_cry_for_help_gets_the_number_and_a_question_even_without_the_model(client, text):
    out = chat(client, text)
    assert "108" in out["message"] and out["engine"] == "step" and not out.get("degraded"), out["message"]


def test_plain_help_is_still_how_to_use_vita(client):
    assert "108" not in chat(client, "help")["message"]
