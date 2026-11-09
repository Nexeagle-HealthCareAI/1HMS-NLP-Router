"""Builds the Postman collection for Vita, the patient chat agent (POST /v2/agent/*), and its environments.

    python scripts/make_vita_postman.py        -> postman/collections/Vita-Chat-Agent.postman_collection.json
                                                  postman/environments/Vita-{Dev,Local}.postman_environment.json

Each folder is one conversation, run top to bottom (Collection Runner or newman): the first request starts a new
conversation, the next ones carry {{sessionId}}; the first doctor shown is kept in {{doctorId}}.
tests/test_postman_collection.py plays every request against the app, so the collection stays in step with the API.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "postman"
NS = uuid.UUID("6f1c2a52-4c5e-4f43-9a51-6a7f0a5e9b11")       # stable ids: the same file every run

# ------------------------------------------------------------------------------------------------ test scripts
CHAT_TEST = """pm.test("200 OK", () => pm.response.to.have.status(200));
const r = pm.response.json();
if (r.session_id) pm.collectionVariables.set("sessionId", r.session_id);
const docs = (r.doctors && r.doctors.items) || [];
if (docs.length) pm.collectionVariables.set("doctorId", docs[0].doctorId);
pm.test("Vita answered (" + r.kind + ", engine " + r.engine + ")", () => pm.expect(r.message).to.be.a("string").and.not.empty);
if (r.degraded) console.warn("degraded:", r.degraded, "- is Groq on? GET /health");"""
NEW_CONVERSATION = 'pm.collectionVariables.unset("sessionId");'

CHECKS = {
    "emergency": """pm.test("emergency, 108 first", () => { pm.expect(r.kind).to.eql("emergency"); pm.expect(r.message).to.include("108"); });""",
    "not_emergency": """pm.test("not an emergency", () => pm.expect(r.kind).to.not.eql("emergency"));""",
    "distress": """pm.test("108 and a question, not the full emergency", () => { pm.expect(r.message).to.include("108"); pm.expect(r.kind).to.not.eql("emergency"); });""",
    "warning": """pm.test("warning-sign question first", () => pm.expect((r.clarify || {}).kind).to.eql("warning_signs"));""",
    "doctor_list": """pm.test("doctors listed", () => pm.expect(docs.length).to.be.above(0));""",
    "selected": """pm.test("doctor chosen", () => pm.expect(r.kind).to.eql("doctor_selected"));""",
    "hinglish": """pm.test("answers in Hinglish (the patient asked)", () => pm.expect(r.language).to.eql("hinglish"));""",
    "english": """pm.test("answers in English (the patient asked)", () => pm.expect(r.language).to.eql("english"));""",
    "whatsapp": """pm.test("ready-to-send WhatsApp messages", () => pm.expect(r.whatsapp).to.be.an("array").that.is.not.empty);""",
    "handover": """pm.test("booking: Vita books, or hands over to the bot", () => { pm.expect(r.kind).to.eql("doctor_selected"); if (!r.booking) pm.expect(r.handover).to.eql({to: "booking", doctorId: r.selectedDoctorId}); });""",
    "no_menu": """pm.test("a goodbye has nothing to tap", () => pm.expect(r.clarify).to.not.be.ok);""",
}
OK_TEST = """pm.test("200 OK", () => pm.response.to.have.status(200));"""


def _id(*parts: str) -> str:
    return str(uuid.uuid5(NS, "/".join(parts)))


def request(name: str, method: str, path: str, *, body: dict | None = None, test: str = OK_TEST, pre: str = "",
            desc: str = "", query: dict | None = None, auth: str | None = None) -> dict:
    raw = "{{baseUrl}}" + path + ("?" + "&".join(f"{k}={v}" for k, v in query.items()) if query else "")
    url = {"raw": raw, "host": ["{{baseUrl}}"], "path": [p for p in path.strip("/").split("/") if p]}
    if query:
        url["query"] = [{"key": k, "value": v} for k, v in query.items()]
    req: dict = {"method": method, "header": [], "url": url, "description": desc}
    if body is not None:
        req["header"].append({"key": "Content-Type", "value": "application/json"})
        req["body"] = {"mode": "raw", "raw": json.dumps(body, ensure_ascii=False, indent=2),
                       "options": {"raw": {"language": "json"}}}
    if auth:
        req["auth"] = {"type": "apikey", "apikey": [{"key": "key", "value": "X-Api-Key", "type": "string"},
                                                    {"key": "value", "value": "{{" + auth + "}}", "type": "string"},
                                                    {"key": "in", "value": "header", "type": "string"}]}
    events = [{"listen": "test", "script": {"type": "text/javascript", "exec": test.split("\n")}}]
    if pre:
        events.insert(0, {"listen": "prerequest", "script": {"type": "text/javascript", "exec": pre.split("\n")}})
    return {"id": _id(name, method, path), "name": name, "event": events, "request": req}


def say(name: str, text: str | None = None, *, first: bool = False, checks: tuple = (), desc: str = "",
        **extra) -> dict:
    """One message to Vita. first: starts a new conversation (no session_id)."""
    body: dict = {} if first else {"session_id": "{{sessionId}}"}
    if text is not None:
        body["text"] = text
    body.update(extra)
    body.setdefault("city", "{{city}}")
    test = CHAT_TEST + "".join("\n" + CHECKS[c] for c in checks)
    return request(name, "POST", "/v2/agent/chat", body=body, test=test, pre=NEW_CONVERSATION if first else "",
                   desc=desc)


def folder(name: str, desc: str, items: list[dict], auth: str | None = None) -> dict:
    f = {"id": _id("folder", name), "name": name, "description": desc, "item": items}
    if auth:
        f["auth"] = {"type": "apikey", "apikey": [{"key": "key", "value": "X-Api-Key", "type": "string"},
                                                  {"key": "value", "value": "{{" + auth + "}}", "type": "string"},
                                                  {"key": "in", "value": "header", "type": "string"}]}
    return f


MODEL = " Needs the model (Groq): without it Vita answers from the fallback (`engine: fallback`)."
WA = {"channel": "whatsapp"}


def build() -> dict:
    items = [
        folder("0. System", "Is the router up, and is Vita's model on? `vita.llm` should be `groq` and "
               "`breakerOpen` false. If `llm` is null, Vita answers from its fallback only.", [
                   request("Health (Vita model, mode, breaker)", "GET", "/health", test=OK_TEST + """
const v = pm.response.json().vita || {};
pm.test("Vita's model is on (" + v.llm + " " + v.model + ", mode " + v.mode + ")", () => pm.expect(v.llm).to.be.ok);
pm.test("breaker closed", () => pm.expect(v.breakerOpen).to.not.eql(true));"""),
                   request("Ready", "GET", "/ready"),
               ]),
        folder("1. Symptoms to the right doctor", "A patient describes symptoms; Vita asks at most what it needs, "
               "names the kind of doctor and lists doctors; the patient chooses one." + MODEL, [
                   say("Fever for 3 days", "mujhe 3 din se bukhar hai", first=True),
                   say("And body ache", "body pain bhi hai, aur kuch nahi"),
                   say("Choose the first doctor shown", "pehla wala"),
               ]),
        folder("2. Doctor search and refining", "Plain doctor searches work even without the model. Filters "
               "carry over: lady doctor, cheaper, more, then choose by tapping (select_doctor_id).", [
                   say("Skin doctor", "skin doctor chahiye", first=True, checks=("doctor_list",)),
                   say("Only lady doctors", "lady doctor chahiye"),
                   say("Cheaper first", "sasta wala"),
                   say("Show more", "aur dikhao"),
                   say("Tap Choose on a doctor", "Choose", select_doctor_id="{{doctorId}}", checks=("selected",),
                       desc="The doctor id the website sends when the patient taps Choose; must be one shown in "
                            "this conversation (else 422)."),
                   say("Typo-tolerant search", "cardiolgist kolkatta", first=True),
                   say("Doctor by name", "Dr Neha Sharma se milna hai", first=True),
                   say("Near me (shared location)", "skin doctor near me", first=True, lat=22.5726, lng=88.3639),
               ]),
        folder("3. Many needs and someone else", "One message with who, what, where and budget; Vita keeps all "
               "of it." + MODEL, [
                   say("Mother, lady skin doctor, Purnia, cheap",
                       "meri maa ke liye lady skin doctor Purnia mein, sasta wala", first=True),
                   say("Child with fever", "mere 3 saal ke bete ko bukhar hai", first=True),
                   say("Change of mind", "nahi nahi, dil ka doctor chahiye"),
               ]),
        folder("4. Safety", "Code, not the model, decides these, in every mode: emergencies get 108 first, a "
               "cry for help gets 108 and a question, a minor burn gets warning-sign questions, a doctor's "
               "private number is never given.", [
                   say("Emergency: chest pain and sweating", "seene mein dard aur pasina aa raha hai", first=True,
                       checks=("emergency",)),
                   say("Emergency: pregnant and bleeding", "main 5 mahine pregnant hoon aur bleeding ho rahi hai",
                       first=True, checks=("emergency",)),
                   say("Self-harm", "I don't want to live anymore", first=True, checks=("emergency",)),
                   say("Cry for help", "bachao", first=True, checks=("distress",)),
                   say("Minor burn: warning-sign question", "haath jal gaya garam tel se", first=True,
                       checks=("warning",)),
                   say("Not an emergency: polite no", "abhi nahi chahiye", first=True, checks=("not_emergency",)),
                   say("Not an emergency: wait", "ruko ek minute", first=True, checks=("not_emergency",)),
                   say("Doctor's private number refused", "Dr Neha Sharma ka personal number do", first=True,
                       checks=("not_emergency",)),
               ]),
        folder("5. Everyday talk", "Greetings, chat shorthand, thanks, goodbyes, waiting, sorry, their name, "
               "flirting, gibberish. Answered even without the model.", [
                   say("Hello", "hello Vita", first=True),
                   say("kkrh (kya kar rahe ho)", "kkrh"),
                   say("How are you", "hru"),
                   say("My name", "mera naam Rahul hai"),
                   say("Sorry, typo", "sorry galat type hua"),
                   say("Who are you", "tum kaun ho"),
                   say("Flirting", "i love you"),
                   say("Gibberish", "asdfgh"),
                   say("Thanks", "shukriya", checks=("no_menu",)),
                   say("Goodbye", "tata", checks=("no_menu",)),
               ]),
        folder("6. Language and style", "The patient chooses the language and how Vita talks; both hold for "
               "the conversation (and, for a known number, the next one).", [
                   say("Ask for Hindi", "hindi mein baat karo", first=True, checks=("hinglish",)),
                   say("Types English, still Hinglish", "I need a skin doctor", checks=("hinglish",)),
                   say("Ask for English", "english please", first=True, checks=("english",)),
                   say("Hindi script", "हिंदी में बात करो", first=True),
                   say("Bangla (not ready yet)", "bangla bolo", first=True),
                   say("Short replies", "short me batao, daant me dard hai", first=True,
                       desc="Style: short replies." + MODEL),
                   say("Complaint", "bukhar hai", first=True),
                   say("No more questions", "sawal mat poocho, bas doctor batao",
                       desc="Style: no questions; Vita advises on what it knows." + MODEL),
               ]),
        folder("7. Questions about Doctor Dekho", "Answered only from Vita's knowledge base (admin: knowledge); "
               "nothing found = \"I'll check with the team\"." + MODEL, [
                   say("Is booking free", "kya booking free hai?", first=True),
                   say("Ayushman card", "ayushman card chalta hai", first=True),
                   say("Video consult", "video call pe dikha sakte hain?", first=True),
                   say("Fees", "fees kitni hai", first=True),
                   say("Not in the knowledge base", "parking hai kya", first=True),
               ]),
        folder("8. WhatsApp channel", "What the WhatsApp bot sends: `channel: whatsapp`; the reply carries "
               "`whatsapp`, ready-to-send Cloud API messages. Taps come back as `reply_id` exactly as Vita wrote "
               "them: doc:<doctorId>, more, refine:<fee|female|today|near|experience>, say:<text>. "
               "`verified_mobile` needs a key with the verified_mobile scope ({{waApiKey}}).", [
                   say("Skin doctor (WhatsApp)", "skin doctor", first=True, checks=("whatsapp", "doctor_list"), **WA,
                       verified_mobile="{{verifiedMobile}}"),
                   say("Tap: lady doctors", None, reply_id="refine:female", checks=("whatsapp",), **WA,
                       verified_mobile="{{verifiedMobile}}"),
                   say("Tap: more", None, reply_id="more", checks=("whatsapp",), **WA,
                       verified_mobile="{{verifiedMobile}}"),
                   say("Tap a doctor row (booking hand-over)", None, reply_id="doc:{{doctorId}}",
                       checks=("whatsapp", "handover"), **WA, verified_mobile="{{verifiedMobile}}",
                       desc="Router booking off: `handover: {to: booking}` and the bot starts its own booking flow."),
                   say("Shared location", None, first=True, lat=25.7771, lng=87.4753, checks=("whatsapp",), **WA,
                       verified_mobile="{{verifiedMobile}}"),
                   say("Emergency on WhatsApp", "saans nahi aa rahi", first=True, checks=("whatsapp", "emergency"),
                       **WA, verified_mobile="{{verifiedMobile}}"),
               ], auth="waApiKey"),
        folder("9. Session, feedback, memory", "What Vita remembers about a conversation, thumbs up/down, and "
               "forgetting a patient.", [
                   say("Start a conversation", "skin doctor chahiye", first=True),
                   request("Get session", "GET", "/v2/agent/session/{{sessionId}}"),
                   request("Feedback: thumbs up", "POST", "/v2/agent/feedback",
                           body={"session_id": "{{sessionId}}", "rating": "up", "comment": "helpful"}),
                   request("Forget session", "DELETE", "/v2/agent/session/{{sessionId}}"),
                   request("Forget a patient (verified mobile)", "DELETE", "/v2/agent/patient-memory",
                           body={"verified_mobile": "{{verifiedMobile}}"}, auth="waApiKey",
                           desc="Or {\"patient_token\": \"...\"}: the 1HMS patient JWT."),
               ]),
        folder("10. Search bar (search-intent)", "One query from the website's search bar -> what Vita "
               "understands (specialty ids, city, doctor name, emergency...). Key scope: route.", [
                   request("Specialty with typos", "POST", "/v2/agent/search-intent",
                           body={"query": "cardiolgist kolkatta"}, auth="routeApiKey"),
                   request("Symptom", "POST", "/v2/agent/search-intent",
                           body={"query": "pet me jalan", "city": "{{city}}"}, auth="routeApiKey"),
                   request("Doctor name", "POST", "/v2/agent/search-intent",
                           body={"query": "Dr Neha Sharma"}, auth="routeApiKey"),
                   request("Lady doctor near me today", "POST", "/v2/agent/search-intent",
                           body={"query": "lady doctor near me aaj"}, auth="routeApiKey"),
                   request("Child (answer to an age question)", "POST", "/v2/agent/search-intent",
                           body={"query": "bukhar aur khansi", "context": {"ageGroup": "child"}}, auth="routeApiKey"),
                   request("Emergency in the search bar", "POST", "/v2/agent/search-intent",
                           body={"query": "seene mein tez dard"}, auth="routeApiKey"),
               ]),
        folder("11. Admin: control, review, examples, knowledge", "Key scope: admin. Mode switch (agent / "
               "planner / off, per channel), conversation review, approved examples and the promotion gate, and "
               "the knowledge base.", [
                   request("Get mode", "GET", "/v2/agent/admin/control"),
                   request("Mode: agent everywhere", "PUT", "/v2/agent/admin/control", body={"mode": "agent"}),
                   request("Mode: planner on WhatsApp only", "PUT", "/v2/agent/admin/control",
                           body={"mode": "agent", "channels": {"whatsapp": "planner"}}),
                   request("Mode: off (kill switch)", "PUT", "/v2/agent/admin/control", body={"mode": "off"},
                           desc="Code steps and the fallback only. Run 'Mode: agent everywhere' after."),
                   request("Mode: back to agent", "PUT", "/v2/agent/admin/control", body={"mode": "agent"}),
                   say("A conversation to review", "daant me dard hai", first=True),
                   request("Recent conversations", "GET", "/v2/agent/admin/conversations", query={"limit": "20"}),
                   request("Booked / emergency / thumbs-down only", "GET", "/v2/agent/admin/conversations",
                           query={"outcome": "emergency", "limit": "20"}),
                   request("One conversation, turn by turn", "GET", "/v2/agent/admin/conversations/{{sessionId}}"),
                   request("Examples (draft + active)", "GET", "/v2/agent/admin/examples"),
                   request("Approve turn 0 as an example", "POST", "/v2/agent/admin/examples",
                           body={"session_id": "{{sessionId}}", "turn": 0, "note": "good triage"},
                           test=OK_TEST + '\nconst e = pm.response.json(); if (e.id) pm.collectionVariables.set("exampleId", e.id);'),
                   request("Correct a reply (write the example)", "POST", "/v2/agent/admin/examples",
                           body={"patient": "daant me bahut dard", "reply": "Daant ke dard ke liye Dentist sahi "
                                 "rahenge. Neeche doctor hain.", "specialist": "Dentist", "note": "correction"}),
                   request("Remove an example", "DELETE", "/v2/agent/admin/examples/{{exampleId}}"),
                   request("Promote drafts (runs the scenario suite)", "POST", "/v2/agent/admin/examples/promote",
                           desc="Makes drafts live only if the scenario suite is not worse. Slow with the model."),
                   request("Knowledge: all facts", "GET", "/v2/agent/admin/knowledge"),
                   request("Knowledge: search as Vita would", "GET", "/v2/agent/admin/knowledge",
                           query={"q": "Sunday ko khula hai", "hospital": "Sadar Hospital Purnia"}),
                   request("Knowledge: add a hospital fact", "POST", "/v2/agent/admin/knowledge",
                           body={"title": "Sadar Hospital Purnia OPD timings",
                                 "text": "OPD runs Monday to Saturday, 9 am to 2 pm. Closed on Sunday.",
                                 "scope": "Sadar Hospital Purnia", "tags": "timing, sunday, khula, opd, kitne baje"},
                           test=OK_TEST + '\nconst k = pm.response.json(); if (k.id) pm.collectionVariables.set("knowledgeId", k.id);'),
                   request("Knowledge: remove that fact", "DELETE", "/v2/agent/admin/knowledge/{{knowledgeId}}"),
               ], auth="adminApiKey"),
        folder("12. Related", "Endpoints Vita builds on, and the old /chat alias.", [
                   request("Route symptom (specialty engine)", "POST", "/v1/route-symptom",
                           body={"query": "3 din se bukhar aur sar dard", "topK": 3}, auth="routeApiKey"),
                   request("Taxonomy (specialty ids)", "GET", "/v1/taxonomy", auth="routeApiKey"),
                   request("Model info", "GET", "/model-info", auth="routeApiKey"),
                   request("Deprecated: POST /chat (answers with Vita)", "POST", "/chat",
                           body={"text": "skin doctor chahiye", "city": "{{city}}"},
                           desc="Kept while Doctor Dekho's /api/vita/app-chat still calls it. Use /v2/agent/chat."),
               ]),
    ]
    return {
        "info": {"_postman_id": _id("collection"), "name": "Vita Chat Agent (1HMS NLP Router)",
                 "description": (
                     "Vita, Doctor Dekho's patient chat agent: `POST /v2/agent/chat` and the endpoints around it.\n\n"
                     "**Setup**: pick the environment (Vita Dev / Vita Local) and fill the keys: `apiKey` (scope "
                     "chat), `waApiKey` (chat + verified_mobile, the WhatsApp bot's key), `routeApiKey` (route), "
                     "`adminApiKey` (admin). Locally, with HMS_API_KEYS empty, auth is off and any value works.\n\n"
                     "**Conversations**: each folder runs top to bottom. The first request of a conversation "
                     "starts fresh; the next ones send `{{sessionId}}`, saved from each reply, and the first "
                     "doctor shown is saved in `{{doctorId}}`.\n\n"
                     "**Model**: check folder 0 first. With Groq on, `vita.llm` is `groq`; otherwise Vita answers "
                     "from its fallback (`engine: fallback`), and the requests marked 'Needs the model' won't "
                     "show triage.\n\n"
                     "Generated by scripts/make_vita_postman.py; tests/test_postman_collection.py plays every "
                     "request against the app."),
                 "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
        "auth": {"type": "apikey", "apikey": [{"key": "key", "value": "X-Api-Key", "type": "string"},
                                              {"key": "value", "value": "{{apiKey}}", "type": "string"},
                                              {"key": "in", "value": "header", "type": "string"}]},
        "variable": [{"key": k, "value": v} for k, v in (
            ("sessionId", ""), ("doctorId", ""), ("exampleId", ""), ("knowledgeId", ""))],
        "item": items,
    }


def environment(name: str, base_url: str) -> dict:
    vals = [("baseUrl", base_url, "default"), ("apiKey", "", "secret"), ("waApiKey", "", "secret"),
            ("routeApiKey", "", "secret"), ("adminApiKey", "", "secret"), ("city", "Kolkata", "default"),
            ("verifiedMobile", "919876543210", "default")]
    return {"id": _id("env", name), "name": name, "_postman_variable_scope": "environment",
            "values": [{"key": k, "value": v, "type": t, "enabled": True} for k, v, t in vals]}


def main() -> None:
    (OUT / "collections").mkdir(parents=True, exist_ok=True)
    (OUT / "environments").mkdir(parents=True, exist_ok=True)
    files = {OUT / "collections" / "Vita-Chat-Agent.postman_collection.json": build(),
             OUT / "environments" / "Vita-Dev.postman_environment.json": environment(
                 "Vita Dev", "http://151.185.45.77:5003"),
             OUT / "environments" / "Vita-Local.postman_environment.json": environment(
                 "Vita Local", "http://localhost:5003")}
    for path, data in files.items():
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
