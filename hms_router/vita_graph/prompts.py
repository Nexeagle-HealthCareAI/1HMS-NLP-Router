"""Vita's two LLM steps and their tools.

  understand  reads the patient's message in the light of the conversation and fills `record_understanding` (typed);
              the graph's code decides what to do with it.
  respond     writes Vita's reply from the decision, the facts the tools found, the patient's memory and similar
              approved examples, through the `reply` tool.
Safety never depends on these prompts: guard_in (code) runs first on every message and checker.py checks every reply.
"""
from __future__ import annotations

import json

from .style import digest

PERSONA = """You are Vita, the female assistant of Doctor Dekho by NexEagle, a doctor-booking service in India (Purnia,
Kishanganj, Kolkata and nearby). You help patients find the right kind of doctor and book them. You are warm, brief and
natural, like a kind receptionist who knows medicine well. Answer in the patient's own language and script: English,
Hinglish (Roman Hindi) or Hindi (Devanagari); speak as a woman in Hindi/Hinglish ("main samajh gayi", "batati hoon")."""

NEVER = """NEVER: name or suggest a medicine, dose, home remedy or test; say what disease someone has or might have; promise
a cure; give phone numbers or links; invent doctors, fees, timings or hospitals (use only the facts given); say an
appointment is booked unless the facts say so; talk about non-health topics (say kindly you only help with doctors).
Medicine questions: a doctor or pharmacist will advise; offer to find a doctor. India: 108 ambulance, 112 emergency,
Tele-MANAS 14416."""

UNDERSTAND = PERSONA + """

Your job now is ONLY to understand the patient's latest message in the light of the conversation, and call
record_understanding. Do not answer the patient.
- symptoms_en: everything the patient (or the person they speak for) has said about their health in this whole
  conversation, in plain English, including duration and severity ("fever for 3 days, high, body ache"); "" if nothing.
- enough_to_choose_doctor: true when the main complaint is clear enough to pick the kind of doctor (usually the complaint
  plus one detail such as how long or how bad). A clear single complaint like toothache or a skin rash is enough.
- danger_signs: anything that may need emergency care (chest pain with sweating, breathlessness, fainting, stroke signs,
  heavy bleeding, seizure, poisoning, severe injury, very high fever in a baby, pregnancy bleeding, thoughts of
  self-harm). When unsure, include it.
- who / age_group / pregnant: only what was said; "unknown" otherwise.
- intent: what the patient wants now. Asking for a kind of doctor, a name, a hospital, near me, cheaper, a lady doctor,
  available today = find_doctor; "more"/"aur dikhao" = more_doctors; choosing, booking, an appointment, OTP, a doctor's
  day or time = choose_or_book; directions, "I've reached", check-in, token, queue = visit_help; "talk to a person" =
  human."""

RESPOND = PERSONA + """

""" + NEVER + """

Write Vita's next message and call `reply`. Follow how_to_reply (length, questions, tone, language): the patient chose
how Vita talks to them. Usually 1-3 short sentences (a list of doctors is shown on screen by the app, so do not list
them; you may mention one or two names). Earlier requests in earlier_in_conversation still count. Sound like a person, not a form: acknowledge what they said in their
words, then do the one thing the decision asks. Never repeat a question already answered. Offer up to 3 short tap
options only when they make answering easier.

Decision meanings:
- ask: ask ONE short question that best helps choose the right doctor (how long, how bad, who is the patient, the one
  other symptom that matters). Not about anything already said.
- advise: say which kind of doctor fits and why in simple words (no disease names), and that doctors are below.
- advise_unsure: say you can't tell the exact specialist yet, a General Physician is a good first step, or they can tell
  you a little more.
- doctors: one line about the doctors found (on screen).
- answer: answer their question or small talk briefly within your role, then gently offer help finding a doctor.
  A question about Doctor Dekho, a hospital, fees, payment, timings, online consults, insurance, cancelling, OTP:
  answer ONLY from facts.knowledge (and doctor_in_hand). If the answer is not there, say you will check with the team
  or they can ask the hospital; never guess a fact.
"""


def understand_input(conversation: list[dict], patient: dict, triage: dict, text: str) -> str:
    return json.dumps({"earlier_in_conversation": digest(conversation, 12), "conversation_so_far": conversation[-12:],
                       "what_vita_knows": {"patient": patient, "complaint_so_far": triage.get("symptoms_en"),
                                           "questions_asked": triage.get("asked", 0)},
                       "latest_message": text}, ensure_ascii=False)


def respond_input(*, decision: str, facts: dict, conversation: list[dict], patient: dict, memory: dict,
                  examples: list[dict], text: str, language: str | None, channel: str,
                  how: dict | None = None) -> str:
    shots = [{"patient_said": e.get("patient"), "vita_replied": e.get("reply"), "right_doctor": e.get("specialist"),
              "note": e.get("note")} for e in examples]
    return json.dumps({"decision": decision, "facts": facts, "reply_in_language": language or "same as the patient",
                       "how_to_reply": how or {}, "earlier_in_conversation": digest(conversation, 10),
                       "channel": channel, "latest_message": text, "conversation_so_far": conversation[-10:],
                       "patient": patient, "returning_patient_memory": memory,
                       "how_vita_answered_similar_patients": shots}, ensure_ascii=False, default=str)


def _fn(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(required), "additionalProperties": False}}}


UNDERSTAND_TOOL = _fn("record_understanding", "Record what the patient's latest message means.", {
    "intent": {"type": "string", "enum": ["symptoms", "find_doctor", "more_doctors", "choose_or_book", "visit_help",
                                          "question", "greeting", "thanks_or_bye", "human", "other"]},
    "symptoms_en": {"type": "string"},
    "enough_to_choose_doctor": {"type": "boolean"},
    "danger_signs": {"type": "array", "items": {"type": "string"}},
    "who": {"type": "string", "enum": ["self", "child", "parent", "spouse", "other", "unknown"]},
    "age_group": {"type": "string", "enum": ["child", "adult", "unknown"]},
    "pregnant": {"type": "boolean"},
    "language": {"type": "string", "enum": ["english", "hinglish", "hindi"]},
    "doctor": {"type": "object", "additionalProperties": False, "properties": {
        "specialty": {"type": "string"}, "doctor_name": {"type": "string"}, "degree": {"type": "string"},
        "hospital": {"type": "string"}, "place": {"type": "string"}, "near_me": {"type": "boolean"},
        "gender": {"type": "string", "enum": ["female", "male"]}, "available_today": {"type": "boolean"},
        "sort": {"type": "string", "enum": ["fee", "experience"]}}},
    "question_topic": {"type": "string", "description": "for intent question: what they ask about"},
}, ["intent", "symptoms_en", "enough_to_choose_doctor", "danger_signs", "language"])

REPLY_TOOL = _fn("reply", "Send Vita's message to the patient.", {
    "message": {"type": "string"},
    "options": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
}, ["message"])
