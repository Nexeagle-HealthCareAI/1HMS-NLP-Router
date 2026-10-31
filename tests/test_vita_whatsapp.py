"""WhatsApp contract: Vita must stay usable on WhatsApp, whatever is added for other channels.

The WhatsApp bot (Whatsapp-Webhook) will send `channel: "whatsapp"`, send each message in `whatsapp` as it is,
and send a tapped row / button back as `reply_id`. These tests hold that contract:

  1. every reply Vita can give, in every language, renders to messages WhatsApp accepts (validate())
  2. every id Vita writes leads somewhere when tapped (round trips)
  3. the shapes the bot will rely on stay fixed (emergency = one text with 108, doctors = one list, ...)
  4. extreme content (long names, long text, ten doctors, duplicate names) still fits the limits

A change that breaks any of these fails CI before it can reach WhatsApp patients.
"""
import pytest

from hms_router.channels.whatsapp import (BODY_MAX, ROW_DESC_MAX, ROW_TITLE_MAX, read_reply_id, render, validate)
from tests.test_vita_doctor_chat import URL, chat, client, hms  # noqa: F401 (pytest fixtures)

WA = {"channel": "whatsapp"}


def wa(client, text=None, **kw):
    body = {**WA, **({"text": text} if text is not None else {}), **kw}
    r = client.post(URL, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["whatsapp"], "a WhatsApp reply always carries at least one message"
    for msg in out["whatsapp"]:
        assert validate(msg) == [], (msg, validate(msg))
    return out


def tap(client, reply, row_id, **kw):
    return wa(client, session_id=reply["session_id"], reply_id=row_id, **kw)


def rows(msg):
    return [r for s in msg["interactive"]["action"]["sections"] for r in s["rows"]]


def ids(out):
    msg = out["whatsapp"][0]
    it = msg.get("interactive") or {}
    if it.get("type") == "list":
        return [r["id"] for r in rows(msg)]
    if it.get("type") == "button":
        return [b["reply"]["id"] for b in it["action"]["buttons"]]
    return []


# ============================================================================ 1. every reply renders
SCENARIOS = [
    ("skin ka doctor chahiye", {"city": "Kolkata"}),
    ("I need a skin doctor", {"city": "Kolkata"}),
    ("मुझे स्किन डॉक्टर चाहिए", {"city": "Kolkata"}),
    ("paas mein cardiologist", {}),                                    # location request
    ("near me cardiologist", {"lat": 22.58, "lng": 88.40}),
    ("dr sharma", {"city": "Kolkata"}),                               # which one?
    ("Dr Anita Sen", {"city": "Kolkata"}),
    ("MD medicine doctor", {"city": "Kolkata"}),
    ("lady skin doctor", {"city": "Kolkata"}),
    ("aaj available cardiologist", {"city": "Kolkata"}),
    ("Sadar Hospital Purnia ke doctor", {}),
    ("Zyx Hospital ke doctor", {}),                                    # nothing found
    ("is Dr Neha Sharma available on Monday?", {}),                   # book offer
    ("dr zzyzxq", {"city": "Kolkata"}),
    ("hello", {}), ("नमस्ते", {}), ("namaste ji", {}),                  # what kind of doctor?
    ("doctor chahiye", {}),
    ("bas dekh raha tha", {}), ("just looking, thanks", {}),
    ("kisi insaan se baat karni hai", {}), ("I want to talk to a person", {}),
    ("Dr Anita Sen ka personal number do", {"city": "Kolkata"}),
    ("3 din se bukhar aur sar dard hai", {"city": "Kolkata"}),        # triage question
    ("I have had fever and headache for 3 days", {"city": "Kolkata"}),
    ("मुझे तीन दिन से बुखार और सिर दर्द है", {"city": "Kolkata"}),
    ("seene mein tez dard aur saans nahi aa rahi", {}),               # emergency
    ("सीने में तेज़ दर्द और साँस नहीं आ रही", {}),
    ("severe chest pain and I can't breathe", {}),
    ("আমার জ্বর হয়েছে", {}),                                           # Bangla
]


@pytest.mark.parametrize("text, extra", SCENARIOS, ids=[s[0][:30] for s in SCENARIOS])
def test_every_reply_renders_to_messages_whatsapp_accepts(client, hms, text, extra):
    wa(client, text, **extra)


def test_multi_turn_triage_stays_valid(client, hms):
    from hms_router.vita_graph.engine import VitaRuntime
    from tests.test_vita_graph import ScriptedLLM, say, u
    a = client.app.state
    a.vita_runtime = VitaRuntime(ScriptedLLM(u("symptoms", "fever and headache for 3 days"), say("Khansi bhi hai?"),
                                             u("symptoms", "fever, headache, mild cough, 3 days", enough=True),
                                             say("General Physician ko dikhayein.")), a.redis,
                                 route_service=lambda: a.route)
    q = wa(client, "3 din se bukhar aur sar dard hai", city="Kolkata")
    rec = wa(client, "haan thodi khasi bhi", session_id=q["session_id"], city="Kolkata")
    assert rec["kind"] == "recommendation" and rec["whatsapp"]
    a.vita_runtime = VitaRuntime(None, a.redis, route_service=lambda: a.route)


def test_web_replies_do_not_carry_whatsapp(client, hms):
    out = chat(client, "skin ka doctor chahiye", city="Kolkata")
    assert "whatsapp" not in out


# ============================================================================ 2. every id leads somewhere
def test_doctor_row_chooses_that_doctor(client, hms):
    lst = wa(client, "skin ka doctor chahiye", city="Kolkata")
    doc = next(i for i in ids(lst) if i.startswith("doc:"))
    picked = tap(client, lst, doc)
    assert picked["kind"] == "doctor_selected" and picked["selectedDoctorId"] == doc[4:]


def test_more_row_shows_the_next_page(client, hms):
    hms_dir = client.app.state.doctors
    hms_dir.limit = 1
    try:
        lst = wa(client, "dermatologist", city="Kolkata")
        assert "more" in ids(lst)
        nxt = tap(client, lst, "more", city="Kolkata")
        assert nxt["kind"] == "doctor_list" and nxt["doctors"]["page"] == 2
    finally:
        hms_dir.limit = 3


@pytest.mark.parametrize("text, lang", [("I need a skin doctor", "english"), ("skin ka doctor chahiye", "hinglish"),
                                        ("मुझे स्किन डॉक्टर चाहिए", "hindi")])
def test_refine_rows_narrow_without_changing_the_language(client, hms, text, lang):
    lst = wa(client, text, city="Kolkata")
    assert "refine:female" in ids(lst)
    lady = tap(client, lst, "refine:female", city="Kolkata")
    assert lady["language"] == lang and lady["doctors"]["filters"].get("gender") == "female"
    cheap = tap(client, lady, "refine:fee", city="Kolkata")
    assert cheap["language"] == lang and cheap["doctors"]["filters"].get("sort") == "fee"


def test_which_doctor_rows_pick_one(client, hms):
    which = wa(client, "dr sharma", city="Kolkata")
    assert which["clarify"]["kind"] == "which_doctor"
    doc = next(i for i in ids(which) if i.startswith("doc:"))
    assert tap(client, which, doc)["kind"] == "doctor_selected"


def test_book_offer_buttons_book_or_decline(client, hms):
    offer = wa(client, "is Dr Neha Sharma available on Monday?")
    assert offer["whatsapp"][0]["interactive"]["type"] == "button"
    yes, no = ids(offer)
    assert yes.startswith("doc:") and no.startswith("say:")
    assert tap(client, offer, yes)["kind"] == "doctor_selected"
    offer2 = wa(client, "is Dr Neha Sharma available on Monday?")
    assert tap(client, offer2, ids(offer2)[1])["kind"] == "info"


def test_location_request_then_shared_location(client, hms):
    ask = wa(client, "paas mein cardiologist")
    assert ask["whatsapp"][0]["interactive"]["type"] == "location_request_message"
    found = wa(client, session_id=ask["session_id"], lat=22.58, lng=88.40)      # no text: just the location
    assert found["kind"] == "doctor_list" and found["language"] == "hinglish"


def test_ask_kind_options_become_a_list(client, hms):
    out = wa(client, "doctor chahiye")
    assert out["whatsapp"][0]["interactive"]["type"] == "list" and len(ids(out)) == 5
    nxt = tap(client, out, ids(out)[1])                                         # "Specialty chunein" as text
    assert nxt["kind"] in ("follow_up", "doctor_list", "recommendation")


def test_text_reply_id_or_location_is_required(client, hms):
    r = client.post(URL, json={**WA})
    assert r.status_code == 422


def test_a_stale_doctor_row_is_a_clean_422(client, hms):
    lst = wa(client, "skin ka doctor chahiye", city="Kolkata")
    r = client.post(URL, json={**WA, "session_id": lst["session_id"],
                               "reply_id": "doc:00000000-0000-0000-0000-00000000dead"})
    assert r.status_code == 422                                                 # the bot shows "please search again"


# ============================================================================ 3. shapes the bot relies on
@pytest.mark.parametrize("text", ["seene mein tez dard aur saans nahi aa rahi", "severe chest pain and I can't breathe",
                                  "सीने में तेज़ दर्द और साँस नहीं आ रही"])
def test_emergency_is_one_text_with_108_and_nothing_to_tap(client, hms, text):
    out = wa(client, text)
    assert [m["type"] for m in out["whatsapp"]] == ["text"]
    assert "108" in out["whatsapp"][0]["text"]["body"].splitlines()[0]


def test_doctor_list_is_one_list_doctors_first(client, hms):
    out = wa(client, "skin ka doctor chahiye", city="Kolkata")
    msg = out["whatsapp"][0]
    assert len(out["whatsapp"]) == 1 and msg["interactive"]["type"] == "list"
    first = msg["interactive"]["action"]["sections"][0]["rows"]
    assert [r["id"][4:] for r in first] == [d["doctorId"] for d in out["doctors"]["items"]]
    assert msg["interactive"]["action"]["button"] == "Doctor chunein"
    assert "'Doctor chunein'" in msg["interactive"]["body"]["text"]           # the text names the real button
    assert "'Choose'" not in msg["interactive"]["body"]["text"]


def test_hindi_list_words_are_hindi(client, hms):
    out = wa(client, "मुझे स्किन डॉक्टर चाहिए", city="Kolkata")
    action = out["whatsapp"][0]["interactive"]["action"]
    assert action["button"] == "डॉक्टर चुनें" and action["sections"][0]["title"] == "डॉक्टर"


def test_selected_doctor_is_text_for_the_bot_to_take_over(client, hms):
    lst = wa(client, "skin ka doctor chahiye", city="Kolkata")
    picked = tap(client, lst, ids(lst)[0])
    assert [m["type"] for m in picked["whatsapp"]] == ["text"]
    assert picked["doctor"]["doctorId"] == picked["selectedDoctorId"]           # booking starts from this


# ============================================================================ 4. extreme content
def _fake_reply(n=12, name="Dr. Venkataraghavan Subramaniam Iyer", message="x " * 1800):
    items = [{"doctorId": f"00000000-0000-0000-0000-{i:012d}", "name": name,
              "specialtyName": "Cardiothoracic and Vascular Surgeon (Heart)", "consultFee": 1500.5,
              "hospital": {"name": "Shri Ramakrishna Institute of Medical Sciences and Research Centre"},
              "distanceKm": 12.345, "availableToday": True} for i in range(n)]
    return {"kind": "doctor_list", "language": "english", "message": message,
            "doctors": {"items": items, "hasMore": True, "filters": {}}, "clarify": None}


def test_ten_long_duplicate_doctors_still_fit():
    msg = render(_fake_reply())[0]
    assert validate(msg) == []
    r = rows(msg)
    assert len(r) == 10 and all(len(x["title"]) <= ROW_TITLE_MAX and len(x.get("description", "")) <= ROW_DESC_MAX for x in r)
    assert len(msg["interactive"]["body"]["text"]) <= BODY_MAX


def test_long_text_and_long_options_fit():
    reply = {"kind": "follow_up", "language": "hindi", "message": "बहुत लंबा संदेश। " * 300,
             "clarify": {"kind": "ask_kind", "options": [{"id": f"o{i}", "label": "बहुत लंबा विकल्प का नाम जो बीस से ज़्यादा है"}
                                                         for i in range(14)]}}
    msg = render(reply)[0]
    assert validate(msg) == [] and msg["interactive"]["type"] == "list"
    assert validate(render({"kind": "info", "message": "y" * 9000})[0]) == []


def test_validate_catches_what_whatsapp_rejects():
    bad = {"type": "interactive", "interactive": {"type": "button", "body": {"text": "hi"}, "action": {"buttons": [
        {"type": "reply", "reply": {"id": "a", "title": "This title is far too long"}},
        {"type": "reply", "reply": {"id": "b", "title": "Same"}}, {"type": "reply", "reply": {"id": "c", "title": "Same"}},
        {"type": "reply", "reply": {"id": "d", "title": "Four"}}]}}}
    problems = validate(bad)
    assert any("4 buttons" in p for p in problems) and any("not unique" in p for p in problems)
    assert any("button title" in p for p in problems)


@pytest.mark.parametrize("rid, lang, expect", [
    ("doc:00000000-0000-0000-0000-000000000003", "english",
     {"text": "Choose", "select_doctor_id": "00000000-0000-0000-0000-000000000003"}),
    ("more", "hindi", {"text": "और डॉक्टर दिखाओ"}),
    ("refine:today", "hinglish", {"text": "inme se aaj available"}),
    ("say:No, just checking", "english", {"text": "No, just checking"}),
    ("something-else", "english", {"text": "something-else"}),
])
def test_read_reply_id(rid, lang, expect):
    assert read_reply_id(rid, lang) == expect
