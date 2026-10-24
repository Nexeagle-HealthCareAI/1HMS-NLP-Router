"""Vita on WhatsApp: every Vita reply as ready-to-send WhatsApp Cloud API messages, and every tap read back.

The WhatsApp bot (Whatsapp-Webhook) does not need to know Vita's reply kinds. It sends the patient's message
to POST /v2/agent/chat with `channel: "whatsapp"`, sends each message in the reply's `whatsapp` list as it
is (adding only `messaging_product` and `to`), and when the patient taps a list row or a button it sends
that row's / button's id back as `reply_id`. Location shared from the location button goes back as lat/lng.

    render(reply)                    -> [message, ...]      (Cloud API "message" objects, minus "to")
    read_reply_id(reply_id, lang)    -> {"text": ..., "select_doctor_id"?: ..., "page"?: ...}
    validate(message)                -> [problem, ...]       (empty = WhatsApp accepts it)

Ids the renderer writes (and read_reply_id understands):
    doc:<doctorId>       choose this doctor                  (doctor rows, "Yes, book", "which one?")
    more                 the next page of the last list
    refine:<key>         narrow the last list (fee, female, today, near, experience), said in the chat's language
    say:<text>           send this text, exactly as if the patient typed it (other options)

Limits (strictest published values, so a message never bounces):
    text body 4096 · interactive body 1024 · header / footer 60 · list button 20 · rows 10 in all ·
    section title 24 · row title 24 (unique) · row description 72 · row id 200 · reply buttons 1-3,
    title 20 (unique), id 256.
"""
from __future__ import annotations

import re
from typing import Any

TEXT_MAX, BODY_MAX, FOOTER_MAX, LIST_BUTTON_MAX = 4096, 1024, 60, 20
ROWS_MAX, SECTION_TITLE_MAX, ROW_TITLE_MAX, ROW_DESC_MAX, ROW_ID_MAX = 10, 24, 24, 72, 200
BUTTONS_MAX, BUTTON_TITLE_MAX, BUTTON_ID_MAX = 3, 20, 256

_UUID = re.compile(r"^[0-9a-fA-F-]{32,36}$")
_CODES = {"english": 0, "hinglish": 1, "hindi": 2}

# (English, Hinglish, Hindi); every one fits the limit of the place it is used
_WORDS = {
    "emergency_share": ("Want the hospitals to call you?", "Kya hospital aapko call karein?", "क्या अस्पताल आपको कॉल करें?"),
    "emergency_location": ("Send your location for the nearest emergency hospitals.",
                           "Sabse nazdeeki emergency hospital ke liye location bhejein.",
                           "सबसे नज़दीकी इमरजेंसी अस्पताल के लिए लोकेशन भेजें।"),
    "choose_doctor": ("Choose doctor", "Doctor chunein", "डॉक्टर चुनें"),          # list button, <= 20
    "choose": ("Choose", "Chunein", "चुनें"),                                       # list button, <= 20
    "doctors": ("Doctors", "Doctors", "डॉक्टर"),                                     # section title, <= 24
    "options": ("More options", "Aur options", "और विकल्प"),                        # section title, <= 24
    "more": ("Show more doctors", "Aur doctor dikhao", "और डॉक्टर दिखाओ"),          # row title, <= 24
    "fee": ("Lower fee", "Kam fees", "कम फ़ीस"),
    "female": ("Lady doctor", "Lady doctor", "महिला डॉक्टर"),
    "today": ("Available today", "Aaj available", "आज उपलब्ध"),
    "near": ("Nearest", "Sabse paas", "सबसे पास"),
    "experience": ("Most experienced", "Sabse experienced", "सबसे अनुभवी"),
    "today_badge": ("today", "aaj", "आज"),
    "km": ("km", "km", "km"),
}
# What a tap on a refine / more row says, in the conversation's language (checked against the parser).
_SAY = {
    "more": ("show more doctors", "aur doctor dikhao", "और डॉक्टर दिखाओ"),
    "fee": ("lowest fee among these", "inme se kam fees wala", "इनमें से सबसे सस्ता वाला"),
    "female": ("lady doctor among these", "inme se koi lady doctor", "इनमें से कोई महिला डॉक्टर"),
    "today": ("only those available today", "inme se aaj available", "इनमें से आज उपलब्ध"),
    "near": ("nearest among these", "inme se paas wala", "इनमें से पास वाला"),
    "experience": ("most experienced among these", "inme se sabse experienced", "इनमें से सबसे अनुभवी"),
}
REFINES = ("fee", "female", "today", "near", "experience")


def _w(key: str, language: str | None) -> str:
    return _WORDS[key][_CODES.get(language or "", 0)]


def _cut(text: str | None, limit: int) -> str:
    """Shorten to `limit` characters at a word where possible, with an ellipsis."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t[: limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip(" ,.;:-·") + "…"


def _body(text: str | None, limit: int = BODY_MAX) -> str:
    """Message text for an interactive body: whole sentences up to the limit, never empty."""
    t = (text or "").strip()
    if len(t) <= limit:
        return t or "…"
    out = ""
    for part in re.split(r"(?<=[.!?।])\s+", t):
        if len(out) + len(part) + 1 > limit - 1:
            break
        out = f"{out} {part}".strip()
    return out or _cut(t, limit)


def _text(body: str) -> dict:
    return {"type": "text", "text": {"preview_url": False, "body": _cut(body, TEXT_MAX) if len(body) > TEXT_MAX else body}}


def _fee(d: dict) -> Any:
    return d.get("discountedFee") if d.get("discountedFee") is not None else d.get("consultFee")


def _doctor_row(d: dict, language: str | None) -> dict:
    """Name as the title; specialty, fee, hospital, distance and 'today' in the description."""
    bits = []
    if d.get("specialtyName"):
        bits.append(re.sub(r"\s*\(.*?\)", "", d["specialtyName"]))
    if _fee(d) is not None:
        bits.append(f"₹{int(_fee(d)) if float(_fee(d)).is_integer() else _fee(d)}")
    hosp = (d.get("hospital") or {}).get("name")
    if d.get("distanceKm") is not None:
        bits.append(f"{d['distanceKm']:.1f} {_w('km', language)}")
    if d.get("availableToday"):
        bits.append(_w("today_badge", language))
    desc = " · ".join(bits)
    if hosp:
        desc = f"{desc} · {hosp}" if desc else hosp
    return {"id": f"doc:{d['doctorId']}", "title": _cut(d.get("name") or "Doctor", ROW_TITLE_MAX),
            "description": _cut(desc, ROW_DESC_MAX)}


def _unique_titles(rows: list[dict], limit: int) -> list[dict]:
    """WhatsApp rejects two rows / buttons with the same title: number the repeats."""
    seen: dict[str, int] = {}
    for r in rows:
        t = r["title"]
        if t in seen:
            seen[t] += 1
            suffix = f" ({seen[t]})"
            r["title"] = _cut(t, limit - len(suffix)) + suffix
        else:
            seen[t] = 1
    return rows


def _option_id(opt: dict) -> str:
    oid = str(opt.get("id") or "")
    if _UUID.match(oid):
        return f"doc:{oid}"
    return _cut(f"say:{opt.get('label') or oid}", ROW_ID_MAX)


def _refines(block: dict) -> list[str]:
    f = block.get("filters") or {}
    out = []
    if f.get("sort") != "fee":
        out.append("fee")
    if not f.get("gender"):
        out.append("female")
    if not f.get("availableToday"):
        out.append("today")
    if not f.get("near") and f.get("location") != "shared":
        out.append("near")
    if f.get("sort") != "experience":
        out.append("experience")
    return out


def _options_message(body: str, options: list[dict], language: str | None) -> dict:
    """Up to three short options as reply buttons; more, or longer, as a list."""
    opts = [o for o in options if o.get("label")]
    if 0 < len(opts) <= BUTTONS_MAX and all(len(o["label"]) <= BUTTON_TITLE_MAX for o in opts):
        buttons = _unique_titles([{"id": _option_id(o), "title": o["label"]} for o in opts], BUTTON_TITLE_MAX)
        return {"type": "interactive", "interactive": {
            "type": "button", "body": {"text": _body(body)},
            "action": {"buttons": [{"type": "reply", "reply": b} for b in buttons]}}}
    rows = _unique_titles([{"id": _option_id(o), "title": _cut(o["label"], ROW_TITLE_MAX),
                            **({"description": _cut(o["label"], ROW_DESC_MAX)} if len(o["label"]) > ROW_TITLE_MAX else {})}
                           for o in opts[:ROWS_MAX]], ROW_TITLE_MAX)
    return {"type": "interactive", "interactive": {
        "type": "list", "body": {"text": _body(body)},
        "action": {"button": _w("choose", language), "sections": [{"title": _cut(_w("options", language), SECTION_TITLE_MAX), "rows": rows}]}}}


def _name_the_button(message: str, button: str) -> str:
    """Vita's text names the web's "Choose" button; on WhatsApp the patient opens the list with `button`."""
    return message.replace("'Choose'", f"'{button}'").replace("Tap Choose", f"Tap '{button}'")


def render(reply: dict) -> list[dict]:
    """One Vita reply -> WhatsApp messages, in the order to send them."""
    language = reply.get("language")
    message = reply.get("message") or ""
    kind = reply.get("kind")
    clarify = reply.get("clarify") or {}
    block = reply.get("doctors") or {}
    items = list(block.get("items") or [])

    if kind == "emergency":                                  # the number first, nothing to book
        msgs = [_text(message)]                              # numbers and the nearest places, as text (numbers are tappable)
        em = reply.get("emergency") or {}
        if em.get("needsLocation"):                          # one tap sends the location, then the places follow
            msgs.append({"type": "interactive", "interactive": {
                "type": "location_request_message", "body": {"text": _body(_w("emergency_location", language))},
                "action": {"name": "send_location"}}})
        options = [o for o in clarify.get("options") or [] if o.get("label")]
        if options:                                          # "Share my number" (consent, one tap)
            msgs.append(_options_message(_w("emergency_share", language), options, language))
        return msgs

    if items:                                                # a doctor list: one list message
        doctor_rows = _unique_titles([_doctor_row(d, language) for d in items[: ROWS_MAX - 1]], ROW_TITLE_MAX)
        option_rows: list[dict] = []
        if block.get("hasMore"):
            option_rows.append({"id": "more", "title": _w("more", language)})
        if len(items) > 1:
            option_rows += [{"id": f"refine:{k}", "title": _w(k, language)} for k in _refines(block)]
        option_rows = option_rows[: ROWS_MAX - len(doctor_rows)]
        sections = [{"title": _cut(_w("doctors", language), SECTION_TITLE_MAX), "rows": doctor_rows}]
        if option_rows:
            sections.append({"title": _cut(_w("options", language), SECTION_TITLE_MAX), "rows": option_rows})
        # a "which one?" question rides on the same list: its rows are the doctors to choose from
        button = _w("choose_doctor", language)
        return [{"type": "interactive", "interactive": {
            "type": "list", "body": {"text": _body(_name_the_button(message, button))},
            "action": {"button": button, "sections": sections}}}]

    if clarify.get("kind") == "location":                    # WhatsApp's own "send location" button
        return [{"type": "interactive", "interactive": {
            "type": "location_request_message", "body": {"text": _body(message)},
            "action": {"name": "send_location"}}}]

    options = [o for o in clarify.get("options") or [] if o.get("label")]
    if options:
        return [_options_message(message, options, language)]

    return [_text(message)]


def read_reply_id(reply_id: str, language: str | None) -> dict:
    """A tapped row / button id -> what to send to /v2/agent/chat. Unknown ids are sent as text."""
    rid = (reply_id or "").strip()
    i = _CODES.get(language or "", 0)
    if rid.startswith("doc:") and _UUID.match(rid[4:]):
        return {"text": "Choose", "select_doctor_id": rid[4:]}
    if rid == "more":
        return {"text": _SAY["more"][i]}
    if rid.startswith("refine:") and rid[7:] in REFINES:
        return {"text": _SAY[rid[7:]][i]}
    if rid.startswith("say:") and rid[4:].strip():
        return {"text": rid[4:].strip()}
    return {"text": rid or "hi"}


def validate(msg: dict) -> list[str]:
    """Everything WhatsApp would reject in this message (empty list = fine)."""
    problems: list[str] = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            problems.append(what)

    kind = msg.get("type")
    if kind == "text":
        body = (msg.get("text") or {}).get("body") or ""
        check(0 < len(body) <= TEXT_MAX, f"text body length {len(body)}")
        return problems
    check(kind == "interactive", f"unknown message type {kind!r}")
    it = msg.get("interactive") or {}
    body = (it.get("body") or {}).get("text") or ""
    check(0 < len(body) <= BODY_MAX, f"body length {len(body)}")
    if it.get("footer"):
        check(len(it["footer"].get("text") or "") <= FOOTER_MAX, "footer too long")
    action = it.get("action") or {}
    if it.get("type") == "list":
        check(0 < len(action.get("button") or "") <= LIST_BUTTON_MAX, f"list button {action.get('button')!r}")
        sections = action.get("sections") or []
        check(1 <= len(sections) <= 10, f"{len(sections)} sections")
        rows = [r for s in sections for r in s.get("rows") or []]
        check(1 <= len(rows) <= ROWS_MAX, f"{len(rows)} rows")
        check(len({r.get("title") for r in rows}) == len(rows), "row titles not unique")
        for s in sections:
            check(0 < len(s.get("title") or "") <= SECTION_TITLE_MAX, f"section title {s.get('title')!r}")
            check(bool(s.get("rows")), "empty section")
        for r in rows:
            check(0 < len(r.get("id") or "") <= ROW_ID_MAX, f"row id {r.get('id')!r}")
            check(0 < len(r.get("title") or "") <= ROW_TITLE_MAX, f"row title {r.get('title')!r}")
            check(len(r.get("description") or "") <= ROW_DESC_MAX, f"row description {r.get('description')!r}")
    elif it.get("type") == "button":
        buttons = action.get("buttons") or []
        check(1 <= len(buttons) <= BUTTONS_MAX, f"{len(buttons)} buttons")
        titles = [(b.get("reply") or {}).get("title") or "" for b in buttons]
        check(len(set(titles)) == len(titles), "button titles not unique")
        for b in buttons:
            r = b.get("reply") or {}
            check(0 < len(r.get("title") or "") <= BUTTON_TITLE_MAX, f"button title {r.get('title')!r}")
            rid = r.get("id") or ""
            check(0 < len(rid) <= BUTTON_ID_MAX and rid == rid.strip(), f"button id {rid!r}")
    elif it.get("type") == "location_request_message":
        check(action.get("name") == "send_location", "location request without send_location")
    else:
        problems.append(f"unknown interactive type {it.get('type')!r}")
    return problems
