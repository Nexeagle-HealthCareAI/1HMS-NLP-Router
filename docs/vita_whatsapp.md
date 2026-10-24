# Vita on WhatsApp: integration contract

Vita is ready for the WhatsApp bot (`Whatsapp-Webhook`) without the bot learning Vita's reply kinds. The
bot is **not** wired to Vita yet; this is the contract it will use, and `tests/test_vita_whatsapp.py`
keeps it working: any router change that would break WhatsApp fails CI.

## The loop

```
patient message ──► bot ──► POST /v2/agent/chat  {channel: "whatsapp", text, session_id, city?}
                                     │
                    reply.whatsapp = [message, ...]   (Cloud API objects, minus "to")
                                     │
bot sends each one: {"messaging_product": "whatsapp", "to": <wa_id>, **message}

patient taps a row / button ──► bot ──► POST /v2/agent/chat {channel, session_id, reply_id: <row or button id>}
patient shares location     ──► bot ──► POST /v2/agent/chat {channel, session_id, lat, lng}   (no text)
```

- Keep Vita's `session_id` per WhatsApp number (it lasts 30 minutes after the last message).
- Send `city` when the bot knows the patient's city; `ui_language` if the patient picked a language.
- The bot never builds Vita messages itself and never parses row ids: it sends `reply_id` back as it came.

## What Vita sends

| Situation | WhatsApp message |
|---|---|
| Doctors found (any way of asking) | One **list**: section "Doctors" (one row per doctor: name; specialty · fee · distance · today · hospital), section "More options" (show more, lower fee, lady doctor, available today, nearest, most experienced) |
| "Which one?" (two similar names) | The same list: the rows are the doctors to choose from |
| Vita needs the location | A **location request** (WhatsApp's own "Send location" button) |
| Up to 3 short options (e.g. "Yes, book" / "No, just checking") | **Reply buttons** |
| More, or longer, options (e.g. the five ways to ask) | A **list** |
| Emergency | One **text**, 108 / 112 in the first line, nothing to tap or book |
| Everything else (triage question, doctor chosen, handoff, Bangla note) | **Text** |

All words are in the conversation's language (English, Hinglish or Hindi script), including list buttons and
section titles. Limits are enforced: body 1024, list button 20, rows 10, row title 24 (unique), row
description 72, row id 200, buttons 1-3 with titles of 20.

## Ids Vita writes

| Id | Means |
|---|---|
| `doc:<doctorId>` | Choose this doctor. A stale one (from an old list) returns **422**: show "please search again" |
| `more` | The next page of the last list |
| `refine:fee` · `female` · `today` · `near` · `experience` | Narrow the last list, said in the chat's language |
| `say:<text>` | Send this text as if the patient typed it |

## After a doctor is chosen

The reply is `kind: "doctor_selected"` with `doctor` (the 1HMS record) and `selectedDoctorId`. That is where
the bot's own booking flow (shift, patient details, confirmation) takes over: map `doctor.doctorId`,
`doctor.name`, fee, `doctor.hospital.*` into its booking context, exactly as it does for its own list today.
Booking inside Vita itself comes later (vision Phase 2).

## If Vita is unavailable

Vita answers within the router's timeout; on 5xx, timeout or an unknown shape, the bot should fall back to its
own finder for that message. Vita's `degraded: ["doctor_lookup"]` (1HMS down) still comes with a valid text
message to send.
