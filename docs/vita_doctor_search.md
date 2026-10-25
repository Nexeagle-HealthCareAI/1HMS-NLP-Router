# Vita: `POST /v2/agent/chat`

Vita Chat requirements, Part 2 (VITA-1 to VITA-13), and the Phase 1 rules of the NLP Router Agentic AI
vision ("How Vita talks"). Vita is its own endpoint; **`/chat` is unchanged** (it still adds the older
`doctors` block on a confident recommendation while lookups are on). Code: `hms_router/doctor_intent.py` (what kind of ask),
`hms_router/doctors.py` (`DoctorDirectory`, the only way to 1HMS), `hms_router/application/doctor_chat.py`
(the turn logic). Tests: `tests/test_vita_doctor_chat.py`.

## Turn it on

| Setting | |
|---|---|
| `HMS_DOCTORS_MCP_URL` (or `HMS_MCP_URL`) | 1HMSAPI MCP endpoint, e.g. `https://<1hms-api>/mcp` |
| `HMS_DOCTORS_MCP_KEY` (or `HMS_MCP_KEY`) | this router's key in 1HMSAPI `Mcp:ServiceKeys` (sent as `X-Service-Key`) |
| `HMS_DOCTORS_LIMIT` | doctors per reply, 1–10, default 3 |
| `HMS_DOCTORS_TIMEOUT` | seconds per MCP call, default 2.0 (each call is retried once) |

Unset = Vita still talks and triages, but every doctor search answers with `degraded: ["doctor_lookup"]`.

## Request

```json
POST /v2/agent/chat          (same API keys and rate limit as /chat)
{
  "text": "skin ka doctor chahiye",
  "session_id": "…",              // from the previous reply; omit on the first message
  "city": "Kolkata",               // optional, used when no location is shared
  "lat": 22.58, "lng": 88.41,      // optional; used for this search only, never stored
  "hospital_id": "…",              // optional: Vita in a hospital's own channel (VITA-11)
  "select_doctor_id": "…",         // optional: the doctor the patient tapped Choose on (VITA-8)
  "page": 2,                       // optional: "Show more doctors" button (or the patient types "aur doctor dikhao")
  "new_search": true,              // optional: search with exactly the asks in `text` (a filter chip was removed)
  "ui_language": "hi"              // optional: the channel's UI language (en, hi, hinglish, bn), see below
}
```

## The five ways to ask

| Ask | Example | Reply on the same turn |
|---|---|---|
| symptom | "2 din se bukhar aur khansi" | triage as before; on a confident `recommendation`, a `doctors` block for the PRIMARY specialty (then SECONDARY, then General Physician, `fallbackUsed: true`) |
| specialty | "skin ka doctor", "dil ka doctor", "cardiologist" | `doctor_list` |
| name | "dr sarma se milna hai", "Anita Sen se milna hai" | `doctor_list`; several equal matches → `clarify.kind = "narrow"` with specialty chips |
| degree | "MS ortho", "MD medicine wala" | `doctor_list` |
| distance | "paas mein", "near me", "5 km ke andar" | `doctor_list` nearest first; no location → `follow_up` with `clarify.kind = "location"` once |

| hospital | "Sadar Hospital Purnia ke doctor", "AMRI hospital mein cardiologist" | `search_hospitals`, then that hospital's doctors only; two equal matches → `clarify.kind="which_hospital"` |
| availability | "Is Dr Sharma available Sunday?", "dr sen kal milenge", "kab baithte hain" | `kind=info` with `availability.days` (shift hours from 1HMS), the next working day if needed, and `clarify.kind="book_offer"` (Yes, book → send the id as `select_doctor_id`; No, just checking) |

Filters that combine with any of the above, and refine the last list: "lady doctor" / "mahila doctor"
(`gender`), "aaj available" (`availableToday`), a place at the end ("near Purnia", "kolkata mein") used as the
city when no location is shared. "mahila rog" / "women's doctor" still means gynaecology.

1HMS has no bookable time slots for public requests: availability is working days and shift hours, and the
hospital confirms the exact time (`availability.note`).

Combinations work in one turn ("MD cardiologist near Salt Lake"). Refinements reuse the last search:
"inme se koi paas wala?", "kam fees wala", "sabse experienced". A bare "doctor chahiye" gets
`clarify.kind = "ask_kind"` with the five options.

Body words alone ("pet mein dard", "kaan mein dard") are symptoms and go to triage; "pet ka doctor" is a
specialty ask. Triage asks **at most two** follow-up questions before showing something (`/chat` keeps 3).

## Booking (vision Phase 2)

Choosing a doctor starts booking when `HMS_PUBLIC_API_URL` / `HMS_PUBLIC_API_KEY` are set (otherwise
`doctor_selected` says "Tap 'Book a slot'" as before). One question at a time, `clarify.kind` says which:

| Step | clarify.kind | Asked |
|---|---|---|
| Day | `book_day` | the doctor's next 4 working days (1HMS availability), "Not now" |
| Time | `book_shift` | that day's shifts; skipped when there is one |
| For whom | `book_who` | Myself / saved family members (signed in) / Someone else |
| Details | `book_name`, `book_age`, `book_sex` | only what is missing |
| Mobile | `book_mobile`, `book_otp` | guests only: a WhatsApp code to the number |
| Confirm | `book_confirm` | "Shall I book Dr X … for Rina (32, F), fee Rs 600?" Yes / Change / Cancel |

- `kind: "booked"` with `booking` (appointmentId, doctor, hospital, date, shift, fee, mapUrl). 1HMS creates a
  pre-appointment; the hospital confirms the exact time.
- **Signed in = no OTP:** send `patient_token` (the 1HMS patient JWT, server to server). Vita reads the
  mobile from it, offers the saved profile and family, and books as that patient.
  A saved family member is booked without their own code: the signed-in account holder is the one booking, and 1HMS records
  them as BookedByMobile (decided 2026-10-25; the vision doc's "member's own OTP" answer is superseded).
- **A channel that verified the number** (WhatsApp sender): `verified_mobile`, honoured only for API keys
  with the `verified_mobile` scope.
- After an OTP the reply carries `patient_session` (token + mobile) so the website can sign the patient in.
  It is never in telemetry and never in WhatsApp replies.
- `client_ip`: the visitor's IP, forwarded to 1HMS with `HMS_PROXY_FORWARDING_SECRET` so 1HMS limits each
  visitor (20 bookings, 8 codes a minute) rather than the router as a whole.
- Never books without "yes". A refusal ("doctor on leave") offers the next free days or other doctors; 1HMS
  down books nothing and says so; no silent retries. "Cancel" / "rehne do" ends it; an emergency still wins.
- Telemetry for booking turns records "[booking details]", never names, numbers or codes.

## Emergency takeover (vision Phase 3)

With booking's 1HMS public API set up (`HMS_PUBLIC_API_KEY`), an emergency reply also carries the nearest emergency-ready
places and 1HMS alerts them (`POST /public/emergency-alerts`; 1HMSAPI `docs/emergency_takeover.md`). The message still starts
with 108 / 112 in the patient's language; the places follow, nearest first, with their numbers:

```
emergency: {places: [{settingId, name, kind, phone, alternatePhone, distanceKm, address, landmark, city, lat, lng, mapUrl}],
            eventId, alerted, repeated, needsLocation, numberShared, canShareNumber}
```

- **No location:** `needsLocation: true` and the message asks for it. The next turn with `lat`/`lng` (text `location`, or a
  WhatsApp location) within 30 minutes gets the places and the alert.
- **Alerted once** per chat session per 30 minutes (1HMS answers `repeated`).
- **Share my number:** when the channel knows the number (WhatsApp `verified_mobile`, or a signed-in `patient_token`), the reply
  offers `clarify.kind = "emergency_share"`; "Share my number" sends it to the alerted places. On the website a guest types it in
  the emergency card and the site sends it to 1HMS itself (`eventId`).
- **Capability** sent to 1HMS from what was said: cardiac, maternity, paediatric or trauma (puts the right hospital first among
  nearly-as-close ones).
- 1HMS unreachable: the numbers still go out, `degraded: ["emergency_places"]`. No doctor list, no booking, ever.

## After booking (vision Phase 4)

Vita stays with the patient until they are inside the clinic. With the 1HMS public API set up:

| The patient says | Reply (`kind: "visit"`, `visit_step`) |
|---|---|
| "rasta batao", "how do I get there", "रास्ता बताओ" | `directions`: map link, landmark, entrance, OPD, parking |
| "main pahunch gayi", "I've reached", "मैं पहुँच गया" | `arrival`: registration desk + booking ID, "Check in now" (`clarify.kind = "visit_checkin"`) when 1HMS allows it today, OPD, pay at the counter |
| "check in" + the location (or the location after "Check in now") | `checked_in`: token, who is being seen now, expected wait (1HMS `POST /public/tokens`, geofenced) |
| "mera number kab aayega", "how long" | `status`: token and wait |

`visit` is 1HMS's visit page data (`GET /public/appointments/{id}/visit`). Which visit: the booking made in this chat, else
`visit_appointment_id` (the website's visit page or latest booking), else a signed-in patient's next visit (`patient_token`).
Directions and status need a known visit; "I've reached" without one points to "My appointments". The emergency check still
runs first. Bookings now carry the patient's language (`en` / `hi` / `hinglish`) to 1HMS for the WhatsApp confirmation and
reminders.

## Channels

`channel: "whatsapp"` adds `whatsapp`: the reply as ready-to-send WhatsApp messages; taps come back as
`reply_id`. Contract and examples: [vita_whatsapp.md](vita_whatsapp.md).

## Language: Vita mirrors the patient

Vita answers in the language the patient writes in, for the whole conversation: English, Hinglish (Roman
Hindi) or Hindi (Devanagari). `language` on every reply says which.

| Patient | Vita |
|---|---|
| writes English | English, including triage questions, chips and the emergency text |
| writes Hinglish | Hinglish |
| writes Hindi script | Hindi script; Hindi asks ("मुझे स्किन डॉक्टर चाहिए") are understood like Hinglish ones |
| writes Bangla | a short note in Bangla, Hindi and English: "please write in Hindi or English", with 108 |
| sends a neutral reply ("Kolkata", a name, "ok", "thank you", a tapped chip) | the conversation keeps its language |
| clearly switches language mid-chat | Vita switches too, from that reply on |

- One message: `hms_router/language.py` scores Hindi words against English words; script (Devanagari,
  Bengali) decides outright. Words that are the same in both ("me", "to") count for neither.
- The conversation: the session's `context.language`, updated only by a message with clear evidence.
- `ui_language` is used only when the first message is neutral (a doctor's name, a city).
- Every phrase is in `hms_router/voice.py` in all three languages. Emergency replies lead with 108 / 112 in
  the patient's language and keep one English line as a backup.
- The triage result is said like a person ("Samajh gaya: 3 din se bukhar aur khansi. Iske liye General
  Physician ko dikhana sahi rahega…"), not the printed card; `card` still carries the full detail.
- `/chat` keeps its own wording; only its language detection improved.

## How Vita talks (conversation moves)

| Patient | Vita |
|---|---|
| "Dr Sharma ka number do" | `kind=info`: never a personal number; offers to book, or the hospital front desk number from 1HMS |
| "bas dekh raha tha", "no thanks", "thank you" | `kind=info`: "No problem. I'm here whenever you need a doctor…", no push to book |
| "hello", "namaste" | `follow_up` with the five ways to ask as chips |
| "talk to a person", "kisi insaan se baat" | `kind=handoff`, `handoff.reason="asked"`: the channel passes the chat to support |
| two dead ends in a row | the chips also offer `{"id": "human", "label": "Talk to a person"}` |
| "dr sharma" matching two doctors | `clarify.kind="which_doctor"`, options "Dr. Neha Sharma · Salt Lake Care, Kolkata" (id = doctorId: send it as `select_doctor_id`); typing "howrah wale" also picks |
| "aur doctor dikhao" / `page` | the next page of the last list; `doctors.page`, `doctors.hasMore`; past the end `clarify.kind="no_more"` |

A short "nahi" or "ok" while triage is waiting for an answer goes to triage, not to small talk.

**Changing your mind mid-way.** The patient is never held to the first flow:

| When | Patient | Vita |
|---|---|---|
| after a list | "dil ka", "nahi, haddi ka", "eye wala" | a new list for that specialty (a one-word answer counts; "dil ki dhadkan tez hai" is still a symptom) |
| after a list | "Patna mein dikhao", "nahi, kolkata me" | the same search in that place ("subah mein", "agle hafte mein" are times, not places) |
| after triage | "nahi, mujhe bachche ke liye doctor chahiye" | children's doctors ("X ke liye doctor" names the specialty) |
| while booking (any step but mobile / OTP) | a specialty, a doctor's name, a place, a symptom said as a sentence, "talk to a person" | the booking ends, nothing is booked, and the message is answered as a new ask |
| while booking | "koi aur doctor dikhao" | the list the doctor was chosen from, again |
| name / age / sex / confirm | "actually mummy ke liye", "nahi, mere liye" | switches who it is for: a known family member or the patient's own profile is filled in, otherwise "Patient ka poora naam?" |
| name step | anything that is a request, not a name ("dil ka doctor chahiye") | never taken as the name |
Specialties are named in plain words first: "bone doctors (Orthopaedic)", "haddi ke doctor (Orthopaedic)".

## Reply

Every reply has the AgentReply fields (`session_id`, `turn`, `kind`, `message`, `question`, `card`,
`state`, `severity`, `keywords`, `language`, `degraded`) plus `clarify` on new kinds. New kinds: `doctor_list`, `doctor_selected`, `info`, `handoff`.

- `doctor_list`: `doctors` block below.
- `doctor_selected`: `doctor` (current profile from `get_doctor`) and `selectedDoctorId`.
  `GET /v2/agent/session/{id}` then shows `selectedDoctorId` and `doctorSearch`.

```json
"doctors": {
  "mode": ["specialty", "distance"],
  "filters": {"specialtyId": "Dermatologist (Skin)", "near": true, "radiusKm": 5, "location": "shared"},
  "items": [ /* McpDoctor objects exactly as 1HMS returned them, in 1HMS order */ ],
  "total": 7,
  "fallbackUsed": false,
  "source": "1hms-mcp",
  "lookupMs": 182.4,
  "page": 1, "hasMore": true,
  "groups": [{"title": "Salt Lake Care", "doctorIds": ["…"]}, {"title": "Other doctors nearby", "doctorIds": ["…"]}],
  "specialist": "Dermatologist", "specialtyId": "Dermatologist (Skin)", "doctors": [ /* same as items */ ]
}
```

`groups` appears only with `hospital_id`. `specialist`, `specialtyId` and `doctors` keep the first version
of this block working. Items are only re-sorted when the patient asked for lower fees.

## Rules the code enforces

- **Emergency first (VITA-9).** Any message that would skip triage is checked with the same emergency
  detector `/v1/route-symptom` uses. If it flags, the agent answers with the emergency card and no doctors.
- **Selection (VITA-8).** `select_doctor_id` must be a doctor shown in this session; otherwise 422.
- **1HMS down (VITA-10).** After the retry, the reply still goes out: `doctors` is absent, `degraded`
  contains `doctor_lookup`, and the message says why (symptom answers keep their specialty card).
- **Privacy (NFR-2.6).** Only filters reach 1HMS (specialty, name, degree, city, location, hospital, sort),
  never the patient's sentence. Telemetry adds `searchMode`, `doctorsShown`, `doctorSelected`, `lookupMs`,
  `doctorFallbackUsed`.
- **Caching (VITA-1).** Specialties 1 h, doctor names 15 min, identical searches 5 min.

Session state for doctor search lives in Redis at `vita:doctor:{session_id}` (30 min): last filters,
doctors shown, a pending location ask, narrowing options and the selected doctor.
`DELETE /v2/agent/session/{id}` clears it.

Dead ends (no doctor found, "what kind of doctor?", handoff) are sent with the text to the CMS review
queue (`reviewText`), like abstained route-symptom queries, so the router can learn the phrase.
