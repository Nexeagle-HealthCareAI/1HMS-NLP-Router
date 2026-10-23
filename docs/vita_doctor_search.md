# Vita doctor search in `/chat`

Vita Chat requirements, Part 2 (VITA-1 to VITA-13). Code: `hms_router/doctor_intent.py` (what kind of ask),
`hms_router/doctors.py` (`DoctorDirectory`, the only way to 1HMS), `hms_router/application/doctor_chat.py`
(the turn logic). Tests: `tests/test_vita_doctor_chat.py`.

## Turn it on

| Setting | |
|---|---|
| `HMS_DOCTORS_MCP_URL` (or `HMS_MCP_URL`) | 1HMSAPI MCP endpoint, e.g. `https://<1hms-api>/mcp` |
| `HMS_DOCTORS_MCP_KEY` (or `HMS_MCP_KEY`) | this router's key in 1HMSAPI `Mcp:ServiceKeys` (sent as `X-Service-Key`) |
| `HMS_DOCTORS_LIMIT` | doctors per reply, 1–10, default 3 |
| `HMS_DOCTORS_TIMEOUT` | seconds per MCP call, default 2.0 (each call is retried once) |

Unset = `/chat` behaves exactly as before (no new kinds, no `doctors` block).

## Request

```json
POST /chat
{
  "text": "skin ka doctor chahiye",
  "session_id": "…",              // from the previous reply; omit on the first message
  "city": "Kolkata",               // optional, used when no location is shared
  "lat": 22.58, "lng": 88.41,      // optional; used for this search only, never stored
  "hospital_id": "…",              // optional: Vita in a hospital's own channel (VITA-11)
  "select_doctor_id": "…"          // optional: the doctor the patient tapped Choose on (VITA-8)
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

Combinations work in one turn ("MD cardiologist near Salt Lake"). Refinements reuse the last search:
"inme se koi paas wala?", "kam fees wala", "sabse experienced". A bare "doctor chahiye" gets
`clarify.kind = "ask_kind"` with the five options.

Body words alone ("pet mein dard", "kaan mein dard") are symptoms and go to triage; "pet ka doctor" is a
specialty ask.

## Reply

Every reply has the AgentReply fields (`session_id`, `turn`, `kind`, `message`, `question`, `card`,
`state`, `severity`, `keywords`, `language`, `degraded`) plus `clarify` on new kinds. New kinds:

- `doctor_list`: `doctors` block below.
- `doctor_selected`: `doctor` (current profile from `get_doctor`) and `selectedDoctorId`.
  `GET /session/{id}` then shows `selectedDoctorId` and `doctorSearch`.

```json
"doctors": {
  "mode": ["specialty", "distance"],
  "filters": {"specialtyId": "Dermatologist (Skin)", "near": true, "radiusKm": 5, "location": "shared"},
  "items": [ /* McpDoctor objects exactly as 1HMS returned them, in 1HMS order */ ],
  "total": 7,
  "fallbackUsed": false,
  "source": "1hms-mcp",
  "lookupMs": 182.4,
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
`DELETE /session/{id}` clears it too.
