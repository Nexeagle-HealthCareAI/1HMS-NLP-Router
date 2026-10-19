"""Golden conversations for /chat, /session, /feedback. Shared by the recorder and the contract test.

The /chat path must not change while /route-symptom is rebuilt. tests/fixtures/chat_golden.json holds
what /chat returned before the refactor; tests/test_chat_contract.py replays the same conversations
and requires identical output (session ids and timestamps aside).

Re-record ONLY for an intended /chat change:  python -m tests.chat_contract --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "chat_golden.json"

CONVERSATIONS: dict[str, list[str]] = {
    "emergency_chest_pain": ["seene mein dard ho raha hai"],
    "emergency_devanagari": ["सांस नहीं आ रही"],
    "followup_then_recommend": ["pet mein dard hai", "haan ulti bhi", "2 din se", "bahut tez"],
    "negation": ["chest pain nahi hai bas khansi hai", "3 din se", "halki"],
    "english_multi": ["I have a headache and dizziness for 2 weeks", "nothing else"],
    "unknown_words": ["aankh laal hai", "aankh mein khujli", "aankh se paani"],
    "fever": ["bukhar hai 3 din se bahut tez"],
}

_VOLATILE = {"session_id", "ts", "created_at", "last_active"}


def scrub(obj):
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    return obj


def run(client) -> dict:
    out: dict = {"health": client.get("/health").json()}
    for name, turns in CONVERSATIONS.items():
        sid, replies = None, []
        for text in turns:
            r = client.post("/chat", json={"text": text, "session_id": sid})
            body = r.json()
            sid = body.get("session_id", sid)
            replies.append({"status": r.status_code, "body": scrub(body)})
        sess = client.get(f"/session/{sid}")
        out[name] = {"replies": replies, "session": {"status": sess.status_code, "body": scrub(sess.json())}}
    out["errors"] = {
        "empty_text": client.post("/chat", json={"text": ""}).status_code,
        "too_long": client.post("/chat", json={"text": "a" * 1001}).status_code,
        "unknown_session": client.get("/session/does-not-exist").status_code,
    }
    out["feedback"] = {
        "bad_kind": client.post("/feedback", json={"kind": "nope"}).status_code,
        "new_phrase": client.post("/feedback", json={"kind": "new_symptom_phrase", "phrase": "khujli"}).json(),
    }
    return out


def _client():
    """The /chat app with chat-era settings: no API keys (dev), fakeredis."""
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


if __name__ == "__main__" and "--record" in sys.argv:
    with _client() as c:
        data = run(c)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"recorded {FIXTURE}")
