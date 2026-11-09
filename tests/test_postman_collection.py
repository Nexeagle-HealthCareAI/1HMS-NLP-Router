"""The Postman collection for Vita (postman/collections/Vita-Chat-Agent.postman_collection.json) stays in step with the
API: every request is played in order against the app, without a model, with the collection's own variables, and the
code-decided checks it carries (emergency, cry for help, warning signs, language, WhatsApp shape, hand-over) hold.
Regenerate the collection with: python scripts/make_vita_postman.py"""
import importlib.util
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hms_router.adapters.http import create_app
from hms_router.adapters.settings import ServiceSettings
from tests.test_vita_doctor_chat import Fake1Hms, make_directory

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "postman" / "collections" / "Vita-Chat-Agent.postman_collection.json"
ENV = ROOT / "postman" / "environments" / "Vita-Local.postman_environment.json"
SLOW = {"Promote drafts (runs the scenario suite)"}       # covered by test_vita_graph's promotion-gate test
SCOPED = {"Forget a patient (verified mobile)"}           # needs a key with the verified_mobile scope: 422 with auth off


def _generator():
    spec = importlib.util.spec_from_file_location("make_vita_postman", ROOT / "scripts" / "make_vita_postman.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _requests(items, path=()):
    for it in items:
        if "item" in it:
            yield from _requests(it["item"], path + (it["name"],))
        else:
            yield path, it


def test_the_collection_on_disk_is_what_the_generator_makes():
    assert json.loads(COLLECTION.read_text(encoding="utf-8")) == json.loads(json.dumps(_generator().build())), \
        "run: python scripts/make_vita_postman.py"


def test_every_request_has_a_unique_name_in_its_folder_and_a_test():
    seen = set()
    for path, it in _requests(json.loads(COLLECTION.read_text(encoding="utf-8"))["item"]):
        assert (path, it["name"]) not in seen
        seen.add((path, it["name"]))
        assert any(e["listen"] == "test" for e in it["event"]), it["name"]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ServiceSettings(rate_limit_chat=10_000))) as c:
        c.app.state.doctors = make_directory(Fake1Hms())
        yield c


def _fill(text: str, vars_: dict) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(vars_.get(m.group(1), "")), text)


def test_play_the_whole_collection(client):
    checks = _generator().CHECKS
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    vars_ = {v["key"]: v["value"] for v in json.loads(ENV.read_text(encoding="utf-8"))["values"]}
    vars_.update({v["key"]: v["value"] for v in col["variable"]})
    played, problems = 0, []
    for path, it in _requests(col["item"]):
        if it["name"] in SLOW:
            continue
        scripts = {e["listen"]: "\n".join(e["script"]["exec"]) for e in it["event"]}
        if 'unset("sessionId")' in scripts.get("prerequest", ""):
            vars_["sessionId"] = ""
        req = it["request"]
        url = _fill(req["url"]["raw"], vars_).replace(vars_["baseUrl"], "")
        body = json.loads(_fill(req["body"]["raw"], vars_)) if req.get("body") else None
        r = client.request(req["method"], url, json=body)
        played += 1
        where = f"{' / '.join(path)} / {it['name']}"
        if it["name"] in SCOPED and r.status_code == 422:
            continue
        if r.status_code != 200:
            problems.append(f"{where}: HTTP {r.status_code} {r.text[:200]}")
            continue
        out = r.json()
        if isinstance(out, dict):
            vars_["sessionId"] = out.get("session_id") or vars_["sessionId"]
            items = (out.get("doctors") or {}).get("items") or [] if isinstance(out.get("doctors"), dict) else []
            if items:
                vars_["doctorId"] = items[0]["doctorId"]
            if "admin/examples" in url and req["method"] == "POST" and out.get("id"):
                vars_["exampleId"] = out["id"]
            if "admin/knowledge" in url and req["method"] == "POST" and out.get("id"):
                vars_["knowledgeId"] = out["id"]
        test = scripts["test"]
        for name, js in checks.items():
            if js in test and not _holds(name, out):
                problems.append(f"{where}: check '{name}' failed: kind={out.get('kind')} lang={out.get('language')} "
                                f"msg={str(out.get('message'))[:120]!r}")
    assert played >= 80
    assert not problems, "\n".join(problems)


def _holds(name: str, r: dict) -> bool:
    docs = (r.get("doctors") or {}).get("items") or []
    msg = r.get("message") or ""
    return {
        "emergency": r.get("kind") == "emergency" and "108" in msg,
        "not_emergency": r.get("kind") != "emergency",
        "distress": "108" in msg and r.get("kind") != "emergency",
        "warning": (r.get("clarify") or {}).get("kind") == "warning_signs",
        "doctor_list": len(docs) > 0,
        "selected": r.get("kind") == "doctor_selected",
        "hinglish": r.get("language") == "hinglish",
        "english": r.get("language") == "english",
        "whatsapp": isinstance(r.get("whatsapp"), list) and len(r["whatsapp"]) > 0,
        "handover": r.get("kind") == "doctor_selected" and (
            bool(r.get("booking")) or r.get("handover") == {"to": "booking", "doctorId": r.get("selectedDoctorId")}),
        "no_menu": not r.get("clarify"),
    }[name]
