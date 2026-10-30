"""Check Vita's doctor search against the REAL doctors on Doctor Dekho. Run from your own Terminal (it needs the
internet), in the 1HMS-NLP-Router folder:

    python3 scripts/check_doctor_search.py                       # Dev: 1HMS dev API + the Dev router
    python3 scripts/check_doctor_search.py --router-key KEY      # if the Dev router has HMS_API_KEYS set (chat scope)
    python3 scripts/check_doctor_search.py --hms https://1hms-api.nexeagle.com --router https://nlp-api.nexeagle.com \
        --router-key KEY                                         # Prod

What it does
  1. Reads every listed doctor and specialty from 1HMS (GET /public/doctors, /public/specialties) and reports data
     problems that break search: specialty categories Vita cannot map, doctors with no degree, the same city spelt
     differently, hospitals with no map pin (invisible to "near me").
  2. Builds searches from those real doctors and sends them to Vita (POST /v2/agent/chat) the way the app does:
     name (full, without "Dr", surname, from another city), degree, specialty, nearby, and a few symptoms.
  3. Prints PASS / FAIL per search and writes doctor_search_report.json.
Standard library only; nothing is booked or changed.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hms_router.doctors import SPECIALIST_TITLES, specialty_key  # noqa: E402

DEGREES = ["MBBS", "MD", "MS", "DM", "MCH", "DNB", "BDS", "MDS", "DGO", "DCH"]
SYMPTOMS = [("daant mein bahut dard hai", "Dentist"), ("aankh laal hai aur paani aa raha hai", "Eye Specialist"),
            ("skin pe khujli aur daane", "Dermatologist"), ("mujhe 3 din se bukhar hai", "General Physician")]


def get(url: str, key: str | None = None) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", **({"X-Api-Key": key} if key else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def vita(router: str, key: str | None, text: str, **extra) -> dict:
    body = json.dumps({"text": text, "channel": "web", **extra}).encode()
    req = urllib.request.Request(router.rstrip("/") + "/v2/agent/chat", data=body, method="POST",
                                 headers={"Content-Type": "application/json", **({"X-Api-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"kind": f"HTTP {e.code}", "message": e.read()[:200].decode(errors="replace")}


def degrees_of(q: str | None) -> set[str]:
    toks = {re.sub(r"[^A-Z]", "", t) for t in re.split(r"[\s,/()+&]+", (q or "").upper().replace(".", ""))}
    return {t for t in toks if t in DEGREES}


def category_of(d: dict) -> str | None:
    return d.get("primaryMedicalSpecialityCategory") or d.get("departmentName")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hms", default="https://1hms-dev-api.nexeagle.com")
    ap.add_argument("--hms-key", default=None, help="1HMS public API key (X-Api-Key), if your API needs one")
    ap.add_argument("--router", default="http://151.185.45.77:5003")
    ap.add_argument("--router-key", default=None, help="router API key with the chat scope")
    ap.add_argument("--max", type=int, default=8, help="doctors to build name searches from")
    a = ap.parse_args()

    # ---------------------------------------------------------------- 1. the real data
    doctors, page = [], 1
    while True:
        data = get(f"{a.hms.rstrip('/')}/public/doctors?page={page}&pageSize=100", a.hms_key)
        doctors += data.get("doctors") or []
        if len(doctors) >= (data.get("totalCount") or 0) or not data.get("doctors"):
            break
        page += 1
    specs = get(f"{a.hms.rstrip('/')}/public/specialties", a.hms_key).get("specialties") or []
    print(f"\n{len(doctors)} listed doctors, {len(specs)} specialty categories on {a.hms}\n")

    problems = []
    print("Specialty categories -> what Vita understands")
    for s in specs:
        cat = s.get("category") or s.get("specialtyId") or s.get("name")
        key = specialty_key(cat) or specialty_key(s.get("displayName"))
        flag = "" if key else "   <-- Vita cannot search this category by specialty"
        print(f"  {cat!s:40} {s.get('doctorCount', '?'):>3} doctors -> {key or '-'}{flag}")
        if not key:
            problems.append(f"unmapped category: {cat}")
    cities = collections.defaultdict(set)
    for d in doctors:
        if d.get("city"):
            cities[re.sub(r"[^a-z]", "", d["city"].lower())].add(d["city"])
    for variants in cities.values():
        if len(variants) > 1:
            problems.append(f"same city spelt differently: {sorted(variants)} (city search is an exact match)")
    no_deg = [d["fullName"] for d in doctors if not degrees_of(d.get("qualification"))]
    no_pin = sorted({d.get("hospitalName") or "?" for d in doctors if d.get("latitude") is None})
    if no_deg:
        problems.append(f"{len(no_deg)} doctors with no recognised degree, e.g. {no_deg[:5]}")
    if no_pin:
        problems.append(f"hospitals with no map pin (never found by 'near me'): {no_pin[:10]}")
    print("\nData problems:" if problems else "\nNo data problems found.")
    for p in problems:
        print("  - " + p)

    # ---------------------------------------------------------------- 2. searches built from that data
    all_cities = sorted({d["city"] for d in doctors if d.get("city")})
    cases = []

    def other_city(c):
        return next((x for x in all_cities if x.lower() != (c or "").lower()), c)

    for d in doctors[: a.max]:
        full = re.sub(r"^(dr\.?\s*)", "", d["fullName"], flags=re.I).strip()
        surname = full.split()[-1]
        cases += [("name", f"Dr {full}", {"city": other_city(d.get("city"))}, {"doctorId": d["doctorId"]}),
                  ("name", full, {"city": d.get("city")}, {"doctorId": d["doctorId"]}),
                  ("name", f"Dr {surname} se milna hai", {"city": d.get("city")}, {"doctorId": d["doctorId"]})]
    for deg in sorted({g for d in doctors for g in degrees_of(d.get("qualification"))}):
        city = next(d.get("city") for d in doctors if deg in degrees_of(d.get("qualification")))
        cases.append(("degree", f"{deg} doctor chahiye", {"city": city}, {"degree": deg}))
    for s in specs:
        key = specialty_key(s.get("category") or "")
        if key and key in SPECIALIST_TITLES:
            city = next((d.get("city") for d in doctors if specialty_key(category_of(d) or "") == key), None)
            cases.append(("specialty", f"{SPECIALIST_TITLES[key]} doctor", {"city": city}, {"key": key}))
    for d in [d for d in doctors if d.get("latitude") is not None][:3]:
        key = specialty_key(category_of(d) or "")
        if key in SPECIALIST_TITLES:
            cases.append(("nearby", f"paas mein {SPECIALIST_TITLES[key]}", {"lat": float(d["latitude"]) + 0.005,
                          "lng": float(d["longitude"]) + 0.005}, {"doctorId": d["doctorId"]}))
    for text, want in SYMPTOMS:
        cases.append(("symptoms", text, {"city": all_cities[0] if all_cities else None}, {"specialist": want}))

    results = []
    print(f"\nSending {len(cases)} searches to {a.router} ...\n")
    for kind, text, extra, expect in cases:
        out = vita(a.router, a.router_key, text, **{k: v for k, v in extra.items() if v is not None})
        sid = out.get("session_id")
        for answer in ("3 din se", "nahi"):                       # symptoms: answer Vita's follow-up questions
            if kind == "symptoms" and out.get("kind") == "follow_up" and sid:
                out = vita(a.router, a.router_key, answer, session_id=sid, **extra)
        items = (out.get("doctors") or {}).get("items") or []
        ok, why = True, ""
        if "doctorId" in expect:
            ok = any(i.get("doctorId") == expect["doctorId"] for i in items)
            total = (out.get("doctors") or {}).get("total") or 0
            if not ok and total > len(items) and "se milna" in text:
                ok, why = None, f"{total} doctors share the surname; not in the first {len(items)}"
        elif "degree" in expect:
            ok = bool(items) and all(expect["degree"] in degrees_of(i.get("degrees") or i.get("qualification"))
                                     for i in items)
        elif "key" in expect:
            ok = bool(items) and all(specialty_key(i.get("specialtyName") or i.get("specialtyId") or "") == expect["key"]
                                     for i in items) and not (out.get("doctors") or {}).get("fallbackUsed")
        elif "specialist" in expect:
            prim = next((o.get("specialist") for o in (out.get("card") or {}).get("outputs") or []
                         if o.get("role") == "PRIMARY"), None)
            ok, why = prim == expect["specialist"], f"got {prim}"
        status = "PASS" if ok else ("CHECK" if ok is None else "FAIL")
        results.append({"type": kind, "text": text, "sent": extra, "status": status, "kind": out.get("kind"),
                        "doctors": [i.get("name") for i in items], "message": (out.get("message") or "")[:160],
                        "note": why})
        print(f"  {status:5} {kind:9} {text[:42]:42} -> {out.get('kind')!s:14} {[i.get('name') for i in items][:3]}"
              + (f"  ({why})" if why and status != "PASS" else ""))

    summary = collections.Counter((r["type"], r["status"]) for r in results)
    print("\nSummary:")
    for t in ("name", "degree", "specialty", "nearby", "symptoms"):
        print(f"  {t:9} pass {summary[(t, 'PASS')]}  fail {summary[(t, 'FAIL')]}  check {summary[(t, 'CHECK')]}")
    Path("doctor_search_report.json").write_text(json.dumps({"problems": problems, "results": results}, indent=1,
                                                            ensure_ascii=False))
    print("\nFull report: doctor_search_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
