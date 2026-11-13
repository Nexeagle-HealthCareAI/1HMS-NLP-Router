"""Start the route-model rebuild inside a running router container and wait for its result.

Used by .github/workflows/retrain.yml, which pipes this file over SSH into
`docker exec -i <container> python -c "<read key line; exec rest>"`, so the admin key travels on stdin and
never appears in a command line. Runs with the container's own Python (standard library only).

Exit 0: the rebuild finished (promoted or kept the current model). Exit 1: it failed or timed out.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

KEY = globals().get("KEY") or os.environ.get("NLP_ROUTER_ADMIN_KEY", "")
BASE = f"http://127.0.0.1:{os.environ.get('HMS_PORT', '5005')}/admin/rebuild"
TIMEOUT_S, EVERY_S = 20 * 60, 15


def call(method):
    req = urllib.request.Request(BASE, method=method, headers={"X-Api-Key": KEY, "Content-Type": "application/json"},
                                 data=b"{}" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


code, body = call("POST")
if code == 409:
    print("A rebuild is already running; waiting for it.")
elif code != 202:
    print(f"Could not start the rebuild: HTTP {code} {body}")
    sys.exit(1)
else:
    print("Rebuild started.")

deadline = time.time() + TIMEOUT_S
while time.time() < deadline:
    time.sleep(EVERY_S)
    code, status = call("GET")
    if code != 200:
        print(f"Status check failed: HTTP {code} {status}")
        continue
    state = status.get("state")
    print(f"  {state}: {status.get('step', '')}")
    if state in ("done", "failed"):
        print(json.dumps(status, indent=2))
        verdict = "promoted a new model" if status.get("promoted") else "kept the current model"
        print(f"RESULT: {state} - {verdict}. {status.get('verdict') or ''}")
        sys.exit(0 if state == "done" else 1)
print("Timed out waiting for the rebuild.")
sys.exit(1)
