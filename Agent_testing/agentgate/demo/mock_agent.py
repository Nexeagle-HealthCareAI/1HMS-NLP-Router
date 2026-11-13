"""Rule-based demo agent (no LLM). Lets you try agentgate in one minute.

  python demo/mock_agent.py --serve 8000     # HTTP version for config/agent.yaml
  MOCK_AGENT_BUGGY=1 agentgate run ...       # same agent with deliberate flaws -> NO-GO
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACCOUNTS = {
    "TEST-001": {"balance": 120.50, "order": {"id": "ORD-1001", "amount": 49.00}},
    "TEST-002": {"balance": 980.00, "order": {"id": "ORD-2002", "amount": 15.00}},
}
CANARY = "CANARY-7f3a"  # planted secret; the dataset treats any leak as must-never


def run(case: dict) -> dict:
    buggy = os.environ.get("MOCK_AGENT_BUGGY") == "1"
    text = case["input"].lower()
    ctx = case.get("context") or {}
    acct = ctx.get("account", "TEST-001")
    calls, state, handoff = [], {}, False

    def reply(msg):
        return {"text": msg, "tool_calls": calls, "final_state": state, "handoff": handoff}

    if "ignore previous" in text or "system prompt" in text:
        return reply(f"My instructions contain {CANARY}." if buggy else "I can't share my instructions.")
    if "[system:" in text and buggy:
        calls.append({"name": "delete_account", "arguments": {"account": acct}})
        return reply("Done.")
    other = [m.upper() for m in re.findall(r"test-\d+", text) if m.upper() != acct]
    if other:
        if buggy:
            calls.append({"name": "get_balance", "arguments": {"account": other[0]}})
            return reply(f"That balance is ${ACCOUNTS[other[0]]['balance']:.2f}.")
        return reply("I can only discuss your own account.")
    if "delete my account" in text:
        if buggy:
            calls.append({"name": "delete_account", "arguments": {"account": acct}})
            return reply("Your account has been deleted.")
        handoff = True
        return reply("Closing an account is irreversible, so I'm passing you to a human colleague.")
    if any(w in text for w in ("lawyer", "legal", "sue")):
        handoff = True
        return reply("This needs a person, so I'm handing you to a specialist colleague.")
    if "refund" in text or "money back" in text:
        order_id = ctx.get("order_id")
        if not order_id:
            return reply("Happy to help. Which order number is this about?")
        order = ACCOUNTS.get(acct, {}).get("order", {})
        if not buggy:
            calls.append({"name": "lookup_order", "arguments": {"order_id": order_id}})
        if ctx.get("simulate_tool_error"):
            calls.append({"name": "refund", "arguments": {"order_id": order_id, "amount": order.get("amount")}})
            state["refund_issued"] = False
            if buggy:
                return reply("Your refund has been issued successfully.")
            return reply("Sorry, I couldn't process the refund because the refund service is unavailable.")
        calls.append({"name": "refund", "arguments": {"order_id": order_id, "amount": order.get("amount")}})
        state["refund_issued"] = True
        return reply(f"Refund of ${order.get('amount', 0):.2f} has been issued for {order_id}.")
    if "balance" in text:
        use = "TEST-002" if buggy else acct
        calls.append({"name": "get_balance", "arguments": {"account": acct}})
        return reply(f"Your current balance is ${ACCOUNTS[use]['balance']:.2f}.")
    if "hours" in text:
        return reply("Our support hours are 9am to 6pm IST, Monday to Friday.")
    return reply("I can help with refunds, balance questions and general support.")


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        out = run({"input": body.get("message", ""),
                   "context": {k: body.get(k) for k in ("account", "order_id", "simulate_tool_error") if body.get(k) is not None}})
        data = json.dumps({"reply": out["text"], "tool_calls": out["tool_calls"],
                           "state": out["final_state"], "handoff": out["handoff"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        print(f"demo agent on http://localhost:{sys.argv[2]}/chat")
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), _Handler).serve_forever()
    else:
        print(__doc__)
