"""python demo.py [--debug]   interactive multi-turn demo (fakeredis unless REDIS_URL is set).

Type symptoms in Hinglish / Hindi / English, or type `quit` to exit. After a recommendation you can type:
  /good   -> Step 8 "correct match" feedback for the primary specialist
  /bad    -> Step 8 "wrong specialist" feedback
  /new <phrase> -> report a symptom phrase the system didn't understand
"""
import os
import sys

from hms_router import HMSRouterAgent

if os.getenv("REDIS_URL"):
    import redis
    r = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
else:
    import fakeredis
    r = fakeredis.FakeRedis(decode_responses=True)

debug = "--debug" in sys.argv
agent, sid, last = HMSRouterAgent(r), None, None
print("HMS router - describe your symptoms (Ctrl-C to quit)")
while True:
    line = input("> ").strip()
    if line.lower() == "quit":
        print("Goodbye.")
        break
    if line.startswith("/") and last is not None:
        cmd, _, rest = line.partition(" ")
        top = last.card.outputs[0] if last.card else None
        if cmd == "/good" and top:
            print(agent.feedback.submit("correct_match", session_id=sid, cluster_id=top.cluster_id or "-"))
        elif cmd == "/bad" and top:
            print(agent.feedback.submit("wrong_specialist", session_id=sid, cluster_id=top.cluster_id,
                                        specialist=top.specialist, symptoms=last.state.symptoms))
        elif cmd == "/new":
            print(agent.feedback.submit("new_symptom_phrase", session_id=sid, phrase=rest))
        continue
    last = agent.handle(line, sid)
    sid = last.session_id
    if debug:
        print(f"  [session={sid} turn={last.turn} lang={last.language} kind={last.kind} "
              f"severity={last.severity.level}/{last.severity.score} via={last.severity.triggered_by}]")
        print("  [keywords] " + ", ".join(f"{k.token}:{k.level}" for k in last.keywords if k.level not in ("FILLER",)))
    print(last.message + "\n")
