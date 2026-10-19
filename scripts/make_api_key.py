"""Create an API key for one client. Prints the raw key ONCE (give it to the client) and the line to add to
HMS_API_KEYS (store only that). Usage:  python scripts/make_api_key.py doctordekho route
                                        python scripts/make_api_key.py whatsapp chat route
                                        python scripts/make_api_key.py cms route admin   (CMS AI Agents tab)
"""
import hashlib
import secrets
import sys

if len(sys.argv) < 3 or not set(sys.argv[2:]) <= {"route", "chat", "admin"}:
    sys.exit("usage: make_api_key.py <client-name> <scope> [scope...]   scopes: route, chat, admin")
name, scopes = sys.argv[1], "|".join(sys.argv[2:])
if not name.replace("-", "").replace("_", "").isalnum():
    sys.exit("client name: letters, digits, - and _ only")
raw = secrets.token_urlsafe(32)
print(f"raw key for {name} (send securely, it is not stored):\n  {raw}")
print(f"HMS_API_KEYS entry:\n  {name}:{scopes}:{hashlib.sha256(raw.encode()).hexdigest()}")
