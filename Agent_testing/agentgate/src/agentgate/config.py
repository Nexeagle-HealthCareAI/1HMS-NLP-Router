"""Config loading: YAML with ${VAR} / ${VAR:-default} substitution, tiny .env reader."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _subst(value):
    if isinstance(value, str):
        def repl(m):
            env = os.environ.get(m.group(1))
            if env:
                return env
            return m.group(2) if m.group(2) is not None else ""
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: _subst(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v) for v in value]
    return value


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return _subst(yaml.safe_load(f) or {})


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"] if isinstance(data, dict) else data


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
