"""Adapters: turn "call my agent" (HTTP API or Python function) into one normalised result.

Output contract (step 3, schema checks): every agent response is mapped to
text (str), tool_calls (list of {name, arguments}), final_state (dict),
handoff (bool) and optional tokens (int). A response that breaks the contract
becomes an error result instead of crashing the run.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests


@dataclass
class AgentResult:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    handoff: bool = False
    tokens: int | None = None
    error: str | None = None
    latency_ms: float = 0.0


def dig(obj, path):
    """Read a dotted path (a.b.0.c) from nested dicts/lists. Missing -> None."""
    if not path:
        return None
    cur = obj
    for part in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
        if cur is None:
            return None
    return cur


def _norm_calls(raw) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("tool_calls must be a list")
    out = []
    for c in raw:
        if not isinstance(c, dict):
            raise ValueError("each tool call must be an object")
        fn = c.get("function") or {}
        name = c.get("name") or fn.get("name")
        if not name:
            raise ValueError("tool call without a name")
        args = c.get("arguments", fn.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                raise ValueError(f"tool call '{name}' has non-JSON arguments")
        out.append({"name": name, "arguments": args or {}})
    return out


def normalise(mapped: dict) -> AgentResult:
    text = mapped.get("text")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    state = mapped.get("final_state") or {}
    if not isinstance(state, dict):
        raise ValueError("final_state must be an object")
    tokens = mapped.get("tokens")
    return AgentResult(
        text=text,
        tool_calls=_norm_calls(mapped.get("tool_calls")),
        final_state=state,
        handoff=bool(mapped.get("handoff")),
        tokens=int(tokens) if isinstance(tokens, (int, float)) else None,
    )


_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _lookup(key: str, case: dict):
    if key == "input":
        return case.get("input", "")
    if key == "case_id":
        return case.get("id", "")
    if key.startswith("context."):
        return (case.get("context") or {}).get(key[len("context."):])
    return None


def render(template, case: dict):
    """Fill {{input}}, {{case_id}}, {{context.key}} in a request template.
    A string that is exactly one placeholder keeps the original type."""
    if isinstance(template, str):
        m = _PLACEHOLDER.fullmatch(template.strip())
        if m:
            return _lookup(m.group(1), case)
        return _PLACEHOLDER.sub(lambda mm: str(_lookup(mm.group(1), case) or ""), template)
    if isinstance(template, dict):
        return {k: render(v, case) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, case) for v in template]
    return template


class HttpAgent:
    def __init__(self, cfg: dict):
        h = cfg.get("http") or {}
        if not h.get("url"):
            raise ValueError("agent config: http.url is required")
        self.url = h["url"]
        self.method = h.get("method", "POST").upper()
        self.headers = {k: v for k, v in (h.get("headers") or {}).items() if v}
        self.timeout = h.get("timeout_s", 30)
        self.template = h.get("request_template") or {"message": "{{input}}"}
        self.response_map = h.get("response_map") or {"text": "text"}

    def describe(self) -> str:
        u = urlparse(self.url)
        return f"http {u.scheme}://{u.hostname}{':' + str(u.port) if u.port else ''}{u.path}"

    def __call__(self, case: dict) -> AgentResult:
        start = time.perf_counter()
        body = render(self.template, case)
        try:
            r = requests.request(self.method, self.url, json=body,
                                 headers=self.headers, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000
            if r.status_code >= 400:
                return AgentResult(error=f"HTTP {r.status_code}", latency_ms=latency)
            data = r.json()
            res = normalise({k: dig(data, p) for k, p in self.response_map.items()})
            res.latency_ms = latency
            return res
        except ValueError as e:
            return AgentResult(error=f"malformed response: {e}",
                               latency_ms=(time.perf_counter() - start) * 1000)
        except requests.RequestException as e:
            return AgentResult(error=f"request failed: {type(e).__name__}",
                               latency_ms=(time.perf_counter() - start) * 1000)


class PythonAgent:
    """Calls module:function(case) -> dict with text/tool_calls/final_state/handoff."""

    def __init__(self, cfg: dict):
        spec = (cfg.get("python") or {}).get("callable")
        if not spec or ":" not in spec:
            raise ValueError("agent config: python.callable must look like 'module:function'")
        sys.path.insert(0, os.getcwd())
        mod, fn = spec.split(":", 1)
        self.spec = spec
        self.fn = getattr(importlib.import_module(mod), fn)

    def describe(self) -> str:
        return f"python {self.spec}"

    def __call__(self, case: dict) -> AgentResult:
        start = time.perf_counter()
        try:
            out = self.fn({"id": case.get("id"), "input": case.get("input"),
                           "context": case.get("context") or {}})
            res = normalise(out or {})
        except Exception as e:  # the agent is user code: never let it crash the run
            return AgentResult(error=f"{type(e).__name__}: {e}",
                               latency_ms=(time.perf_counter() - start) * 1000)
        res.latency_ms = (time.perf_counter() - start) * 1000
        return res


def build_adapter(cfg: dict):
    kind = cfg.get("type", "http")
    if kind == "http":
        return HttpAgent(cfg)
    if kind == "python":
        return PythonAgent(cfg)
    raise ValueError(f"agent config: unknown type '{kind}' (use http or python)")
