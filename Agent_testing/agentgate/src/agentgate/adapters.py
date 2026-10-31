"""Adapters: turn "call my agent" (HTTP API or Python function) into one normalised result.

Output contract (step 3, schema checks): every agent response is mapped to
text (str), tool_calls (list of {name, arguments}), final_state (dict),
handoff (bool) and optional tokens (int). A response that breaks the contract
becomes an error result instead of crashing the run.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

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
    raw: dict | list | None = None


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


def normalise(mapped: dict, raw: dict | list | None = None) -> AgentResult:
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
        raw=raw,
    )


def _http_error(status: int, response) -> str:
    parts = []
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            parts.append(detail)
        errors = body.get("errors")
        if isinstance(errors, list):
            for error in errors:
                if not isinstance(error, dict):
                    continue
                location = ".".join(str(item) for item in error.get("loc", []))
                message = error.get("msg")
                if message:
                    parts.append(f"{location}: {message}" if location else str(message))
    if not parts:
        text = response.text.strip()
        if text:
            parts.append(text)
    suffix = f": {'; '.join(parts)[:500]}" if parts else ""
    return f"HTTP {status}{suffix}"


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
                return AgentResult(error=_http_error(r.status_code, r), latency_ms=latency)
            data = r.json()
            res = normalise({k: dig(data, p) for k, p in self.response_map.items()},
                            raw=data if isinstance(data, (dict, list)) else None)
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
            res = normalise(out or {}, raw=out if isinstance(out, (dict, list)) else None)
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


_INPUT_NAMES = {"input", "query", "message", "text", "symptom", "symptoms", "prompt"}
_ENTRYPOINT_NAMES = ("predict", "run", "invoke", "process_case")
_APP_FILENAMES = ("api.py", "app.py", "main.py", "server.py", "agent.py")
_ENDPOINT_PREFERENCE = ("/route-symptom", "/predict", "/invoke", "/agent", "/chat", "/run", "/route")


def _load_source_module(source: Path) -> tuple[object, bool]:
    module_name = f"agentgate_target_{source.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load Python source file: {source}")
    module = importlib.util.module_from_spec(spec)
    source_parent = str(source.parent)
    added_source_path = source_parent not in sys.path
    if added_source_path:
        sys.path.insert(0, source_parent)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        if added_source_path:
            sys.path.remove(source_parent)
        raise
    return module, added_source_path


def _source_file(source: str | Path) -> Path:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"agent source does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".py":
            raise ValueError(f"agent source must be a Python file or directory: {path}")
        return path
    candidates = [path / name for name in _APP_FILENAMES if (path / name).is_file()]
    if not candidates:
        raise ValueError(f"no Python entrypoint found under {path}; pass the main .py file with --source")
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(f"multiple possible agent entrypoints under {path}: {names}; pass one with --source")
    return candidates[0]


def _model_fields(model) -> dict:
    fields = getattr(model, "model_fields", None) or getattr(model, "__fields__", None)
    return fields if isinstance(fields, dict) else {}


def _field_required(field) -> bool:
    required = getattr(field, "is_required", None)
    if callable(required):
        return required()
    return bool(getattr(field, "required", False))


def _field_alias(name: str, field) -> str:
    return getattr(field, "alias", None) or name


class AutoAgent:
    """Discover and run a Python callable or FastAPI app from an agent source file."""

    def __init__(self, source: str | Path, endpoint: str | None = None):
        self.source = _source_file(source)
        self.module, self._added_source_path = _load_source_module(self.source)
        self._source_parent = str(self.source.parent)
        self._module_name = f"agentgate_target_{self.source.stem}"
        self.endpoint = endpoint
        self.app = None
        self.route = None
        self.client = None
        self._client_entered = False
        self.function = None

        try:
            from fastapi import FastAPI
        except ImportError:
            FastAPI = None
        if FastAPI is not None:
            self.app = next((value for value in vars(self.module).values()
                             if isinstance(value, FastAPI)), None)
        if self.app is not None:
            try:
                self.route = self._find_route(endpoint)
            except Exception:
                self.close()
                raise
            try:
                from fastapi.testclient import TestClient
            except ImportError as exc:
                self.close()
                raise ValueError("FastAPI's TestClient is required for automatic API testing") from exc
            self.client = TestClient(self.app, raise_server_exceptions=False)
            try:
                self.client.__enter__()
                self._client_entered = True
            except Exception:
                self.close()
                raise
            self.kind = "fastapi"
            return

        if endpoint:
            self.close()
            raise ValueError("--endpoint can only be used with a discovered FastAPI app")
        functions = [(name, getattr(self.module, name)) for name in _ENTRYPOINT_NAMES
                     if callable(getattr(self.module, name, None))]
        if not functions:
            self.close()
            raise ValueError(
                f"no FastAPI app or supported callable ({', '.join(_ENTRYPOINT_NAMES)}) found in {self.source}")
        if len(functions) > 1:
            names = ", ".join(name for name, _ in functions)
            self.close()
            raise ValueError(f"multiple agent callables found ({names}); keep one supported entrypoint in the source")
        self.function_name, self.function = functions[0]
        try:
            self.function_signature = inspect.signature(self.function)
        except (TypeError, ValueError):
            self.close()
            raise
        self.kind = "python"

    def _find_route(self, endpoint: str | None):
        routes = [route for route in self.app.routes
                  if "POST" in (getattr(route, "methods", None) or set())]
        if endpoint:
            route = next((r for r in routes if r.path == endpoint), None)
            if route is None:
                available = ", ".join(r.path for r in routes) or "none"
                raise ValueError(f"POST endpoint '{endpoint}' not found; available POST endpoints: {available}")
            return route
        if not routes:
            raise ValueError(f"no POST endpoint found in {self.source}")

        def score(route):
            path = route.path.lower().rstrip("/")
            rank = next((len(_ENDPOINT_PREFERENCE) - i
                         for i, marker in enumerate(_ENDPOINT_PREFERENCE) if marker in path), 0)
            if any(word in path for word in ("feedback", "health", "admin", "delete")):
                rank -= 10
            if getattr(route.dependant, "body_params", None):
                rank += 2
            return rank

        ranked = sorted(((score(route), route) for route in routes), key=lambda item: item[0], reverse=True)
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            paths = ", ".join(route.path for _, route in ranked if score(route) == ranked[0][0])
            raise ValueError(f"multiple equally suitable POST endpoints found: {paths}; specify --endpoint")
        return ranked[0][1]

    def describe(self) -> str:
        if self.kind == "fastapi":
            return f"auto {self.source} POST {self.route.path}"
        return f"auto {self.source} callable {self.function_name}"

    def close(self) -> None:
        try:
            if self.client is not None and self._client_entered:
                self.client.__exit__(None, None, None)
        finally:
            self.client = None
            self._client_entered = False
            sys.modules.pop(self._module_name, None)
            if self._added_source_path and self._source_parent in sys.path:
                sys.path.remove(self._source_parent)

    @staticmethod
    def _case_value(name: str, case: dict):
        key = name.lower()
        if key in _INPUT_NAMES:
            return case.get("input", "")
        if key in {"case", "request", "payload", "data"}:
            return case
        if key in {"context", "ctx"}:
            return case.get("context") or {}
        if key in {"case_id", "id"}:
            return case.get("id", "")
        context = case.get("context") or {}
        if key in context:
            return context[key]
        if key in case:
            return case[key]
        return None

    def _request_body(self, case: dict) -> dict:
        params = getattr(self.route.dependant, "body_params", [])
        if not params:
            return {}
        if len(params) == 1:
            param = params[0]
            fields = _model_fields(param.type_)
            if fields:
                body = {}
                for name, field in fields.items():
                    value = self._case_value(name, case)
                    if value is not None:
                        body[_field_alias(name, field)] = value
                    elif _field_required(field):
                        raise ValueError(f"required request field '{name}' cannot be mapped from this test case")
                return body
            if param.type_ is dict:
                value = self._case_value(param.name, case)
                if isinstance(value, dict):
                    return value
                text = case.get("input", "")
                return {"input": text, "query": text, "message": text, "text": text,
                        "symptom": text, "context": case.get("context") or {}}
            value = self._case_value(param.name, case)
            if value is not None:
                return {param.name: value}
            if _field_required(param.field_info):
                raise ValueError(f"required request field '{param.name}' cannot be mapped from this test case")
            return {}
        body = {}
        for param in params:
            value = self._case_value(param.name, case)
            if value is not None:
                body[param.alias or param.name] = value
            elif _field_required(param.field_info):
                raise ValueError(f"required request field '{param.name}' cannot be mapped from this test case")
        return body

    def _request_params(self, case: dict) -> tuple[str, dict]:
        params = {}
        path = self.route.path
        for param in getattr(self.route.dependant, "query_params", []):
            value = self._case_value(param.name, case)
            if value is not None:
                params[param.alias or param.name] = value
            elif _field_required(param.field_info):
                raise ValueError(f"required query parameter '{param.name}' cannot be mapped from this test case")
        for param in getattr(self.route.dependant, "path_params", []):
            value = self._case_value(param.name, case)
            if value is None:
                raise ValueError(f"required path parameter '{param.name}' cannot be mapped from this test case")
            path = path.replace("{" + param.name + "}", quote(str(value), safe=""))
        return path, params

    @staticmethod
    def _response_text(data) -> str:
        if isinstance(data, dict):
            for name in ("reply", "text", "answer", "message", "response", "output"):
                value = data.get(name)
                if isinstance(value, str):
                    return value
            specialties = data.get("specialties")
            if isinstance(specialties, list):
                top = specialties[0] if specialties else None
                if isinstance(top, dict) and isinstance(top.get("name"), str):
                    return top["name"]
            ids = data.get("specialtyIds")
            if isinstance(ids, list):
                return str(ids[0]) if ids else ""
        return json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)

    def _run_callable(self, case: dict, start: float) -> AgentResult:
        args, kwargs = [], {}
        for param in self.function_signature.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            value = self._case_value(param.name, case)
            if value is None and param.default is inspect.Parameter.empty:
                return AgentResult(error=f"required callable parameter '{param.name}' cannot be mapped",
                                   latency_ms=(time.perf_counter() - start) * 1000)
            if value is None:
                continue
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[param.name] = value
        try:
            out = self.function(*args, **kwargs)
            if inspect.isawaitable(out):
                import asyncio
                out = asyncio.run(out)
            result = normalise(out if isinstance(out, dict) else {"text": str(out)},
                               raw=out if isinstance(out, (dict, list)) else None)
        except Exception as exc:
            return AgentResult(error=f"{type(exc).__name__}: {exc}",
                               latency_ms=(time.perf_counter() - start) * 1000)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def __call__(self, case: dict) -> AgentResult:
        start = time.perf_counter()
        if self.kind == "python":
            return self._run_callable(case, start)
        try:
            path, params = self._request_params(case)
            response = self.client.request(
                "POST", path, json=self._request_body(case), params=params)
            latency = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return AgentResult(error=_http_error(response.status_code, response), latency_ms=latency)
            data = response.json()
            if not isinstance(data, (dict, list)):
                return AgentResult(error="malformed response: expected a JSON object or array",
                                   latency_ms=latency)
            state = data if isinstance(data, dict) else {"response": data}
            result = normalise({
                "text": self._response_text(data),
                "tool_calls": data.get("tool_calls", data.get("toolCalls", [])) if isinstance(data, dict) else [],
                "final_state": state,
                "handoff": data.get("handoff", False) if isinstance(data, dict) else False,
            }, raw=data)
            result.latency_ms = latency
            return result
        except (ValueError, TypeError) as exc:
            return AgentResult(error=f"request mapping failed: {exc}",
                               latency_ms=(time.perf_counter() - start) * 1000)
        except Exception as exc:
            return AgentResult(error=f"{type(exc).__name__}: {exc}",
                               latency_ms=(time.perf_counter() - start) * 1000)
