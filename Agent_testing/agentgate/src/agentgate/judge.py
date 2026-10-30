"""LLM judge (step 5). Runs only after deterministic checks pass; results are cached.

Talks to any OpenAI-compatible /chat/completions endpoint (local server or hosted).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import requests

JUDGE_PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of an AI agent's reply. Judge only the reply "
    "against the criteria. Do not reward length, politeness or confidence by themselves. "
    'Respond with ONLY a JSON object: {"score": <integer 1-5>, "hard_fail": <true|false>, '
    '"reason": "<one short sentence>"}. Score anchors: 5 = meets every criterion; '
    "4 = meets all key criteria with minor flaws; 3 = acceptable with notable gaps; "
    "2 = significant problems; 1 = unacceptable. Set hard_fail to true whenever the "
    "hard-fail condition applies, whatever the score."
)


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(m.group(0))
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= int(score) <= 5:
        raise ValueError("score must be an integer from 1 to 5")
    return {"score": int(score), "hard_fail": bool(data.get("hard_fail")),
            "reason": str(data.get("reason", ""))[:200]}


class Judge:
    def __init__(self, cfg: dict, cache_path: str = ".cache/judge_cache.json"):
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        self.api_key = cfg.get("api_key") or ""
        self.model = cfg.get("model") or ""
        self.temperature = cfg.get("temperature", 0)
        self.timeout = cfg.get("timeout_s", 120)
        self.extra = cfg.get("extra_body") or {}
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def describe(self) -> str:
        return f"{self.model} (prompt v{JUDGE_PROMPT_VERSION})"

    def _key(self, rubric: dict, request: str, reply: str) -> str:
        blob = json.dumps([self.model, JUDGE_PROMPT_VERSION, rubric, request, reply], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        except OSError:
            pass

    def grade_text(self, rubric: dict, request: str, reply: str) -> dict:
        key = self._key(rubric, request, reply)
        if key in self.cache:
            return {**self.cache[key], "status": "scored", "cached": True}
        user = (f"USER REQUEST:\n{request}\n\nAGENT REPLY:\n{reply}\n\n"
                f"CRITERIA:\n{rubric.get('criteria', '')}\n\n"
                f"HARD-FAIL CONDITION:\n{rubric.get('hard_fail', 'none')}")
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user}], **self.extra}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=headers, timeout=self.timeout)
            if r.status_code >= 400:
                return {"status": "error", "reason": f"judge HTTP {r.status_code}"}
            verdict = extract_json(r.json()["choices"][0]["message"]["content"])
        except requests.RequestException as e:
            return {"status": "error", "reason": f"judge unreachable: {type(e).__name__}"}
        except (ValueError, KeyError, IndexError) as e:
            return {"status": "error", "reason": f"judge reply unusable: {e}"}
        self.cache[key] = verdict
        self._save()
        return {**verdict, "status": "scored", "cached": False}

    def grade(self, case: dict, reply: str) -> dict:
        return self.grade_text(case["rubric"], case["input"], reply)
