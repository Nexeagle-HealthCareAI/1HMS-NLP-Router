"""Timeout + circuit breaker around any `complete(system, user) -> str` function.

After `max_failures` consecutive failures the breaker opens for `cooldown` seconds and calls fail fast,
so a slow or dead provider costs the deterministic answer nothing.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable


class CircuitOpen(RuntimeError):
    pass


class GuardedCompletion:
    def __init__(self, complete: Callable[[str, str], str], timeout: float = 2.0, max_failures: int = 3,
                 cooldown: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._complete, self.timeout, self.max_failures, self.cooldown = complete, timeout, max_failures, cooldown
        self._clock = clock
        self._failures, self._opened_at = 0, None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if self._clock() - self._opened_at >= self.cooldown:     # half-open: let the next call try
                self._opened_at = None
                self._failures = self.max_failures - 1
                return False
            return True

    def __call__(self, system: str, user: str) -> str:
        if self.is_open:
            raise CircuitOpen("llm circuit open")
        future = self._pool.submit(self._complete, system, user)
        try:
            result = future.result(timeout=self.timeout)
        except (FutureTimeout, Exception) as exc:
            future.cancel()
            with self._lock:
                self._failures += 1
                if self._failures >= self.max_failures:
                    self._opened_at = self._clock()
            raise exc if not isinstance(exc, FutureTimeout) else TimeoutError("llm timeout") from exc
        with self._lock:
            self._failures = 0
        return result
