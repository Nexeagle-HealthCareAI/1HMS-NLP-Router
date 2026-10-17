"""HTTP client for the FastAPI layer. This is the ONLY way the Voice layer
reaches the NLP Brain -- it has no import of nlp_brain and no knowledge of
how symptom routing actually works, just the HTTP contract."""
import requests

from .config import API_BASE_URL


class SymptomRouterClient:
    """A small, focused HTTP client for the api/ layer's two endpoints this
    layer actually needs. Construct one per voice session (voice/cli.py's
    interactive() does), not per-request -- it's cheap and stateless, but
    there's no reason to rebuild it. Point `base_url` at Dev/Prod/localhost
    via the NLP_API_BASE_URL env var (see config.py) rather than
    hardcoding it here."""

    def __init__(self, base_url: str = API_BASE_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def route_symptom(self, query: str) -> dict:
        """POSTs to /route-symptom and returns the parsed JSON response.
        Raises requests.RequestException on network failure or a non-2xx
        status (including 429 from the API's rate limiter) -- callers
        decide how to surface that to the user."""
        response = requests.post(
            f"{self.base_url}/route-symptom",
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """GETs /health. Call this once before starting an interactive
        session (voice/cli.py does) to fail fast with a clear message if
        the API isn't reachable yet, instead of the first real query's
        error being the user's only clue."""
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
