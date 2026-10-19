"""/chat must behave exactly as it did before the /route-symptom rebuild (see tests/chat_contract.py)."""
import json

import pytest

from tests.chat_contract import FIXTURE, _client, run


@pytest.fixture(scope="module")
def recorded():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current():
    with _client() as c:
        return json.loads(json.dumps(run(c), ensure_ascii=False, sort_keys=True))


@pytest.mark.parametrize("key", ["emergency_chest_pain", "emergency_devanagari", "followup_then_recommend",
                                 "negation", "english_multi", "unknown_words", "fever", "errors", "feedback"])
def test_chat_unchanged(recorded, current, key):
    assert current[key] == recorded[key]


def test_health_still_reports_redis(recorded, current):
    assert current["health"]["status"] == recorded["health"]["status"]
    assert current["health"]["redis"] == recorded["health"]["redis"]
