import fakeredis
import pytest

from hms_router import HMSRouterPipeline
from hms_router.script import normalize_script
from hms_router.spelling import phonetic_key


@pytest.fixture
def redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def pipe(redis):
    return HMSRouterPipeline(redis)


# ---------- Step 1
def test_new_then_continued_session(pipe, redis):
    a = pipe.ingest("pet dard ho raha hai")
    b = pipe.ingest("aur ulti bhi aa rahi hai", a.session_id)
    assert a.is_new_session and not b.is_new_session
    assert (a.turn, b.turn) == (1, 2)
    assert "pet dard" in b.combined_context and "ulti" in b.combined_context
    assert 0 < redis.ttl(f"session:{a.session_id}") <= 1800


def test_expired_session_starts_fresh(pipe, redis):
    a = pipe.ingest("sir dard")
    redis.delete(f"session:{a.session_id}")           # simulate TTL expiry
    b = pipe.ingest("bukhar", a.session_id)
    assert b.is_new_session and b.session_id != a.session_id and b.turn == 1


@pytest.mark.parametrize("text,lang", [
    ("kal raat se sir dard ho raha hai", "hinglish"),
    ("मुझे बुखार है", "hindi"),
    ("I have a fever since morning", "english"),
    ("mujhe fever hai", "hinglish"),
])
def test_language(pipe, text, lang):
    assert pipe.ingest(text).language.language == lang


def test_devanagari_to_roman():
    assert normalize_script("बुखार") == "bukhaar"
    assert normalize_script("सर दर्द!!") == "sar dard"
    assert normalize_script("Dardddd") == "dardd"


def test_empty_input_rejected(pipe):
    with pytest.raises(ValueError):
        pipe.ingest("   ")


# ---------- Step 2
def test_phonetic_keys():
    assert phonetic_key("bukhar") == phonetic_key("bukhaar") == "bkr"
    assert phonetic_key("sir") == phonetic_key("sar")


@pytest.mark.parametrize("text,symptom", [
    ("bukhar", "fever"), ("bukhaar", "fever"), ("bukar", "fever"),
    ("sir dard", "headache"), ("sar dard", "headache"), ("sardard", "headache"),
    ("pet me dard", "stomach pain"), ("pet mein dard", "stomach pain"),
    ("khasee", "cough"),
])
def test_doc_spelling_variants(pipe, text, symptom):
    assert symptom in pipe.process(text).correction.symptoms


def test_phrase_and_multi_symptom(pipe):
    r = pipe.process("seene mein dard aur saans nahi aa raha").correction
    assert r.symptoms == ["chest pain", "breathlessness"]


def test_unknown_token_flagged_and_queued(pipe, redis):
    r = pipe.process("mujhe ghutna jakad gaya hai").correction
    assert "ghutna" in r.unknown_tokens
    assert any("ghutna" in x for x in redis.lrange("review_queue", 0, -1))


def test_short_words_not_fuzzy_matched(pipe):
    # 3-letter junk must not be "corrected" into a symptom
    assert pipe.process("xyz").correction.symptoms == []
