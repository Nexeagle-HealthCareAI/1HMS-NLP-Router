"""Step 8: feedback, nightly review, human-approved lexicon growth, and the guards against poisoning."""
import csv
import json

import fakeredis
import pytest

from hms_router import HMSRouterAgent
from hms_router.config import Settings
from hms_router.learning import (CORRECT, NEW_PHRASE, WRONG, FeedbackCollector, NovelTokenReviewer, apply_approved)
from hms_router.lexicon import Lexicon


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def agent(r):
    return HMSRouterAgent(r, settings=Settings(novel_min_count=2))


# ---------------------------------------------------------------- feedback
def test_reinforce_boost_grows_but_is_capped(r):
    fb = FeedbackCollector(r, Settings(feedback_max_boost=0.05, feedback_boost_step=0.01))
    assert fb.boost("GI2") == 0
    for _ in range(3):
        fb.submit(CORRECT, cluster_id="GI2")
    assert fb.boost("GI2") == pytest.approx(0.03)
    for _ in range(50):
        fb.submit(CORRECT, cluster_id="GI2")
    assert fb.boost("GI2") == pytest.approx(0.05)                       # cap


def test_negative_votes_never_give_negative_boost_and_queue_for_human_review(r):
    fb = FeedbackCollector(r)
    fb.submit(WRONG, cluster_id="GI2", specialist="Gastroenterologist", symptoms=["fever"], suggested_specialist="General Physician")
    assert fb.boost("GI2") == 0
    c = fb.pending_corrections()
    assert c[0]["shown"] == "Gastroenterologist" and c[0]["suggested"] == "General Physician"


def test_one_vote_per_session_and_target(r):
    fb = FeedbackCollector(r)
    assert fb.submit(CORRECT, session_id="s1", cluster_id="GI2")["accepted"]
    assert fb.submit(CORRECT, session_id="s1", cluster_id="GI2") == {"accepted": False, "reason": "duplicate"}
    assert fb.submit(CORRECT, session_id="s2", cluster_id="GI2")["accepted"]
    assert fb.boost("GI2") == pytest.approx(0.02)                       # ballot stuffing from one session ignored


def test_new_phrase_is_normalised_validated_and_queued(r):
    fb = FeedbackCollector(r)
    assert fb.submit(NEW_PHRASE, phrase="घबराहट")["accepted"]
    assert not fb.submit(NEW_PHRASE, phrase="x")["accepted"]
    assert not fb.submit(NEW_PHRASE, phrase="call 98765 43210")["accepted"]       # digits / phone numbers rejected
    queued = [json.loads(x)["token"] for x in r.lrange("review_queue", 0, -1)]
    assert len(queued) == 1 and queued[0].isascii() and queued[0].startswith("ghab")   # Roman before queuing
    with pytest.raises(ValueError):
        fb.submit("bogus")


def test_feedback_stores_no_free_text_sentence(r):
    fb = FeedbackCollector(r)
    fb.submit(CORRECT, cluster_id="GI2")
    assert all("raw" not in x and "text" not in x for x in map(json.loads, r.lrange("feedback:log", 0, -1)))


# ---------------------------------------------------------------- nightly review
def _flood(agent, token, n):
    for _ in range(n):
        agent.feedback.submit(NEW_PHRASE, phrase=token)


def test_nightly_counts_filters_rare_and_suggests_nearest_word(agent, tmp_path):
    _flood(agent, "jakdan", 3)
    _flood(agent, "oncetoken", 1)                      # below novel_min_count
    csv_path = tmp_path / "novel_symptoms_review.csv"
    out = agent.learning.nightly(csv_path)
    assert out["drained"] == 4 and out["new_candidates"] == 1
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["token"] == "jakdan" and rows[0]["count"] == "3" and rows[0]["status"] == "pending"
    assert rows[0]["suggested_word"] == "jukaam"


def test_rerun_never_overwrites_human_decisions(agent, tmp_path):
    p = tmp_path / "rev.csv"
    _flood(agent, "jakdan", 3)
    agent.learning.nightly(p)
    rows = list(csv.DictReader(open(p)))
    rows[0].update(status="approved", canonical="anxiety", category="symptom")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    _flood(agent, "jakdan", 3)
    agent.learning.nightly(p)
    again = list(csv.DictReader(open(p)))
    assert len(again) == 1 and again[0]["status"] == "approved"


def test_queue_is_drained_atomically_and_tokens_pushed_later_survive(agent, tmp_path):
    _flood(agent, "jakdan", 3)
    NovelTokenReviewer(agent.r, agent.pipeline.lex, Settings(novel_min_count=2)).drain()
    assert agent.r.llen("review_queue") == 0
    agent.feedback.submit(NEW_PHRASE, phrase="jakdan")
    assert agent.r.llen("review_queue") == 1


# ---------------------------------------------------------------- applying approved rows
def _review(tmp_path, *rows):
    p = tmp_path / "rev.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, ["token", "count", "suggested_word", "distance", "status", "canonical", "category", "lang"])
        w.writeheader()
        for row in rows:
            w.writerow({"count": 5, "suggested_word": "", "distance": "", "lang": "hi", **row})
    return p


def test_pending_rows_change_nothing(tmp_path):
    lex = Lexicon.from_csv()
    grown, rep = apply_approved(_review(tmp_path, {"token": "jakdan", "status": "pending"}), lex)
    assert grown is lex and not rep.added_entries


def test_approved_needs_canonical_and_valid_category(tmp_path):
    lex = Lexicon.from_csv()
    p = _review(tmp_path,
                {"token": "jakdan", "status": "approved", "canonical": "", "category": "symptom"},
                {"token": "jhatka", "status": "approved", "canonical": "spasm", "category": "particle"})
    grown, rep = apply_approved(p, lex)
    assert grown is lex and set(rep.skipped) == {"jakdan", "jhatka"}


def test_approved_token_cannot_override_existing_meaning(tmp_path):
    lex = Lexicon.from_csv()
    p = _review(tmp_path, {"token": "bukhar", "status": "approved", "canonical": "cold", "category": "symptom"})
    grown, rep = apply_approved(p, lex)
    assert "bukhar" in rep.skipped and grown.phrase_index["bukhar"].canonical == "fever"


def test_alias_must_point_at_known_word(tmp_path):
    lex = Lexicon.from_csv()
    p = _review(tmp_path, {"token": "bukhaarr", "status": "alias", "canonical": "bukhar"},
                {"token": "evil", "status": "alias", "canonical": "nonexistentword"})
    grown, rep = apply_approved(p, lex)
    assert rep.added_aliases == ["bukhaarr"] and "evil" in rep.skipped and "evil" not in grown.aliases


def test_end_to_end_learning_changes_what_the_agent_understands(agent, tmp_path):
    before = agent.handle("mujhe jalan ho rahi hai")
    assert "burning sensation" not in before.state.symptoms
    _flood(agent, "jalan", 3)
    p = tmp_path / "rev.csv"
    agent.learning.nightly(p)
    rows = list(csv.DictReader(open(p)))
    rows[0].update(status="approved", canonical="burning sensation", category="symptom", lang="hi")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    rep = agent.learning.apply(p)
    assert rep.added_entries == ["jalan"]
    after = agent.handle("mujhe jalan ho rahi hai")
    assert "burning sensation" in after.state.symptoms
    assert agent.r.hget("novel_counts", "jalan") is None               # handled tokens leave the counter


def test_rejected_tokens_are_suppressed_forever(agent, tmp_path):
    p = tmp_path / "rev.csv"
    _flood(agent, "blahblah", 3)
    agent.learning.nightly(p)
    rows = list(csv.DictReader(open(p)))
    rows[0]["status"] = "rejected"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    agent.learning.apply(p)
    _flood(agent, "blahblah", 5)
    assert agent.learning.nightly(tmp_path / "rev2.csv")["new_candidates"] == 0


def test_learning_cannot_weaken_the_safety_net(agent, tmp_path):
    """Even if a reviewer 'approves' an emergency word as something harmless, the hardcoded net still fires."""
    p = _review(tmp_path, {"token": "dardd", "status": "approved", "canonical": "harmless", "category": "function"})
    agent.learning.apply(p)
    assert agent.handle("seene mein dard ho raha hai").kind == "emergency"


def test_feedback_boost_reaches_the_retriever(agent):
    base = agent.retriever.route({"symptoms": ["cough", "fever"]}).recommendations[0]
    for _ in range(20):
        agent.feedback.submit(CORRECT, cluster_id=base.cluster_id)
    boosted = agent.retriever.route({"symptoms": ["cough", "fever"]}).recommendations[0]
    assert boosted.confidence >= base.confidence and boosted.specialist == base.specialist
