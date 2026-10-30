"""Step 8 - long-term memory & continuous learning.

Flowchart                                         Here
  User feedback: helpful / not helpful
    Correct match      -> Reinforce (vector weight boost)   FeedbackCollector.submit("correct_match") -> boost()
    Wrong specialist   -> Log correction, human review      FeedbackCollector.submit("wrong_specialist") -> correction_queue
    New symptom phrase -> Novel Token Store (LPUSH)         FeedbackCollector.submit("new_symptom_phrase") -> review_queue
                          (Step 2 also pushes unknown tokens onto the same queue)
  Nightly batch review                                      NovelTokenReviewer.run()   -> novel_symptoms_review.csv
  Valid medical symptom? YES -> lexicon + re-embed          apply_approved() + LearningLoop.apply()
                         NO  -> discard / flag as noise     status = rejected
  Weekly NER fine-tune                                      NOT built here (needs a labelled corpus + a model)

Safety rules baked in
  * Tokens come from untrusted users. NOTHING is added to the lexicon automatically: a human sets status=approved
    (and the canonical meaning) in the review CSV. The nightly job only counts, de-noises and suggests.
  * Learning can never touch the emergency safety net (severity.EMERGENCY_RULES). Extra emergency phrases are
    added only by editing emergency_keywords.csv (severity.load_extra_rules), additive and reviewed.
  * Positive feedback is a small, capped confidence boost; it cannot create a match that retrieval didn't find.
  * Only canonical symptom names / single tokens are stored - never the patient's full sentence (health data).
  * One vote per (session, kind, cluster); queues are length-capped.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .config import Settings
from .lexicon import Entry, Lexicon
from .script import normalize_script

CORRECT, WRONG, NEW_PHRASE = "correct_match", "wrong_specialist", "new_symptom_phrase"
KINDS = (CORRECT, WRONG, NEW_PHRASE)
ALLOWED_CATEGORIES = {"symptom", "body_part", "time", "intensity", "function"}
REVIEW_COLUMNS = ["token", "count", "suggested_word", "distance", "status", "canonical", "category", "lang"]
MAX_QUEUE = 10_000


# ---------------------------------------------------------------------------- feedback collector
class FeedbackCollector:
    VOTES, CORRECTIONS, LOG = "feedback:cluster_votes", "correction_queue", "feedback:log"

    def __init__(self, redis_client, settings: Settings | None = None, review_queue_key: str | None = None):
        self.r = redis_client
        self.cfg = settings or Settings()
        self.review_key = review_queue_key or self.cfg.review_queue_key

    def submit(self, kind: str, *, session_id: str | None = None, cluster_id: str | None = None,
               specialist: str | None = None, symptoms: list[str] | None = None, phrase: str | None = None,
               suggested_specialist: str | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if session_id:                                          # one vote per (session, kind, target)
            vk = f"feedback:voted:{session_id}"
            if not self.r.sadd(vk, f"{kind}:{cluster_id or phrase or specialist}"):
                return {"accepted": False, "reason": "duplicate"}
            self.r.expire(vk, 86_400)

        if kind == CORRECT:
            if not cluster_id:
                raise ValueError("cluster_id required for correct_match")
            self.r.hincrby(self.VOTES, cluster_id, 1)
        elif kind == WRONG:
            if cluster_id:
                self.r.hincrby(self.VOTES, cluster_id, -1)
            self._push(self.CORRECTIONS, {"cluster_id": cluster_id, "shown": specialist,
                                          "symptoms": list(symptoms or [])[:10], "suggested": suggested_specialist,
                                          "ts": time.time()})
        else:
            token = normalize_script(phrase or "")
            if not (3 <= len(token) <= 60) or any(c.isdigit() for c in token):
                return {"accepted": False, "reason": "invalid phrase"}
            self._push(self.review_key, {"token": token, "source": "user"})
        self._push(self.LOG, {"kind": kind, "cluster_id": cluster_id, "ts": time.time()}, cap=50_000)
        return {"accepted": True}

    def _push(self, key: str, obj: dict, cap: int = MAX_QUEUE) -> None:
        pipe = self.r.pipeline()
        pipe.lpush(key, json.dumps(obj, ensure_ascii=False))
        pipe.ltrim(key, 0, cap - 1)
        pipe.execute()

    def boost(self, cluster_id: str) -> float:
        """Feedback weight for a specialty id (kept for feedback statistics; routing no longer uses it). Never negative, capped."""
        votes = int(self.r.hget(self.VOTES, cluster_id) or 0)
        return min(self.cfg.feedback_max_boost, max(0, votes) * self.cfg.feedback_boost_step)

    def pending_corrections(self) -> list[dict]:
        return [json.loads(x) for x in self.r.lrange(self.CORRECTIONS, 0, -1)]


# ---------------------------------------------------------------------------- nightly batch review
@dataclass
class ReviewCandidate:
    token: str
    count: int
    suggested_word: str = ""        # nearest known word - a likely misspelling if distance is small
    distance: int | None = None


class NovelTokenReviewer:
    COUNTS, REJECTED, PROCESSING = "novel_counts", "novel_rejected", "review_queue:processing"

    def __init__(self, redis_client, lexicon: Lexicon, settings: Settings | None = None):
        self.r, self.lex = redis_client, lexicon
        self.cfg = settings or Settings()

    def drain(self) -> int:
        """Move the queue into a persistent counter. RENAME makes it atomic: tokens pushed while the job
        runs go to a fresh queue and are picked up tomorrow."""
        try:
            self.r.rename(self.cfg.review_queue_key, self.PROCESSING)
        except Exception:                                       # queue empty / missing
            return 0
        items = [json.loads(x) for x in self.r.lrange(self.PROCESSING, 0, -1)]
        pipe = self.r.pipeline()
        for it in items:
            pipe.hincrby(self.COUNTS, it["token"], 1)
        pipe.delete(self.PROCESSING)
        pipe.execute()
        return len(items)

    def candidates(self) -> list[ReviewCandidate]:
        rejected = set(self.r.smembers(self.REJECTED))
        vocab = sorted(self.lex.vocab)
        # A token is "already known" only if it has a standalone phrase_index entry
        # (i.e. it maps directly to a canonical symptom/body_part/etc.).
        # Words that appear only as sub-words within multi-word phrase variants are
        # still genuinely novel and should be reviewed.
        standalone_known = self.lex.phrase_index
        out = []
        for tok, n in self.r.hgetall(self.COUNTS).items():
            n = int(n)
            if n < self.cfg.novel_min_count or tok in rejected or tok in standalone_known or tok in self.lex.aliases:
                continue
            best = process.extractOne(tok, vocab, scorer=Levenshtein.distance, score_cutoff=3)
            out.append(ReviewCandidate(tok, n, best[0] if best else "", int(best[1]) if best else None))
        return sorted(out, key=lambda c: -c.count)


    def run(self, csv_path: str | Path) -> dict:
        """Nightly job: drain -> count -> write NEW candidates to novel_symptoms_review.csv (status=pending).
        Existing rows (and any human decisions in them) are never overwritten."""
        drained = self.drain()
        path = Path(csv_path)
        existing = _read_review(path)
        known = {r["token"] for r in existing}
        fresh = [c for c in self.candidates() if c.token not in known]
        mode = "a" if path.exists() else "w"
        with open(path, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, REVIEW_COLUMNS)
            if mode == "w":
                w.writeheader()
            for c in fresh:
                w.writerow({"token": c.token, "count": c.count, "suggested_word": c.suggested_word,
                            "distance": "" if c.distance is None else c.distance, "status": "pending",
                            "canonical": "", "category": "", "lang": ""})
        return {"drained": drained, "new_candidates": len(fresh), "csv": str(path)}


def _read_review(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------- apply human decisions
@dataclass
class ApplyReport:
    added_entries: list[str] = field(default_factory=list)
    added_aliases: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)          # approved but invalid (reason in `notes`)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def apply_approved(csv_path: str | Path, lexicon: Lexicon, redis_client=None,
                   lexicon_csv: str | Path | None = None, variants_csv: str | Path | None = None
                   ) -> tuple[Lexicon, ApplyReport]:
    """status=approved -> new lexicon entry (needs canonical + category); status=alias -> spelling variant of an
    existing word (canonical = that word); status=rejected -> suppressed from future reviews. Rows still `pending`
    are left alone. Returns a NEW Lexicon; pass `lexicon_csv` / `variants_csv` to persist the additions."""
    rep, new_entries, new_aliases = ApplyReport(), [], {}
    rows = _read_review(Path(csv_path))
    for r in rows:
        tok, status = r["token"].strip().lower(), (r.get("status") or "").strip().lower()
        canon, cat = (r.get("canonical") or "").strip(), (r.get("category") or "").strip()
        lang = (r.get("lang") or "hi").strip() or "hi"
        if status == "approved":
            if not canon or cat not in ALLOWED_CATEGORIES:
                rep.skipped.append(tok)
                rep.notes.append(f"{tok}: approved but needs canonical + category in {sorted(ALLOWED_CATEGORIES)}")
            elif tok in lexicon.phrase_index or tok in lexicon.aliases:
                rep.skipped.append(tok)
                rep.notes.append(f"{tok}: already known - existing meaning wins")
            else:
                new_entries.append(Entry(tok, canon, cat, lang))
                rep.added_entries.append(tok)
        elif status == "alias":
            if canon in lexicon.vocab and tok not in lexicon.vocab:
                new_aliases[tok] = canon
                rep.added_aliases.append(tok)
            else:
                rep.skipped.append(tok)
                rep.notes.append(f"{tok}: alias target '{canon}' is not a known word")
        elif status == "rejected":
            rep.rejected.append(tok)

    grown = Lexicon(lexicon.entries + new_entries, {**lexicon.aliases, **new_aliases}) if (new_entries or new_aliases) else lexicon
    if lexicon_csv and new_entries:
        lp = Path(lexicon_csv)
        # Bug #9 fix: write the header row when creating the file fresh so that
        # Lexicon.from_csv() (which uses DictReader) does not treat the first data
        # row as a header on the next load.
        needs_header = not lp.exists() or lp.stat().st_size == 0
        with open(lp, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if needs_header:
                w.writerow(["variant", "canonical", "category", "lang"])
            w.writerows([[e.variant, e.canonical, e.category, e.lang] for e in new_entries])
    if variants_csv and new_aliases:
        vp = Path(variants_csv)
        needs_header = not vp.exists() or vp.stat().st_size == 0
        with open(vp, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if needs_header:
                w.writerow(["variant", "canonical"])
            w.writerows(new_aliases.items())
    if redis_client is not None:                              # handled tokens leave the counter / stay suppressed
        done = rep.added_entries + rep.added_aliases + rep.rejected
        if done:
            redis_client.hdel(NovelTokenReviewer.COUNTS, *done)
        if rep.rejected:
            redis_client.sadd(NovelTokenReviewer.REJECTED, *rep.rejected)
    return grown, rep


# ---------------------------------------------------------------------------- facade
class LearningLoop:
    """Wires Step 8 into a running system: feedback is collected for review, approved tokens grow the lexicon.
    (Specialty learning happens in the route-symptom model: CMS-approved examples + /admin/rebuild.)"""

    def __init__(self, redis_client, pipeline, settings: Settings | None = None):
        self.r, self.pipeline = redis_client, pipeline
        self.cfg = settings or pipeline.cfg
        self.feedback = FeedbackCollector(redis_client, self.cfg)
        self.reviewer = NovelTokenReviewer(redis_client, pipeline.lex, self.cfg)

    def nightly(self, review_csv: str | Path) -> dict:
        self.reviewer.lex = self.pipeline.lex
        return self.reviewer.run(review_csv)

    def apply(self, review_csv: str | Path, persist: bool = False) -> ApplyReport:
        from .lexicon import DEFAULT_LEXICON, DEFAULT_VARIANTS
        grown, rep = apply_approved(review_csv, self.pipeline.lex, self.r,
                                    DEFAULT_LEXICON if persist else None, DEFAULT_VARIANTS if persist else None)
        if grown is not self.pipeline.lex:
            self.pipeline.reload_lexicon(grown)                # spelling indexes rebuilt
            self.reviewer.lex = grown
        return rep
