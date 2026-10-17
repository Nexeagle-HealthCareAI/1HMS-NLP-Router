"""Layer 2 - Memory & Retrieval  (flowchart: Redis Cache → RAG Retriever → Long-Term Memory Store).

This module is the single entry point for ALL data retrieval in the HMS Router.
It wires together the three retrieval sub-systems shown in the flowchart:

  ┌─────────────────────────────────────────────────────────────────┐
  │  LAYER 2 — MEMORY & RETRIEVAL                                   │
  │                                                                 │
  │  1. Redis Cache          session state + symptom context        │
  │     └─► SessionCache     thin read interface on top of          │
  │                          session.py / accumulator.py keys       │
  │                                                                 │
  │  2. RAG Retriever        Vector DB → symptom-specialist corpus  │
  │     └─► VectorRetriever  wraps specialist_rag.SpecialistRetriever│
  │                          embed → ANN search → rerank → top-K    │
  │                                                                 │
  │  3. Long-Term Memory Store  novel symptom phrases → feedback    │
  │     └─► LongTermStore    wraps learning.py Redis keys           │
  │                          (novel_counts, feedback:cluster_votes, │
  │                           correction_queue, review_queue)        │
  └─────────────────────────────────────────────────────────────────┘

Public API
----------
  RetrievalLayer(redis_client, ...)
      .session   → SessionCache
      .vector    → VectorRetriever
      .ltm       → LongTermStore
      .retrieve(state, severity) → RetrievalResult  # one-call convenience

The RetrievalLayer.retrieve() method executes the full Layer-2 flowchart:
  1. Pull accumulated symptom context from Redis Cache.
  2. Run the RAG retrieval (embed → vector search → rerank → top-K clusters).
  3. Apply long-term feedback boosts from the LTM store.
  4. Return a RetrievalResult with everything the routing engine (Layer 4) needs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .accumulator import AccumulatedState, SymptomAccumulator
from .config import Settings
from .specialist_rag import (
    Cluster,
    Recommendation,
    RoutingResult,
    SpecialistRetriever,
    load_corpus,
)

log = logging.getLogger("hms_router.retrieval")

# Set HMS_USE_CHROMA=1 to enable persistent ChromaDB vector store.
# Falls back to in-memory NumPy store if chromadb is not installed.
_USE_CHROMA_DEFAULT = os.getenv("HMS_USE_CHROMA", "0") == "1"


# =============================================================================
# 1. SESSION CACHE  — Redis Cache layer (session state + symptom context)
# =============================================================================
class SessionCache:
    """Read-only view of the Redis session keys written by session.py and accumulator.py.

    The cache does NOT write to Redis — all writes stay in SessionManager /
    SymptomAccumulator.  This class is purely a retrieval interface, so Layer 2
    can pull context without importing the full pipeline.

    Redis keys read:
        session:{id}            HASH  created_at, last_active
        session:{id}:turns      INT   turn counter
        session:{id}:buffer     LIST  raw message records  (session.py)
        session:{id}:context    HASH  language, script, intensity, asked, done
        session:{id}:symptoms   LIST  symptom objects per turn  (accumulator.py)
    """

    def __init__(self, redis_client, ttl_seconds: int = 1800):
        self.r = redis_client
        self.ttl = ttl_seconds

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _k(sid: str, suffix: str = "") -> str:
        return f"session:{sid}{suffix}"

    # ------------------------------------------------------------------ reads
    def exists(self, session_id: str) -> bool:
        """True when the session has not expired."""
        return bool(self.r.exists(self._k(session_id)))

    def turn_count(self, session_id: str) -> int:
        v = self.r.get(self._k(session_id, ":turns"))
        return int(v) if v else 0

    def context(self, session_id: str) -> dict[str, str]:
        """Returns the context hash: language, script, intensity, asked, done."""
        return self.r.hgetall(self._k(session_id, ":context")) or {}

    def message_history(self, session_id: str) -> list[dict]:
        """Ordered list of raw + normalized message records (oldest first)."""
        return [json.loads(x) for x in self.r.lrange(self._k(session_id, ":buffer"), 0, -1)]

    def symptom_history(self, session_id: str) -> list[dict]:
        """Ordered list of symptom objects stored by the accumulator (oldest first)."""
        return [json.loads(x) for x in self.r.lrange(self._k(session_id, ":symptoms"), 0, -1)]

    def all_normalized_texts(self, session_id: str) -> list[str]:
        """Every normalized user message in this session (for the safety net)."""
        return [m.get("normalized", "") for m in self.message_history(session_id)]

    def session_summary(self, session_id: str) -> dict[str, Any]:
        """Convenience: everything the routing engine needs from the cache in one call."""
        meta = self.r.hgetall(self._k(session_id)) or {}
        return {
            "session_id":       session_id,
            "active":           self.exists(session_id),
            "turn":             self.turn_count(session_id),
            "context":          self.context(session_id),
            "message_history":  self.message_history(session_id),
            "symptom_history":  self.symptom_history(session_id),
            "created_at":       meta.get("created_at"),
            "last_active":      meta.get("last_active"),
        }


# =============================================================================
# 2. VECTOR RETRIEVER  — RAG Retriever (Vector DB → symptom-specialist corpus)
# =============================================================================
@dataclass
class VectorSearchResult:
    """Raw result from one vector DB query (before the routing engine runs)."""
    cluster_id: str
    specialist: str
    vector_score: float       # raw cosine similarity from ANN search
    rerank_score: float       # score after SymptomOverlapReranker / CrossEncoder
    matched_symptoms: list[str] = field(default_factory=list)
    cluster_description: str = ""
    cluster_symptoms: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


class VectorRetriever:
    """Layer 2 RAG retrieval component.

    Thin wrapper around SpecialistRetriever that exposes:
      - search(query_text, k)       raw ANN results (VectorSearchResult list)
      - retrieve(state, severity)   full embed→search→rerank pipeline

    This class owns the Vector DB lifecycle (corpus load → TF-IDF fit → index).
    Swap ``embedder`` for a SentenceTransformerEmbedder to use MedBERT / dense vectors.
    """

    def __init__(
        self,
        corpus: list[Cluster] | None = None,
        dataset_path: str | Path | None = None,
        top_k: int = 10,
        min_confidence: float = 0.25,
        use_chroma: bool = False,
        chroma_persist_dir: str | Path | None = None,
        chroma_reset: bool = False,
    ):
        if use_chroma:
            # Build the retriever with a persistent ChromaVectorStore.
            # Falls back to in-memory if chromadb is not installed.
            try:
                from .chroma_store import build_chroma_retriever, DEFAULT_PERSIST_DIR
                persist = chroma_persist_dir or DEFAULT_PERSIST_DIR
                self._retriever = build_chroma_retriever(
                    persist_dir=persist,
                    corpus=corpus,
                    dataset_path=dataset_path,
                    top_k=top_k,
                    min_confidence=min_confidence,
                    reset_corpus=chroma_reset,
                )
                log.info("VectorRetriever: using ChromaDB at %s", persist)
            except ImportError:
                log.warning(
                    "chromadb not installed — falling back to InMemoryVectorStore. "
                    "Run: pip install chromadb"
                )
                use_chroma = False

        if not use_chroma:
            self._retriever = SpecialistRetriever(
                corpus=corpus,
                dataset_path=dataset_path,
                top_k=top_k,
                min_confidence=min_confidence,
            )
            log.info("VectorRetriever: using InMemoryVectorStore (NumPy)")

        log.info(
            "VectorRetriever ready: %d clusters in corpus, top_k=%d, min_confidence=%.2f",
            len(self._retriever.corpus), top_k, min_confidence,
        )

    # ------------------------------------------------------------------ corpus info
    @property
    def corpus(self) -> list[Cluster]:
        return self._retriever.corpus

    @property
    def corpus_size(self) -> int:
        return len(self._retriever.corpus)

    # ------------------------------------------------------------------ raw ANN search
    def search(self, query_text: str, k: int | None = None) -> list[VectorSearchResult]:
        """Embed query_text and return the top-K nearest clusters from the vector index.

        This is the pure retrieval step — no reranking, no routing logic.
        Useful for debugging what the vector index returns before the reranker runs.
        """
        retriever = self._retriever
        k = k or retriever.top_k
        vec = retriever.embedder.encode([query_text])[0]
        hits = retriever.store.query(vec, k)
        out: list[VectorSearchResult] = []
        for cid, score in hits:
            if score <= 0:
                continue
            c = retriever.by_id.get(cid)
            if c:
                out.append(VectorSearchResult(
                    cluster_id=cid,
                    specialist=c.specialist,
                    vector_score=round(float(score), 4),
                    rerank_score=0.0,
                    cluster_description=c.description,
                    cluster_symptoms=c.symptoms,
                ))
        return out

    # ------------------------------------------------------------------ full RAG pipeline
    def retrieve(self, state: dict, severity: dict | None = None) -> RoutingResult:
        """Run the full RAG pipeline: embed → ANN search → rerank → top-3 specialists.

        Args:
            state:    AccumulatedState.to_dict() from Layer 3 (accumulator).
            severity: SeverityResult.to_dict() from Layer 4 Step 5 (optional).

        Returns:
            RoutingResult with up to 3 Recommendations + emergency flag.
        """
        return self._retriever.route(state, severity)

    # ------------------------------------------------------------------ reindex (Step 8)
    def reindex(self, corpus: list[Cluster]) -> None:
        """Rebuild the vector index after the lexicon / corpus grows (Step 8)."""
        self._retriever.reindex(corpus)
        log.info("VectorRetriever reindexed: %d clusters", len(corpus))

    def set_boost(self, boost_fn) -> None:
        """Wire in the Step 8 feedback boost function (cluster_id → float)."""
        self._retriever.boost = boost_fn


# =============================================================================
# 3. LONG-TERM MEMORY STORE  — Novel symptom phrases → feedback loop
# =============================================================================
class LongTermStore:
    """Layer 2 read interface for the Long-Term Memory (learning.py Redis keys).

    Covers three Redis stores:
      A. novel_counts          HASH  token → count of unseen occurrences
      B. feedback:cluster_votes HASH  cluster_id → net positive votes (Step 8)
      C. correction_queue      LIST  wrong-routing corrections for human review
      D. review_queue          LIST  unknown tokens nominated for lexicon addition

    This class is READ-ONLY from the routing engine's perspective.
    Writes happen through learning.FeedbackCollector.submit() / NovelTokenReviewer.
    """

    # Redis key constants (must stay in sync with learning.py)
    NOVEL_COUNTS  = "novel_counts"
    CLUSTER_VOTES = "feedback:cluster_votes"
    CORRECTIONS   = "correction_queue"
    REVIEW_QUEUE  = "review_queue"
    REJECTED      = "novel_rejected"

    def __init__(self, redis_client, settings: Settings | None = None):
        self.r = redis_client
        self.cfg = settings or Settings()

    # ------------------------------------------------------------------ A. novel tokens
    def novel_token_count(self, token: str) -> int:
        """How many times this unknown token has been seen across all sessions."""
        v = self.r.hget(self.NOVEL_COUNTS, token)
        return int(v) if v else 0

    def top_novel_tokens(self, n: int = 20) -> list[dict]:
        """Top-N novel tokens by occurrence count (candidates for lexicon review)."""
        rejected = self.r.smembers(self.REJECTED) or set()
        raw = self.r.hgetall(self.NOVEL_COUNTS) or {}
        pairs = [(t, int(c)) for t, c in raw.items() if t not in rejected]
        pairs.sort(key=lambda x: -x[1])
        return [{"token": t, "count": c} for t, c in pairs[:n]]

    def pending_review_count(self) -> int:
        """Number of tokens currently sitting in the review_queue."""
        return self.r.llen(self.REVIEW_QUEUE)

    # ------------------------------------------------------------------ B. feedback votes
    def cluster_votes(self, cluster_id: str) -> int:
        """Net up/down vote count for a specialist cluster."""
        v = self.r.hget(self.CLUSTER_VOTES, cluster_id)
        return int(v) if v else 0

    def confidence_boost(self, cluster_id: str) -> float:
        """Translate votes → capped confidence boost (mirrors FeedbackCollector.boost)."""
        votes = max(0, self.cluster_votes(cluster_id))
        return min(self.cfg.feedback_max_boost, votes * self.cfg.feedback_boost_step)

    def all_cluster_votes(self) -> dict[str, int]:
        """All cluster vote counts as {cluster_id: vote_count}."""
        raw = self.r.hgetall(self.CLUSTER_VOTES) or {}
        return {k: int(v) for k, v in raw.items()}

    # ------------------------------------------------------------------ C. correction queue
    def pending_corrections(self, limit: int = 50) -> list[dict]:
        """Most recent wrong-routing corrections waiting for human review."""
        return [json.loads(x) for x in self.r.lrange(self.CORRECTIONS, 0, limit - 1)]

    def correction_count(self) -> int:
        return self.r.llen(self.CORRECTIONS)

    # ------------------------------------------------------------------ D. health summary
    def health_summary(self) -> dict[str, Any]:
        """Quick summary of the LTM store state — useful for monitoring dashboards."""
        return {
            "novel_token_types":    len(self.r.hgetall(self.NOVEL_COUNTS) or {}),
            "pending_review_queue": self.pending_review_count(),
            "pending_corrections":  self.correction_count(),
            "cluster_vote_entries": len(self.r.hgetall(self.CLUSTER_VOTES) or {}),
            "rejected_tokens":      self.r.scard(self.REJECTED),
            "checked_at":           time.time(),
        }


# =============================================================================
# 4. RETRIEVAL RESULT  — unified output of the Layer 2 pipeline
# =============================================================================
@dataclass
class RetrievalResult:
    """Everything Layer 4 (Specialist Routing Engine) needs from Layer 2.

    Produced by RetrievalLayer.retrieve().
    """
    session_summary: dict[str, Any]       # from SessionCache
    routing: RoutingResult                # from VectorRetriever (embed→search→rerank)
    ltm_health: dict[str, Any]            # snapshot of LTM store state
    query_text: str                       # the query sent to the vector store
    retrieval_ms: float                   # wall-clock time for the vector retrieval step

    def to_dict(self) -> dict:
        d = asdict(self)
        d["routing"] = self.routing.to_dict()
        return d


# =============================================================================
# 5. RETRIEVAL LAYER  — unified facade (the public API of this file)
# =============================================================================
class RetrievalLayer:
    """Facade that wires all three Layer-2 sub-systems together.

    Usage
    -----
    Instantiate once at startup (same as HMSRouterAgent):

        layer = RetrievalLayer(redis_client)

    Then per request:

        result = layer.retrieve(state_dict, severity_dict)
        specialists = result.routing.recommendations

    Or use the sub-systems directly:

        layer.session.message_history(sid)
        layer.vector.search("chest pain stomach pain", k=5)
        layer.ltm.top_novel_tokens(10)
    """

    def __init__(
        self,
        redis_client,
        settings: Settings | None = None,
        corpus: list[Cluster] | None = None,
        dataset_path: str | Path | None = None,
        use_chroma: bool | None = None,
        chroma_persist_dir: str | Path | None = None,
        chroma_reset: bool = False,
    ):
        cfg = settings or Settings()
        # use_chroma: explicit arg > env-var default
        _chroma = use_chroma if use_chroma is not None else _USE_CHROMA_DEFAULT

        self.session = SessionCache(redis_client, cfg.session_ttl_seconds)
        self.vector  = VectorRetriever(
            corpus=corpus,
            dataset_path=dataset_path,
            top_k=cfg.rag_top_k,
            min_confidence=cfg.rag_min_confidence,
            use_chroma=_chroma,
            chroma_persist_dir=chroma_persist_dir,
            chroma_reset=chroma_reset,
        )
        self.ltm = LongTermStore(redis_client, cfg)

        # Wire Step 8 feedback boost into the vector retriever
        self.vector.set_boost(self.ltm.confidence_boost)

        # Attach ChromaDB Long-Term Vector Memory (when enabled)
        self.chroma_ltm = None
        if _chroma:
            try:
                from .chroma_store import build_chroma_ltm
                persist = chroma_persist_dir or None
                self.chroma_ltm = build_chroma_ltm(
                    persist_dir=persist,
                    embedder=self.vector._retriever.embedder,
                )
                log.info("RetrievalLayer: ChromaDB LTM attached")
            except ImportError:
                log.warning("chromadb not installed — ChromaDB LTM disabled")

        log.info("RetrievalLayer ready (chroma=%s)", _chroma)

    # ------------------------------------------------------------------ main pipeline
    def retrieve(self, state: dict, severity: dict | None = None) -> RetrievalResult:
        """Execute the full Layer-2 flowchart for one request.

        Flow:
          1. Pull session summary from Redis Cache.
          2. Run VectorRetriever (embed → ANN search → rerank, with LTM boost).
          3. Snapshot the LTM store health.
          4. Return a RetrievalResult.

        Args:
            state:    AccumulatedState.to_dict() — the merged multi-turn symptom state.
            severity: SeverityResult.to_dict()  — optional, used to flag emergency routing.
        """
        session_id = state.get("session_id", "")

        # Step 1: Redis Cache — pull accumulated context
        sess_summary = self.session.session_summary(session_id) if session_id else {}

        # Step 2: RAG Retrieval — embed → vector DB → rerank → top-3
        t0 = time.perf_counter()
        routing = self.vector.retrieve(state, severity)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Step 3: LTM health snapshot (lightweight — just reads counters)
        ltm_health = self.ltm.health_summary()

        log.debug(
            "retrieve: session=%s query=%r candidates=%d emergency=%s ms=%.1f",
            session_id, routing.query, len(routing.recommendations),
            routing.emergency, retrieval_ms,
        )
        return RetrievalResult(
            session_summary=sess_summary,
            routing=routing,
            ltm_health=ltm_health,
            query_text=routing.query,
            retrieval_ms=round(retrieval_ms, 2),
        )

    # ------------------------------------------------------------------ reindex (Step 8)
    def reindex(self, corpus: list[Cluster]) -> None:
        """Rebuild the vector index after the lexicon grows (wired through LearningLoop)."""
        self.vector.reindex(corpus)
