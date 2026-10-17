"""ChromaDB vector store for the HMS NLP Router — two persistent collections.

Collections
-----------
  hms_corpus : Symptom-specialist cluster embeddings (RAG retrieval corpus).
               Drop-in replacement for the in-memory NumPy InMemoryVectorStore.
               Embeddings persist across restarts; cold-start re-indexing is
               skipped when the cluster count matches.

  hms_ltm    : Long-Term Memory. Stores embeddings of APPROVED novel symptom
               tokens and feedback audit events. Written ONLY after human review
               (status=approved in novel_symptoms_review.csv), enforcing the
               same safety rules as Step 8.

VectorStore Protocol (specialist_rag.py)
----------------------------------------
ChromaVectorStore fully implements:
    reset()             clear the collection (called by SpecialistRetriever.reindex)
    add(ids, vectors)   upsert pre-computed L2-normalised embeddings
    query(vector, k)    cosine ANN search -> list of (id, similarity) tuples

It is a ZERO-CHANGE drop-in for InMemoryVectorStore inside SpecialistRetriever.

Usage
-----
    from hms_router.chroma_store import build_chroma_retriever, build_chroma_ltm

    retriever = build_chroma_retriever()          # persistent RAG retriever
    ltm       = build_chroma_ltm(embedder=...)    # long-term memory store

    ltm.add_approved_token("aankhon mein sujan", "eye swelling", "Ophthalmologist")
    results = ltm.search_similar("eye pain redness", k=5)
    dups    = ltm.find_near_duplicates("eye swelling")

Install
-------
    pip install chromadb
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("hms_router.chroma_store")

# ---------------------------------------------------------------------------
# Paths & collection names
# ---------------------------------------------------------------------------
DEFAULT_PERSIST_DIR = Path(__file__).parent / "data" / "chroma_db"
CORPUS_COLLECTION   = "hms_corpus"    # RAG corpus (symptom-specialist clusters)
LTM_COLLECTION      = "hms_ltm"       # Long-term memory (approved novel symptoms)


# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------
def _chroma():
    """Import chromadb or raise a friendly error."""
    try:
        import chromadb  # noqa: PLC0415
        return chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is not installed. Run:  pip install chromadb"
        ) from exc


# =============================================================================
# ChromaVectorStore  —  DROP-IN replacement for InMemoryVectorStore
# =============================================================================
class ChromaVectorStore:
    """Persistent ChromaDB-backed vector store.

    Satisfies the VectorStore Protocol in specialist_rag.py so it can be passed
    directly as the ``store`` argument of SpecialistRetriever with no other
    changes needed.

    Vectors are pre-computed by the project Embedder (TF-IDF or
    SentenceTransformerEmbedder) and pushed into Chroma as raw float lists.
    Chroma's own embedding function is disabled; cosine similarity is used.

    Args:
        persist_dir:     Directory for ChromaDB WAL + HNSW index files.
        collection_name: Name of the Chroma collection.
        reset_on_init:   Wipe the collection on startup (useful for tests /
                         forced reindex after dataset updates).
    """

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = CORPUS_COLLECTION,
        reset_on_init: bool = False,
    ):
        chromadb = _chroma()
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._col_name = collection_name

        if reset_on_init:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass

        # cosine distance in Chroma (0 = identical, 2 = opposite)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaVectorStore: collection=%r  path=%s  docs=%d",
            collection_name, self._persist_dir, self._col.count(),
        )

    # ------------------------------------------------------------------
    # VectorStore Protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe the collection and re-create it.

        Called by SpecialistRetriever.reindex() before re-embedding the corpus.
        """
        self._client.delete_collection(self._col_name)
        self._col = self._client.get_or_create_collection(
            name=self._col_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.debug("ChromaVectorStore.reset: collection %r cleared", self._col_name)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Upsert pre-computed (L2-normalised) embeddings into the collection.

        Large corpora are batched in chunks of 500 to stay within Chroma limits.
        """
        if not ids:
            return
        embeddings = vectors.tolist()
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            self._col.upsert(
                ids=ids[i : i + BATCH],
                embeddings=embeddings[i : i + BATCH],
            )
        log.debug("ChromaVectorStore.add: upserted %d vectors", len(ids))

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Cosine ANN search. Returns list of (cluster_id, cosine_similarity).

        Chroma returns cosine distance in [0, 2]; converted to similarity with
        sim = 1 - dist/2  so the range matches the in-memory dot-product store.
        """
        n = min(k, self._col.count())
        if n == 0:
            return []
        result = self._col.query(
            query_embeddings=[vector.tolist()],
            n_results=n,
            include=["distances"],
        )
        ids_out   = result["ids"][0]
        distances = result["distances"][0]
        sims      = [max(0.0, 1.0 - d / 2.0) for d in distances]
        return list(zip(ids_out, sims))

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of vectors currently indexed."""
        return self._col.count()

    def collection_name(self) -> str:
        return self._col_name

    def persist_dir(self) -> Path:
        return self._persist_dir


# =============================================================================
# LongTermVectorMemory  —  hms_ltm ChromaDB collection
# =============================================================================
class LongTermVectorMemory:
    """Persistent long-term memory for APPROVED novel symptom knowledge.

    What is stored
    --------------
    approved_symptom entries
        After a human reviews novel_symptoms_review.csv and sets
        status=approved, LearningLoop.apply() calls add_approved_token().
        The canonical symptom phrase is embedded and stored permanently so the
        system can:
          (a) Find semantically similar known symptoms for fuzzy matching.
          (b) Detect near-duplicate submissions in future review batches.
          (c) Build a growing, restartable medical vocabulary index.

    feedback_event entries
        Durable audit trail for correct_match / wrong_specialist feedback events.
        Supplements the capped Redis correction_queue with a persistent record.

    Safety rules (identical to Step 8)
    ------------------------------------
    * NOTHING is written automatically. Only human-approved tokens are stored.
    * Patient sentences are NEVER stored — only canonical symptom names and
      truncated symptom lists (max 10 items).
    * Embeddings come from the shared project Embedder (no extra model needed).

    Args:
        persist_dir: Same directory as ChromaVectorStore (shared Chroma client).
        embedder:    TfidfEmbedder or SentenceTransformerEmbedder from the pipeline.
                     Set later with set_embedder() if not available at construction.
    """

    TYPE_SYMPTOM  = "approved_symptom"
    TYPE_FEEDBACK = "feedback_event"

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        embedder=None,
    ):
        chromadb = _chroma()
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._col = self._client.get_or_create_collection(
            name=LTM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder
        log.info(
            "LongTermVectorMemory ready: %d entries in collection %r",
            self._col.count(), LTM_COLLECTION,
        )

    def set_embedder(self, embedder) -> None:
        """Wire in (or hot-swap) the embedder after construction."""
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_approved_token(
        self,
        token: str,
        canonical: str,
        specialist_hint: str = "",
        category: str = "symptom",
        lang: str = "hi",
        metadata_extra: dict | None = None,
    ) -> bool:
        """Embed and store an approved novel symptom token.

        Args:
            token:           Raw Hinglish variant, e.g. "aankhon mein sujan".
            canonical:       English canonical form, e.g. "eye swelling".
            specialist_hint: Specialist most associated with this symptom.
            category:        Lexicon category: symptom / body_part / time / intensity.
            lang:            Language code (hi / en).
            metadata_extra:  Additional metadata dict to attach.

        Returns:
            True if stored, False if already present (fully idempotent).
        """
        doc_id   = "sym_" + token.lower().replace(" ", "_")
        existing = self._col.get(ids=[doc_id], include=[])
        if existing["ids"]:
            log.debug("LTM: token %r already stored — skipping", token)
            return False

        text = " ".join(filter(None, [token, canonical, specialist_hint]))
        meta: dict[str, Any] = {
            "type":       self.TYPE_SYMPTOM,
            "token":      token,
            "canonical":  canonical,
            "specialist": specialist_hint,
            "category":   category,
            "lang":       lang,
            "added_at":   time.time(),
        }
        if metadata_extra:
            meta.update(metadata_extra)

        kwargs: dict[str, Any] = dict(ids=[doc_id], documents=[text], metadatas=[meta])
        if self._embedder is not None:
            vec = self._embedder.encode([text])[0]
            kwargs["embeddings"] = [vec.tolist()]

        self._col.add(**kwargs)
        log.info(
            "LTM: stored approved token %r -> canonical=%r specialist=%r",
            token, canonical, specialist_hint,
        )
        return True

    def add_feedback_event(
        self,
        session_id: str,
        kind: str,
        cluster_id: str | None = None,
        specialist: str | None = None,
        symptoms: list[str] | None = None,
    ) -> None:
        """Store a feedback audit event in LTM.

        Only canonical symptom names are stored (max 10), never full patient sentences.

        Args:
            session_id:  Session that produced the feedback.
            kind:        correct_match | wrong_specialist | new_symptom_phrase.
            cluster_id:  Cluster the feedback refers to.
            specialist:  Specialist that was shown.
            symptoms:    Active symptom list at the time (canonical names only).
        """
        event_id = f"fb_{session_id}_{kind}_{int(time.time() * 1000)}"
        sym_str  = ";".join((symptoms or [])[:10])
        meta: dict[str, Any] = {
            "type":       self.TYPE_FEEDBACK,
            "session_id": session_id or "",
            "kind":       kind,
            "cluster_id": cluster_id or "",
            "specialist": specialist or "",
            "symptoms":   sym_str,
            "ts":         time.time(),
        }
        self._col.add(
            ids=[event_id],
            documents=[f"feedback {kind} {specialist or cluster_id or ''} {sym_str}"],
            metadatas=[meta],
        )
        log.debug("LTM: feedback event stored: %r", event_id)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query: str,
        k: int = 5,
        type_filter: str | None = None,
    ) -> list[dict]:
        """Semantic search in the LTM collection.

        Args:
            query:       Free-text query (e.g. "eye pain redness swelling").
            k:           Max results to return.
            type_filter: Restrict results to approved_symptom or feedback_event.

        Returns:
            List of dicts with keys: id, score, and all metadata fields.
        """
        if self._embedder is None:
            log.warning("LTM.search_similar: no embedder set — returning empty list")
            return []
        n = min(k, self._col.count())
        if n == 0:
            return []
        vec   = self._embedder.encode([query])[0]
        where = {"type": {"$eq": type_filter}} if type_filter else None
        kw: dict[str, Any] = dict(
            query_embeddings=[vec.tolist()],
            n_results=n,
            include=["metadatas", "distances"],
        )
        if where:
            kw["where"] = where
        res = self._col.query(**kw)
        out = []
        for rid, dist, meta in zip(
            res["ids"][0], res["distances"][0], res["metadatas"][0]
        ):
            out.append({"id": rid, "score": round(max(0.0, 1.0 - dist / 2.0), 4), **meta})
        return out

    def find_near_duplicates(self, token: str, threshold: float = 0.85) -> list[dict]:
        """Find approved symptoms semantically very close to token.

        Used in the nightly review job to flag near-duplicate submissions and
        avoid redundant lexicon entries. Returns entries with score >= threshold.
        """
        return [
            r for r in self.search_similar(token, k=10, type_filter=self.TYPE_SYMPTOM)
            if r["score"] >= threshold
        ]

    def approved_symptoms(self) -> list[dict]:
        """Return all approved symptom metadata entries (no vectors)."""
        res = self._col.get(
            where={"type": {"$eq": self.TYPE_SYMPTOM}},
            include=["metadatas"],
        )
        return list(res.get("metadatas") or [])

    def count(self) -> int:
        """Total number of entries in the LTM collection."""
        return self._col.count()

    def health_summary(self) -> dict:
        """Lightweight status dict for monitoring / health checks."""
        return {
            "collection":    LTM_COLLECTION,
            "persist_dir":   str(self._persist_dir),
            "total_entries": self._col.count(),
            "checked_at":    time.time(),
        }


# =============================================================================
# Factory helpers
# =============================================================================

def build_chroma_retriever(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    corpus=None,
    dataset_path: str | Path | None = None,
    top_k: int = 10,
    min_confidence: float = 0.25,
    reset_corpus: bool = False,
):
    """Build a SpecialistRetriever backed by ChromaDB (persistent vector store).

    This is the recommended factory for production use. The corpus embeddings
    are written to persist_dir on first run. Subsequent starts skip re-indexing
    if the cluster count matches, giving a fast cold-start.

    Pass reset_corpus=True whenever combined_symptom_dataset.csv is updated.

    Args:
        persist_dir:    Directory for ChromaDB files (WAL + HNSW index).
        corpus:         Pre-built Cluster list (overrides dataset_path).
        dataset_path:   Path to combined_symptom_dataset.csv.
        top_k:          ANN retrieval width (candidates before reranking).
        min_confidence: Minimum rerank score to include a recommendation.
        reset_corpus:   Wipe and rebuild the corpus collection on startup.

    Returns:
        SpecialistRetriever configured with ChromaVectorStore as its store.
    """
    from .specialist_rag import SpecialistRetriever  # local import avoids circulars

    store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=CORPUS_COLLECTION,
        reset_on_init=reset_corpus,
    )
    retriever = SpecialistRetriever(
        corpus=corpus,
        dataset_path=dataset_path,
        store=store,
        top_k=top_k,
        min_confidence=min_confidence,
    )

    stored      = store.count()
    corpus_size = len(retriever.corpus)
    if stored != corpus_size:
        log.info(
            "Chroma corpus mismatch (stored=%d, expected=%d): reindexing...",
            stored, corpus_size,
        )
        retriever.reindex(retriever.corpus)
    else:
        log.info("Chroma corpus up-to-date (%d clusters). Skipping reindex.", stored)

    return retriever


def build_chroma_ltm(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    embedder=None,
) -> LongTermVectorMemory:
    """Convenience factory for LongTermVectorMemory.

    Args:
        persist_dir: Same directory as the corpus store (shared Chroma client).
        embedder:    TfidfEmbedder or SentenceTransformerEmbedder instance.
    """
    return LongTermVectorMemory(persist_dir=persist_dir, embedder=embedder)
