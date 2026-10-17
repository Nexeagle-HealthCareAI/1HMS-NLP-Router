"""Step 6 - RAG-based specialist retrieval.

Standalone. Input is the combined Step 4 state (dict) and, optionally, the Step 5 severity result (dict).

  symptom query -> embedding -> vector store (top-K similar clusters, cosine)
                -> specialist corpus (symptom cluster -> specialist, combined_symptom_dataset.csv)
                -> re-ranker (precision) -> top 3 distinct specialists

Everything heavy is behind a small interface so you can swap the defaults for the stack in your doc:
    Embedder    : TfidfEmbedder (default, numpy only)  |  SentenceTransformerEmbedder (MedBERT etc.)
    VectorStore : InMemoryVectorStore (default)        |  write a thin Chroma / Pinecone / Weaviate adapter
    Reranker    : SymptomOverlapReranker (default)     |  CrossEncoderReranker
The model-backed classes (SentenceTransformerEmbedder, CrossEncoderReranker) are not exercised by the tests.
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

DEFAULT_CORPUS = Path(__file__).parent / "data" / "specialist_mapping.csv"
DEFAULT_DATASET = Path(__file__).parent.parent / "Datasets" / "combined_symptom_dataset.csv"
EMERGENCY_SPECIALIST = "Emergency Medicine"
FALLBACKS = ["General Physician", "Telemedicine consult", "OPD triage desk"]     # pad to 3 outputs (the doc's "3-output rule")


# ----------------------------------------------------------------------------- corpus
@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    specialist: str
    symptoms: tuple[str, ...]
    description: str

    @property
    def doc(self) -> str:
        return " ; ".join(self.symptoms) + " ; " + self.description


@lru_cache(maxsize=1)
def _default_corpus() -> tuple:
    """Dataset-derived clusters + curated mapping, built once. Call _default_corpus.cache_clear()
    in a test that swaps DEFAULT_DATASET / DEFAULT_CORPUS."""
    from .dataset_builder import load_corpus_from_dataset
    return tuple(load_corpus_from_dataset(DEFAULT_DATASET)) + tuple(load_corpus(DEFAULT_CORPUS))


def load_corpus(path: str | Path | None = None) -> list[Cluster]:
    """Load the dataset-derived default corpus, or a mapping CSV when given."""
    if path is None:
        return list(_default_corpus())      # cached: the 6k-row dataset is parsed once per process
    with open(path, newline="", encoding="utf-8") as f:
        return [Cluster(r["cluster_id"], r["specialist"], tuple(s.strip() for s in r["symptoms"].split(";")),
                        r["description"]) for r in csv.DictReader(f)]


# ----------------------------------------------------------------------------- embedders
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...        # rows L2-normalised


def _features(text: str) -> list[str]:
    """Words plus whole symptom phrases ('stomach pain' as one feature) so exact symptom matches count more."""
    out = []
    for part in text.lower().split(";"):
        words = re.findall(r"[a-z]+", part)
        out += words
        if len(words) > 1 and len(words) <= 3:
            out.append("_".join(words))
    return out


class TfidfEmbedder:
    """Dependency-free TF-IDF vectors fitted on the corpus. Good enough because queries are canonical
    English symptom names (Steps 2-3 already did the Hinglish -> canonical work)."""

    def __init__(self, corpus_docs: list[str]):
        df: dict[str, int] = defaultdict(int)
        for d in corpus_docs:
            for f in set(_features(d)):
                df[f] += 1
        self.vocab = {f: i for i, f in enumerate(sorted(df))}
        n = len(corpus_docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[f])) + 1 for f in sorted(df)])

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for f in _features(t):
                if f in self.vocab:
                    m[r, self.vocab[f]] += 1
        m *= self.idf
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    """e.g. SentenceTransformerEmbedder('<your MedBERT / fine-tuned sentence model>'). Untested here."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


# ----------------------------------------------------------------------------- vector store
class VectorStore(Protocol):
    def reset(self) -> None: ...
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...   # (id, cosine similarity)


class InMemoryVectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.matrix = np.zeros((0, 0))

    def reset(self) -> None:
        self.ids, self.matrix = [], np.zeros((0, 0))

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids += ids
        self.matrix = vectors if not self.matrix.size else np.vstack([self.matrix, vectors])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        sims = self.matrix @ vector                    # vectors are unit length -> dot product = cosine
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


# ----------------------------------------------------------------------------- re-rankers
@dataclass
class Query:
    symptoms: list[str]
    negated: list[str]
    body_parts: list[str]
    text: str


@dataclass
class Candidate:
    cluster: Cluster
    vector_score: float
    rerank_score: float = 0.0
    matched: list[str] = field(default_factory=list)


class Reranker(Protocol):
    def rerank(self, query: Query, candidates: list[Candidate]) -> list[Candidate]: ...


class SymptomOverlapReranker:
    """Precision pass: how much of what the patient reported does this cluster explain, and does it rely on
    something the patient denied?"""

    def rerank(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        q, neg = set(query.symptoms), set(query.negated)
        for c in candidates:
            cs = set(c.cluster.symptoms)
            c.matched = [s for s in query.symptoms if s in cs]
            coverage = len(c.matched) / len(q) if q else 0.0              # share of the patient's symptoms explained
            fit = len(c.matched) / len(cs) if cs else 0.0                 # share of the cluster the patient has
            body = 0.08 if any(re.search(rf"\b{re.escape(b)}\b", c.cluster.doc.lower()) for b in query.body_parts) else 0.0
            penalty = min(0.3, 0.15 * len(cs & neg))
            c.rerank_score = max(0.0, min(1.0, 0.35 * c.vector_score + 0.4 * coverage + 0.17 * fit + body - penalty))
        return sorted(candidates, key=lambda c: -c.rerank_score)


class CrossEncoderReranker:
    """Cross-encoder precision pass (e.g. a MiniLM / medical cross-encoder). Untested here."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        scores = self.model.predict([(query.text, c.cluster.doc) for c in candidates])
        for c, s in zip(candidates, scores):
            c.rerank_score = float(1 / (1 + math.exp(-s)))
            c.matched = [x for x in query.symptoms if x in c.cluster.symptoms]
        return sorted(candidates, key=lambda c: -c.rerank_score)


# ----------------------------------------------------------------------------- result + retriever
@dataclass
class Recommendation:
    specialist: str
    confidence: float
    matched_symptoms: list[str]
    rationale: str
    cluster_id: str | None
    source: str                    # retrieval | emergency | fallback


@dataclass
class RoutingResult:
    recommendations: list[Recommendation]
    emergency: bool
    low_confidence: bool
    query: str

    def to_dict(self) -> dict:
        return asdict(self)


class SpecialistRetriever:
    def __init__(self, corpus: list[Cluster] | None = None, embedder: Embedder | None = None,
                 store: VectorStore | None = None, reranker: Reranker | None = None,
                 top_k: int = 10, min_confidence: float = 0.25, boost: Callable[[str], float] | None = None,
                 dataset_path: str | Path | None = None):
        """Initialise the retriever.

        Args:
            corpus:       Pre-built list of Cluster objects.  If None the corpus
                          is loaded from *dataset_path* (when given) or from
                          combined_symptom_dataset.csv plus the curated mapping.
            dataset_path: Path to combined_symptom_dataset.csv.  When supplied the
                          corpus is built on-the-fly from that file, giving the
                          retriever full coverage of all 6 292 labelled rows.
                          Ignored when *corpus* is explicitly passed.
        """
        if corpus is not None:
            self.corpus = corpus
        elif dataset_path is not None:
            from .dataset_builder import load_corpus_from_dataset  # lazy import
            self.corpus = load_corpus_from_dataset(dataset_path)
        else:
            self.corpus = load_corpus()
        self._custom_embedder = embedder is not None
        self.embedder = embedder
        self.store = store or InMemoryVectorStore()
        self.reranker = reranker or SymptomOverlapReranker()
        self.top_k, self.min_confidence = top_k, min_confidence
        self.boost = boost             # Step 8: cluster_id -> small additive confidence from positive feedback
        self.reindex(self.corpus)

    def reindex(self, corpus: list[Cluster]) -> None:
        """(Re-)embed the corpus into the vector store. Step 8 calls this after the lexicon / corpus grows.
        The default TF-IDF embedder is re-fitted because its vocabulary depends on the corpus; a custom
        (pre-trained) embedder is kept as is."""
        self.corpus = corpus
        self.by_id = {c.cluster_id: c for c in corpus}
        if not self._custom_embedder:
            self.embedder = TfidfEmbedder([c.doc for c in corpus])
        self.store.reset()
        self.store.add([c.cluster_id for c in corpus], self.embedder.encode([c.doc for c in corpus]))

    def route(self, state: dict, severity: dict | None = None) -> RoutingResult:
        negated = [s for s in state.get("negated_symptoms") or [] if s not in (state.get("symptoms") or [])]
        symptoms = [s for s in dict.fromkeys(state.get("symptoms") or []) if s not in negated]
        parts = list(dict.fromkeys(state.get("body_parts") or []))
        q = Query(symptoms, negated, parts, " ; ".join(symptoms + parts))
        emergency = bool(severity and (severity.get("emergency") or severity.get("level") == "EMERGENCY"))

        recs: list[Recommendation] = []
        if symptoms:
            hits = self.store.query(self.embedder.encode([q.text])[0], self.top_k)
            cands = [Candidate(self.by_id[i], s) for i, s in hits if s > 0]
            # Bug #7 fix: rerank returns a new sorted list. Apply boost and build `best` from
            # that same list so we never iterate a disconnected copy of the candidates.
            reranked = self.reranker.rerank(q, cands)
            for c in reranked:
                if self.boost and c.rerank_score > 0:                      # Step 8 "reinforce": bounded, never invents a match
                    c.rerank_score = min(1.0, c.rerank_score + self.boost(c.cluster.cluster_id))
            best: dict[str, Candidate] = {}
            for c in sorted(reranked, key=lambda c: -c.rerank_score):      # one entry per specialist (best cluster)
                best.setdefault(c.cluster.specialist, c)
            recs = [Recommendation(c.cluster.specialist, round(c.rerank_score, 3), c.matched, c.cluster.description,
                                   c.cluster.cluster_id, "retrieval") for c in best.values()
                    if c.rerank_score >= self.min_confidence]

        low = not recs
        if emergency:                                                      # emergency -> ER first, always
            recs = [r for r in recs if r.specialist != EMERGENCY_SPECIALIST]
            recs.insert(0, Recommendation(EMERGENCY_SPECIALIST, 1.0, symptoms, "Emergency flag raised in Step 5",
                                          None, "emergency"))
        for name in FALLBACKS:                                             # always 3 outputs
            if len(recs) >= 3:
                break
            if all(r.specialist != name for r in recs):
                recs.append(Recommendation(name, 0.0, [], "Fallback when no better match", None, "fallback"))
        return RoutingResult(recs[:3], emergency, low and not emergency, q.text)


if __name__ == "__main__":
    r = SpecialistRetriever()
    for st in ({"symptoms": ["stomach pain", "vomiting", "fever"], "body_parts": ["stomach"]},
               {"symptoms": ["headache", "dizziness"], "body_parts": ["head"]}):
        out = r.route(st)
        print([(x.specialist, x.confidence) for x in out.recommendations])
