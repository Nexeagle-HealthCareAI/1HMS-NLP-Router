"""The 74 curated symptom clusters (data/specialist_mapping.csv) plus clusters derived from
Datasets/combined_symptom_dataset.csv.

Since 2026-10-30 these are keyword DATA only: /chat's keyword tagger (keywords.py) uses how many clusters share a
symptom to tell discriminative words ("chakkar") from common ones ("fever"). No specialist is chosen from them; every
specialty decision comes from the /v1/route-symptom engine (route_bridge.py). The old retrieval engine that ranked
these clusters (specialist_rag.SpecialistRetriever) was removed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).parent / "data" / "specialist_mapping.csv"
DEFAULT_DATASET = Path(__file__).parent.parent / "Datasets" / "combined_symptom_dataset.csv"


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
    dataset_corpus = load_corpus_from_dataset(DEFAULT_DATASET) if DEFAULT_DATASET.exists() else []
    combined = dataset_corpus + load_corpus(DEFAULT_CORPUS)
    unique: dict[str, Cluster] = {}
    for cluster in combined:
        unique.setdefault(cluster.cluster_id, cluster)
    return tuple(unique.values())


def load_corpus(path: str | Path | None = None) -> list[Cluster]:
    """Load the dataset-derived default corpus, or a mapping CSV when given."""
    if path is None:
        return list(_default_corpus())      # cached: the 6k-row dataset is parsed once per process
    with open(path, newline="", encoding="utf-8") as f:
        return [Cluster(r["cluster_id"], r["specialist"], tuple(s.strip() for s in r["symptoms"].split(";")),
                        r["description"]) for r in csv.DictReader(f)]


# ----------------------------------------------------------------------------- embedders
