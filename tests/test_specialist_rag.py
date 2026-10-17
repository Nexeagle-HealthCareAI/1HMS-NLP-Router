import numpy as np

from hms_router.specialist_rag import (CrossEncoderReranker, FALLBACKS, InMemoryVectorStore, SpecialistRetriever,
                            SymptomOverlapReranker, TfidfEmbedder, load_corpus)
from hms_router.dataset_builder import load_corpus_from_dataset

R = SpecialistRetriever()


def test_default_corpus_uses_combined_symptom_dataset():
    corpus = load_corpus()
    dataset_corpus = load_corpus_from_dataset()
    corpus_ids = {cluster.cluster_id for cluster in corpus}

    assert {cluster.cluster_id for cluster in dataset_corpus}.issubset(corpus_ids)
    assert len(corpus) > len(dataset_corpus)
    assert "GI1" in corpus_ids
    assert any(cluster.specialist == "Breast clinic" for cluster in corpus)


def names(res):
    return [r.specialist for r in res.recommendations]


def test_doc_example_gastro_first_gp_in_top3():
    res = R.route({"symptoms": ["stomach pain", "vomiting", "fever"], "body_parts": ["stomach"]})
    assert names(res)[0] == "Gastroenterologist" and "General Physician" in names(res)
    assert res.recommendations[0].matched_symptoms == ["stomach pain", "vomiting", "fever"]


def test_neuro_and_derm_and_cardio():
    assert names(R.route({"symptoms": ["headache", "dizziness"], "body_parts": ["head"]}))[0] == "Neurologist"
    assert names(R.route({"symptoms": ["itching"]}))[0] == "Dermatologist"
    assert names(R.route({"symptoms": ["chest pain", "breathlessness"]}))[0] in ("Cardiologist", "Pulmonologist")


def test_always_three_distinct_outputs():
    for st in ({"symptoms": ["cough"]}, {"symptoms": ["itching"]}, {"symptoms": []},
               {"symptoms": ["stomach pain", "vomiting"]}):
        out = names(R.route(st))
        assert len(out) == 3 and len(set(out)) == 3


def test_no_symptoms_is_low_confidence_and_falls_back():
    res = R.route({"symptoms": []})
    assert res.low_confidence and names(res) == FALLBACKS


def test_emergency_puts_er_first_even_when_ranked_elsewhere():
    res = R.route({"symptoms": ["chest pain"]}, {"emergency": True, "level": "EMERGENCY"})
    assert res.emergency and names(res)[0] == "Emergency Medicine" and res.recommendations[0].source == "emergency"
    assert names(res).count("Emergency Medicine") == 1


def test_negated_symptoms_not_in_query_and_penalised():
    res = R.route({"symptoms": ["cough"], "negated_symptoms": ["fever"]})
    assert "fever" not in res.query
    with_fever = R.route({"symptoms": ["cough", "fever"]})
    assert res.recommendations[0].confidence != with_fever.recommendations[0].confidence


def test_vector_store_and_embedder():
    docs = [c.doc for c in load_corpus()]
    emb = TfidfEmbedder(docs)
    v = emb.encode(["stomach pain ; vomiting"])
    assert np.isclose(np.linalg.norm(v[0]), 1.0)
    store = InMemoryVectorStore()
    ids = [c.cluster_id for c in load_corpus()]
    store.add(ids, emb.encode(docs))
    top = store.query(v[0], 3)
    assert top[0][1] >= top[1][1] >= top[2][1] and top[0][0].startswith("GI")


def test_custom_components_plug_in():
    class Const(SymptomOverlapReranker):
        def rerank(self, query, cands):
            cands = super().rerank(query, cands)
            for c in cands:
                c.rerank_score = 0.9 if c.cluster.specialist == "Orthopedist" else 0.3
            return sorted(cands, key=lambda c: -c.rerank_score)
    res = SpecialistRetriever(reranker=Const()).route({"symptoms": ["stomach pain", "back pain"]})
    assert names(res)[0] == "Orthopedist"
    assert CrossEncoderReranker  # importable; model-backed, not run here
