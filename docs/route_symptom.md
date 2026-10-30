# /route-symptom — developer guide

Stateless specialty routing: one symptom text in, ranked specialties with confidences, an emergency
flag, and a decision (`answer` / `clarify` / `abstain`) out. Built beside `/chat`, which is unchanged
(pinned by `tests/test_chat_contract.py`). Triage support only, never a diagnosis.

## Layout (dependencies point inward)

```
hms_router/
  domain/          models, taxonomy (30 specialtyIds), RoutingPolicy thresholds        pure, stdlib only
  safety/          red-flag rules (data/red_flags.csv), RuleDetector, ModelDetector, AnyOf, DenialAware
  rag/             corpus, TextEncoder, ExampleRetriever (sentence RAG), ClassifierScorer,
                   EvidenceVoteScorer, WeightedFusion, ModelBundle (train/save/load), MedicalVocabulary
  reasoning/       Verifier, Explainer, ClarifyingQuestions, NoReasoner, LlmReasoner (port only)
  application/     RouteSymptomService (use case), ChatService (wraps HMSRouterAgent), container
  adapters/        http/ (FastAPI routers, auth, rate limits, problem+json), llm/ (LangChain providers,
                   timeout + circuit breaker, redaction), settings.py (env)
  vita_graph/      Vita on LangGraph (served on /v2/agent/chat when VITA_LLM_PROVIDER is set)
  route_bridge.py  conversations (/chat, Vita) -> this engine
  textnorm.py      shared normalization (Devanagari -> Roman, "kam nahi" persistence fix, stopwords)
```

`domain`, `safety`, `rag`, `reasoning` and `application` import no FastAPI, Redis or LangChain. The
composition root (`application/container.py`) is the only place that picks concrete classes.

## Request flow

1. **Normalize** – `textnorm.preprocess`.
2. **Safety, always first** – explicitly denied spans are removed first ("saans lene mein koi dikkat
   nahi", "I don't have chest pain"; `safety/denial.py`), then red-flag phrases on the text plus canonical
   symptoms from /chat's spelling + NER stages, OR the emergency classifier at ≥ 0.6. Detectors can only
   add urgency. Bare "nahi" is never a denial ("saans nahi aa raha" stays an emergency).
3. **Retrieve + score** – 25 nearest labelled sentences (evidence) voted by similarity², fused 50/50
   with the classifier's probabilities. `ageGroup: adult` removes paediatrics.
4. **Coverage gate** – nothing similar in the corpus → `abstain` ("see a General Physician first").
   Also `abstain` when the query has no health word at all (`rag/vocabulary.py`: lexicon + seeds + red
   flags + a short clinical list, typo-tolerant), no emergency, and the corpus match is below 0.6
   ("cricket score today"). `notes` says "no health terms in the query".
5. **Decisive** (top ≥ 0.60, or a child with paediatrics in the top 3) → `answer`, 1–3 `specialtyIds`.
6. **Reasoner** (only if `HMS_LLM_PROVIDER` is set) – may choose among the candidates and must cite
   their evidence; the `Verifier` rejects anything outside the candidates, uncited, or with diagnosis
   or dosing language. Rejected → logged in `notes`, deterministic path continues.
7. Otherwise → `clarify`: one question (age first, then curated pairs, then "which fits best") and
   the top-5 hedge in `specialtyIds`, so Doctor Dekho keeps asking Groq exactly as before.

## Contract

```
POST /v1/route-symptom[?include=evidence]      (alias: POST /route-symptom)
headers: X-Api-Key (when HMS_API_KEYS is set), X-Request-Id (optional, echoed)
body:    {"query": "...", "topK": 5, "context": {"ageGroup": "child" | "adult"}}
200:     specialties[{specialtyId,name,confidence,evidence?}], emergency{flag,reasons,source,score,
         recommendedSpecialtyIds}, decision, clarify{kind,question,options,optionIds}|null, reason,
         method, decisive, noMatch, matchScore, specialtyIds, usedDefault, confidence, modelVersion, requestId
errors:  application/problem+json — 401, 403, 413, 422, 429 (Retry-After), 500, 503
```

Callers: check `emergency.flag` first (it can be true with `noMatch`). For `clarify.kind == "age"`,
resend the same query with `context.ageGroup` set to the chosen option id.

## Run, build, measure

```
pip install -r requirements.txt                       # + pip install "langgraph>=1.0" "langchain-core>=1.0" for LangGraph
python -m hms_router.rag.model_bundle build           # ~40 s -> hms_router/data/route_model.joblib
uvicorn api:app --port 5005                           # dev: no auth, in-memory Redis, LLM off
pytest -q                                             # legacy nlp_brain tests are skipped (tests/conftest.py)
python scripts/eval_route_symptom.py --dd-dir ../DoctorDekhoWeb/scripts/search-eval [--sweep]
python scripts/make_api_key.py doctordekho route      # prints a key + its HMS_API_KEYS entry
```

Production: `HMS_ENV=production` refuses to start without `HMS_API_KEYS` and `REDIS_URL`, and refuses
`HMS_CORS_ORIGINS=*`. All variables: `hms_router/adapters/settings.py`.

## Measured (2026-10-19, after the denial and off-topic fixes; model built from all three corpus files)

| Measure | Value |
| --- | --- |
| Held-out dataset sentences (1,259), top-1 / top-3 | 75% / 90% |
| Emergency recall on held-out Emergency rows (111) | 92% (rules alone 34%) |
| False emergency flags on other held-out rows (1,148) | 10% |
| Doctor Dekho set 1 (tuning, 71): answered correctly | 93% (was 87%); non-medical abstained 10/10 |
| Doctor Dekho set 2 (unseen, 50): answered correctly | 80% (was 76%; old nlp_brain router: 68%) |
| Set 2: precision of `answer` decisions | 100% (22 of 50 answered; rest clarify or abstain) |
| Set 2: non-medical abstained / medical wrongly abstained | 7 of 8 (was 5) / 1 of 42 (was 0) |
| Latency per request, warm | about 20 ms |

Thresholds were chosen on set 1 and held-out data only.

## Extending without editing the service

- New retriever (ChromaDB, multilingual embeddings): implement `rag.protocols.Retriever`; wire it in
  `application/container.py`; it must pass `tests/test_rag_safety_reasoning.py::test_retriever_contract`.
- New scorer: implement `rag.protocols.Scorer` and add it to the fusion weights.
- New red flag: add a row to `hms_router/data/red_flags.csv` (needs clinician review).
- LLM reasoner: set `HMS_LLM_PROVIDER` + `HMS_LLM_MODEL` after the patient-data decision is recorded.

## Known limits

- Red flags, English seed phrases, clarifying questions and reply templates need clinician review.
- Set 2 has now been scored after these fixes, so it is no longer unseen. Build a fresh set 3 before the
  next round of tuning.
- "My mother is 72 and needs regular health monitoring" is abstained: "health" and "monitoring" are not
  in the vocabulary. Left as found so set 2 stays honest; add generic words like these with set 3.
- "call me back later" gets a clarify ("back" is a body word). "behosh nahi hua bas chakkar" still raises
  a flag from the emergency classifier, which does not read negation (over-triage, the safe direction).
- Denials are pattern-based (problem noun + nahi; don't/never + have/get). Other phrasings still flag.
- `/chat` still uses its own cluster retriever and safety rules; moving it onto these services is phase 6.
