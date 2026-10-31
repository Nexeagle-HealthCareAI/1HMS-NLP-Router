# HMS NLP Router

Two services in one FastAPI app. **Triage support only - not a diagnosis.**

* **Vita** (`POST /v2/agent/chat`): the patient agent for the Doctor Dekho website and app and WhatsApp. One LangGraph
  conversation (`hms_router/vita_graph/`): code steps for safety, booking, taps and the visit; a Groq LLM agent for
  understanding and replies; the route-symptom engine for the specialist; doctors from 1HMS. See
  [docs/vita_graph.md](docs/vita_graph.md), [docs/vita_doctor_search.md](docs/vita_doctor_search.md),
  [docs/vita_whatsapp.md](docs/vita_whatsapp.md). `POST /v2/agent/search-intent` reads one search-bar query the same
  way (Doctor Dekho's search bar).
* **route-symptom** (`POST /v1/route-symptom`, alias `/route-symptom`): one symptom text -> ranked specialties, an
  emergency flag, and `answer` / `clarify` / `abstain`. A grounded RAG pipeline (labelled-sentence retrieval fused with
  a classifier, red-flag safety net first, optional fenced LLM reasoner). See [docs/route_symptom.md](docs/route_symptom.md).

```
pip install -r requirements.txt
python -m hms_router.rag.model_bundle build     # ~40 s; also built on first start if missing
uvicorn api:app --port 5005                     # dev defaults: no auth, in-memory Redis, LLM off
pytest -q
```

Vita's LLM: `VITA_LLM_PROVIDER=groq`, `GROQ_API_KEY`, `VITA_LLM_MODEL` (default `openai/gpt-oss-120b`). `GET /health`
shows whether it is on (`vita.llm`, `vita.model`, `vita.mode`). Without it Vita still runs its code steps (safety,
booking, plain doctor searches) and says "please try again, or talk to our team" for anything else.

One brain: the old `/chat` triage agent (`HMSRouterAgent`: `agent.py`, `accumulator.py`, `output.py`, `keywords.py`,
`learning.py`, `symptom_clusters.py`, `dataset_builder.py`) and Vita's rules engine were removed on 2026-10-31.
`POST /chat` remains only as a deprecated alias of Vita for Doctor Dekho's `/api/vita/app-chat` until that calls
`/v2/agent/chat` everywhere. Never add a folder named `api/`: it would shadow `api.py`.

## Repository layout

```
hms-nlp-router/
├── hms_router/
│   ├── adapters/http/       # FastAPI routers: agent_routes (Vita, search-intent, /chat alias), route_routes, admin, health
│   ├── application/         # doctor_chat.py (Vita's code steps), booking, visit, journey, emergency help,
│   │                        # route_symptom, search_intent, retraining
│   ├── vita_graph/          # Vita on LangGraph: graph, engine, tools, prompts, checker, memory, learning, evals
│   ├── rag/  domain/  reasoning/  safety/  channels/   # route-symptom engine, shared safety, WhatsApp rendering
│   ├── doctor_intent.py  search_spelling.py   # reading a doctor search (spelling-tolerant)
│   ├── doctors.py  booking_client.py           # 1HMS (MCP) doctor directory, public booking API
│   ├── route_bridge.py      # a conversation -> the route-symptom engine (specialist choice)
│   ├── pipeline.py          # sessions, language, spelling, NER (Vita's ingest)
│   ├── session.py  language.py  script.py  lexicon.py  spelling.py  ner.py  textnorm.py
│   ├── severity.py          # the built-in emergency phrase list (the shared detector adds red_flags.csv + a model)
│   ├── voice.py             # Vita's wording in English, Hinglish and Hindi
│   └── data/                # lexicon, red flags, specialist mapping, the route model
├── Datasets/                # training data for the route-symptom model
├── scripts/                 # maintenance helpers (run from repo root), check_doctor_search.py
├── tests/
├── api.py                   # FastAPI entry point
└── requirements.txt
```

## Safety design
* Vita's safety steps run on **every message**, before the LLM: self-harm first (Tele-MANAS 14416), then the shared
  emergency detector (red-flag rules + model, denial-aware) and the warning-sign questions. The LLM can only add
  urgency. Every LLM reply passes `vita_graph/checker.py` (no medicines, doses, diagnoses, phone numbers, links,
  invented doctors or booking claims).
* New emergency phrases go in `data/red_flags.csv` - additive only.

## Deliberate deviations from the PDF
1. **Order in Step 5.** The flowchart (p.7) shows score -> emergency keyword check; the IMPORTANT box (p.13) says the safety net runs first and independently. Code follows p.13.
2. **Output 3.** The Step 7 flowchart says Alternative = GP/telemedicine; the sample card shows Emergency Medicine third. Code follows the flowchart; the ER-escalation advice appears under "Watch for".
3. **Fuzzy matching** is edit distance <= 2 only for words of 6+ letters (<= 1 below), and never maps onto words shorter than 4 letters. At distance 2 an unknown word like `uthke` otherwise matched the filler "the".
4. No `session:{id}:ttl` key: `EXPIRE` on each key does the job. Soundex/Metaphone replaced by a Hindi consonant-skeleton key (`phonetic_key`).

## NOT built (needs data, a model, or infrastructure you haven't specified)
* PostgreSQL long-term logs; WebSocket gateway; chat frontends.
* Weekly NER fine-tune; `BertNER`, XGBoost severity model, MedBERT embeddings and the cross-encoder re-ranker have hooks but no trained models.
* Corpus scale: the generated data currently contains 934 lexicon entries and 63 specialist clusters, below the doc's 2000+ symptoms / 5000+ mappings / 150+ emergency terms / 40+ specialists. Step 8 is the mechanism for growing it.

## Known limitations
* Devanagari words get a spurious middle "a" (`घबराहट` -> `ghabaraahat`). Known words are rescued by the phonetic key; novel ones reach the review queue in that longer form.
* The safety net errs toward alerting: "stroke patient diet advice" alerts. Intended.
* Self-harm / suicidal statements are **not** in the emergency list and would need a different response than "call 108 / 112 / ER". Decide this deliberately before launch.
* All patient-facing text, symptom weights, score bands and the emergency phrase list need clinician review.
