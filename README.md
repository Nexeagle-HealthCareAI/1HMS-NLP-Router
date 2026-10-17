# HMS NLP Router

Hinglish symptom text -> clinical intent -> specialist routing -> emergency flagging.
Implements `HMS_NLP_Router_Architecture.pdf`. **Triage support only - not a diagnosis.**

```
pip install -r requirements.txt
python -m pytest            # run the test-suite (needs hms_router/data/*.csv)
python demo.py --debug      # interactive; type "quit" to exit
```
```python
import redis
from hms_router import HMSRouterAgent
agent = HMSRouterAgent(redis.Redis(decode_responses=True))
reply = agent.handle("seene mein dard ho raha hai", session_id=None)
reply.kind      # follow_up | emergency | recommendation
reply.message   # ready-to-send text (Hinglish labels for Hinglish users)
reply.session_id  # pass back on the next message
```

## Repository layout

```
hms-nlp-router/
├── hms_router/              # the package
│   ├── __init__.py
│   ├── agent.py             # orchestration, in doc order
│   ├── pipeline.py          # Steps 1-3 glue
│   ├── session.py  language.py  script.py          # L1 (Step 1)
│   ├── lexicon.py  spelling.py                     # L2 (Step 2)
│   ├── keywords.py                                 # L2 keyword tagging
│   ├── ner.py                                      # L2 NER (Step 3)
│   ├── accumulator.py                              # L3 (Step 4)
│   ├── severity.py                                 # L4 (Step 5)
│   ├── specialist_rag.py  dataset_builder.py       # L4 (Step 6)
│   ├── output.py                                   # L5 (Step 7)
│   ├── learning.py                                 # L5 (Step 8)
│   ├── config.py
│   └── data/                # hinglish_lexicon.csv, spelling_variants.csv, specialist_mapping.csv
├── Datasets/
│   └── combined_symptom_dataset.csv
├── scripts/                 # maintenance helpers (run from repo root)
│   ├── inspect_dataset.py
│   ├── cleanup_lexicon.py
│   └── one_off/patch_builder.py   # archived, already applied
├── tests/                   # 8 test modules (~100 test functions)
├── demo.py
├── pytest.ini
└── requirements.txt
```

## Doc -> code

| Doc | Module |
|---|---|
| L1 Session / language / script (Step 1) | `session.py`, `language.py`, `script.py` |
| L2 Spelling, lexicon (Step 2) | `spelling.py`, `lexicon.py`, `data/hinglish_lexicon.csv`, `data/spelling_variants.csv` |
| L2 Keyword extractor / unique keyword tagging (p.12) | `keywords.py` |
| L2 NER -> symptom object (Step 3) | `ner.py` |
| L3 Redis accumulation (Step 4) | `accumulator.py` |
| L4 Severity + emergency safety net (Step 5) | `severity.py` |
| L4 RAG retrieval (Step 6) | `specialist_rag.py`, `Datasets/combined_symptom_dataset.csv` (primary), `data/specialist_mapping.csv` (curated supplement) |
| L5 Three outputs / Emergency card (Step 7) | `output.py` |
| L5 Feedback + continuous learning (Step 8) | `learning.py` |
| Orchestration, in doc order | `agent.py` |

## Safety design
* The hardcoded emergency list in `severity.py` runs on **every turn**, on the merged multi-turn state *and* the raw text, and **before** any ML score. ML can only raise severity.
* If spelling/NER crash, the agent still runs the safety net on raw text (`reply.degraded == ["nlp"]`).
* An emergency is never delayed by follow-up questions.
* Learning never touches the safety net. New emergency phrases go in `emergency_keywords.csv` (`severity.load_extra_rules`) - additive only.
* Nothing is auto-added to the lexicon: the nightly job writes `novel_symptoms_review.csv`; a human sets `status=approved` + canonical + category; `agent.learning.apply(csv)` then reloads spelling and re-indexes RAG.

## Deliberate deviations from the PDF
1. **Order in Step 5.** The flowchart (p.7) shows score -> emergency keyword check; the IMPORTANT box (p.13) says the safety net runs first and independently. Code follows p.13.
2. **Output 3.** The Step 7 flowchart says Alternative = GP/telemedicine; the sample card shows Emergency Medicine third. Code follows the flowchart; the ER-escalation advice appears under "Watch for".
3. **Fuzzy matching** is edit distance <= 2 only for words of 6+ letters (<= 1 below), and never maps onto words shorter than 4 letters. At distance 2 an unknown word like `uthke` otherwise matched the filler "the".
4. No `session:{id}:ttl` key: `EXPIRE` on each key does the job. Soundex/Metaphone replaced by a Hindi consonant-skeleton key (`phonetic_key`).

## NOT built (needs data, a model, or infrastructure you haven't specified)
* PostgreSQL long-term logs; FastAPI/WebSocket gateway; chat frontends.
* Weekly NER fine-tune; `BertNER`, XGBoost severity model, MedBERT embeddings and the cross-encoder re-ranker have hooks but no trained models.
* Corpus scale: ~210 lexicon rows / 20 specialist clusters vs the doc's 2000+ symptoms / 5000+ mappings / 150+ emergency terms / 40+ specialists. Step 8 is the mechanism for growing it.

## Known limitations
* Devanagari words get a spurious middle "a" (`घबराहट` -> `ghabaraahat`). Known words are rescued by the phonetic key; novel ones reach the review queue in that longer form.
* The safety net errs toward alerting: "stroke patient diet advice" alerts. Intended.
* Self-harm / suicidal statements are **not** in the emergency list and would need a different response than "call 112 / ER". Decide this deliberately before launch.
* All patient-facing text, symptom weights, score bands and the emergency phrase list need clinician review.
