"""Utility to build the specialist_mapping.csv RAG corpus and enrich the
hinglish_lexicon.csv from Datasets/combined_symptom_dataset.csv.

Usage (from the hms_router/ project root):
    python -m hms_router.dataset_builder

It writes (or overwrites) two files:
    hms_router/data/specialist_mapping.csv
    hms_router/data/hinglish_lexicon.csv  (merged: existing entries preserved,
                                           new symptom variants appended)

Design
------
* Each (specialty, triage) bucket becomes one Cluster in specialist_mapping.csv.
  The semicolon-separated symptom phrases from the dataset rows are collected
  and de-duplicated to form the cluster's `symptoms` column.
* The `description` column is auto-generated as
  "{specialty} - {triage}: <top symptoms joined>".
* Cluster IDs are compact codes derived from specialty + triage:
      Cardiology + Emergency  -> CARD_EMG
      Psychiatry  + Routine   -> PSYC_RTN
* Symptom phrases in English (or close-to-English canonical forms already in the
  lexicon) are extracted and written as new `symptom` lexicon entries with
  lang=hi (they are Hindi-language symptom descriptions).

Run this script after updating Datasets/combined_symptom_dataset.csv to regenerate
the corpus. The rest of the pipeline is dataset-agnostic.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------ paths
_HERE = Path(__file__).parent
DATASET_PATH = _HERE.parent / "Datasets" / "combined_symptom_dataset.csv"
OUT_MAPPING = _HERE / "data" / "specialist_mapping.csv"
OUT_LEXICON = _HERE / "data" / "hinglish_lexicon.csv"

# Core vocabulary supports the conversational pipeline independently of the
# larger, free-text dataset. Keep these canonical entries ahead of generated
# phrases so the NER and severity layers receive stable symptom names.
_BASE_LEXICON = [
    ("fever", "fever", "symptom", "en"),
    ("cough", "cough", "symptom", "en"),
    ("vomiting", "vomiting", "symptom", "en"),
    ("dizziness", "dizziness", "symptom", "en"),
    ("breathlessness", "breathlessness", "symptom", "en"),
    ("headache", "headache", "symptom", "en"),
    ("stomach pain", "stomach pain", "symptom", "en"),
    ("chest pain", "chest pain", "symptom", "en"),
    ("back pain", "back pain", "symptom", "en"),
    ("sore throat", "sore throat", "symptom", "en"),
    ("vision loss", "vision loss", "symptom", "en"),
    ("numbness", "numbness", "symptom", "en"),
    ("itching", "itching", "symptom", "en"),
    ("rash", "rash", "symptom", "en"),
    ("diarrhea", "diarrhea", "symptom", "en"),
    ("nausea", "nausea", "symptom", "en"),
    ("weakness", "weakness", "symptom", "en"),
    ("fatigue", "fatigue", "symptom", "en"),
    ("swelling", "swelling", "symptom", "en"),
    ("pain", "pain", "symptom", "en"),
    ("blood in vomit", "blood in vomit", "symptom", "en"),
    ("unconscious", "unconscious", "symptom", "en"),
    ("sudden", "sudden", "time", "en"),
    ("severe", "severe", "intensity", "en"),
    ("moderate", "moderate", "intensity", "en"),
    ("mild", "mild", "intensity", "en"),
    ("stomach", "stomach", "body_part", "en"),
    ("head", "head", "body_part", "en"),
    ("chest", "chest", "body_part", "en"),
    ("back", "back", "body_part", "en"),
    ("throat", "throat", "body_part", "en"),
    ("sir dard", "headache", "symptom", "hi"),
    ("sar dard", "headache", "symptom", "hi"),
    ("sardard", "headache", "symptom", "hi"),
    ("sir", "head", "body_part", "hi"),
    ("sar", "head", "body_part", "hi"),
    ("pet dard", "stomach pain", "symptom", "hi"),
    ("pet", "stomach", "body_part", "hi"),
    ("seene mein dard", "chest pain", "symptom", "hi"),
    ("seene me dard", "chest pain", "symptom", "hi"),
    ("sine mein dard", "chest pain", "symptom", "hi"),
    ("sine me dard", "chest pain", "symptom", "hi"),
    ("seena me dard", "chest pain", "symptom", "hi"),
    ("seena", "chest", "body_part", "hi"),
    ("seene", "chest", "body_part", "hi"),
    ("kamar dard", "back pain", "symptom", "hi"),
    ("kamar", "back", "body_part", "hi"),
    ("gala dard", "sore throat", "symptom", "hi"),
    ("gala", "throat", "body_part", "hi"),
    ("bukhar", "fever", "symptom", "hi"),
    ("bukhaar", "fever", "symptom", "hi"),
    ("khansi", "cough", "symptom", "hi"),
    ("khaansi", "cough", "symptom", "hi"),
    ("ulti", "vomiting", "symptom", "hi"),
    ("chakkar", "dizziness", "symptom", "hi"),
    ("saans", "breathlessness", "symptom", "hi"),
    ("dard", "pain", "symptom", "hi"),
    ("halka", "mild", "intensity", "hi"),
    ("thoda", "mild", "intensity", "hi"),
    ("kaafi", "moderate", "intensity", "hi"),
    ("tez", "severe", "intensity", "hi"),
    ("bahut", "very", "intensity", "hi"),
    ("achanak", "sudden", "time", "hi"),
    ("aankhon ke aage andhera", "vision loss", "symptom", "hi"),
    ("haath pair sune", "numbness", "symptom", "hi"),
    ("haath pair sunn", "numbness", "symptom", "hi"),
    ("behosh", "unconscious", "symptom", "hi"),
    ("teen", "3", "number", "hi"),
    ("do", "2", "number", "hi"),
    ("ek", "1", "number", "hi"),
    ("din", "day", "time", "hi"),
    ("raat", "night", "time", "hi"),
    ("subah", "morning", "time", "hi"),
    ("shaam", "evening", "time", "hi"),
    ("dopahar", "afternoon", "time", "hi"),
    ("kal", "yesterday", "time", "hi"),
    ("aaj", "today", "time", "hi"),
    ("mujhe", "mujhe", "function", "hi"),
    ("hai", "hai", "particle", "hi"),
    ("hain", "hain", "particle", "hi"),
    ("ho", "ho", "particle", "hi"),
    ("hoon", "hoon", "particle", "hi"),
    ("raha", "raha", "particle", "hi"),
    ("rahi", "rahi", "particle", "hi"),
    ("rahe", "rahe", "particle", "hi"),
    ("hua", "hua", "particle", "hi"),
    ("hui", "hui", "particle", "hi"),
    ("gaya", "gaya", "particle", "hi"),
    ("gayi", "gayi", "particle", "hi"),
    ("gaye", "gaye", "particle", "hi"),
    ("nahi", "nahi", "particle", "hi"),
    ("nahin", "nahin", "particle", "hi"),
    ("aa", "aa", "particle", "hi"),
    ("me", "in", "particle", "hi"),
    ("mein", "in", "particle", "hi"),
    ("ka", "ka", "particle", "hi"),
    ("ki", "ki", "particle", "hi"),
    ("ke", "ke", "particle", "hi"),
    ("se", "se", "particle", "hi"),
    ("aur", "aur", "function", "hi"),
    ("bhi", "bhi", "function", "hi"),
    ("bas", "bas", "function", "hi"),
    ("no", "no", "function", "en"),
    ("only", "only", "function", "en"),
    ("ghabrana", "ghabrana", "function", "hi"),
]

_CURATED_MAPPING = [
    ("GI1", "Gastroenterologist", "stomach pain;vomiting;fever;diarrhea;nausea",
     "Gastrointestinal symptoms including abdominal pain, vomiting and fever"),
    ("GI2", "Gastroenterologist", "stomach pain;vomiting;diarrhea;nausea;constipation",
     "Gastrointestinal and digestive symptoms"),
    ("NEU1", "Neurologist", "headache;dizziness;vision loss;numbness;weakness",
     "Neurological symptoms including headache and dizziness"),
    ("DERM1", "Dermatologist", "itching;rash;swelling",
     "Skin symptoms including itching and rash"),
    ("CARD1", "Cardiologist", "chest pain;breathlessness;dizziness;weakness",
     "Cardiac symptoms including chest pain and breathlessness"),
    ("PULM1", "Pulmonologist", "cough;breathlessness;sore throat",
     "Respiratory symptoms including cough and breathlessness"),
    ("ORTH1", "Orthopedist", "back pain;joint pain;swelling;pain",
     "Musculoskeletal pain and joint symptoms"),
    ("GP1", "General Physician", "fever;cough;headache;fatigue;weakness",
     "General symptoms for primary assessment"),
    ("ENT1", "ENT Specialist", "sore throat;cough;hearing loss",
     "Ear, nose and throat symptoms"),
    ("OPHT1", "Ophthalmologist", "vision loss;eye pain;blurred vision",
     "Eye symptoms including vision changes"),
]

# ------------------------------------------------------------------ specialist → short code
_SPEC_CODE: dict[str, str] = {
    "Cardiology":                 "CARD",
    "Paediatrics":                "PAED",
    "Endocrinology":              "ENDO",
    "Physiotherapy":              "PHYT",
    "Orthopaedics":               "ORTH",
    "Psychiatry":                 "PSYC",
    "ENT":                        "ENT",
    "Gynaecology":                "GYNA",
    "Pulmonology":                "PULM",
    "Ophthalmology":              "OPHT",
    "Urology":                    "UROL",
    "General Emergency / ER":     "EMER",
    "Dermatology & Venereology":  "DERM",
    "Breast clinic":              "BRST",
    "Rheumatology":               "RHEU",
    "General surgery":            "GSUR",
    "Gastroenterology":           "GAST",
    "Gastroenterologist":         "GAST",
    "General Physician":          "GPCO",
    "Anaesthesiologist":          "ANES",
    "Neurology":                  "NEUR",
    "Nephrology":                 "NEPH",
    "Oncology":                   "ONCO",
    "Haematology":                "HAEM",
}

_TRIAGE_CODE: dict[str, str] = {
    "Emergency":           "EMG",
    "Urgent":              "URG",
    "Routine":             "RTN",
    "Not specified":       "GEN",
    "Doctor review first": "REV",
}


def _spec_code(specialty: str) -> str:
    return _SPEC_CODE.get(specialty, re.sub(r"[^A-Z]", "", specialty.upper())[:4] or "SPEC")


def _triage_code(triage: str) -> str:
    return _TRIAGE_CODE.get(triage, re.sub(r"[^A-Z]", "", triage.upper())[:3] or "GEN")


# ------------------------------------------------------------------ specialist name normalisation
# The dataset says "Neurology"; the curated mapping, output.py and the tests say "Neurologist".
# Everything is mapped to ONE vocabulary here. Keys are lower-case. Extend as needed;
# unknown names pass through unchanged. Cluster IDs are built from the RAW name, so they stay stable.
SPECIALIST_ALIASES: dict[str, str] = {
    "cardiology": "Cardiologist",
    "neurology": "Neurologist",
    "dermatology": "Dermatologist",
    "dermatology & venereology": "Dermatologist",
    "gastroenterology": "Gastroenterologist",
    "pulmonology": "Pulmonologist",
    "orthopaedics": "Orthopedist",
    "orthopedics": "Orthopedist",
    "rheumatology": "Rheumatologist",
    "general medicine": "General Physician",
    "general physician": "General Physician",
    "anaesthesiology": "Anaesthesiologist",
    "general emergency / er": "Emergency Medicine",
    "breast surgery": "Breast clinic",
}


def canonical_specialist(name: str) -> str:
    """Map a raw specialty label to the project's single specialist vocabulary."""
    name = (name or "").strip()
    return SPECIALIST_ALIASES.get(name.lower(), name)


# ------------------------------------------------------------------ symptom phrase splitting
def _split_symptoms(text: str) -> list[str]:
    """Split a semicolon/comma-separated hinglish symptom string into phrases."""
    parts = re.split(r"[;]+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip().strip(",").strip()
        if p:
            out.append(p)
    return out


# ------------------------------------------------------------------ triage → severity label
_TRIAGE_SEVERITY = {
    "Emergency":           "emergency",
    "Urgent":              "urgent",
    "Routine":             "routine",
    "Not specified":       "routine",
    "Doctor review first": "urgent",
}


def _description(specialty: str, triage: str, top_symptoms: list[str]) -> str:
    sev = _TRIAGE_SEVERITY.get(triage, "routine")
    preview = "; ".join(top_symptoms[:5])
    return f"{specialty} ({sev}): {preview}"


# ------------------------------------------------------------------ corpus builder
def build_specialist_mapping(dataset_path: Path = DATASET_PATH) -> list[dict]:
    """Read combined_symptom_dataset.csv, cluster by (specialty, triage), return
    rows suitable for specialist_mapping.csv."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)

    with open(dataset_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            spec = row["specialty"].strip()
            triage = row["triage"].strip()
            for phrase in _split_symptoms(row["symptoms_hinglish"]):
                if phrase:
                    buckets[(spec, triage)].append(phrase)

    rows: list[dict] = []
    seen_ids: dict[str, int] = {}

    for (spec, triage), phrases in sorted(buckets.items()):
        # Deduplicate while preserving order
        seen: dict[str, None] = {}
        for p in phrases:
            seen[p] = None
        unique = list(seen.keys())

        base_id = f"{_spec_code(spec)}_{_triage_code(triage)}"
        n = seen_ids.get(base_id, 0)
        seen_ids[base_id] = n + 1
        cluster_id = base_id if n == 0 else f"{base_id}_{n}"

        rows.append({
            "cluster_id": cluster_id,
            "specialist":  canonical_specialist(spec),
            "symptoms":    ";".join(unique),
            "description": _description(spec, triage, unique),
        })
    return rows


# ------------------------------------------------------------------ lexicon enricher
# Known canonical mappings for common Hinglish symptom phrases that appear
# verbatim or near-verbatim in the dataset.  New variants in the dataset that
# are NOT already in the lexicon will be added as category=symptom, lang=hi.
_KNOWN_CANONICALS: dict[str, str] = {
    "dard":               "pain",
    "bukhaar":            "fever",
    "bukhar":             "fever",
    "khansi":             "cough",
    "ulti":               "vomiting",
    "chakkar":            "dizziness",
    "sans":               "breathlessness",
    "jalan":              "burning",
    "khujli":             "itching",
    "kamzori":            "weakness",
    "thakan":             "fatigue",
    "neend":              "insomnia",
    "bhookh":             "loss of appetite",
    "dast":               "diarrhea",
    "kabriyat":           "constipation",
    "pasina":             "sweating",
    "sujan":              "swelling",
    "dabav":              "pressure",
    "sar dard":           "headache",
    "sir dard":           "headache",
    "pet dard":           "stomach pain",
    "sine mein dard":     "chest pain",
    "kamar dard":         "back pain",
    "gala dard":          "sore throat",
    "aankhon mein dard":  "eye pain",
    "haath mein dard":    "hand pain",
    "pair mein dard":     "leg pain",
}


def _load_existing_lexicon(path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    variants: set[str] = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                variants.add(row["variant"].strip().lower())
    return rows, variants


def build_lexicon_additions(dataset_path: Path = DATASET_PATH,
                            existing_variants: set[str] | None = None) -> list[dict]:
    """Extract short, clean Hinglish symptom phrases from the dataset that are
    not already in the lexicon and return them as new lexicon rows.

    Rules for inclusion:
    * Multi-word phrases (2-6 words): always included; they map to their phrase
      as canonical (the phrase matcher uses them as composite entries and they
      do not pollute the single-word unknown-token queue).
    * Single-word phrases: only included when a mapping to an English canonical
      exists in ``_KNOWN_CANONICALS``.  Without a known canonical the word is
      left unknown so the Step-8 learning loop can surface it for human review.
    """
    existing = existing_variants or set()
    new_rows: list[dict] = []
    seen: set[str] = set()
    for variant, canonical, category, lang in _BASE_LEXICON:
        if variant not in existing and variant not in seen:
            new_rows.append({"variant": variant, "canonical": canonical, "category": category, "lang": lang})
            seen.add(variant)

    with open(dataset_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for phrase in _split_symptoms(row["symptoms_hinglish"]):
                phrase_lc = phrase.lower().strip()
                words = phrase_lc.split()
                n_words = len(words)
                if n_words == 0 or n_words > 6:
                    continue
                if phrase_lc in existing or phrase_lc in seen:
                    continue
                # Skip phrases that contain digits
                if any(ch.isdigit() for ch in phrase_lc):
                    continue
                canonical = _KNOWN_CANONICALS.get(phrase_lc)
                if canonical is None:
                    if n_words == 1:
                        # Single-word without a known English canonical:
                        # leave it unknown so the learning loop can process it.
                        continue
                    canonical = phrase_lc   # multi-word: use phrase itself as canonical
                new_rows.append({
                    "variant":   phrase_lc,
                    "canonical": canonical,
                    "category":  "symptom",
                    "lang":      "hi",
                })
                seen.add(phrase_lc)
    return new_rows


# ------------------------------------------------------------------ writers
def write_specialist_mapping(rows: list[dict], out: Path = OUT_MAPPING) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "specialist", "symptoms", "description"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[dataset_builder] Wrote {len(rows)} clusters -> {out}")


def write_lexicon(existing: list[dict], additions: list[dict], out: Path = OUT_LEXICON) -> None:
    base = {variant: {"variant": variant, "canonical": canonical, "category": category, "lang": lang}
            for variant, canonical, category, lang in _BASE_LEXICON}
    all_rows = list(base.values())
    known = set(base)
    for row in existing + additions:
        variant = row["variant"].strip().lower()
        if variant not in known:
            all_rows.append(row)
            known.add(variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "canonical", "category", "lang"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[dataset_builder] Wrote {len(existing)} existing + {len(additions)} new lexicon entries -> {out}")


# ------------------------------------------------------------------ loader (runtime use)
def load_corpus_from_dataset(dataset_path: Path | str | None = None) -> list:
    """Load a symptom_clusters.Cluster list directly from combined_symptom_dataset.csv.

    This is an alternative to reading specialist_mapping.csv.  Useful when the
    dataset has been updated but the mapping CSV has not been regenerated yet.

    Returns the same Cluster objects that symptom_clusters.load_corpus() returns.
    """
    from .symptom_clusters import Cluster  # local import to avoid circular at module level

    path = Path(dataset_path) if dataset_path else DATASET_PATH
    mapping_rows = build_specialist_mapping(path)
    return [
        Cluster(
            cluster_id=r["cluster_id"],
            specialist=r["specialist"],
            symptoms=tuple(s.strip() for s in r["symptoms"].split(";")),
            description=r["description"],
        )
        for r in mapping_rows
    ]


def _load_existing_mapping(path: Path) -> list[dict]:
    """Read an existing specialist_mapping.csv and return its rows as dicts."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_merged_mapping(dataset_path: Path = DATASET_PATH,
                         existing_path: Path = OUT_MAPPING) -> list[dict]:
    """Merge existing specialist_mapping.csv with new clusters derived from the dataset.

    Existing clusters always win (their cluster_id is preserved).  New clusters
    from the dataset are appended only if their cluster_id is not already present.
    This ensures hand-crafted clusters (which the tests rely on) are never overwritten.
    """
    existing = _load_existing_mapping(existing_path)
    existing_ids = {r["cluster_id"] for r in existing}
    new_rows = build_specialist_mapping(dataset_path)
    additions = [r for r in new_rows if r["cluster_id"] not in existing_ids]
    curated = [
        {"cluster_id": cid, "specialist": specialist, "symptoms": symptoms, "description": description}
        for cid, specialist, symptoms, description in _CURATED_MAPPING
        if cid not in existing_ids and cid not in {r["cluster_id"] for r in additions}
    ]
    return existing + additions + curated


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    import shutil

    print(f"[dataset_builder] Reading {DATASET_PATH}")

    # Back up the original mapping before overwriting
    if OUT_MAPPING.exists():
        backup = OUT_MAPPING.with_suffix(".bak.csv")
        shutil.copy2(OUT_MAPPING, backup)
        print(f"[dataset_builder] Backed up original -> {backup}")

    mapping = build_merged_mapping()
    write_specialist_mapping(mapping)

    existing, existing_variants = _load_existing_lexicon(OUT_LEXICON)
    additions = build_lexicon_additions(existing_variants=existing_variants)
    write_lexicon(existing, additions)

    print("[dataset_builder] Done.")
