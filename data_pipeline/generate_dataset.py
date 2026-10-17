# -*- coding: utf-8 -*-
"""
Builds the expanded Hinglish symptom -> specialist dataset:

  1. Takes the hand-curated seed bank (specialist_seed_data.SEED) covering all 32
     canonical classes and expands it with rule-based Spelling Variation / Variation
     rows (same "type" convention the original CSV already used).
  2. Folds in the existing Hinglish_Symptoms_Reference.csv (1075 rows, 17 classes),
     relabelling its "specialist" column to the SAME canonical patient-facing names
     used everywhere else (applying the model script's own LABEL_ALIASES once, here,
     rather than at every load) so the merged file is self-consistent.
  3. Dedupes on (normalized text, specialist) and writes the merged result to
     Hinglish_Symptoms_Reference_v2.csv (the original file is left untouched).

Run:
    python generate_dataset.py
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

from specialist_seed_data import SEED, CANONICAL_DB_CATEGORIES, NON_DB_CATEGORIES

HERE = Path(__file__).parent
OLD_CSV = HERE.parent / "Hinglish_Symptoms_Reference.csv"
OUT_CSV = HERE.parent / "Hinglish_Symptoms_Reference_v2.csv"

# Same aliasing the model script applies at load time — done once here so the merged
# CSV carries canonical names directly and no longer depends on that step.
# Targeted relabels for a handful of legacy rows whose original label predates the finer
# 32-class taxonomy — these describe a symptom that now has its own, more specific class,
# and were only bucketed under the broader legacy label because nothing finer existed yet.
# (Found by grepping the legacy CSV for vein/nerve/etc. keywords per current label — NOT an
# exhaustive re-audit, just the unambiguous cases surfaced so far.)
RELABEL_OVERRIDES = {
    ("General Surgeon", "Pairon ki naso mein soojan aur neeli-neeli nasein dikhti hain"): "Vascular Surgeon",
    ("General Surgeon", "Pairon ki naso mein sujan aur neeli-neeli nasein dikhti hain"): "Vascular Surgeon",
}

LABEL_ALIASES = {
    "Orthopedist": "Orthopaedic Surgeon (Bone)",
    "Orthopedic Surgeon": "Orthopaedic Surgeon (Bone)",
    "Otolaryngologist - ENT": "ENT Specialist",
    "Family Physician / Internist": "General Physician",
    "Cardiologist": "Cardiologist (Heart)",
    "Dermatologist": "Dermatologist (Skin)",
    "Endocrinologist": "Endocrinologist (Hormones/Diabetes)",
    "Gynecologist": "Gynaecologist",
    "Ophthalmologist": "Ophthalmologist (Eye)",
    "Pediatrician": "Paediatrician",
    "Pulmonologist": "Pulmonologist (Chest/Lungs)",
    "General (Duration)": "General Physician",
}

# ---------------------------------------------------------------------------
# Spelling-variation rules: common Hinglish transliteration inconsistencies.
# Whole-word, case-insensitive; each "from" can expand to multiple "to" spellings.
# Meaning-preserving by construction (same word, different spelling) — safe to
# apply mechanically, unlike the synonym rules below.
# ---------------------------------------------------------------------------
SPELLING_RULES = [
    (r"\bdard\b", ["darad"]),
    (r"\bdarad\b", ["dard"]),
    (r"\baaram\b", ["aram", "aaraam"]),
    (r"\bdoctor\b", ["daktar", "docter"]),
    (r"\bbahut\b", ["bohot", "bahot"]),
    (r"\bzyada\b", ["jyada", "jyaada"]),
    (r"\bchahiye\b", ["chaiye", "chahye"]),
    (r"\bnahi\b", ["nahin", "nhi"]),
    (r"\bmein\b", ["me"]),
    (r"\braha\b", ["rha"]),
    (r"\bkhansi\b", ["khaansi"]),
    (r"\bbukhar\b", ["bukhaar"]),
    (r"\bkamzori\b", ["kamzoori"]),
    (r"\bsujan\b", ["soojan"]),
    (r"\bkhujli\b", ["khujali"]),
    (r"\bbimari\b", ["bimaari"]),
    (r"\bthoda\b", ["thora"]),
    (r"\bpeshaab\b", ["peshab"]),
    (r"\bgaanth\b", ["ganth"]),
]

# ---------------------------------------------------------------------------
# Synonym-substitution rules: genuine paraphrase, not just spelling — swaps one
# content word for a colloquial/English equivalent. Tagged "Variation of X" (same
# convention as the original CSV's semantic-paraphrase rows).
# ---------------------------------------------------------------------------
SYNONYM_RULES = [
    (r"\bdard\b", ["takleef", "problem"]),
    (r"\bpet\b", ["stomach", "pait"]),
    (r"\bsar\b", ["head", "matha"]),
    (r"\bbahut\b", ["kaafi", "itna"]),
    (r"\bhaath\b", ["hand"]),
    (r"\bpairo\b", ["legs", "pairon"]),
    (r"\bkhansi\b", ["cough"]),
    (r"\bbukhar\b", ["fever"]),
    (r"\bkamzori\b", ["weakness"]),
    (r"\bchahiye\b", ["zaroorat hai"]),
    (r"\bdikhana\b", ["consult karna"]),
]

MAX_SPELLING_VARIANTS_PER_PHRASE = 2
MAX_SYNONYM_VARIANTS_PER_PHRASE = 1


def normalize_for_dedupe(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.,!?;:]+$", "", t)
    return t


def apply_rules_once(text: str, rules, max_variants: int):
    """Apply each rule that matches, substituting the FIRST match only (keeps the
    output a single, readable sentence rather than compounding several swaps)."""
    variants = []
    for pattern, alternatives in rules:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        for alt in alternatives:
            new_text = text[:m.start()] + alt + text[m.end():]
            if new_text != text:
                variants.append(new_text)
            if len(variants) >= max_variants:
                return variants
        if len(variants) >= max_variants:
            break
    return variants


def expand_seed_bank():
    """Yields (text, specialist, type) rows from the hand-curated seed bank plus its
    rule-generated spelling/semantic variants."""
    for specialist, buckets in SEED.items():
        for base_type, phrases in (
            ("Original", buckets.get("symptoms", [])),
            ("Specialist Term", buckets.get("doctor_mentions", [])),
            ("Keyword Phrase", buckets.get("keywords", [])),
        ):
            for phrase in phrases:
                yield phrase, specialist, base_type

                for variant in apply_rules_once(phrase, SPELLING_RULES, MAX_SPELLING_VARIANTS_PER_PHRASE):
                    yield variant, specialist, f"Spelling Variation of: {phrase}"

                if base_type == "Original":
                    for variant in apply_rules_once(phrase, SYNONYM_RULES, MAX_SYNONYM_VARIANTS_PER_PHRASE):
                        yield variant, specialist, f"Variation of: {phrase}"


def load_existing_csv():
    if not OLD_CSV.exists():
        print(f"Note: {OLD_CSV} not found, skipping merge of existing data.", file=sys.stderr)
        return
    with open(OLD_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("specialist") or "").strip()
            row_type = (row.get("type") or "").strip() or "Original"
            if not text or not label:
                continue
            label = LABEL_ALIASES.get(label, label)
            label = RELABEL_OVERRIDES.get((label, text), label)
            yield text, label, row_type


def main():
    all_expected = CANONICAL_DB_CATEGORIES | NON_DB_CATEGORIES
    seen = {}  # normalized (text, specialist) -> (text, specialist, type) kept
    order = []

    def add(text, specialist, row_type):
        key = (normalize_for_dedupe(text), specialist)
        if key in seen:
            return
        seen[key] = (text, specialist, row_type)
        order.append(key)

    for text, specialist, row_type in expand_seed_bank():
        add(text, specialist, row_type)

    seed_only_count = len(order)

    unexpected_labels = set()
    for text, specialist, row_type in load_existing_csv():
        if specialist not in all_expected:
            unexpected_labels.add(specialist)
        add(text, specialist, row_type)

    total = len(order)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "specialist", "type"])
        for key in order:
            writer.writerow(seen[key])

    # ---- Summary ----
    from collections import Counter
    counts = Counter(seen[k][1] for k in order)
    print(f"Seed bank (incl. generated variants): {seed_only_count} rows")
    print(f"After merging existing CSV + dedup:   {total} rows")
    print(f"Wrote: {OUT_CSV}\n")
    print(f"Classes covered: {len(counts)} / {len(all_expected)} expected\n")
    print(f"{'Specialist':40s} {'Rows':>5s}")
    for label in sorted(all_expected, key=lambda l: -counts.get(l, 0)):
        flag = "  <-- LOW" if counts.get(label, 0) < 15 else ""
        print(f"{label:40s} {counts.get(label, 0):5d}{flag}")
    missing_entirely = [l for l in all_expected if counts.get(l, 0) == 0]
    if missing_entirely:
        print("\nMISSING ENTIRELY:", missing_entirely)
    if unexpected_labels:
        print("\nWARNING - existing CSV had labels outside the 32-class taxonomy:", unexpected_labels)


if __name__ == "__main__":
    main()
