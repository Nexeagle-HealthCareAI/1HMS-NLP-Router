# -*- coding: utf-8 -*-
"""
ONE-TIME script: splits Hinglish_Symptoms_Reference_v2.csv into a frozen validation_set.csv
(held out, never trained on, never touched again) and prints the remaining "trainable pool"
row count for reference.

Uses the same leakage-free split used for every held-out eval this session: Original /
Specialist Term / Keyword Phrase rows are trainable; Spelling Variation of: */Variation of: *
rows are the genuinely-unseen-phrasing test set.

Run ONCE. validation_set.csv is committed and never regenerated after this — every future
retrain (CMS edits, production feedback) only ever grows the TRAINABLE pool, never this file,
so retrain-over-retrain metrics stay comparable.

    python build_validation_set.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
SOURCE_CSV = HERE.parent / "Hinglish_Symptoms_Reference_v2.csv"
VALIDATION_CSV = HERE / "validation_set.csv"
TRAINABLE_CSV = HERE / "trainable_pool.csv"


def main():
    with open(SOURCE_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    validation_rows = []
    trainable_rows = []
    for row in rows:
        row_type = (row.get("type") or "").strip()
        if row_type.startswith("Spelling Variation of") or row_type.startswith("Variation of"):
            validation_rows.append(row)
        else:
            trainable_rows.append(row)

    with open(VALIDATION_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "specialist", "type"])
        writer.writeheader()
        for row in validation_rows:
            writer.writerow({"text": row["text"], "specialist": row["specialist"], "type": row.get("type", "")})

    with open(TRAINABLE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "specialist", "type"])
        writer.writeheader()
        for row in trainable_rows:
            writer.writerow({"text": row["text"], "specialist": row["specialist"], "type": row.get("type", "")})

    print(f"Validation set (frozen, held out): {len(validation_rows)} rows -> {VALIDATION_CSV}")
    print(f"Trainable pool (goes into the DB seed): {len(trainable_rows)} rows -> {TRAINABLE_CSV}")


if __name__ == "__main__":
    main()
