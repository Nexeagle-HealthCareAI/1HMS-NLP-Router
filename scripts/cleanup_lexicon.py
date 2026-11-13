"""Cleanup script: remove from hinglish_lexicon.csv any entry where:
- it was added by dataset_builder (lang=hi, category=symptom)
- it is a single word
- its canonical equals its variant (no real English mapping)
These entries were added by the old build_lexicon_additions logic and should
not be in the lexicon (they pollute the unknown-token learning queue).

Run from the repo root:  python scripts/cleanup_lexicon.py
"""
import csv
from pathlib import Path

LEX = Path(r'hms_router/data/hinglish_lexicon.csv')

with open(LEX, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f'Before: {len(rows)} entries')

cleaned = []
removed = []
for r in rows:
    variant = r['variant'].strip()
    canonical = r['canonical'].strip()
    n_words = len(variant.split())
    is_self_mapped = (canonical == variant)
    is_single_word = (n_words == 1)
    is_dataset_added = (r['lang'].strip() == 'hi' and r['category'].strip() == 'symptom')

    if is_dataset_added and is_single_word and is_self_mapped:
        removed.append(variant)
    else:
        cleaned.append(r)

print(f'Removing {len(removed)} self-mapped single-word entries')
print(f'After: {len(cleaned)} entries')

assert 'ghabrahat' not in [r['variant'] for r in cleaned], "ghabrahat still present!"
print("ghabrahat correctly removed")

with open(LEX, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['variant', 'canonical', 'category', 'lang'])
    writer.writeheader()
    writer.writerows(cleaned)

print("Done.")
