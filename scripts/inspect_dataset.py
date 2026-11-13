"""Quick look at the dataset: sample rows + which separators the symptom column uses.
Run from the repo root:  python scripts/inspect_dataset.py
"""
import csv

with open(r'Datasets/combined_symptom_dataset.csv', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print('=== Sample rows ===')
for r in rows[0:10]:
    spec = r['specialty']
    triage = r['triage']
    symptoms = r['symptoms_hinglish'][:120]
    print(f'  specialty={spec}, triage={triage}')
    print(f'  symptoms: {symptoms}')
    print()

sep_counts = {}
for r in rows:
    for sep in [';', ',', '|']:
        if sep in r['symptoms_hinglish']:
            sep_counts[sep] = sep_counts.get(sep, 0) + 1
print('Separator usage:', sep_counts)
