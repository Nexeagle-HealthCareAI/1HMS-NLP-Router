"""ARCHIVED one-off helper. Already applied: hms_router/dataset_builder.py now contains the patched
build_lexicon_additions(). Kept only for history - safe to delete.
Run from the repo root.
"""
from pathlib import Path

TARGET = Path(r'hms_router/dataset_builder.py')
src = TARGET.read_text(encoding='utf-8')

old_line = '                canonical = _KNOWN_CANONICALS.get(phrase_lc, phrase_lc)'
new_lines = '''                canonical = _KNOWN_CANONICALS.get(phrase_lc)
                if canonical is None:
                    if n_words == 1:
                        continue
                    canonical = phrase_lc'''

if old_line in src:
    src = src.replace(old_line, new_lines, 1)
    src = src.replace(
        '                words = phrase_lc.split()\n                if len(words) > 6 or len(words) == 0:\n',
        '                words = phrase_lc.split()\n                n_words = len(words)\n                if n_words == 0 or n_words > 6:\n')
    TARGET.write_text(src, encoding='utf-8')
    print("PATCHED OK")
else:
    print("Nothing to patch (already applied).")
