"""Unit tests for fixes made while joining the steps (script, spelling, safety net)."""
import pytest

from hms_router.lexicon import Lexicon
from hms_router.script import normalize_script
from hms_router.severity import SeverityClassifier, emergency_check, load_extra_rules, EMERGENCY_RULES
from hms_router.spelling import SpellingCorrector


@pytest.fixture(scope="module")
def lex():
    return Lexicon.from_csv()


# ---------------------------------------------------------------- script
@pytest.mark.parametrize("dev,roman", [
    ("साँस नहीं आ रही", "saans nahin aa rahi"),   # chandrabindu used to split the word: "saa ns"
    ("तेज़ बुखार", "tez bukhaar"),                # trailing nukta used to leave "teza"
    ("उल्टी", "ulti"), ("सीने में दर्द", "seene men dard"), ("खाँसी", "khaansi"),
    ("मुझे तीन दिन से बुखार है", "mujhe teen din se bukhaar hai"),
])
def test_devanagari_to_roman(dev, roman):
    assert normalize_script(dev) == roman


# ---------------------------------------------------------------- spelling
def test_alias_file_is_loaded_and_validated(lex):
    assert lex.aliases["nai"] == "nahi" and "nai" not in lex.vocab
    bad = Lexicon(lex.entries, {"zzz": "not-a-word", "nhii": "nahi"})
    assert "zzz" not in bad.aliases and bad.aliases["nhii"] == "nahi"      # an alias can't inject a new word


def test_tie_between_same_meaning_words_is_accepted(lex):
    c = SpellingCorrector(lex)
    assert c.repair_word("sinee")[0] == "seene"


def test_fuzzy_never_maps_to_tiny_filler_words(lex):
    c = SpellingCorrector(lex)
    for w in ("uthke", "kuchh", "abhii"):
        assert c.repair_word(w)[0] not in ("the", "hai", "ek", "ke"), w
    assert c.repair_word("uthke") == ("uthke", "unknown")


def test_normal_typos_still_repaired(lex):
    c = SpellingCorrector(lex)
    assert c.repair_word("bukhr")[0] == "bukhar" and c.repair_word("khansee")[0] == "khansi"


# ---------------------------------------------------------------- safety net
def test_extra_rules_are_additive_and_cannot_remove_builtins(tmp_path):
    f = tmp_path / "emergency_keywords.csv"
    f.write_text("phrase,english,canonical,flag\nzehar kha liya,Poisoning,poisoning,EMERGENCY\nchest pain,Chest pain,chest pain,BOGUS\n")
    extra = load_extra_rules(f)
    assert [r[2] for r in extra] == ["poisoning"]                           # invalid flag row dropped
    clf = SeverityClassifier(extra_rules=extra)
    assert clf.classify({"symptoms": []}, ["usne zehar kha liya"]).emergency
    assert clf.classify({"symptoms": ["chest pain"]}).emergency             # built-ins intact
    assert len(clf.rules) == len(EMERGENCY_RULES) + 1


def test_word_order_and_english_variants():
    for t in ("khoon ki ulti", "vomiting blood", "i can t breathe", "cannot breathe", "chhaati dard"):
        assert emergency_check({"symptoms": []}, [t]), t


# ---------------------------------------------------------------- accumulator
