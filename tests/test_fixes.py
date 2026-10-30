"""Unit tests for fixes made while joining the steps (script, spelling, safety net, output, keywords)."""
import fakeredis
import pytest

from hms_router import HMSRouterPipeline
from hms_router.accumulator import SymptomAccumulator
from hms_router.keywords import KeywordTagger
from hms_router.lexicon import Lexicon
from hms_router.output import OutputGenerator
from hms_router.script import normalize_script
from hms_router.severity import EMERGENCY, SeverityClassifier, emergency_check, load_extra_rules, EMERGENCY_RULES
from hms_router.symptom_clusters import Cluster
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
def test_accumulator_obeys_safety_net_flag_and_callable():
    acc = SymptomAccumulator(fakeredis.FakeRedis(decode_responses=True))
    s = acc.add("s1", {"symptoms": ["headache"], "language_mix": "hinglish"}, emergency=True)
    assert s.ready and s.ready_reason == "emergency_safety_net" and s.next_question is None
    seen = []
    s = acc.add("s2", {"symptoms": ["cough"]}, emergency=lambda st: seen.append(st.symptoms) or False)
    assert seen == [["cough"]] and not s.ready and s.next_question                  # callable got the merged state


# ---------------------------------------------------------------- Step 7
_SERVICE = None


def _card(state, **kw):
    """The card /chat builds: severity first, then the route-symptom engine (route_bridge), as agent._route does."""
    global _SERVICE
    if _SERVICE is None:
        from hms_router.application.container import build_route_symptom_service
        _SERVICE = build_route_symptom_service()
    from hms_router.route_bridge import emergency_routing, to_routing
    clf = SeverityClassifier()
    sev = clf.classify(state).to_dict()
    syms = list(state.get("symptoms") or [])
    decision = _SERVICE.route(" ; ".join(syms) or "-", top_k=5)
    routing = (emergency_routing(decision, syms, "") if sev["emergency"] or decision.emergency.flag
               else to_routing(decision, syms, ""))
    return OutputGenerator().build(state, sev, routing)


def test_three_distinct_outputs_always():
    for syms in (["stomach pain", "fever"], ["itching"], ["headache", "dizziness"], ["not-a-symptom"], []):
        card = _card({"symptoms": syms})
        assert card.kind == "recommendation" and len(card.outputs) == 3
        assert len({o.specialist for o in card.outputs}) == 3, syms
        assert [o.role for o in card.outputs] == ["PRIMARY", "SECONDARY", "ALTERNATIVE"]


def test_alternative_is_always_gp_or_telemedicine():
    for syms in (["stomach pain", "fever"], ["cough", "fever"], ["itching"]):
        assert _card({"symptoms": syms}).outputs[2].specialist in ("General Physician", "Telemedicine consult", "OPD triage desk")


def test_emergency_gets_response_card_not_three_outputs():
    card = _card({"symptoms": ["chest pain"]})
    assert card.kind == "emergency" and len(card.outputs) == 1 and card.outputs[0].specialist == "Emergency Medicine"
    assert "108" in card.alert and "Cardiologist" in card.after_stabilisation


def test_watch_for_has_no_duplicates_and_always_has_emergency_line():
    card = _card({"symptoms": ["stomach pain", "vomiting", "fever"], "intensity": "moderate"})
    assert len(card.watch_for) == len(set(card.watch_for)) and "108" in card.watch_for[-1]


# ---------------------------------------------------------------- keywords
def test_keyword_levels_follow_corpus_idf():
    corpus = [Cluster(f"C{i}", f"S{i}", ("fever",) + (("dizziness",) if i == 0 else ()), "d") for i in range(10)]
    tagger = KeywordTagger(corpus)
    assert tagger.idf("dizziness") > tagger.idf("fever")                            # rarer = more discriminative
    pipe = HMSRouterPipeline(fakeredis.FakeRedis(decode_responses=True))
    tags = {t.canonical: t.level for t in tagger.tag(pipe.process("bahut bukhar aur chakkar").correction) if t.canonical}
    assert tags["fever"] == "MEDIUM" and tags["dizziness"] == "HIGH" and tags["very"] == "MODIFIER"
