"""Self-harm and suicidal intent, by meaning rather than one exact phrase.

red_flags.csv lists phrases such as "marna chahta" or "kill myself"; a patient who writes "I don't want to live
anymore", "mann karta hai sab khatam kar doon", "ab jeene ka mann nahi karta" or "मैं मरना चाहता हूँ" matched none
of them (probe 2026-10-30). This detector looks for the shape of the thought in English, Roman Hinglish and
Devanagari: a wish to die / not to live, ending one's life, hurting oneself, a method, or the words themselves.

* It runs on the raw text (Devanagari) and on the normalised text (Roman), on every message and on the
  conversation so far, before any model, and can only ADD urgency.
* It errs on the side of asking: an indirect sign ("zindagi se thak gaya hoon", "no reason to live") is flagged.
  A clear "I am not suicidal" / "suicide ka koi khayal nahi" is not.
* The reason it gives is the red-flag english "Suicidal thoughts / self-harm", so every reply that names it adds
  Tele-MANAS 14416 (voice.emergency) and recommends Emergency Medicine + Psychiatry.

Starting point only: the phrase families need review by a mental-health clinician before patients rely on them.
"""
from __future__ import annotations

import re

from ..domain.models import EmergencyAssessment, Query

REASON = "Suicidal thoughts / self-harm"
CANONICAL = "suicidal thoughts"
SPECIALTIES = ["emergencymedicine", "psychiatry"]

_W = r"(?:\s+\w+){0,3}?"                     # up to three words in between ("jeene ka bilkul bhi mann nahi")
_NEG = r"(?:nahi|nahin|nhi|nai|na|mat)"
_WANT = r"(?:chahta|chahti|chahte|chahata|chaahta|chaahti|chaahata|chaahataa|chaahatee|chaahti|chahiye|chaiye|hai|he|h)"
_SELF = r"(?:khud|khudko|apne\s*aap|apne\s*ko|apneaap|swayam|myself|my\s*self)"

_ROMAN = [
    # the words themselves
    r"\bsuicid\w*", r"\bsuicide\b", r"\bkhud\s*kushi\b", r"\bkhudkhushi\b", r"\bkhudkhusi\b", r"\bkhudkushi\b",
    r"\baa?tm\s*hatya\b", r"\baatmahatyaa?\b", r"\bself[\s-]*harm\w*", r"\bkill\w*\s+(?:my\s*self|myself)\b",
    r"\bhurt\w*\s+(?:my\s*self|myself)\b", r"\bcut\w*\s+(?:my\s*self|myself|my\s+(?:wrist|wrists|hand|arm))\b",
    # wish to die
    r"\b(?:want|wanna|wanted|going|plan\w*|think\w*\s+(?:of|about)|ready)\s+(?:to\s+)?(?:die|end\s+it)\b",
    r"\bi\s+(?:wish|hope)\s+i\s+(?:was|were|could\s+be|am)\s+dead\b", r"\bbetter\s+off\s+dead\b",
    r"\b(?:wish|hope)\s+i\s+(?:never\s+)?(?:wake|woke)\s+up\b", r"\bdon\s?t\s+want\s+to\s+wake\s+up\b",
    # not wanting to live / go on
    r"\b(?:don\s?t|dont|do\s+not|no\s+longer|not)\s+(?:\w+\s+)?want\s+to\s+(?:live|be\s+alive|exist|be\s+here|go\s+on)\b",
    r"\b(?:no|nothing)\s+(?:\w+\s+)?(?:reason|point|purpose)\s+(?:to|in|for)\s+(?:live|living|life|go\s+on|going\s+on)\b",
    r"\bcan\s?t\s+go\s+on\b", r"\bcannot\s+go\s+on\b", r"\btired\s+of\s+(?:living|life|being\s+alive)\b",
    r"\b(?:life|living)\s+(?:is\s+)?(?:not\s+worth|isn\s?t\s+worth|worthless|pointless|meaningless)\b",
    r"\bnot\s+worth\s+living\b",
    r"\b(?:end|take|ending|taking)\s+(?:my|my\s+own)\s+life\b", r"\bend\s+it\s+all\b", r"\bending\s+it\s+all\b",
    r"\boverdose\b", r"\b(?:took|take|taking|swallow\w*)\s+(?:all|many|lots\s+of|a\s+lot\s+of|whole\s+bottle)\s+(?:of\s+)?(?:my\s+)?(?:pills|tablets|medicines?)\b",
    r"\bhang\s+(?:my\s*self|myself)\b",
    # Hinglish: marna / mar jaana
    rf"\b(?:marna|marnaa|maranaa|marne|mar\s+jana|mar\s+jaana|mar\s+jaanaa|mar\s+jaane){_W}\s+{_WANT}\b",
    rf"\b(?:marne|mar\s+jaane)\s+(?:ka|ki|ko)\s+(?:mann|man|mn|dil|ichha|ichchha|iccha|khayal|khyal|vichar|soch)\b",
    r"\bmar\s+(?:jaun|jaaun|jau|jaaoon|jaoon|jaaun\s+main|jaata|jaati)\s+(?:to|toh|main|mai|me|bas)?\b(?!.*\b(?:dard|pain)\b)",
    r"\bmujhe\s+mar\s+(?:jana|jaana)\s+(?:hai|he)\b", r"\bmaut\s+(?:chahiye|chahta|chahti|aa\s+jaye|aa\s+jaaye)\b",
    # Hinglish: jeena nahi
    rf"\b(?:jeena|jina|jeenaa|jeene|jine|zinda\s+rehna|zinda\s+rahna|jeevit\s+rehna){_W}\s+{_NEG}\s+(?:chahta|chahti|chahte|hai|he|karta|karti|karte|lagta|lagti|hota|hoti)\b",
    rf"\b(?:jeene|jine)\s+(?:ka|ki|ko)\s+(?:mann|man|mn|dil|ichha|ichchha|iccha|wajah|vajah|matlab|koi\s+wajah|koi\s+matlab){_W}\s+{_NEG}\b",
    rf"\b(?:koi|ab\s+koi)\s+(?:wajah|vajah|matlab|reason)\s+{_NEG}\s+(?:jeene|jine)\b",
    r"\b(?:zindagi|zindagee|jindagi|jeevan|life)\s+(?:se\s+)?(?:thak|tang|pareshan)\s+(?:gaya|gayi|gyi|gya|aa\s+gaya|aa\s+gayi|ho\s+gaya|ho\s+gayi)\b",
    r"\b(?:zindagi|zindagee|jindagi|jeevan|life|jaan)\s+(?:khatam|khatm|samapt)\s+(?:kar|karna|karni|karne|kar\s+lu|kar\s+lun|kar\s+loon|kar\s+dun|kar\s+doon|kar\s+du)\b",
    # Hinglish: ending it / harming oneself
    r"\bsab\s+(?:kuch\s+)?(?:khatam|khatm)\s+kar\s+(?:du|dun|doon|dunga|dungi|deta|deti|dena|lu|lun|loon|lunga|lungi|lena)\b",
    rf"\b{_SELF}\s+(?:ko\s+)?(?:maar|mar|khatam|khatm|nuksan|nuksaan|nukasan|chot|hurt|kaat|kat|jala|khatam\s+kar)\w*",
    r"\bapni\s+(?:jaan|zindagi|jindagi)\s+(?:le|lena|lene|lu|lun|loon|de|dena|dene|du|dun|doon|khatam)\b",
    r"\bjaan\s+(?:de|dena|dene|du|dun|doon|dunga|dungi)\s+(?:du|dun|doon|dunga|dungi|chahta|chahti|hai|he)?\b",
    r"\b(?:fansi|phansi|faansi|phaansi|fanda|phanda|fandaa)\s+(?:laga|lagana|lagane|lagaa|lagaun|lagaunga|lagaungi|le|lena|lu|lun|loon|dal)\w*",
    r"\b(?:nas|nass|kalai|wrist)\s+(?:kaat|kat)\w*",
    r"\b(?:zeher|zehar|zahar|jahar|jeher|poison)\s+(?:kha|pi|pee|le|lena|khana|peena|khaa)\w*",
    r"\b(?:saari|sari|bahut\s+saari|sab|poori|puri|bahut)\s+(?:goli|goliyan|goliya|goliyaan|tablet|tablets|dawai|dawaiyan|dawa)\s+(?:kha|khaa|le|lee)\s*(?:li|liya|lee|lu|lun|loon|lunga|lungi|lena|leni|chuka|chuki)?\b",
    r"\b(?:duniya|duniyaa)\s+(?:se\s+)?(?:chhod|chod|chale|chala|chali)\s*(?:du|dun|doon|dena|jaun|jau|jaana|jana|jaunga|jaungi)?\b(?=.*\b(?:chahta|chahti|hai|he|mann|man|dil|jaun|jau|dun|doon)\b)",
]
_DEV = [
    r"आत्महत्या", r"ख़ुदकुशी", r"खुदकुशी", r"सुसाइड", r"सुसाईड",
    r"मरना\s*(?:\S+\s*){0,2}चाह", r"मर\s*जाना\s*(?:\S+\s*){0,2}(?:चाह|है)", r"मर\s*जा(?:ऊँ|ऊं|उं|ऊ)", r"मौत\s*(?:चाहिए|चाह)",
    r"मरने\s*(?:का|की|को)\s*(?:मन|दिल|इच्छा|ख़याल|खयाल|ख्याल|विचार)",
    r"(?:जीना|जीने|ज़िंदा\s*रहना|जिंदा\s*रहना)\s*(?:\S+\s*){0,3}(?:नहीं|नही|ना)\s*(?:चाह|है|कर|लग|हो)",
    r"(?:जीने)\s*(?:का|की|को)\s*(?:मन|दिल|इच्छा|वजह|मतलब)",
    r"(?:ख़ुद|खुद|अपने\s*आप)\s*(?:को)?\s*(?:मार|ख़त्म|खत्म|नुक़सान|नुकसान|चोट|काट)",
    r"अपनी\s*(?:जान|ज़िंदगी|जिंदगी)\s*(?:ले|दे|ख़त्म|खत्म)", r"(?:ज़िंदगी|जिंदगी|जीवन)\s*(?:ख़त्म|खत्म|समाप्त)\s*कर",
    r"(?:ज़िंदगी|जिंदगी|जीवन)\s*से\s*(?:थक|तंग)",
    r"सब\s*(?:कुछ\s*)?(?:ख़त्म|खत्म)\s*कर\s*(?:दूँ|दूं|दू|दूंगा|दूँगा|दूंगी|लूँ|लूं|लू|लूंगा|दे)",
    r"(?:फांसी|फाँसी|फंदा|फन्दा)\s*(?:लगा|ले|डाल)", r"(?:नस|कलाई)\s*काट", r"(?:ज़हर|जहर)\s*(?:खा|पी|ले)",
]
# a clear statement that the thought is absent ("I am not suicidal", "suicide ka koi khayal nahi")
_ABSENT = [
    r"\b(?:not|never|no)\s+(?:been\s+|feeling\s+|having\s+)?(?:suicidal|self[\s-]*harm\w*)\b",
    r"\b(?:no|never\s+had|don\s?t\s+have|do\s+not\s+have)\s+(?:any\s+)?(?:suicidal\s+thoughts|thoughts\s+of\s+(?:suicide|self[\s-]*harm))\b",
    r"\b(?:suicide|suicidal|khudkushi|aatmhatya|marne)\s+(?:ka|ki|ke)?\s*(?:koi\s+)?(?:khayal|khyal|vichar|soch|thoughts?|irada)?\s*(?:bilkul\s+)?(?:nahi|nahin|nhi)\b",
    r"(?:आत्महत्या|ख़ुदकुशी|खुदकुशी)\s*(?:का|की|के)?\s*(?:कोई\s*)?(?:ख़याल|खयाल|विचार|इरादा)?\s*नहीं",
]
_ROMAN_RE = [re.compile(p, re.I) for p in _ROMAN]
_DEV_RE = [re.compile(p) for p in _DEV]
_ABSENT_RE = [re.compile(p, re.I) for p in _ABSENT]


def _clean(text: str) -> str:
    text = (text or "").lower().replace("’", "'").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def mentions_self_harm(*texts: str) -> bool:
    """True when any text expresses suicidal intent or self-harm (Roman or Devanagari)."""
    for raw in texts:
        if not raw:
            continue
        roman = _clean(raw)
        if any(r.search(roman) or r.search(raw) for r in _ABSENT_RE):
            stripped = roman
            for r in _ABSENT_RE:
                stripped = r.sub(" ", stripped)
            raw_stripped = raw
            for r in _ABSENT_RE:
                raw_stripped = r.sub(" ", raw_stripped)
            roman, raw = stripped, raw_stripped
        if any(r.search(roman) for r in _ROMAN_RE) or any(r.search(raw) for r in _DEV_RE):
            return True
    return False


class SelfHarmDetector:
    """EmergencyDetector: flags suicidal intent / self-harm on the raw and the normalised text."""

    def detect(self, query: Query) -> EmergencyAssessment:
        if mentions_self_harm(query.raw, query.normalized):
            return EmergencyAssessment(flag=True, reasons=[REASON], source="rules", score=1.0,
                                       recommendedSpecialtyIds=list(SPECIALTIES))
        return EmergencyAssessment(flag=False)
