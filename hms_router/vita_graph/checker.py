"""Every LLM-written reply is checked before a patient sees it. A failed check never reaches the patient: the turn
falls back to Vita's reviewed rules engine. Deliberately strict; tune with Vita TestAgent's ai-agent pack."""
from __future__ import annotations

import re

MAX_CHARS = 700

_MEDICINE = re.compile(
    r"\b(paracetamol|acetaminophen|crocin|dolo|calpol|ibuprofen|brufen|combiflam|aspirin|disprin|diclofenac|voveran|"
    r"azithro\w*|amoxi\w*|augmentin|cipro\w*|oflox\w*|norflox\w*|metronidazole|flagyl|doxy\w*|cefixime|antibiotics?|"
    r"cetirizine|cetzine|levocet\w*|allegra|montair|montelukast|sinarest|saridon|vicks|pantoprazole|pan\s?40|pan\s?d|"
    r"omeprazole|ranitidine|rantac|digene|eno|gelusil|antacids?|domperidone|ondansetron|emeset|loperamide|ors|"
    r"metformin|insulin|glimepiride|amlodipine|telmisartan|atorvastatin|thyroxine|eltroxin|steroids?|prednisolone|"
    r"dexamethasone|betadine|neosporin|soframycin|burnol|ointment|syrup|dawai\s+(lo|lein|le\s+lo|kha\s+lo)|"
    r"medicine\s+(lo|le\s+lo|lein|kha\s+lo)|take\s+(a\s+|an\s+|this\s+|some\s+)?(tablet|pill|medicine)|"
    r"पैरासिटामोल|दवा\s+(लें|लो|खाएं|खा\s+लें))\b", re.I)
_DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|gm|g|tablets?|tabs?|goli|goliyan|capsules?|drops|puffs?|units?|iu|"
                   r"times\s+a\s+day|baar\s+(din|roz))\b", re.I)
_DISEASE = (r"(dengue|malaria|typhoid|covid|corona|cancer|tumou?r|tb|tuberculosis|diabetes|migraine|pneumonia|"
            r"jaundice|hepatitis|ulcer|appendicitis|heart\s+attack|stroke|infection|uti|kidney\s+stone|pathri|"
            r"thyroid|asthma|arthritis|gastritis|acidity|allergy|viral|flu|anemia|anaemia|pcod|pcos|depression|anxiety\s+disorder)")
_DIAGNOSE = re.compile(
    rf"\b(you\s+(have|might\s+have|may\s+have|could\s+have|probably\s+have|likely\s+have|are\s+suffering\s+from)|"
    rf"(it|this)\s+(is|looks\s+like|sounds\s+like|could\s+be|might\s+be|may\s+be)|"
    rf"aapko|apko|lagta\s+hai|ho\s+sakta|ho\s+sakti|yeh|ye)\b[^.?!]{{0,40}}\b{_DISEASE}\b"
    rf"|\b{_DISEASE}\b[^.?!]{{0,25}}\b(ho\s+sakta|ho\s+sakti|lag\s+raha|lagta|hai\s+aapko|hua\s+hai)\b", re.I)
_PHONE = re.compile(r"(?<!\d)(\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)|(?<!\d)\d{10,}(?!\d)")
_LINK = re.compile(r"https?://|www\.|\.com\b|\.in\b", re.I)
_BOOKED = re.compile(r"\b(appointment\s+(is\s+)?(booked|confirmed)|booked\s+(your|an|the)\s+appointment|"
                     r"book\s+(ho\s+gaya|ho\s+gayi|kar\s+diya|kar\s+di)|confirm\s+ho\s+gaya)\b", re.I)
_DR_NAME = re.compile(r"\b(?:Dr|Doctor|डॉ)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")


def problems(message: str, known_names: set[str] | None = None) -> list[str]:
    """What is wrong with an LLM reply (empty = fine to show)."""
    out: list[str] = []
    text = message or ""
    if not text.strip():
        return ["empty"]
    if len(text) > MAX_CHARS:
        out.append("too_long")
    if _MEDICINE.search(text) or _DOSE.search(text):
        out.append("medicine")
    if _DIAGNOSE.search(text):
        out.append("diagnosis")
    if _PHONE.search(text):
        out.append("phone_number")
    if _LINK.search(text):
        out.append("link")
    if _BOOKED.search(text):
        out.append("booking_claim")
    known = {n.lower() for n in (known_names or set())}
    for m in _DR_NAME.finditer(text):
        name = m.group(1).lower()
        if not any(name in k or k.endswith(name) for k in known):
            out.append("unknown_doctor")
            break
    return out
