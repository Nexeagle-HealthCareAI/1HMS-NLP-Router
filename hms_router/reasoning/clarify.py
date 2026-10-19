"""Pick ONE question that best separates the top candidates - what a triage nurse would ask next.

Order: the age question when paediatrics is in contention and the age is unknown; then curated
questions for the confusions seen most in testing; otherwise "which fits best" with the candidates
as options. Wording needs clinician review before patients see it.
"""
from __future__ import annotations

from ..domain.models import Candidate, Clarification, Query

# frozenset of two specialtyIds -> (English, Hinglish)
_PAIR_QUESTIONS: dict[frozenset, tuple[str, str]] = {
    frozenset({"cardiology", "gastroenterology"}): (
        "Is the pain in the middle of the chest, or in the upper stomach after eating?",
        "Dard seene ke beech mein hai, ya khana khane ke baad pet ke upar?"),
    frozenset({"cardiology", "pulmonology"}): (
        "Is it mainly chest pain or pressure, or mainly cough and breathlessness?",
        "Zyada seene mein dard ya dabaav hai, ya khansi aur saans phoolna?"),
    frozenset({"orthopedics", "physiotherapy"}): (
        "Did it start after an injury or a fall, or has it built up slowly?",
        "Chot ya girne ke baad shuru hua, ya dheere dheere badha?"),
    frozenset({"orthopedics", "rheumatology"}): (
        "Is it one joint after an injury, or several joints that are stiff in the morning?",
        "Chot ke baad ek jod mein hai, ya kai jodon mein subah akdan rehti hai?"),
    frozenset({"urology", "gynecology"}): (
        "Is the problem mainly with passing urine, or with periods or vaginal discharge?",
        "Takleef zyada peshab mein hai, ya periods ya safed paani mein?"),
    frozenset({"urology", "nephrology"}): (
        "Is it pain or burning when passing urine, or swelling and less urine than usual?",
        "Peshab mein dard ya jalan hai, ya sujan aur peshab kam aana?"),
    frozenset({"neurology", "general"}): (
        "Did the headache start suddenly and severely, or has it come and gone for days?",
        "Sir dard achanak tez shuru hua, ya kuch dinon se aata jaata hai?"),
    frozenset({"neurology", "neurosurgery"}): (
        "Was there a head or back injury, or did this start on its own?",
        "Sir ya kamar mein chot lagi thi, ya apne aap shuru hua?"),
    frozenset({"cardiology", "cardiothoracicsurgery"}): (
        "Has a doctor already advised heart surgery, or is this a new problem?",
        "Kya doctor ne pehle heart surgery ki salah di hai, ya yeh nayi takleef hai?"),
    frozenset({"ent", "general"}): (
        "Is the main problem in the ear, nose or throat, or is it a general fever and body ache?",
        "Takleef zyada kaan, naak ya gale mein hai, ya bukhar aur badan dard hai?"),
}
_AGE = (("Is this for a child or an adult?", ["Child", "Adult"]),
        ("Yeh bachche ke liye hai ya bade ke liye?", ["Bachcha", "Bada"]))
_GENERIC = ("Which of these sounds closest to what you need?", "Inmein se kaunsa aapki takleef se zyada milta hai?")


class ClarifyingQuestions:
    def __init__(self, options: int = 3):
        self.options = options

    def pick(self, query: Query, candidates: list[Candidate]) -> Clarification | None:
        top = candidates[: self.options]
        if len(top) < 2:
            return None
        hinglish = query.language in ("hinglish", "hindi")
        ids = [c.specialtyId for c in top]
        if "pediatrics" in ids[:2] and query.age_group is None:
            q, opts = _AGE[1] if hinglish else _AGE[0]
            return Clarification(question=q, options=opts, optionIds=["child", "adult"], kind="age")
        pair = frozenset(ids[:2])
        q2 = _PAIR_QUESTIONS.get(pair, _GENERIC)
        return Clarification(question=q2[1] if hinglish else q2[0], options=[c.name for c in top], optionIds=ids)
