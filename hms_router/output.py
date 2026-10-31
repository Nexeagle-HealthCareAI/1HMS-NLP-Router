"""Step 7 - three-output generation (+ the Emergency Response Card from Layer 4/5).

Flowchart (Layer 4):  Emergency flag? -- NO  -> 3 specialist outputs: Primary / Secondary / Alternative
                                      -- YES -> EMERGENCY ALERT -> Emergency Response Card (no 3-output list)

  Output 1  PRIMARY      highest-confidence match
  Output 2  SECONDARY    next distinct specialist (differential-diagnosis route)
  Output 3  ALTERNATIVE  GP or telemedicine option (never a repeat of 1 or 2)

Texts that are shown to patients (labels, watch-for lines) are deliberately few and taken from the architecture doc.
Have a clinician review them before go-live.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .severity import EMERGENCY_NUMBER
from .specialist_rag import EMERGENCY_SPECIALIST, FALLBACKS, RoutingResult

PRIMARY, SECONDARY, ALTERNATIVE = "PRIMARY", "SECONDARY", "ALTERNATIVE"
ROLE_HINT = {PRIMARY: "Highest confidence match", SECONDARY: "Differential-diagnosis route",
             ALTERNATIVE: "GP / telemedicine option"}

# doc, Step 7 sample card + Step 5 emergency table
GENERAL_WATCH_FOR = ("Chest pain, trouble breathing, fainting, heavy bleeding, face drooping or sudden weakness -> "
                     f"call {EMERGENCY_NUMBER} / go to ER immediately")
WATCH_FOR: dict[str, list[str]] = {
    "Gastroenterologist": ["Blood in vomit or stool, or pain that keeps worsening -> go to ER immediately"],
    "General Physician": ["Fever above 103 F or symptoms getting worse -> consult Emergency Medicine"],
}
LABELS = {  # en / hinglish
    "summary": ("Symptom summary", "Aapke symptoms"), "severity": ("Severity", "Gambhirta"),
    "specialists": ("Recommended specialists", "Kin doctors ko dikhayein"), "watch": ("Watch for", "Dhyan rakhein"),
    "alert": ("EMERGENCY", "EMERGENCY"),
    PRIMARY: ("Primary", "Pehla option"), SECONDARY: ("Secondary", "Doosra option"),
    ALTERNATIVE: ("Alternative", "Vikalp"),
}
DISCLAIMER = ("Triage support only - not a diagnosis.", "Yeh sirf margdarshan hai, diagnosis nahi.")
ICON = {"MILD": "GREEN", "MODERATE": "YELLOW", "SEVERE": "ORANGE", "EMERGENCY": "RED"}


@dataclass
class SpecialistOutput:
    role: str                      # PRIMARY | SECONDARY | ALTERNATIVE | EMERGENCY
    specialist: str
    confidence: float
    reason: str
    matched_symptoms: list[str] = field(default_factory=list)
    cluster_id: str | None = None  # feedback (Step 8) refers to this


@dataclass
class ResponseCard:
    kind: str                      # "recommendation" | "emergency"
    summary: str
    severity_level: str
    severity_score: int
    disposition: str
    outputs: list[SpecialistOutput]
    watch_for: list[str]
    alert: str | None = None
    emergency_reasons: list[str] = field(default_factory=list)
    after_stabilisation: list[str] = field(default_factory=list)   # specialists to see once the ER is done
    red_flags: list[str] = field(default_factory=list)
    low_confidence: bool = False
    disclaimer: str = DISCLAIMER[0]

    def to_dict(self) -> dict:
        return asdict(self)


def symptom_summary(state: dict) -> str:
    syms = state.get("symptoms") or []
    text = " + ".join(syms) if syms else "no clear symptom recognised yet"
    return f"{text} ({state['duration']})" if state.get("duration") else text


class OutputGenerator:
    def build(self, state: dict, severity: dict, routing: RoutingResult) -> ResponseCard:
        summary = symptom_summary(state)
        base = dict(summary=summary, severity_level=severity["level"], severity_score=severity["score"],
                    disposition=severity["disposition"], red_flags=list(severity.get("red_flags") or []))
        if severity.get("emergency") or routing.emergency:
            others = [r.specialist for r in routing.recommendations if r.specialist != EMERGENCY_SPECIALIST]
            reasons = [h["english"] for h in severity.get("emergency_hits") or []]
            return ResponseCard(
                kind="emergency", outputs=[SpecialistOutput("EMERGENCY", EMERGENCY_SPECIALIST, 1.0,
                                                            "Emergency red flag detected", list(state.get("symptoms") or []))],
                watch_for=[], alert=severity.get("alert"), emergency_reasons=reasons,
                after_stabilisation=others, **base)

        pool = [r for r in routing.recommendations if r.source in ("retrieval", "route-symptom")]
        chosen: list[SpecialistOutput] = []
        used: set[str] = set()

        def take(role: str, name: str, conf: float, why: str, matched: list[str], cid: str | None) -> None:
            chosen.append(SpecialistOutput(role, name, conf, why, matched, cid))
            used.add(name)

        def generic() -> str:                     # next unused GP-type option
            return next((f for f in FALLBACKS if f not in used), FALLBACKS[-1])

        for role, rec in zip((PRIMARY, SECONDARY), pool):                  # outputs 1-2 from retrieval
            take(role, rec.specialist, rec.confidence, rec.rationale, rec.matched_symptoms, rec.cluster_id)
        for role in (PRIMARY, SECONDARY)[len(chosen):]:                    # 3-output rule: pad, never leave a gap
            take(role, generic(), 0.0, "No closer match - start with a general assessment", [], None)
        take(ALTERNATIVE, generic(), 0.0, "Start with a general / tele-consult first", [], None)   # output 3

        watch: list[str] = []
        for o in chosen[:2]:                                   # specialist-specific lines first, in rank order
            watch += WATCH_FOR.get(o.specialist, [])
        watch.append(GENERAL_WATCH_FOR)
        return ResponseCard(kind="recommendation", outputs=chosen, watch_for=list(dict.fromkeys(watch)),
                            low_confidence=routing.low_confidence, **base)

    # ------------------------------------------------------------------ text rendering
    @staticmethod
    def render(card: ResponseCard, lang: str = "en") -> str:
        i = 1 if lang in ("hinglish", "hindi") else 0
        L = lambda k: LABELS[k][i]
        lines: list[str] = []
        if card.kind == "emergency":
            lines += [f"[!] {L('alert')}", card.alert or "", ""]
            if card.emergency_reasons:
                lines.append(", ".join(card.emergency_reasons))
            lines.append(f"{L('summary')}: {card.summary}")
            if card.after_stabilisation:
                lines.append("After the ER: " + ", ".join(card.after_stabilisation))
            lines += ["", DISCLAIMER[i]]
            return "\n".join(x for x in lines if x is not None)
        lines += [f"{L('summary')}: {card.summary}",
                  f"{L('severity')}: {ICON.get(card.severity_level, '')} {card.severity_level} - {card.disposition}"]
        for f in card.red_flags:
            lines.append(f"  ! {f}")
        lines += ["", f"{L('specialists')}:"]
        for n, o in enumerate(card.outputs, 1):
            why = f" -> {o.reason}" if o.reason else ""
            lines.append(f"  {n}. {o.specialist.upper():22} ({L(o.role)}){why}")
        if card.low_confidence:
            lines.append("  (No close symptom match - please describe more, or start with a general assessment.)")
        lines += ["", f"{L('watch')}:"] + [f"  - {w}" for w in card.watch_for] + ["", DISCLAIMER[i]]
        return "\n".join(lines)
