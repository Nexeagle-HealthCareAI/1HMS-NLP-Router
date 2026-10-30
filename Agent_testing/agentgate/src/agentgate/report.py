"""One-page PDF report. KeepInFrame(shrink) guarantees it never spills to page 2."""
from __future__ import annotations

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

GREEN, RED, AMBER, GREY, INK = (colors.HexColor(h) for h in
                                ("#1a7f4b", "#b3261e", "#b26a00", "#5f6368", "#202124"))
VERDICT_COLOR = {"GO": GREEN, "SMOKE PASS": GREEN, "NO-GO": RED, "SMOKE FAIL": RED,
                 "PENDING MANUAL CHECKS": AMBER}
STATUS_COLOR = {"pass": GREEN, "fail": RED, "warn": AMBER, "pending": AMBER}


def _p(text, size=8, bold=False, color=INK, align=0, leading=None):
    style = ParagraphStyle("x", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                           leading=leading or size + 2.5, textColor=color, alignment=align)
    return Paragraph(text, style)


def _h(text):
    return _p(escape(text), 10, True, INK, leading=13)


def _next_actions(res: dict) -> list[str]:
    gate, cases = res["gate"], res["cases"]
    actions = []
    failing = [c for c in cases if not c["passed"]]
    if gate["verdict"] in ("NO-GO", "SMOKE FAIL"):
        actions.append("Fix the agent (prompt, tool or data) and add every failed case below to the golden dataset.")
        actions.append("Re-run from step 3; do not release until the gate returns GO.")
    if gate["totals"]["must_never_events"]:
        actions.append("Treat must-never events as release blockers, even if all percentages look good.")
    if any(c["status"] == "warn" and c["name"] == "Sample size" for c in gate["checks"]):
        actions.append("Grow thin categories toward the minimum case count using real failures from pilots.")
    if gate["verdict"] == "PENDING MANUAL CHECKS":
        actions.append("Rehearse rollback, then re-run with --rollback-rehearsed to finalise the verdict.")
    if gate["verdict"] in ("GO", "SMOKE PASS") and not failing:
        actions.append("Release gradually (limited users first), watch the dashboards, and feed findings back (step 8).")
    return actions[:4] or ["No action needed."]


def build_pdf(res: dict, path: str) -> None:
    meta, gate, cases = res["meta"], res["gate"], res["cases"]
    t = gate["totals"]
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title="AI Agent Test Report", author="agentgate")
    width = A4[0] - 28 * mm - 12
    height = A4[1] - 24 * mm - 12
    s = []

    s.append(_p("AI Agent Test Report", 18, True, INK, leading=22))
    s.append(_p(escape(f"Tier: {meta['tier']}  |  Run: {meta['run_id']} UTC  |  Agent: {meta['agent']}"), 8, color=GREY))
    s.append(Spacer(1, 5))

    banner = Table([[_p(escape(gate["verdict"]), 17, True, colors.white, leading=21),
                     _p(escape(f"{t['passed']} of {t['cases']} cases passed  |  "
                               f"{t['must_never_events']} must-never event(s)"),
                        9, False, colors.white, align=2)]],
                   colWidths=[width * 0.55, width * 0.45])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), VERDICT_COLOR.get(gate["verdict"], GREY)),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    s += [banner, Spacer(1, 6)]

    judge = f"{t['judge_avg']:.2f}/5" if t["judge_avg"] is not None else "n/a"
    stats = [("Cases", str(t["cases"])), ("Pass rate", f"{t['passed'] / t['cases']:.0%}" if t["cases"] else "n/a"),
             ("p95 latency", f"{t['p95_latency_ms']:.0f} ms"), ("Error rate", f"{t['error_rate']:.1%}"),
             ("Judge avg", judge)]
    cells = [[_p(f"<b>{escape(v)}</b>", 13, align=1, leading=16) for _, v in stats],
             [_p(escape(k), 7, color=GREY, align=1) for k, _ in stats]]
    st = Table(cells, colWidths=[width / len(stats)] * len(stats))
    st.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f8fa")),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    s += [st, Spacer(1, 8), _h("Results by category (step 7 thresholds, set in step 1)")]

    rows = [[_p("<b>Category</b>", 8), _p("<b>Cases</b>", 8), _p("<b>Passed</b>", 8),
             _p("<b>Pass rate</b>", 8), _p("<b>Required</b>", 8), _p("<b>Status</b>", 8)]]
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.6, GREY), ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.lightgrey),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    for i, c in enumerate(gate["categories"], start=1):
        req = f"{c['required']:.0%}" + (" (zero tol.)" if c["zero_tolerance"] else "")
        col = GREEN if c["ok"] else RED
        rows.append([_p(escape(c["category"]), 8), _p(str(c["cases"]), 8), _p(str(c["passed"]), 8),
                     _p(f"{c['rate']:.0%}", 8), _p(req, 8),
                     _p("PASS" if c["ok"] else "FAIL", 8, True, col)])
    ct = Table(rows, colWidths=[width * x for x in (0.26, 0.1, 0.12, 0.14, 0.22, 0.16)])
    ct.setStyle(TableStyle(style))
    s += [ct, Spacer(1, 8), _h("Release gate checks")]

    rows = [[_p("<b>Check</b>", 8), _p("<b>Status</b>", 8), _p("<b>Detail</b>", 8)]]
    for c in gate["checks"]:
        rows.append([_p(escape(c["name"]), 8), _p(c["status"].upper(), 8, True, STATUS_COLOR[c["status"]]),
                     _p(escape(c["detail"][:140]), 7.5, color=GREY)])
    gt = Table(rows, colWidths=[width * 0.24, width * 0.12, width * 0.64])
    gt.setStyle(TableStyle(style))
    s += [gt, Spacer(1, 8)]

    failing = [c for c in cases if not c["passed"]]
    s.append(_h(f"Failed cases ({len(failing)})"))
    if failing:
        for c in failing[:6]:
            why = "; ".join(c["must_never"] + c["failed_checks"])[:170] or "see results.json"
            s.append(_p(f"- <b>{escape(c['id'])}</b> ({escape(c['category'])}, {c['passes']}/{c['repeats']} runs): {escape(why)}", 7.5))
        if len(failing) > 6:
            s.append(_p(f"...and {len(failing) - 6} more in results.json", 7.5, color=GREY))
    else:
        s.append(_p("None.", 8))
    s += [Spacer(1, 8), _h("Next actions")]
    for a in _next_actions(res):
        s.append(_p(f"- {escape(a)}", 8))
    s.append(Spacer(1, 8))
    s.append(_p(escape(
        f"Pinned for reproducibility | dataset {meta['dataset_hash']} | thresholds {meta['thresholds_hash']} | "
        f"judge {meta['judge']} | commit {meta['commit']} | agentgate {meta['version']}"), 6.5, color=GREY))

    doc.build([KeepInFrame(width, height, s, mode="shrink")])
