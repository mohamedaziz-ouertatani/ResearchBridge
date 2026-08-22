"""Render a ResearchAssessment as a downloadable .docx or .pdf (blueprint Sec 2A).

Both formats print the same content in the same order as the web report
(AssessmentReport.tsx) and by the same design principle: an assessment is
never its own source of truth, so every gradeable field's full supporting
evidence is printed inline here rather than collapsed the way the web UI's
<details> element hides it - the exported file has to stand on its own
without a browser to expand anything in.

build_report_sections() is the one place that decides *what* goes in the
report; build_docx()/build_pdf() only decide *how* to lay it out for their
format. A future third format is a new renderer over the same sections,
not a rework of the content logic.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from researchbridge.api.schemas import AssessmentEvidenceOut, ResearchAssessmentOut

OPPORTUNITIES_REASON = (
    "Not generated. Naming a product opportunity means inventing a claim the "
    "literature does not make, so this is left to a human reviewer."
)


@dataclass
class ReportSection:
    label: str
    level: str | None = None
    """A rule-based category (novelty/feasibility level), not a measurement."""
    body: str | None = None
    unassessed_reason: str | None = None
    """Shown instead of body when body is None."""
    evidence: list[AssessmentEvidenceOut] = field(default_factory=list)


def build_report_sections(assessment: ResearchAssessmentOut) -> list[ReportSection]:
    by_role: dict[str, list[AssessmentEvidenceOut]] = {}
    for item in assessment.evidence:
        by_role.setdefault(item.role, []).append(item)

    applications_body = None
    if assessment.potential_applications:
        applications_body = "\n".join(
            f"- {app['application']} (source: {app['source_paper']})" for app in assessment.potential_applications
        )

    research_gap_body = assessment.research_gap_text
    if research_gap_body and assessment.research_gap_source:
        source_note = (
            "reused a reviewed candidate gap"
            if assessment.research_gap_source == "reused_candidate_gap"
            else "found for this input"
        )
        research_gap_body = f"{research_gap_body}\n({source_note})"

    return [
        ReportSection(
            label="Existing solutions",
            body=assessment.comparison_summary,
            unassessed_reason="No retrieved paper had extracted claims to compare against.",
            evidence=by_role.get("comparison", []),
        ),
        ReportSection(
            label="Novelty assessment",
            level=assessment.novelty_level,
            body=assessment.novelty_reasoning,
            unassessed_reason="Not enough evidence to judge novelty.",
            evidence=by_role.get("novelty", []),
        ),
        ReportSection(
            label="Research gap",
            body=research_gap_body,
            unassessed_reason="No gap was found in the retrieved literature for this input.",
            evidence=by_role.get("research_gap", []),
        ),
        ReportSection(
            label="Potential applications",
            body=applications_body,
            unassessed_reason="No retrieved paper stated an application.",
            evidence=by_role.get("application", []),
        ),
        ReportSection(
            label="Product / technology opportunities",
            body=None,
            unassessed_reason=OPPORTUNITIES_REASON,
            evidence=by_role.get("opportunity", []),
        ),
        ReportSection(
            label="Technical feasibility",
            level=assessment.technical_feasibility_level,
            body=assessment.technical_feasibility_reasoning,
            unassessed_reason="Nothing close enough to ground a feasibility judgement.",
            evidence=by_role.get("feasibility", []),
        ),
        ReportSection(
            label="Risks / limitations",
            body=assessment.risks_and_limitations,
            unassessed_reason="No retrieved paper stated a limitation.",
            evidence=by_role.get("risk", []),
        ),
        ReportSection(
            label="External validation needed",
            body=assessment.external_validation_needed,
        ),
    ]


def _related_papers(assessment: ResearchAssessmentOut) -> list[str]:
    seen: dict[str, str] = {}
    for item in assessment.evidence:
        seen[str(item.paper_id)] = item.paper_title
    return list(seen.values())


def build_docx(assessment: ResearchAssessmentOut) -> bytes:
    import docx

    document = docx.Document()

    document.add_heading("Research Assessment", level=0)
    document.add_paragraph(f"Recommendation: {assessment.recommendation or 'Not assessed'}")
    document.add_paragraph(f"Confidence: {assessment.confidence or 'Not assessed'}")
    document.add_paragraph(f"Human reviewed: {'yes' if assessment.human_reviewed else 'no'}")

    document.add_heading("Input", level=1)
    document.add_paragraph(assessment.research_input.raw_text)
    document.add_paragraph(f"Type: {assessment.research_input.input_type}")

    related = _related_papers(assessment)
    if related:
        document.add_heading("Related research", level=1)
        for title in related:
            document.add_paragraph(title, style="List Bullet")

    for section in build_report_sections(assessment):
        heading = document.add_heading(section.label, level=1)
        if section.level:
            heading.add_run(f"  ({section.level.replace('_', ' ')})")

        if section.body:
            document.add_paragraph(section.body)
        elif section.unassessed_reason:
            document.add_paragraph(section.unassessed_reason)

        for item in section.evidence:
            p = document.add_paragraph(style="List Bullet")
            p.add_run(f"“{item.text}”").italic = True
            p.add_run(f" — {item.paper_title}")
            if item.section:
                p.add_run(f" ({item.section})")

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def build_pdf(assessment: ResearchAssessmentOut) -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = []

    def para(text: str, style: str = "BodyText") -> None:
        story.append(Paragraph(_escape(text), styles[style]))

    para("Research Assessment", "Title")
    para(f"Recommendation: {assessment.recommendation or 'Not assessed'}")
    para(f"Confidence: {assessment.confidence or 'Not assessed'}")
    para(f"Human reviewed: {'yes' if assessment.human_reviewed else 'no'}")
    story.append(Spacer(1, 12))

    para("Input", "Heading2")
    para(assessment.research_input.raw_text)
    para(f"Type: {assessment.research_input.input_type}")
    story.append(Spacer(1, 12))

    related = _related_papers(assessment)
    if related:
        para("Related research", "Heading2")
        for title in related:
            para(f"- {title}")
        story.append(Spacer(1, 12))

    for section in build_report_sections(assessment):
        heading = section.label
        if section.level:
            heading += f" ({section.level.replace('_', ' ')})"
        para(heading, "Heading2")

        if section.body:
            para(section.body)
        elif section.unassessed_reason:
            para(section.unassessed_reason)

        for item in section.evidence:
            suffix = f" ({item.section})" if item.section else ""
            para(f"“{item.text}” — {item.paper_title}{suffix}")

        story.append(Spacer(1, 12))

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=LETTER).build(story)
    return buffer.getvalue()


def _escape(text: str) -> str:
    """reportlab's Paragraph interprets its text as a small XML/markup dialect."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
