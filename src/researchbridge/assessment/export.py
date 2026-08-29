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
from datetime import datetime, timezone
from pathlib import Path

from researchbridge.api.schemas import AssessmentEvidenceOut, ResearchAssessmentOut

# The same ink-gray palette and font pairing as the web report (globals.css /
# layout.tsx) - Space Grotesk for headings/labels, Source Serif 4 for body
# text and quotes. No color accents: teal and amber are reserved for cosine
# distance and the active/typing state respectively, neither of which applies
# to a static export.
INK = "#14181d"
INK_SOFT = "#4a545f"
INK_FAINT = "#78838f"
RULE = "#c6cbd2"

FONTS_DIR = Path(__file__).parent / "fonts"

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


_LEVEL_RANK = {"low": 1, "medium": 2, "high": 3}


def _level_dots(level: str | None) -> str | None:
    """A filled/hollow-dot readout of a rule-based level (low/medium/high),
    for skimming the eyebrow line without reading the word. Bullet (U+2022)
    and middle dot (U+00B7) are both in WinAnsi/Latin-1, so they render in
    every font this module uses - no risk of a missing glyph."""
    rank = _LEVEL_RANK.get(level or "")
    if rank is None:
        return None
    return "•" * rank + "·" * (3 - rank)


@dataclass
class RelatedPaper:
    paper_id: str
    title: str
    link: str | None


def _source_link(url: str | None, doi: str | None) -> str | None:
    if url:
        return url
    if doi:
        return f"https://doi.org/{doi}"
    return None


def _related_papers(assessment: ResearchAssessmentOut) -> list[RelatedPaper]:
    """Every distinct paper cited as evidence, in first-seen order - this is
    the report's reference list (rendered at the end), and also where each
    evidence quote's citation looks up a link to hyperlink to."""
    seen: dict[str, RelatedPaper] = {}
    for item in assessment.evidence:
        pid = str(item.paper_id)
        if pid not in seen:
            seen[pid] = RelatedPaper(
                paper_id=pid, title=item.paper_title, link=_source_link(item.paper_url, item.paper_doi)
            )
    return list(seen.values())


def _docx_rgb(hex_color: str):
    from docx.shared import RGBColor

    return RGBColor.from_string(hex_color.lstrip("#"))


def _docx_bottom_border(paragraph, hex_color: str) -> None:
    """python-docx has no paragraph-border API, so this drops down to the
    underlying XML - the same recipe used across the python-docx ecosystem
    for a bare horizontal rule (there's no built-in flowable for one)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "8")
    bottom.set(qn("w:color"), hex_color.lstrip("#"))
    borders.append(bottom)
    p_pr.append(borders)


def _docx_eyebrow(document, text: str, *, color: str = INK_FAINT, heading: bool = False, dots: str | None = None):
    """A small uppercase label, standing in for the web report's `.eyebrow`
    class - Space Grotesk is referenced by name only (Word substitutes if a
    reader doesn't have it installed), unlike the PDF path which embeds it.

    heading=True gives the paragraph Word's built-in "Heading 2" style on
    top of the same direct run formatting - direct formatting always wins
    visually, but the style is what makes the paragraph show up in Word's
    Navigation pane, which is the report's table of contents."""
    from docx.shared import Pt

    p = document.add_paragraph(style="Heading 2" if heading else None)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text.upper())
    run.font.name = "Space Grotesk"
    run.font.size = Pt(8.5)
    run.font.bold = True
    run.font.color.rgb = _docx_rgb(color)
    if dots:
        dots_run = p.add_run("  " + dots)
        dots_run.font.name = "Source Serif 4"
        dots_run.font.size = Pt(9)
        dots_run.font.bold = False
        dots_run.font.color.rgb = _docx_rgb(INK_SOFT)
    return p


def _docx_hyperlink(paragraph, text: str, url: str, *, color: str = INK_SOFT, size_pt: float | None = None) -> None:
    """python-docx has no hyperlink API, so this drops to the underlying
    XML - the standard recipe for a `w:hyperlink` run pointing at an
    external relationship (there's no flowable for one, same situation as
    _docx_bottom_border above)."""
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    r_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    run_pr = OxmlElement("w:rPr")
    color_el = OxmlElement("w:color")
    color_el.set(qn("w:val"), color.lstrip("#"))
    run_pr.append(color_el)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    run_pr.append(underline)
    if size_pt is not None:
        size_el = OxmlElement("w:sz")
        size_el.set(qn("w:val"), str(int(size_pt * 2)))
        run_pr.append(size_el)
    run.append(run_pr)

    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _docx_footer(document, assessment: ResearchAssessmentOut, generated_at: datetime) -> None:
    from docx.shared import Pt

    section = document.sections[0]
    section.footer.is_linked_to_previous = False
    footer_p = section.footer.paragraphs[0]
    run = footer_p.add_run(
        f"Assessment {str(assessment.id)[:8]}  ·  Exported {generated_at:%Y-%m-%d %H:%M UTC}"
    )
    run.font.name = "Source Serif 4"
    run.font.size = Pt(8)
    run.font.color.rgb = _docx_rgb(INK_FAINT)


def build_docx(assessment: ResearchAssessmentOut) -> bytes:
    import docx
    from docx.shared import Pt

    document = docx.Document()

    normal = document.styles["Normal"]
    normal.font.name = "Source Serif 4"
    normal.font.size = Pt(11)
    normal.font.color.rgb = _docx_rgb(INK)

    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(2)
    title_run = title.add_run("Research Assessment")
    title_run.font.name = "Space Grotesk"
    title_run.font.size = Pt(15)
    title_run.font.bold = True
    title_run.font.color.rgb = _docx_rgb(INK_FAINT)

    _docx_eyebrow(document, "recommendation")
    headline = document.add_paragraph()
    headline_run = headline.add_run(assessment.recommendation or "Not assessed")
    headline_run.font.name = "Space Grotesk"
    headline_run.font.size = Pt(24)
    headline_run.font.bold = True
    headline_run.font.color.rgb = _docx_rgb(INK)

    meta = document.add_paragraph()
    meta.paragraph_format.space_after = Pt(10)
    meta_run = meta.add_run(
        f"confidence: {assessment.confidence or '—'}   ·   "
        f"human reviewed: {'yes' if assessment.human_reviewed else 'no'}"
    )
    meta_run.font.size = Pt(9.5)
    meta_run.font.color.rgb = _docx_rgb(INK_SOFT)
    _docx_bottom_border(meta, RULE)

    _docx_eyebrow(document, "input")
    document.add_paragraph(assessment.research_input.raw_text)
    input_type = document.add_paragraph()
    input_type_run = input_type.add_run(f"Type: {assessment.research_input.input_type}")
    input_type_run.font.size = Pt(9.5)
    input_type_run.font.color.rgb = _docx_rgb(INK_FAINT)

    related = _related_papers(assessment)
    link_by_paper_id = {paper.paper_id: paper.link for paper in related}

    for section in build_report_sections(assessment):
        label = section.label
        dots = _level_dots(section.level)
        if section.level:
            label += f"  ·  {section.level.replace('_', ' ')}"
        _docx_eyebrow(document, label, heading=True, dots=dots)

        if section.body:
            document.add_paragraph(section.body)
        elif section.unassessed_reason:
            reason = document.add_paragraph()
            reason_run = reason.add_run(section.unassessed_reason)
            reason_run.italic = True
            reason_run.font.color.rgb = _docx_rgb(INK_FAINT)

        for item in section.evidence:
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Pt(14)
            quote_run = p.add_run(f"“{item.text}”  — ")
            quote_run.italic = True
            quote_run.font.color.rgb = _docx_rgb(INK_SOFT)
            link = link_by_paper_id.get(str(item.paper_id))
            if link:
                _docx_hyperlink(p, item.paper_title, link, size_pt=9)
            else:
                source_run = p.add_run(item.paper_title)
                source_run.font.size = Pt(9)
                source_run.font.color.rgb = _docx_rgb(INK_FAINT)
            if item.section:
                section_run = p.add_run(f" ({item.section})")
                section_run.font.size = Pt(9)
                section_run.font.color.rgb = _docx_rgb(INK_FAINT)

    if related:
        _docx_eyebrow(document, "references", heading=True)
        for i, paper in enumerate(related, start=1):
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Pt(14)
            number_run = p.add_run(f"{i}. ")
            number_run.font.size = Pt(9.5)
            number_run.font.color.rgb = _docx_rgb(INK_FAINT)
            if paper.link:
                _docx_hyperlink(p, paper.title, paper.link, color=INK, size_pt=9.5)
            else:
                title_run = p.add_run(paper.title)
                title_run.font.size = Pt(9.5)
                title_run.font.color.rgb = _docx_rgb(INK)

    _docx_footer(document, assessment, datetime.now(timezone.utc))

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


_PDF_FONTS_REGISTERED = False


def _register_pdf_fonts() -> None:
    """Registers the bundled static font instances with reportlab. Idempotent
    and cheap to call per-export - reportlab has no "is this font already
    registered" check of its own, so a module-level flag avoids re-reading
    the font files off disk on every call."""
    global _PDF_FONTS_REGISTERED
    if _PDF_FONTS_REGISTERED:
        return

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont("SpaceGrotesk-Bold", str(FONTS_DIR / "SpaceGrotesk-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSerif4", str(FONTS_DIR / "SourceSerif4-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSerif4-Italic", str(FONTS_DIR / "SourceSerif4-Italic.ttf")))
    _PDF_FONTS_REGISTERED = True


def _pdf_styles():
    from reportlab.lib.styles import ParagraphStyle

    return {
        "title": ParagraphStyle(
            "title", fontName="SpaceGrotesk-Bold", fontSize=13, textColor=INK_FAINT, spaceAfter=2
        ),
        "headline": ParagraphStyle(
            "headline", fontName="SpaceGrotesk-Bold", fontSize=26, textColor=INK, leading=30, spaceAfter=6
        ),
        "meta": ParagraphStyle("meta", fontName="SourceSerif4", fontSize=9.5, textColor=INK_SOFT),
        "eyebrow": ParagraphStyle(
            "eyebrow", fontName="SpaceGrotesk-Bold", fontSize=9, textColor=INK_FAINT, spaceBefore=16, spaceAfter=4
        ),
        # Same look as "eyebrow" - a distinct style name so afterFlowable
        # (see _AssessmentDocTemplate) can tell a report section heading
        # apart from a meta label like "Input" and only bookmark the former.
        "section": ParagraphStyle(
            "section", fontName="SpaceGrotesk-Bold", fontSize=9, textColor=INK_FAINT, spaceBefore=16, spaceAfter=4
        ),
        "body": ParagraphStyle("body", fontName="SourceSerif4", fontSize=10.5, textColor=INK, leading=15),
        "reason": ParagraphStyle(
            "reason", fontName="SourceSerif4-Italic", fontSize=10, textColor=INK_FAINT, leading=14
        ),
        "quote": ParagraphStyle(
            "quote",
            fontName="SourceSerif4-Italic",
            fontSize=9.5,
            textColor=INK_SOFT,
            leading=13,
            leftIndent=12,
            spaceAfter=3,
        ),
        "quote_source": ParagraphStyle(
            "quote_source", fontName="SourceSerif4", fontSize=8, textColor=INK_FAINT, leftIndent=12, spaceAfter=8
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="SourceSerif4", fontSize=10.5, textColor=INK, leading=15, leftIndent=12, spaceAfter=2
        ),
    }


def _make_pdf_footer(assessment: ResearchAssessmentOut, generated_at: datetime):
    """A page-number-and-provenance footer. Built as a closure (rather than
    a plain onFirstPage/onLaterPages callback) because SimpleDocTemplate
    only ever passes it (canvas, doc) - the assessment id and export
    timestamp have to come from here instead."""

    def footer(canvas, doc) -> None:
        from reportlab.lib.colors import HexColor

        canvas.saveState()
        canvas.setStrokeColor(HexColor(RULE))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 0.6 * 72, doc.pagesize[0] - doc.rightMargin, 0.6 * 72)
        canvas.setFont("SourceSerif4", 8)
        canvas.setFillColor(HexColor(INK_FAINT))
        canvas.drawString(
            doc.leftMargin,
            0.4 * 72,
            f"Assessment {str(assessment.id)[:8]}  ·  Exported {generated_at:%Y-%m-%d %H:%M UTC}",
        )
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.4 * 72, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    return footer


class _AssessmentDocTemplate:
    """Mixed in over SimpleDocTemplate (built lazily in build_pdf, since the
    class itself is only importable once reportlab is) to add a PDF outline
    (bookmarks) entry for each report section - the same "read structure I
    already lay out" idea as the docx Heading 2 style in _docx_eyebrow."""

    _bookmark_counter = 0

    def afterFlowable(self, flowable) -> None:
        style = getattr(flowable, "style", None)
        if style is not None and style.name == "section":
            key = f"section-{self._bookmark_counter}"
            self._bookmark_counter += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)


def _pdf_link(text: str, url: str | None, *, color: str) -> str:
    """reportlab's `<link>` tag (not HTML's `<a>`) is the documented way to
    make part of a Paragraph clickable - it's the one markup tag guaranteed
    to accept a `color` attribute across reportlab versions."""
    escaped = _escape(text)
    if not url:
        return escaped
    return f'<link href="{_escape(url)}" color="{color}">{escaped}</link>'


def build_pdf(assessment: ResearchAssessmentOut) -> bytes:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate
    from reportlab.platypus.flowables import HRFlowable

    class _DocTemplate(_AssessmentDocTemplate, SimpleDocTemplate):
        pass

    _register_pdf_fonts()
    styles = _pdf_styles()
    story = []

    def para(text: str, style: str, *, markup: str | None = None) -> None:
        story.append(Paragraph(markup if markup is not None else _escape(text), styles[style]))

    def rule() -> None:
        story.append(HRFlowable(width="100%", thickness=0.75, color=HexColor(RULE), spaceBefore=6, spaceAfter=14))

    para("RESEARCH ASSESSMENT", "title")
    para(assessment.recommendation or "Not assessed", "headline")
    para(
        f"confidence: {assessment.confidence or '—'}"
        f"   ·   human reviewed: {'yes' if assessment.human_reviewed else 'no'}",
        "meta",
    )
    rule()

    para("Input", "eyebrow")
    para(assessment.research_input.raw_text, "body")
    para(f"Type: {assessment.research_input.input_type}", "meta")

    if assessment.research_gap_text:
        para("Gap", "eyebrow")
        para(assessment.research_gap_text.splitlines()[0], "body")

    story.append(PageBreak())

    related = _related_papers(assessment)
    link_by_paper_id = {paper.paper_id: paper.link for paper in related}

    for section in build_report_sections(assessment):
        heading = _escape(section.label)
        if section.level:
            heading += f"  ·  {section.level.replace('_', ' ')}"
        dots = _level_dots(section.level)
        if dots:
            heading += f'  <font name="SourceSerif4" size="9" color="{INK_SOFT}">{dots}</font>'
        para(section.label, "section", markup=heading)

        if section.body:
            para(section.body, "body")
        elif section.unassessed_reason:
            para(section.unassessed_reason, "reason")

        for item in section.evidence:
            suffix = f" ({_escape(item.section)})" if item.section else ""
            para(f"“{item.text}”", "quote")
            link = link_by_paper_id.get(str(item.paper_id))
            source_markup = f"— {_pdf_link(item.paper_title, link, color=INK_FAINT)}{suffix}"
            para(item.paper_title, "quote_source", markup=source_markup)

    if related:
        para("References", "section")
        for i, paper in enumerate(related, start=1):
            markup = f"{i}. {_pdf_link(paper.title, paper.link, color=INK)}"
            para(paper.title, "bullet", markup=markup)

    buffer = io.BytesIO()
    doc = _DocTemplate(
        buffer, pagesize=LETTER, topMargin=0.9 * 72, bottomMargin=0.9 * 72, leftMargin=0.9 * 72, rightMargin=0.9 * 72
    )
    generated_at = datetime.now(timezone.utc)
    footer = _make_pdf_footer(assessment, generated_at)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def _escape(text: str) -> str:
    """reportlab's Paragraph interprets its text as a small XML/markup dialect."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
