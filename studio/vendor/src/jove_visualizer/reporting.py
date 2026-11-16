from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

from .models import Candidate, MatchResult
from .utils import ms_to_timestamp


PALETTE = [
    "D9EAF7", "FCE4D6", "E2F0D9", "FFF2CC", "E4DFEC", "DDEBF7", "F4CCCC", "D9D2E9",
    "CFE2F3", "FCE5CD", "D0E0E3", "EAD1DC", "D9EAD3", "FFE599", "C9DAF8", "F9CB9C",
]


def _candidate_to_dict(candidate: Candidate | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "journal_id": candidate.article_id,
        "title": candidate.article_title,
        "sections": candidate.sections,
        "start_timestamp": ms_to_timestamp(candidate.start_ms),
        "end_timestamp": ms_to_timestamp(candidate.end_ms),
        "caption_passage": candidate.text,
        "confidence_score": round(candidate.final_score, 4),
        "ai_suitability": candidate.ai_suitability,
        "reason": candidate.ai_reason,
        "vtt_evidence_uncertainty": candidate.ai_visual_uncertainty,
        "vtt_evidence_risk": candidate.vtt_evidence_risk,
        "relay_semantic_score": round(candidate.relay_semantic_score, 4),
        "protocol_semantic_score": round(candidate.protocol_semantic_score, 4),
        "sequence_continuity_score": round(candidate.sequence_continuity_score, 4),
        "state_semantic_score": round(candidate.state_semantic_score, 4),
        "source_coverage_score": round(candidate.source_coverage_score, 4),
        "sequence_transition_score": round(candidate.sequence_transition_score, 4),
        "sequence_path_score": round(candidate.sequence_path_score, 4),
        "sequence_path_selected": candidate.sequence_path_selected,
        "sequence_reason": candidate.sequence_reason,
        "constraint_match_score": round(candidate.constraint_match_score, 4),
        "learned_ranker_score": round(candidate.learned_ranker_score, 4),
        "reviewer_memory": {
            "label": candidate.reviewer_memory_label,
            "similarity": round(candidate.reviewer_memory_similarity, 4),
            "note": candidate.reviewer_memory_note,
        },
        "contradictions": candidate.contradictions,
        "source_file": candidate.source_file,
    }


def results_to_json(results: list[MatchResult], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        payload.append({
            "step_number": result.step.number,
            "script_title": result.step.script_title,
            "script_step": result.step.text,
            "production_notes": result.step.production_notes,
            "protocol_phase": result.step.protocol_phase,
            "core_action": result.step.core_action,
            "core_entities": result.step.core_entities,
            "state_before": result.step.state_before,
            "state_after": result.step.state_after,
            "state_transition": result.step.state_transition,
            "status": result.status,
            "confidence": result.confidence,
            "best_match": _candidate_to_dict(result.best),
            "additional_timestamps_same_id": [
                _candidate_to_dict(candidate) for candidate in result.additional_timestamps_same_id
            ],
            "alternatives": [_candidate_to_dict(candidate) for candidate in result.alternatives],
            "review_note": result.no_match_reason,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
        })
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def results_to_excel(results: list[MatchResult], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Visualizer Matches"
    sheet.sheet_view.showGridLines = False

    # Column N intentionally remains the reviewer-feedback column for the team's workflow.
    headers = [
        "Step", "Script Step", "Result Type", "Status", "Confidence", "Journal ID", "Journal Title",
        "Subject / Section", "Start", "End", "Matching CC Text", "Score",
        "Contradictions / Review Note", "Feedback Notes", "AI Suitability", "AI Reason",
        "Learned Ranker Score", "Reviewer Memory Label", "Reviewer Memory Similarity",
        "VTT Evidence Risk", "Elapsed Seconds", "Relay Score", "Protocol Score",
        "Source Continuity", "Constraint Match", "State Alignment", "Source Coverage",
        "Global Sequence Selected", "Global Sequence Score",
    ]
    sheet.append(headers)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 38
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    id_colors: dict[str, str] = {}
    thin_gray = Side(style="thin", color="D9E1F2")

    def append_candidate(result: MatchResult, label: str, candidate: Candidate | None, note: str = "") -> None:
        if candidate is None:
            sheet.append([
                result.step.number, result.step.text, label, result.status, result.confidence,
                "", "", "", "", "", "", "", note or result.no_match_reason or "No reliable match found",
                "", "", "", "", "", "", "Not applicable", round(result.elapsed_seconds, 3),
                "", "", "", "", "", "", "", "",
            ])
            return
        color = id_colors.setdefault(candidate.article_id, PALETTE[len(id_colors) % len(PALETTE)])
        review_notes = list(candidate.contradictions)
        if note:
            review_notes.append(note)
        row = [
            result.step.number,
            result.step.text,
            label,
            result.status,
            result.confidence,
            candidate.article_id,
            candidate.article_title,
            " | ".join(candidate.sections),
            ms_to_timestamp(candidate.start_ms),
            ms_to_timestamp(candidate.end_ms),
            candidate.text,
            round(candidate.final_score, 3),
            " | ".join(dict.fromkeys(item for item in review_notes if item)),
            "",
            candidate.ai_suitability or "Not AI-reviewed",
            candidate.ai_reason or "",
            round(candidate.learned_ranker_score, 3),
            candidate.reviewer_memory_label or "",
            round(candidate.reviewer_memory_similarity, 3),
            candidate.vtt_evidence_risk.title(),
            round(result.elapsed_seconds, 3),
            round(candidate.relay_semantic_score, 3),
            round(candidate.protocol_semantic_score, 3),
            round(candidate.sequence_continuity_score, 3),
            round(candidate.constraint_match_score, 3),
            round(candidate.state_semantic_score, 3),
            round(candidate.source_coverage_score, 3),
            "Yes" if candidate.sequence_path_selected else "",
            round(candidate.sequence_path_score, 3),
        ]
        sheet.append(row)
        current_row = sheet.max_row
        for col in (6, 9, 10):
            sheet.cell(current_row, col).fill = PatternFill("solid", fgColor=color)
        if candidate.ai_suitability == "preferred":
            sheet.cell(current_row, 15).fill = PatternFill("solid", fgColor="C6E0B4")
        elif candidate.ai_suitability == "acceptable":
            sheet.cell(current_row, 15).fill = PatternFill("solid", fgColor="FFF2CC")
        elif candidate.ai_suitability == "rejected":
            sheet.cell(current_row, 15).fill = PatternFill("solid", fgColor="F4CCCC")

    for result in results:
        if result.best:
            append_candidate(result, "Best match", result.best, result.no_match_reason or "")
            for candidate in result.additional_timestamps_same_id:
                append_candidate(result, "Additional timestamp (same ID)", candidate)
            for candidate in result.alternatives:
                append_candidate(result, "Alternative", candidate)
        else:
            append_candidate(result, "No reliable match", None)
            for candidate in result.alternatives:
                append_candidate(result, "Review candidate", candidate)

    widths = {
        1: 8, 2: 48, 3: 26, 4: 16, 5: 14, 6: 13, 7: 42, 8: 28,
        9: 14, 10: 14, 11: 65, 12: 10, 13: 48, 14: 36, 15: 16,
        16: 48, 17: 18, 18: 20, 19: 20, 20: 18, 21: 16,
        22: 14, 23: 14, 24: 16, 25: 16, 26: 15, 27: 15, 28: 20, 29: 18,
    }
    for col_index, width in widths.items():
        sheet.column_dimensions[get_column_letter(col_index)].width = width

    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=thin_gray)
        for index in (1, 4, 5, 6, 9, 10, 12, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29):
            row[index - 1].alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)

    workbook.save(output_path)
    return output_path


def _set_cell_shading(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_run_shading(run, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    r_pr = run._r.get_or_add_rPr()
    shd = r_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        r_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _add_match_line(document: Document, label: str, candidate: Candidate, color: str, confidence: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(3)
    prefix = paragraph.add_run(f"{label}: ")
    prefix.bold = True
    text = (
        f"Journal ID {candidate.article_id} | "
        f"{ms_to_timestamp(candidate.start_ms)}–{ms_to_timestamp(candidate.end_ms)} | "
        f"{confidence} | Score {candidate.final_score:.3f}"
    )
    run = paragraph.add_run(text)
    run.bold = label == "Best match"
    _set_run_shading(run, color)

    details = document.add_paragraph()
    details.paragraph_format.left_indent = Inches(0.18)
    details.paragraph_format.space_after = Pt(3)
    details.add_run(candidate.article_title).italic = True
    details.add_run(f" | Section: {' | '.join(candidate.sections)}")
    if label == "Best match" and candidate.ai_reason:
        details.add_run(f"\nSelection note: {candidate.ai_reason}")
    if label == "Best match" and candidate.sequence_reason:
        details.add_run(f"\nSequence note: {candidate.sequence_reason}")
    if candidate.ai_visual_uncertainty:
        warning = details.add_run("\nVisual check required: caption text cannot confirm the exact footage.")
        warning.bold = True
        warning.font.color.rgb = RGBColor(156, 87, 0)


def results_to_docx(
    results: list[MatchResult],
    output_path: str | Path,
    title: str = "JoVE Visualizer Match Report",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = document.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.5)
    styles["Title"].font.name = "Arial"
    styles["Title"].font.size = Pt(20)
    styles["Heading 1"].font.name = "Arial"
    styles["Heading 1"].font.size = Pt(13)

    script_title = next((result.step.script_title for result in results if result.step.script_title), title)
    document.add_heading(script_title or title, 0)
    subtitle = document.add_paragraph("JoVE Visualizer — V3 Context Relay (VTT-only) match output")
    subtitle.runs[0].italic = True
    subtitle.runs[0].font.color.rgb = RGBColor(89, 89, 89)

    summary_text = (
        f"Steps: {len(results)} | High: {sum(r.confidence == 'High' for r in results)} | "
        f"Medium: {sum(r.confidence == 'Medium' for r in results)} | "
        f"Low: {sum(r.confidence == 'Low' for r in results)} | "
        f"No reliable match: {sum(r.status == 'NO_MATCH' for r in results)}"
    )
    summary = document.add_paragraph(summary_text)
    summary.runs[0].bold = True

    note = document.add_paragraph(
        "Timestamp colours are consistent by Journal ID. Matching uses VTT captions only; no video-frame or footage inspection is performed."
    )
    note.runs[0].font.size = Pt(9)
    note.runs[0].font.color.rgb = RGBColor(89, 89, 89)

    id_colors: dict[str, str] = {}
    references: dict[str, Candidate] = {}

    for result in results:
        heading = document.add_paragraph()
        heading.style = document.styles["Heading 1"]
        heading.paragraph_format.keep_with_next = True
        heading.add_run(f"{result.step.number}. {result.step.text}")

        if result.step.production_notes:
            production = document.add_paragraph()
            production.paragraph_format.left_indent = Inches(0.18)
            production.add_run("Production notes: ").bold = True
            production.add_run(" | ".join(result.step.production_notes))

        if not result.best:
            no_match = document.add_paragraph()
            run = no_match.add_run("No reliable match found")
            run.bold = True
            run.font.highlight_color = WD_COLOR_INDEX.RED
            if result.no_match_reason:
                reason = document.add_paragraph(result.no_match_reason)
                reason.paragraph_format.left_indent = Inches(0.18)
            if result.alternatives:
                review = document.add_paragraph("Closest review candidates (not approved as matches):")
                review.runs[0].italic = True
                for candidate in result.alternatives:
                    color = id_colors.setdefault(candidate.article_id, PALETTE[len(id_colors) % len(PALETTE)])
                    references.setdefault(candidate.article_id, candidate)
                    _add_match_line(document, "Review candidate", candidate, color, "Review only")
        else:
            candidates = [("Best match", result.best)]
            candidates.extend(("Additional timestamp", item) for item in result.additional_timestamps_same_id)
            candidates.extend(("Alternative", item) for item in result.alternatives)
            for label, candidate in candidates:
                color = id_colors.setdefault(candidate.article_id, PALETTE[len(id_colors) % len(PALETTE)])
                references.setdefault(candidate.article_id, candidate)
                _add_match_line(document, label, candidate, color, result.confidence if label == "Best match" else "Alternative")

        if result.no_match_reason and result.best:
            review = document.add_paragraph()
            review.paragraph_format.left_indent = Inches(0.18)
            review.add_run("Review note: ").bold = True
            review.add_run(result.no_match_reason)

        divider = document.add_paragraph()
        divider.paragraph_format.space_after = Pt(4)
        divider.add_run("—" * 42).font.color.rgb = RGBColor(191, 191, 191)

    document.add_page_break()
    document.add_heading("Journal reference list", level=1)
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.autofit = False
    widths = [Inches(1.0), Inches(3.6), Inches(1.6), Inches(1.3)]
    for index, width in enumerate(widths):
        table.columns[index].width = width
    header = table.rows[0].cells
    for index, text in enumerate(["Journal ID", "Title", "Section", "Link"]):
        header[index].text = text
        _set_cell_shading(header[index], "1F4E78")
        header[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in header[index].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.bold = True

    for article_id, candidate in references.items():
        cells = table.add_row().cells
        cells[0].text = article_id
        cells[1].text = candidate.article_title
        cells[2].text = " | ".join(candidate.sections)
        cells[3].text = f"https://www.jove.com/v/{article_id}"
        _set_cell_shading(cells[0], id_colors[article_id])
        for cell in cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    document.save(output_path)
    return output_path

