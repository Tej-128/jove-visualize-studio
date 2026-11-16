from __future__ import annotations

import io
import re
from pathlib import Path

from .context_engine import build_protocol_context
from .models import ScriptStep


NUMBERED_RE = re.compile(r"^\s*(\d{1,3})\s*[.)-]\s*(.+)$")
SUBSTEP_RE = re.compile(r"^\s*(\d{1,3})\.(\d{1,2})\s*[:.)-]?\s*(.+)$")
URL_RE = re.compile(r"https?://", re.IGNORECASE)
REFERENCE_PREFIXES = (
    "reference paper", "manuscript link", "jove articles", "jove reference",
    "reference videos", "original paper", "references", "reference no.",
)
SCIENTIFIC_NOTE_PREFIXES = (
    "add onscreen txt:", "add on-screen txt:", "onscreen txt:", "on-screen txt:", "txt:",
)
PRODUCTION_NOTE_PREFIXES = (
    "video editor:", "show title", "modification needed", "pause frame",
    "show for ", "talent ",
)
PRODUCTION_PAREN_RE = re.compile(
    r"\((?:[^()]*(?:pronounce|read as|video editor|show title|modification needed|pause frame|show for)[^()]*)\)",
    re.IGNORECASE,
)


def _clean_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [re.sub(r"\s+", " ", line).strip() for line in normalized.split("\n") if line.strip()]


def _looks_like_reference(line: str) -> bool:
    lower = line.lower().strip()
    return lower.startswith(REFERENCE_PREFIXES) or bool(URL_RE.match(lower))


def _is_scientific_note(line: str) -> bool:
    return line.lower().strip().startswith(SCIENTIFIC_NOTE_PREFIXES)


def _strip_note_prefix(line: str) -> str:
    lower = line.lower().strip()
    for prefix in SCIENTIFIC_NOTE_PREFIXES:
        if lower.startswith(prefix):
            return line.strip()[len(prefix):].strip()
    return line.strip()


def _is_production_note(line: str) -> bool:
    lower = line.lower().strip()
    return lower.startswith(PRODUCTION_NOTE_PREFIXES)


def _split_production_parentheticals(text: str) -> tuple[str, list[str]]:
    notes = [match.group(0).strip("() ") for match in PRODUCTION_PAREN_RE.finditer(text)]
    cleaned = PRODUCTION_PAREN_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, notes


def _repair_standalone_notes(items: list[dict]) -> list[dict]:
    """Repair any note accidentally interpreted as its own numbered step.

    This is a defensive pass for Word documents where list formatting or copy/paste can
    make an on-screen/production note look numbered. It never changes genuine protocol
    steps unless the entire step begins with a known note marker.
    """
    repaired: list[dict] = []
    for item in items:
        joined = re.sub(r"\s+", " ", " ".join(item.get("parts", []))).strip()
        if repaired and _is_scientific_note(joined):
            detail = _strip_note_prefix(joined)
            if detail:
                repaired[-1]["parts"].append(f"On-screen scientific detail: {detail}")
            repaired[-1]["production_notes"].extend(item.get("production_notes", []))
            continue
        if repaired and _is_production_note(joined):
            repaired[-1]["production_notes"].append(joined)
            repaired[-1]["production_notes"].extend(item.get("production_notes", []))
            continue
        repaired.append(item)
    return repaired


def parse_steps_from_text(text: str) -> list[ScriptStep]:
    lines = _clean_lines(text)
    if not lines:
        return []

    title = ""
    first_step_index = next((i for i, line in enumerate(lines) if NUMBERED_RE.match(line)), None)
    if first_step_index is not None and first_step_index > 0:
        title = " ".join(lines[:first_step_index]).strip()
    elif first_step_index is None and len(lines) > 1:
        title = lines[0]

    numbered: list[dict] = []
    current: dict | None = None
    references_started = False

    source_lines = lines[first_step_index:] if first_step_index is not None else lines[1:]
    for line in source_lines:
        if _looks_like_reference(line):
            references_started = True
            continue
        if references_started:
            continue

        substep = SUBSTEP_RE.match(line)
        match = NUMBERED_RE.match(line) if not substep else None
        if match:
            if current is not None:
                numbered.append(current)
            step_text, parenthetical_notes = _split_production_parentheticals(match.group(2).strip())
            current = {
                "number": int(match.group(1)),
                "parts": [step_text] if step_text else [],
                "production_notes": parenthetical_notes,
            }
            continue

        if current is None:
            continue

        if substep:
            # Decimal substeps are supporting details for the parent step, not new search queries.
            current["production_notes"].append(f"{substep.group(1)}.{substep.group(2)} {substep.group(3).strip()}")
        elif _is_scientific_note(line):
            detail = _strip_note_prefix(line)
            if detail:
                current["parts"].append(f"On-screen scientific detail: {detail}")
        elif _is_production_note(line):
            current["production_notes"].append(line)
        else:
            cleaned, parenthetical_notes = _split_production_parentheticals(line)
            if cleaned:
                current["parts"].append(cleaned)
            current["production_notes"].extend(parenthetical_notes)

    if current is not None:
        numbered.append(current)

    if not numbered:
        # Fallback for unnumbered pasted text. Note/reference lines are never promoted to steps.
        candidate_lines = lines[1:] if title and len(lines) > 1 else lines
        numbered = []
        for line in candidate_lines:
            if _looks_like_reference(line):
                break
            if _is_scientific_note(line) and numbered:
                detail = _strip_note_prefix(line)
                if detail:
                    numbered[-1]["parts"].append(f"On-screen scientific detail: {detail}")
                continue
            if _is_production_note(line) and numbered:
                numbered[-1]["production_notes"].append(line)
                continue
            cleaned, parenthetical_notes = _split_production_parentheticals(line)
            if cleaned:
                numbered.append({
                    "number": len(numbered) + 1,
                    "parts": [cleaned],
                    "production_notes": parenthetical_notes,
                })

    numbered = _repair_standalone_notes(numbered)

    steps: list[ScriptStep] = []
    for index, item in enumerate(numbered):
        step_text = re.sub(r"\s+", " ", " ".join(item["parts"])).strip()
        if not step_text:
            continue
        previous_text = re.sub(r"\s+", " ", " ".join(numbered[index - 1]["parts"])).strip() if index > 0 else ""
        next_text = (
            re.sub(r"\s+", " ", " ".join(numbered[index + 1]["parts"])).strip()
            if index + 1 < len(numbered) else ""
        )
        steps.append(
            ScriptStep(
                number=item["number"],
                text=step_text,
                previous_text=previous_text,
                next_text=next_text,
                script_title=title,
                production_notes=list(dict.fromkeys(item["production_notes"])),
            )
        )

    # Ensure previous/next references are based on the repaired final step list.
    for index, step in enumerate(steps):
        step.previous_text = steps[index - 1].text if index > 0 else ""
        step.next_text = steps[index + 1].text if index + 1 < len(steps) else ""
    return build_protocol_context(steps)


def parse_steps_from_docx(content: bytes | str | Path) -> list[ScriptStep]:
    from docx import Document

    if isinstance(content, bytes):
        document = Document(io.BytesIO(content))
    else:
        document = Document(str(content))

    parts: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            # Preserve internal line breaks: on-screen notes are frequently stored inside
            # the same Word paragraph as the numbered step.
            parts.extend(line.strip() for line in paragraph.text.splitlines() if line.strip())
    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if values:
                parts.append(" | ".join(values))
    return parse_steps_from_text("\n".join(parts))

