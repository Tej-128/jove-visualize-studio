from __future__ import annotations

import re
from pathlib import Path

from .models import CaptionCue
from .normalizer import clean_caption_text
from .utils import timestamp_to_ms


TIMING_RE = re.compile(
    r"^(?P<start>(?:\d{1,3}:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{1,3}:)?\d{2}:\d{2}[.,]\d{3})(?:\s+.*)?$"
)


class VTTParseError(ValueError):
    pass


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise VTTParseError(f"Unable to decode VTT file: {path}")


def parse_vtt(path: str | Path) -> list[CaptionCue]:
    path = Path(path)
    text = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    cues: list[CaptionCue] = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "REGION", "X-TIMESTAMP-MAP")):
            index += 1
            if line.startswith(("NOTE", "STYLE", "REGION")):
                while index < len(lines) and lines[index].strip():
                    index += 1
            continue

        timing_match = TIMING_RE.match(line)
        if not timing_match and index + 1 < len(lines):
            timing_match = TIMING_RE.match(lines[index + 1].strip())
            if timing_match:
                index += 1

        if not timing_match:
            index += 1
            continue

        start_ms = timestamp_to_ms(timing_match.group("start"))
        end_ms = timestamp_to_ms(timing_match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        cue_text = clean_caption_text(" ".join(text_lines))
        if not cue_text:
            continue
        if end_ms <= start_ms:
            continue
        if cues and cue_text == cues[-1].text and start_ms <= cues[-1].end_ms + 500:
            cues[-1].end_ms = max(cues[-1].end_ms, end_ms)
            continue
        cues.append(CaptionCue(start_ms=start_ms, end_ms=end_ms, text=cue_text))

    if not cues:
        raise VTTParseError(f"No valid caption cues found in {path.name}")
    return cues

