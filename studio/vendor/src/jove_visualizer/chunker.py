from __future__ import annotations

from .models import ArticleMetadata, CaptionCue, Passage
from .normalizer import normalize_text


def build_passages(
    cues: list[CaptionCue],
    metadata: ArticleMetadata,
    source_file: str,
    target_words: int = 75,
    max_words: int = 120,
    max_seconds: int = 45,
    overlap_seconds: int = 12,
) -> list[Passage]:
    if not cues:
        return []

    passages: list[Passage] = []
    start_index = 0
    max_duration_ms = max_seconds * 1000
    overlap_ms = overlap_seconds * 1000

    while start_index < len(cues):
        end_index = start_index
        word_count = 0
        start_ms = cues[start_index].start_ms
        while end_index < len(cues):
            cue = cues[end_index]
            candidate_words = len(cue.text.split())
            duration = cue.end_ms - start_ms
            if end_index > start_index and (word_count + candidate_words > max_words or duration > max_duration_ms):
                break
            word_count += candidate_words
            end_index += 1
            if word_count >= target_words:
                break

        if end_index == start_index:
            end_index += 1

        selected = cues[start_index:end_index]
        text_parts: list[str] = []
        for cue in selected:
            if not text_parts or cue.text != text_parts[-1]:
                text_parts.append(cue.text)
        passage_text = " ".join(text_parts).strip()
        normalized = normalize_text(passage_text)
        if normalized:
            passages.append(
                Passage(
                    passage_id=None,
                    article_id=metadata.article_id,
                    article_title=metadata.article_title,
                    sections=metadata.sections,
                    start_ms=selected[0].start_ms,
                    end_ms=selected[-1].end_ms,
                    text=passage_text,
                    normalized_text=normalized,
                    source_file=source_file,
                )
            )

        if end_index >= len(cues):
            break
        next_start_time = max(cues[start_index].start_ms + 1, selected[-1].end_ms - overlap_ms)
        next_index = start_index + 1
        while next_index < end_index and cues[next_index].start_ms < next_start_time:
            next_index += 1
        if next_index <= start_index:
            next_index = start_index + 1
        start_index = next_index

    return passages

