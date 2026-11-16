from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .models import ArticleMetadata
from .utils import canonical_id


COLUMN_ALIASES = {
    "article_id": {"article_id", "id", "lesson_id", "video_id", "journal_id"},
    "article_title": {"article_title", "title", "lesson_name", "video_name", "article_name"},
    "journal_section": {"journal_section", "section", "subject", "subject_section"},
    "article_url": {"article_url", "url", "video_url", "lesson_url"},
}


def _resolve_columns(columns: list[str]) -> dict[str, str]:
    normalized = {column.strip().lower(): column for column in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                resolved[canonical] = normalized[alias]
                break
    required = {"article_id", "article_title", "journal_section"}
    missing = required - resolved.keys()
    if missing:
        raise ValueError(
            "Metadata CSV is missing required columns: " + ", ".join(sorted(missing))
        )
    return resolved


def load_metadata(path: str | Path) -> tuple[dict[str, ArticleMetadata], pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    resolved = _resolve_columns(list(frame.columns))

    clean = pd.DataFrame(
        {
            "article_id": frame[resolved["article_id"]].map(canonical_id),
            "article_title": frame[resolved["article_title"]].map(lambda value: html.unescape(str(value).strip())),
            "journal_section": frame[resolved["journal_section"]].map(lambda value: str(value).strip()),
            "article_url": (
                frame[resolved["article_url"]].map(lambda value: str(value).strip())
                if "article_url" in resolved
                else ""
            ),
        }
    )
    if (clean["article_id"] == "").any():
        raise ValueError("Metadata contains one or more blank article IDs.")

    metadata: dict[str, ArticleMetadata] = {}
    for article_id, group in clean.groupby("article_id", sort=False):
        titles = [value for value in group["article_title"].unique().tolist() if value]
        sections = sorted({value for value in group["journal_section"].tolist() if value})
        urls = [value for value in group["article_url"].unique().tolist() if value]
        metadata[article_id] = ArticleMetadata(
            article_id=article_id,
            article_title=titles[0] if titles else f"Journal ID {article_id}",
            sections=sections or ["Unclassified"],
            article_url=urls[0] if urls else None,
        )
    return metadata, clean

