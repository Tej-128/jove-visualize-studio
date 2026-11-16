from __future__ import annotations

import html
import re
import unicodedata


# Generic orthographic/unit normalization only.
# IMPORTANT: Do not add assay-, organism-, reagent-, protocol-, or Journal-specific aliases here.
UNIT_REPLACEMENTS = [
    (r"\bmicrograms?\s+per\s+millilit(?:er|re)\b", "ug/ml"),
    (r"\bmilligrams?\s+per\s+millilit(?:er|re)\b", "mg/ml"),
    (r"\bnanograms?\s+per\s+millilit(?:er|re)\b", "ng/ml"),
    (r"\bnanograms?\s+per\s+microlit(?:er|re)\b", "ng/ul"),
    (r"\bmicrograms?\s+per\s+microlit(?:er|re)\b", "ug/ul"),
    (r"\bmicrolit(?:er|re)s?\b", "ul"),
    (r"\bmillilit(?:er|re)s?\b", "ml"),
    (r"\bmicromolars?\b", "um"),
    (r"\bnanomolars?\b", "nm"),
    (r"\bmillimolars?\b", "mm"),
    (r"\bmolars?\b", "m"),
    (r"\bdegrees?\s*c(?:elsius)?\b", "celsius"),
    (r"\bdegrees?\s*f(?:ahrenheit)?\b", "fahrenheit"),
    (r"\brevolutions?\s+per\s+minute\b", "rpm"),
]


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]{0,80}\]", " ", text)
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\bph\s*([0-9]+(?:\.[0-9]+)?)", r"ph \1", text)
    for pattern, replacement in UNIT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    # Generic presentation normalization. No domain-specific synonym expansion.
    text = re.sub(r"\broom\s+temperature\b", "room-temperature", text)
    text = re.sub(r"\b([0-9]{1,4})\s*[- ]?\s*well\b", r"\1-well", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"[^a-z0-9%+./-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_caption_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\\[^}]+\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

