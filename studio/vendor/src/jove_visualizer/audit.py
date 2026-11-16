from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .metadata import load_metadata
from .models import AuditSummary
from .utils import canonical_id
from .vtt_parser import VTTParseError, parse_vtt

ProgressCallback = Callable[[int, int, str], None]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_corpus(
    metadata_csv: str | Path,
    vtt_folder: str | Path,
    output_dir: str | Path,
    validate_content: bool = True,
    progress: ProgressCallback | None = None,
) -> AuditSummary:
    metadata_csv = Path(metadata_csv)
    vtt_folder = Path(vtt_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, metadata_frame = load_metadata(metadata_csv)
    files = sorted(vtt_folder.glob("*.vtt"), key=lambda path: path.name.lower())
    file_rows: list[dict] = []
    id_to_files: dict[str, list[Path]] = defaultdict(list)
    malformed: list[dict] = []
    empty: list[dict] = []
    nonstandard: list[dict] = []

    for position, path in enumerate(files, start=1):
        exact_numeric = path.stem.isdigit()
        article_id = canonical_id(path.stem)
        id_to_files[article_id].append(path)
        status = "OK"
        cue_count = ""
        if path.stat().st_size == 0:
            status = "EMPTY"
            empty.append({"article_id": article_id, "filename": path.name, "reason": "Zero-byte file"})
        elif validate_content:
            try:
                cues = parse_vtt(path)
                cue_count = len(cues)
            except VTTParseError as exc:
                status = "MALFORMED"
                malformed.append({"article_id": article_id, "filename": path.name, "reason": str(exc)})
        if not exact_numeric:
            nonstandard.append({"article_id": article_id, "filename": path.name, "reason": "Filename stem is not numeric"})
        file_rows.append(
            {
                "article_id": article_id,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "cue_count": cue_count,
                "status": status,
                "metadata_found": "YES" if article_id in metadata else "NO",
            }
        )
        if progress:
            progress(position, len(files), f"Auditing {path.name}")

    metadata_ids = set(metadata)
    vtt_ids = set(id_to_files)
    metadata_without_vtt = [
        {
            "article_id": article_id,
            "article_title": metadata[article_id].article_title,
            "sections": " | ".join(metadata[article_id].sections),
        }
        for article_id in sorted(metadata_ids - vtt_ids, key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    ]
    vtt_without_metadata = [
        {"article_id": article_id, "filenames": " | ".join(path.name for path in id_to_files[article_id])}
        for article_id in sorted(vtt_ids - metadata_ids, key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    ]
    duplicate_vtts = [
        {"article_id": article_id, "file_count": len(paths), "filenames": " | ".join(path.name for path in paths)}
        for article_id, paths in id_to_files.items()
        if len(paths) > 1
    ]
    multi_section = [
        {
            "article_id": article_id,
            "article_title": item.article_title,
            "sections": " | ".join(item.sections),
            "section_count": len(item.sections),
        }
        for article_id, item in metadata.items()
        if len(item.sections) > 1
    ]

    _write_csv(output_dir / "vtt_inventory.csv", file_rows, ["article_id", "filename", "size_bytes", "cue_count", "status", "metadata_found"])
    _write_csv(output_dir / "metadata_without_vtt.csv", metadata_without_vtt, ["article_id", "article_title", "sections"])
    _write_csv(output_dir / "vtt_without_metadata.csv", vtt_without_metadata, ["article_id", "filenames"])
    _write_csv(output_dir / "duplicate_vtt_ids.csv", duplicate_vtts, ["article_id", "file_count", "filenames"])
    _write_csv(output_dir / "malformed_vtts.csv", malformed, ["article_id", "filename", "reason"])
    _write_csv(output_dir / "empty_vtts.csv", empty, ["article_id", "filename", "reason"])
    _write_csv(output_dir / "nonstandard_filenames.csv", nonstandard, ["article_id", "filename", "reason"])
    _write_csv(output_dir / "multi_section_ids.csv", multi_section, ["article_id", "article_title", "sections", "section_count"])

    summary_rows = [
        {"metric": "Metadata rows", "value": len(metadata_frame)},
        {"metric": "Metadata unique IDs", "value": len(metadata)},
        {"metric": "VTT files", "value": len(files)},
        {"metric": "VTT unique IDs", "value": len(vtt_ids)},
        {"metric": "Matched IDs", "value": len(metadata_ids & vtt_ids)},
        {"metric": "Metadata IDs without VTT", "value": len(metadata_without_vtt)},
        {"metric": "VTT IDs without metadata", "value": len(vtt_without_metadata)},
        {"metric": "Duplicate VTT IDs", "value": len(duplicate_vtts)},
        {"metric": "Malformed VTTs", "value": len(malformed)},
        {"metric": "Empty VTTs", "value": len(empty)},
        {"metric": "Nonstandard filenames", "value": len(nonstandard)},
        {"metric": "IDs in multiple Sections", "value": len(multi_section)},
    ]
    _write_csv(output_dir / "audit_summary.csv", summary_rows, ["metric", "value"])

    return AuditSummary(
        metadata_rows=len(metadata_frame),
        metadata_unique_ids=len(metadata),
        vtt_files=len(files),
        vtt_unique_ids=len(vtt_ids),
        matched_ids=len(metadata_ids & vtt_ids),
        metadata_without_vtt=len(metadata_without_vtt),
        vtt_without_metadata=len(vtt_without_metadata),
        duplicate_vtt_ids=len(duplicate_vtts),
        malformed_vtts=len(malformed),
        empty_vtts=len(empty),
        nonstandard_filenames=len(nonstandard),
        output_dir=output_dir,
    )

