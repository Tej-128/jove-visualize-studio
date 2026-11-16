from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .audit import audit_corpus
from .chunker import build_passages
from .config import Settings
from .embeddings import create_embedder
from .metadata import load_metadata
from .models import IndexManifest
from .utils import atomic_write_json, batched, canonical_id, sha256_file
from .vtt_parser import VTTParseError, parse_vtt

ProgressCallback = Callable[[str, int, int, str], None]
INDEX_VERSION = "1.0.0"


def _progress(callback: ProgressCallback | None, stage: str, current: int, total: int, message: str) -> None:
    if callback:
        callback(stage, current, total, message)


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY,
            article_title TEXT NOT NULL,
            sections_json TEXT NOT NULL,
            article_url TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE passages (
            passage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT NOT NULL,
            article_title TEXT NOT NULL,
            sections_json TEXT NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            source_file TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX idx_passages_article_id ON passages(article_id)")
    connection.execute(
        "CREATE VIRTUAL TABLE passage_fts USING fts5(normalized_text, content='passages', content_rowid='passage_id')"
    )
    return connection


def _select_vtt_files(vtt_folder: Path) -> list[Path]:
    by_id: dict[str, list[Path]] = {}
    for path in vtt_folder.glob("*.vtt"):
        by_id.setdefault(canonical_id(path.stem), []).append(path)
    selected: list[Path] = []
    for paths in by_id.values():
        # Prefer the largest file if duplicate files exist for the same ID.
        selected.append(max(paths, key=lambda item: (item.stat().st_size, item.name)))
    return sorted(selected, key=lambda path: (int(canonical_id(path.stem)) if canonical_id(path.stem).isdigit() else 10**18, path.name))


def build_index(
    metadata_csv: str | Path,
    vtt_folder: str | Path,
    index_dir: str | Path,
    settings: Settings,
    progress: ProgressCallback | None = None,
    run_audit: bool = True,
) -> IndexManifest:
    metadata_csv = Path(metadata_csv).resolve()
    vtt_folder = Path(vtt_folder).resolve()
    index_dir = Path(index_dir).resolve()

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not vtt_folder.is_dir():
        raise NotADirectoryError(f"VTT folder not found: {vtt_folder}")

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{index_dir.name}.building-", dir=index_dir.parent))
    audit_dir = temp_dir / "audit"

    try:
        if run_audit:
            _progress(progress, "audit", 0, 1, "Auditing metadata and VTT files")
            audit_corpus(
                metadata_csv,
                vtt_folder,
                audit_dir,
                validate_content=False,
                progress=lambda current, total, message: _progress(progress, "audit", current, total, message),
            )

        metadata, _ = load_metadata(metadata_csv)
        vtt_files = _select_vtt_files(vtt_folder)
        database_path = temp_dir / "corpus.sqlite"
        connection = _create_database(database_path)

        article_rows = [
            (item.article_id, item.article_title, json.dumps(item.sections, ensure_ascii=False), item.article_url)
            for item in metadata.values()
        ]
        connection.executemany(
            "INSERT INTO articles(article_id, article_title, sections_json, article_url) VALUES (?, ?, ?, ?)",
            article_rows,
        )
        connection.commit()

        parsed_files = 0
        skipped_unmapped = 0
        skipped_malformed = 0
        inserted_passages = 0
        insert_buffer: list[tuple] = []

        for position, path in enumerate(vtt_files, start=1):
            article_id = canonical_id(path.stem)
            article = metadata.get(article_id)
            if article is None:
                if not settings.include_unmapped_vtts:
                    skipped_unmapped += 1
                    _progress(progress, "parse", position, len(vtt_files), f"Skipped unmapped {path.name}")
                    continue
                from .models import ArticleMetadata
                article = ArticleMetadata(article_id, f"Journal ID {article_id}", ["Unmapped"], None)
            try:
                cues = parse_vtt(path)
            except (VTTParseError, OSError, ValueError):
                skipped_malformed += 1
                _progress(progress, "parse", position, len(vtt_files), f"Skipped malformed {path.name}")
                continue

            passages = build_passages(
                cues,
                article,
                path.name,
                target_words=settings.target_passage_words,
                max_words=settings.max_passage_words,
                max_seconds=settings.max_passage_seconds,
                overlap_seconds=settings.overlap_seconds,
            )
            for passage in passages:
                insert_buffer.append(
                    (
                        passage.article_id,
                        passage.article_title,
                        json.dumps(passage.sections, ensure_ascii=False),
                        passage.start_ms,
                        passage.end_ms,
                        passage.text,
                        passage.normalized_text,
                        passage.source_file,
                    )
                )
            if len(insert_buffer) >= 2000:
                connection.executemany(
                    """
                    INSERT INTO passages(
                        article_id, article_title, sections_json, start_ms, end_ms,
                        text, normalized_text, source_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_buffer,
                )
                inserted_passages += len(insert_buffer)
                insert_buffer.clear()
                connection.commit()
            parsed_files += 1
            _progress(progress, "parse", position, len(vtt_files), f"Parsed {path.name}")

        if insert_buffer:
            connection.executemany(
                """
                INSERT INTO passages(
                    article_id, article_title, sections_json, start_ms, end_ms,
                    text, normalized_text, source_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_buffer,
            )
            inserted_passages += len(insert_buffer)
            connection.commit()

        passage_count = int(connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0])
        if passage_count == 0:
            raise RuntimeError("No searchable passages were created. Check the VTT files and metadata mapping.")

        _progress(progress, "fts", 0, 1, "Building the local exact-text index")
        connection.execute("INSERT INTO passage_fts(passage_fts) VALUES('rebuild')")
        connection.commit()

        embedder = create_embedder(
            settings.embedding_backend,
            settings.embedding_model,
            device=settings.embedding_device,
        )
        embeddings_path = temp_dir / "embeddings.npy"
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(passage_count, embedder.dimension),
        )

        cursor = connection.execute("SELECT passage_id, normalized_text FROM passages ORDER BY passage_id")
        completed = 0
        while True:
            rows = cursor.fetchmany(settings.embedding_batch_size)
            if not rows:
                break
            texts = [row[1] for row in rows]
            matrix = embedder.encode_documents(texts, batch_size=settings.embedding_batch_size)
            if matrix.shape != (len(rows), embedder.dimension):
                raise RuntimeError(f"Unexpected embedding shape: {matrix.shape}")
            embeddings[completed : completed + len(rows)] = matrix
            completed += len(rows)
            embeddings.flush()
            _progress(progress, "embed", completed, passage_count, f"Embedded {completed:,} of {passage_count:,} passages")

        # Windows keeps the memory-mapped embeddings file locked until its
        # underlying mmap handle is explicitly closed. Leaving it open prevents
        # the completed temporary index directory from being renamed into place.
        embeddings.flush()
        mmap_handle = getattr(embeddings, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        del embeddings
        cursor.close()

        connection.execute(
            "CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        build_info = {
            "parsed_files": str(parsed_files),
            "skipped_unmapped": str(skipped_unmapped),
            "skipped_malformed": str(skipped_malformed),
            "passage_count": str(passage_count),
        }
        connection.executemany("INSERT INTO build_info(key, value) VALUES (?, ?)", build_info.items())
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()

        manifest = IndexManifest(
            version=INDEX_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            passage_count=passage_count,
            embedding_dim=embedder.dimension,
            embedding_backend=settings.embedding_backend,
            embedding_model=embedder.name,
            metadata_file=str(metadata_csv),
            metadata_sha256=sha256_file(metadata_csv),
            vtt_folder=str(vtt_folder),
            vtt_count=len(vtt_files),
            settings=settings.to_dict(),
        )
        atomic_write_json(temp_dir / "manifest.json", asdict(manifest))

        backup_dir = index_dir.with_name(index_dir.name + ".previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if index_dir.exists():
            os.replace(index_dir, backup_dir)
        os.replace(temp_dir, index_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        _progress(progress, "complete", 1, 1, f"Index ready: {index_dir}")
        return manifest
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

