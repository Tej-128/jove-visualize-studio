from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Settings
from .embeddings import create_embedder
from .features import build_fts_query, score_feature_compatibility
from .feedback_memory import ReviewerFeedbackMemory
from .models import Candidate, ScriptStep
from .normalizer import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RetrievalInfo:
    fusion: float = 0.0
    best_rank: int = 10_000
    source_hits: int = 0
    sources: set[str] = field(default_factory=set)

    def add(self, source: str, rank: int, weight: float) -> None:
        self.fusion += float(weight) / (30.0 + float(rank))
        self.best_rank = min(self.best_rank, int(rank))
        if source not in self.sources:
            self.sources.add(source)
            self.source_hits += 1


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _token_scores(query: str, candidate: str) -> tuple[float, float]:
    left = set(normalize_text(query).split())
    right = set(normalize_text(candidate).split())
    if not left or not right:
        return 0.0, 0.0
    overlap = len(left & right)
    return overlap / max(1, len(left | right)), overlap / max(1, min(len(left), len(right)))


class CorpusIndex:
    def __init__(self, index_dir: str | Path, settings: Settings | None = None) -> None:
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.settings = settings or Settings(**{
            key: value
            for key, value in self.manifest.get("settings", {}).items()
            if key in Settings.__dataclass_fields__
        })
        self.connection = sqlite3.connect(self.index_dir / "corpus.sqlite", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self.embedder = create_embedder(
            self.manifest["embedding_backend"],
            self.manifest["embedding_model"],
            device=self.settings.embedding_device,
        )
        if self.embeddings.shape[1] != self.embedder.dimension:
            raise RuntimeError("Embedding model dimension does not match the stored index.")

        self.learned_artifact: dict | None = None
        self.learned_model_path = _resolve_project_path(self.settings.learned_ranker_path)
        if self.settings.learned_ranker_enabled and self.learned_model_path.exists():
            try:
                import joblib
                artifact = joblib.load(self.learned_model_path)
                if isinstance(artifact, dict) and not artifact.get("hardcoded_article_rules", False):
                    self.learned_artifact = artifact
            except Exception:
                self.learned_artifact = None

        self.feedback_memory: ReviewerFeedbackMemory | None = None
        feedback_path = _resolve_project_path(self.settings.reviewer_feedback_path)
        if self.settings.reviewer_feedback_enabled and feedback_path.exists():
            try:
                self.feedback_memory = ReviewerFeedbackMemory(
                    feedback_path,
                    self.embedder,
                    min_similarity=self.settings.reviewer_memory_min_similarity,
                )
            except Exception:
                self.feedback_memory = None

        self._article_ids: list[str] | None = None
        self._article_titles: list[str] | None = None
        self._article_embeddings: np.ndarray | None = None
        self._article_index: dict[str, int] | None = None

    @property
    def learned_ranker_available(self) -> bool:
        return self.learned_artifact is not None

    @property
    def reviewer_feedback_available(self) -> bool:
        return bool(self.feedback_memory and self.feedback_memory.available)

    def close(self) -> None:
        self.connection.close()

    def _semantic_top_k(self, query_texts: list[str], top_k: int) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
        query_matrix = self.embedder.encode_queries(query_texts, batch_size=max(1, len(query_texts)))
        total = self.embeddings.shape[0]
        block_size = 50_000
        best_ids = [np.empty(0, dtype=np.int64) for _ in query_texts]
        best_scores = [np.empty(0, dtype=np.float32) for _ in query_texts]

        for start in range(0, total, block_size):
            end = min(total, start + block_size)
            block = np.asarray(self.embeddings[start:end], dtype=np.float32)
            scores = block @ query_matrix.T
            for query_index in range(len(query_texts)):
                column = scores[:, query_index]
                local_k = min(top_k, len(column))
                if local_k == 0:
                    continue
                positions = np.argpartition(column, -local_k)[-local_k:]
                ids = positions.astype(np.int64) + start + 1
                values = column[positions].astype(np.float32)
                combined_ids = np.concatenate([best_ids[query_index], ids])
                combined_scores = np.concatenate([best_scores[query_index], values])
                keep = min(top_k, len(combined_scores))
                selected = np.argpartition(combined_scores, -keep)[-keep:]
                best_ids[query_index] = combined_ids[selected]
                best_scores[query_index] = combined_scores[selected]

        results: list[list[tuple[int, float]]] = []
        for ids, scores in zip(best_ids, best_scores):
            order = np.argsort(scores)[::-1]
            results.append([(int(ids[pos]), float(scores[pos])) for pos in order])
        return results, np.asarray(query_matrix, dtype=np.float32)

    def _lexical_top_k(self, query_text: str, top_k: int) -> list[tuple[int, float]]:
        fts_query = build_fts_query(query_text)
        if not fts_query:
            return []
        try:
            rows = self.connection.execute(
                """
                SELECT rowid AS passage_id, bm25(passage_fts) AS bm25_score
                FROM passage_fts
                WHERE passage_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(row["passage_id"]), 1.0 / (1.0 + rank)) for rank, row in enumerate(rows)]

    def _fetch_passages(self, passage_ids: Iterable[int]) -> dict[int, sqlite3.Row]:
        ids = sorted(set(int(value) for value in passage_ids))
        if not ids:
            return {}
        result: dict[int, sqlite3.Row] = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT * FROM passages WHERE passage_id IN ({placeholders})", chunk
            ).fetchall()
            for row in rows:
                result[int(row["passage_id"])] = row
        return result

    def _article_passage_ids(self, article_ids: list[str]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = defaultdict(list)
        for start in range(0, len(article_ids), 800):
            chunk = article_ids[start:start + 800]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT passage_id, article_id FROM passages WHERE article_id IN ({placeholders})", chunk
            ).fetchall()
            for row in rows:
                result[str(row["article_id"])].append(int(row["passage_id"]))
        return result

    def _load_article_title_cache(self) -> tuple[list[str], list[str], np.ndarray, dict[str, int]]:
        if self._article_ids is not None and self._article_embeddings is not None and self._article_index is not None:
            return self._article_ids, self._article_titles or [], self._article_embeddings, self._article_index

        rows = self.connection.execute("SELECT article_id, article_title FROM articles ORDER BY article_id").fetchall()
        article_ids = [str(row["article_id"]) for row in rows]
        article_titles = [str(row["article_title"]) for row in rows]
        cache_dir = self.index_dir / "v2_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = cache_dir / "article_title_embeddings.npy"
        ids_path = cache_dir / "article_ids.json"
        titles_path = cache_dir / "article_titles.json"
        meta_path = cache_dir / "article_cache_meta.json"
        expected = {
            "embedding_model": self.manifest.get("embedding_model", ""),
            "article_count": len(article_ids),
        }
        valid = False
        if all(path.exists() for path in (embeddings_path, ids_path, titles_path, meta_path)):
            try:
                valid = (
                    json.loads(meta_path.read_text(encoding="utf-8")) == expected
                    and json.loads(ids_path.read_text(encoding="utf-8")) == article_ids
                )
            except Exception:
                valid = False
        if valid:
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if embeddings.shape[0] != len(article_ids):
                valid = False
        if not valid:
            embeddings = self.embedder.encode_queries(article_titles, batch_size=128).astype(np.float32)
            np.save(embeddings_path, embeddings)
            ids_path.write_text(json.dumps(article_ids, ensure_ascii=False), encoding="utf-8")
            titles_path.write_text(json.dumps(article_titles, ensure_ascii=False), encoding="utf-8")
            meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")

        self._article_ids = article_ids
        self._article_titles = article_titles
        self._article_embeddings = embeddings
        self._article_index = {article_id: i for i, article_id in enumerate(article_ids)}
        return article_ids, article_titles, embeddings, self._article_index

    @staticmethod
    def _score01(raw: float) -> float:
        return max(0.0, min(1.0, (float(raw) + 1.0) / 2.0))

    def _construct_candidate(
        self,
        row: sqlite3.Row,
        step: ScriptStep,
        vectors: dict[str, np.ndarray],
        lexical_score: float,
        retrieval: RetrievalInfo,
    ) -> Candidate:
        passage_id = int(row["passage_id"])
        passage_vector = np.asarray(self.embeddings[passage_id - 1], dtype=np.float32)
        semantic = self._score01(passage_vector @ vectors["step"])
        contextual = self._score01(passage_vector @ vectors["context"])
        title_step = self._score01(passage_vector @ vectors["title_step"])
        relay_semantic = self._score01(passage_vector @ vectors.get("relay", vectors["context"]))
        protocol_semantic = self._score01(passage_vector @ vectors.get("protocol", vectors["title_step"]))
        phase_semantic = self._score01(passage_vector @ vectors.get("phase", vectors["step"]))
        state_semantic = self._score01(passage_vector @ vectors.get("state", vectors["context"]))
        feature = score_feature_compatibility(step.text, row["text"])
        local_score = (
            0.31 * semantic
            + 0.10 * contextual
            + 0.08 * title_step
            + self.settings.relay_semantic_weight * relay_semantic
            + self.settings.protocol_semantic_weight * protocol_semantic
            + self.settings.phase_semantic_weight * phase_semantic
            + 0.23 * float(feature["feature_score"])
            + self.settings.hard_constraint_weight * float(feature.get("constraint_match", 0.5))
        )
        # Persistent experiment-state alignment is a bounded supporting signal only.
        local_score += 0.05 * (state_semantic - 0.5)
        if feature["critical_contradiction"]:
            local_score *= 0.48
        return Candidate(
            passage_id=passage_id,
            article_id=str(row["article_id"]),
            article_title=str(row["article_title"]),
            sections=json.loads(row["sections_json"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            text=str(row["text"]),
            source_file=str(row["source_file"]),
            semantic_score=semantic,
            context_semantic_score=contextual,
            lexical_score=float(lexical_score),
            feature_score=float(feature["feature_score"]),
            local_score=float(local_score),
            action_match=float(feature["action_match"]),
            entity_match=float(feature["entity_match"]),
            condition_match=float(feature["condition_match"]),
            numeric_match=float(feature["numeric_match"]),
            relay_semantic_score=float(relay_semantic),
            protocol_semantic_score=float(protocol_semantic),
            phase_semantic_score=float(phase_semantic),
            state_semantic_score=float(state_semantic),
            constraint_match_score=float(feature.get("constraint_match", 0.5)),
            vtt_evidence_risk=str(feature.get("vtt_evidence_risk", "medium")),
            contradictions=list(feature["contradictions"]),
            critical_contradiction=bool(feature["critical_contradiction"]),
            retrieval_fusion=float(retrieval.fusion),
            retrieval_source_coverage=float(retrieval.source_hits),
            retrieval_best_rank=float(retrieval.best_rank),
        )

    def _passage_features(
        self,
        step: ScriptStep,
        candidate: Candidate,
        vectors: dict[str, np.ndarray],
        article_embeddings: np.ndarray,
        article_index: dict[str, int],
        fusion_max: float,
        article_support: int,
    ) -> np.ndarray:
        title_idx = article_index.get(candidate.article_id)
        if title_idx is None:
            title_step = title_protocol = 0.5
        else:
            title_vector = np.asarray(article_embeddings[title_idx], dtype=np.float32)
            title_step = self._score01(title_vector @ vectors["title_step"])
            title_protocol = self._score01(title_vector @ vectors["title"])
        passage_vector = np.asarray(self.embeddings[candidate.passage_id - 1], dtype=np.float32)
        title_step_passage = self._score01(passage_vector @ vectors["title_step"])
        jaccard, containment = _token_scores(step.text, candidate.text)
        fusion = candidate.retrieval_fusion / max(fusion_max, 1e-9)
        source_coverage = min(1.0, candidate.retrieval_source_coverage / 6.0)
        best_rank = 0.0 if candidate.retrieval_best_rank >= 10_000 else 1.0 / math.log2(candidate.retrieval_best_rank + 2.0)
        return np.asarray([
            candidate.semantic_score,
            candidate.context_semantic_score,
            title_step_passage,
            title_step,
            title_protocol,
            candidate.lexical_score,
            jaccard,
            containment,
            candidate.feature_score,
            candidate.action_match,
            candidate.entity_match,
            candidate.condition_match,
            candidate.numeric_match,
            candidate.local_score,
            1.0 - min(1.0, len(candidate.contradictions) / 4.0),
            0.0 if candidate.critical_contradiction else 1.0,
            fusion,
            source_coverage,
            best_rank,
            min(1.0, article_support / float(max(1, self.settings.max_passages_per_article))),
        ], dtype=np.float64)

    @staticmethod
    def _aggregate_article_features(records: list[tuple[Candidate, np.ndarray]]) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for candidate, features in records:
            grouped[candidate.article_id].append(features)
        output: dict[str, np.ndarray] = {}
        for article_id, feature_rows in grouped.items():
            matrix = np.vstack(feature_rows)
            maximum = matrix.max(axis=0)
            top2_step = sorted(matrix[:, 0], reverse=True)[:2]
            top2_local = sorted(matrix[:, 13], reverse=True)[:2]
            top2_fusion = sorted(matrix[:, 16], reverse=True)[:2]
            mean2 = lambda values: float(np.mean(values)) if values else 0.0
            output[article_id] = np.asarray([
                maximum[0], maximum[1], maximum[2], maximum[3], maximum[4],
                maximum[5], maximum[8], maximum[9], maximum[10], maximum[11],
                maximum[12], maximum[13], maximum[14], maximum[15], maximum[16],
                maximum[17], maximum[18], min(1.0, len(feature_rows) / 4.0),
                mean2(top2_step), mean2(top2_local), mean2(top2_fusion),
            ], dtype=np.float64)
        return output

    def _apply_learned_scores(
        self,
        step: ScriptStep,
        candidates: list[Candidate],
        vectors: dict[str, np.ndarray],
        article_embeddings: np.ndarray,
        article_index: dict[str, int],
    ) -> None:
        if not candidates:
            return
        support: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            support[candidate.article_id] += 1
            candidate.article_support = min(1.0, support[candidate.article_id] / float(max(1, self.settings.max_passages_per_article)))
        fusion_max = max((candidate.retrieval_fusion for candidate in candidates), default=1.0)
        features = [
            self._passage_features(
                step, candidate, vectors, article_embeddings, article_index,
                fusion_max, support[candidate.article_id],
            )
            for candidate in candidates
        ]

        if self.learned_artifact:
            passage_weights = np.asarray(self.learned_artifact.get("passage_weights", []), dtype=np.float64)
            article_weights = np.asarray(self.learned_artifact.get("article_weights", []), dtype=np.float64)
            if len(passage_weights) == len(features[0]):
                passage_scores = np.asarray(np.vstack(features) @ passage_weights, dtype=np.float64)
            else:
                passage_scores = np.asarray([candidate.local_score for candidate in candidates], dtype=np.float64)
            article_features = self._aggregate_article_features(list(zip(candidates, features)))
            article_scores = {
                article_id: float(values @ article_weights) if len(article_weights) == len(values) else 0.5
                for article_id, values in article_features.items()
            }
            for index, candidate in enumerate(candidates):
                candidate.learned_article_score = max(0.0, min(1.0, article_scores.get(candidate.article_id, 0.5)))
                candidate.learned_ranker_score = max(
                    0.0,
                    min(1.0, 0.68 * candidate.learned_article_score + 0.32 * float(passage_scores[index])),
                )

        if self.feedback_memory and self.feedback_memory.available:
            memory_results = self.feedback_memory.assess_many(step, candidates)
            for candidate, memory in zip(candidates, memory_results):
                candidate.reviewer_memory_score = memory.score
                candidate.reviewer_memory_similarity = memory.similarity
                candidate.reviewer_memory_label = memory.label
                candidate.reviewer_memory_note = memory.note

        for candidate in candidates:
            pre_ai = candidate.local_score
            if self.learned_artifact:
                pre_ai += self.settings.learned_signal_weight * (candidate.learned_ranker_score - 0.5)
            if candidate.reviewer_memory_similarity >= self.settings.reviewer_memory_min_similarity:
                pre_ai += self.settings.reviewer_memory_weight * (candidate.reviewer_memory_score - 0.5)
            # Context relay and hard constraints are explicit VTT-only signals.
            pre_ai += 0.035 * (candidate.relay_semantic_score - 0.5)
            pre_ai += 0.025 * (candidate.protocol_semantic_score - 0.5)
            pre_ai += 0.035 * (candidate.state_semantic_score - 0.5)
            pre_ai += 0.045 * (candidate.constraint_match_score - 0.5)
            pre_ai += 0.04 * min(1.0, candidate.retrieval_fusion / max(fusion_max, 1e-9))
            if (
                candidate.reviewer_memory_label == "rejected"
                and candidate.reviewer_memory_similarity >= self.settings.reviewer_memory_exact_similarity
            ):
                pre_ai *= 0.72
            candidate.final_score = max(0.0, min(1.0, pre_ai))

    def retrieve(self, step: ScriptStep) -> list[Candidate]:
        title = step.script_title.strip()
        context_text = " ".join(part for part in [
            f"Protocol title: {title}" if title else "",
            f"Previous step: {step.previous_text}" if step.previous_text else "",
            f"Current step: {step.text}",
            f"Next step: {step.next_text}" if step.next_text else "",
        ] if part)
        title_step_text = f"{title}. {step.text}" if title else step.text
        relay_text = step.relay_text or context_text
        protocol_text = " ".join(part for part in [step.protocol_signature, f"Current step: {step.text}"] if part)
        phase_text = " ".join(part for part in [
            f"Protocol phase: {step.protocol_phase}" if step.protocol_phase else "",
            f"Active entities: {', '.join(step.active_entities)}" if step.active_entities else "",
            f"Current step: {step.text}",
        ] if part)
        state_text = step.state_transition or " ".join(part for part in [step.state_before, step.text, step.state_after] if part)
        query_texts = [
            step.text,
            context_text,
            title_step_text,
            title or step.text,
            relay_text,
            protocol_text or title_step_text,
            phase_text or step.text,
            state_text or relay_text,
        ]
        semantic_sets, query_matrix = self._semantic_top_k(query_texts, top_k=self.settings.semantic_top_k)
        vectors = {
            "step": query_matrix[0],
            "context": query_matrix[1],
            "title_step": query_matrix[2],
            "title": query_matrix[3],
            "relay": query_matrix[4],
            "protocol": query_matrix[5],
            "phase": query_matrix[6],
            "state": query_matrix[7],
        }

        info_by_id: dict[int, RetrievalInfo] = {}
        semantic_scores: dict[str, dict[int, float]] = {}
        for source, results, weight in [
            ("semantic_step", semantic_sets[0], 1.00),
            ("semantic_context", semantic_sets[1], 0.72),
            ("semantic_title_step", semantic_sets[2], 0.82),
            ("semantic_relay", semantic_sets[4], 0.94),
            ("semantic_protocol", semantic_sets[5], 0.62),
            ("semantic_phase", semantic_sets[6], 0.68),
            ("semantic_state", semantic_sets[7], 0.88),
        ]:
            semantic_scores[source] = dict(results)
            for rank, (passage_id, _) in enumerate(results, start=1):
                info_by_id.setdefault(passage_id, RetrievalInfo()).add(source, rank, weight)

        lexical_by_id: dict[int, float] = {}
        for source, text, weight in [
            ("lexical_step", step.text, 0.92),
            ("lexical_title_step", title_step_text, 0.55),
            ("lexical_relay", relay_text, 0.62),
            ("lexical_phase", phase_text, 0.45),
            ("lexical_state", state_text, 0.52),
        ]:
            lexical = self._lexical_top_k(text, self.settings.lexical_top_k)
            for rank, (passage_id, lexical_score) in enumerate(lexical, start=1):
                info_by_id.setdefault(passage_id, RetrievalInfo()).add(source, rank, weight)
                lexical_by_id[passage_id] = max(lexical_by_id.get(passage_id, 0.0), float(lexical_score))

        article_ids, _, article_embeddings, article_index = self._load_article_title_cache()
        title_scores = np.asarray(article_embeddings @ vectors["title_step"], dtype=np.float32)
        article_k = min(self.settings.article_title_top_k, len(title_scores))
        if article_k:
            positions = np.argpartition(title_scores, -article_k)[-article_k:]
            positions = positions[np.argsort(title_scores[positions])[::-1]]
            top_article_ids = [article_ids[int(i)] for i in positions]
            passage_map = self._article_passage_ids(top_article_ids)
            for article_rank, article_id in enumerate(top_article_ids, start=1):
                passage_ids = passage_map.get(article_id, [])
                if not passage_ids:
                    continue
                passage_vectors = np.asarray(self.embeddings[np.asarray(passage_ids, dtype=np.int64) - 1], dtype=np.float32)
                values = passage_vectors @ vectors["step"]
                take = min(2, len(passage_ids))
                local_positions = np.argpartition(values, -take)[-take:]
                local_positions = local_positions[np.argsort(values[local_positions])[::-1]]
                for within_rank, local_pos in enumerate(local_positions, start=1):
                    passage_id = int(passage_ids[int(local_pos)])
                    info_by_id.setdefault(passage_id, RetrievalInfo()).add(
                        "article_title", article_rank + within_rank - 1, 1.15
                    )

        rows = self._fetch_passages(info_by_id)
        candidates: list[Candidate] = []
        for passage_id, retrieval in info_by_id.items():
            row = rows.get(passage_id)
            if row is None:
                continue
            candidate = self._construct_candidate(
                row, step, vectors, lexical_by_id.get(passage_id, 0.0), retrieval
            )
            candidates.append(candidate)

        candidates.sort(key=lambda item: (item.retrieval_fusion, item.local_score), reverse=True)
        selected: list[Candidate] = []
        article_counts: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            if article_counts[candidate.article_id] >= self.settings.max_passages_per_article:
                continue
            article_counts[candidate.article_id] += 1
            selected.append(candidate)
            if len(selected) >= self.settings.broad_candidate_pool:
                break

        self._apply_learned_scores(step, selected, vectors, article_embeddings, article_index)
        selected.sort(key=lambda item: item.final_score, reverse=True)
        return selected[: self.settings.shortlist_per_step]

