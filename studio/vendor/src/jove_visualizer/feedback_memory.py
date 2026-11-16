from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Candidate, ScriptStep


LABEL_SCORES = {"preferred": 1.0, "acceptable": 0.65, "rejected": 0.0}


@dataclass(slots=True)
class FeedbackMatch:
    score: float = 0.5
    similarity: float = 0.0
    label: str | None = None
    note: str | None = None


class ReviewerFeedbackMemory:
    """Similarity-based reviewer memory.

    It stores no Journal IDs and does not implement any ID-specific rule. It compares the
    current step/candidate wording with reviewer-labelled examples and returns a soft signal.
    """

    def __init__(self, path: str | Path, embedder, min_similarity: float = 0.72) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.min_similarity = float(min_similarity)
        self.examples: list[dict] = []
        self.vectors: np.ndarray | None = None
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.examples = [item for item in payload.get("examples", []) if item.get("label") in LABEL_SCORES]
        if self.examples:
            texts = [self._pair_text(item["step_text"], item["candidate_title"], item["candidate_text"]) for item in self.examples]
            self.vectors = self.embedder.encode_queries(texts, batch_size=min(64, len(texts)))

    @staticmethod
    def _pair_text(step_text: str, title: str, passage: str) -> str:
        return f"Script step: {step_text}\nCandidate title: {title}\nCandidate caption: {passage}"

    @property
    def available(self) -> bool:
        return bool(self.examples) and self.vectors is not None

    def assess_many(self, step: ScriptStep, candidates: list[Candidate]) -> list[FeedbackMatch]:
        if not candidates:
            return []
        if not self.available:
            return [FeedbackMatch() for _ in candidates]
        texts = [self._pair_text(step.text, item.article_title, item.text) for item in candidates]
        vectors = self.embedder.encode_queries(texts, batch_size=min(64, len(texts)))
        matrix = np.asarray(vectors @ self.vectors.T, dtype=np.float64)
        results: list[FeedbackMatch] = []
        for similarities in matrix:
            if not len(similarities):
                results.append(FeedbackMatch())
                continue
            order = np.argsort(similarities)[::-1][:3]
            best_idx = int(order[0])
            best_similarity = float(similarities[best_idx])
            best_example = self.examples[best_idx]
            if best_similarity < self.min_similarity:
                results.append(FeedbackMatch(similarity=best_similarity))
                continue
            weights = np.asarray([max(0.0, float(similarities[int(i)])) ** 6 for i in order], dtype=np.float64)
            values = np.asarray([LABEL_SCORES[self.examples[int(i)]["label"]] for i in order], dtype=np.float64)
            score = float(np.average(values, weights=weights)) if weights.sum() > 0 else 0.5
            results.append(
                FeedbackMatch(
                    score=max(0.0, min(1.0, score)),
                    similarity=best_similarity,
                    label=str(best_example["label"]),
                    note=str(best_example.get("reviewer_note") or ""),
                )
            )
        return results

    def assess(self, step: ScriptStep, candidate: Candidate) -> FeedbackMatch:
        return self.assess_many(step, [candidate])[0]

