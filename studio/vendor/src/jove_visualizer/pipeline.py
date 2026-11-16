from __future__ import annotations

import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .confidence import classify_match, finalize_scores
from .config import Settings
from .context_engine import build_protocol_context
from .models import MatchResult, RerankOutcome, ScriptStep
from .rerank import ScientificReranker
from .retrieval import CorpusIndex
from .sequence_optimizer import optimize_sequence

ProgressCallback = Callable[[int, int, str], None]


def _unique_article_candidates(candidates, limit: int):
    selected = []
    seen = set()
    for candidate in candidates:
        if candidate.article_id in seen:
            continue
        seen.add(candidate.article_id)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _article_strength(candidates: list, limit: int) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for candidate in candidates[:limit]:
        strengths[candidate.article_id] = max(
            strengths.get(candidate.article_id, 0.0),
            float(candidate.final_score or candidate.local_score),
        )
    return strengths


def _apply_sequence_context(
    steps: list[ScriptStep],
    retrieved: dict[int, list],
    settings: Settings,
) -> None:
    """Softly prefer coherent source reuse across neighbouring protocol steps.

    This is deliberately gated by each candidate's own scientific/VTT evidence. A source
    never gets continuity credit if its current-step match is weak or contradictory.
    """
    if not settings.context_relay_enabled or len(steps) < 2:
        return

    top_n = max(3, settings.continuity_top_articles_per_step)
    strengths = {
        step.number: _article_strength(retrieved.get(step.number, []), top_n)
        for step in steps
    }
    global_counts: Counter[str] = Counter()
    global_strength_sum: defaultdict[str, float] = defaultdict(float)
    for step in steps:
        for article_id, strength in strengths[step.number].items():
            if strength >= settings.continuity_min_local_score:
                global_counts[article_id] += 1
                global_strength_sum[article_id] += strength

    for position, step in enumerate(steps):
        previous_strength = strengths[steps[position - 1].number] if position > 0 else {}
        next_strength = strengths[steps[position + 1].number] if position + 1 < len(steps) else {}

        for candidate in retrieved.get(step.number, []):
            if candidate.critical_contradiction or candidate.local_score < settings.continuity_min_local_score:
                candidate.sequence_continuity_score = 0.0
                continue
            if (
                candidate.reviewer_memory_label == "rejected"
                and candidate.reviewer_memory_similarity >= settings.reviewer_memory_exact_similarity
            ):
                candidate.sequence_continuity_score = 0.0
                continue

            adjacent_values = []
            if candidate.article_id in previous_strength:
                adjacent_values.append(previous_strength[candidate.article_id])
            if candidate.article_id in next_strength:
                adjacent_values.append(next_strength[candidate.article_id])
            adjacent = sum(adjacent_values) / len(adjacent_values) if adjacent_values else 0.0

            count = global_counts.get(candidate.article_id, 0)
            global_support = 0.0
            if count > 1:
                mean_strength = global_strength_sum[candidate.article_id] / count
                coverage = min(1.0, (count - 1) / max(1, len(steps) - 1))
                global_support = mean_strength * coverage

            continuity = min(1.0, 0.82 * adjacent + 0.18 * global_support)
            candidate.sequence_continuity_score = continuity
            candidate.final_score = min(
                1.0,
                candidate.final_score
                + settings.adjacent_article_boost * adjacent
                + settings.global_article_boost * global_support,
            )
        retrieved.get(step.number, []).sort(key=lambda item: item.final_score, reverse=True)


def match_steps(
    index: CorpusIndex,
    steps: list[ScriptStep],
    settings: Settings,
    api_key: str | None = None,
    progress: ProgressCallback | None = None,
) -> list[MatchResult]:
    if not steps:
        return []

    # Always rebuild context after any edits made in the Streamlit table.
    build_protocol_context(steps)

    retrieved: dict[int, list] = {}
    started: dict[int, float] = {}
    for position, step in enumerate(steps, start=1):
        started[step.number] = time.perf_counter()
        retrieved[step.number] = index.retrieve(step)
        if progress:
            progress(position, len(steps), f"Retrieved VTT candidates with protocol context for step {step.number}")

    _apply_sequence_context(steps, retrieved, settings)

    reranker = ScientificReranker(
        settings,
        api_key=api_key,
        cache_path=Path(index.index_dir) / "rerank_cache.sqlite",
    )
    rerank_meta: dict[int, RerankOutcome] = {}

    def rerank_one(step: ScriptStep):
        candidates = retrieved[step.number]
        return step.number, reranker.rerank(step, candidates)

    max_workers = min(settings.openai_max_concurrency, len(steps)) if api_key and settings.openai_enabled else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(rerank_one, step): step for step in steps}
        completed = 0
        for future in as_completed(futures):
            step_number, outcome = future.result()
            rerank_meta[step_number] = outcome
            completed += 1
            if progress:
                progress(completed, len(steps), f"Context-aware scientific reranking completed for step {step_number}")

    # Finalize every candidate list first so all outputs share one scoring pass.
    final_candidates: dict[int, list] = {}
    for step in steps:
        outcome = rerank_meta.get(
            step.number,
            RerankOutcome(False, False, "Scientific reranking was not completed."),
        )
        candidates = retrieved[step.number]
        if outcome.ai_used and outcome.reviewed_passage_ids:
            # An unreviewed local candidate cannot overtake the AI-reviewed shortlist.
            candidates = [item for item in candidates if item.passage_id in outcome.reviewed_passage_ids]
        final_candidates[step.number] = finalize_scores(candidates, settings, ai_was_used=outcome.ai_used)

    # V3.2: optimize the complete protocol as one conservative path. Local scientific
    # suitability remains primary; continuity can only reorder close, independently usable candidates.
    optimize_sequence(steps, final_candidates, settings)
    if progress:
        progress(len(steps), len(steps), "Whole-script sequence and source-coverage optimization completed")

    results: list[MatchResult] = []
    for step in steps:
        outcome = rerank_meta.get(
            step.number,
            RerankOutcome(False, False, "Scientific reranking was not completed."),
        )
        candidates = final_candidates[step.number]

        status, confidence, classification_reason = classify_match(
            candidates,
            settings,
            outcome.no_match_recommended,
            outcome.no_match_reason,
            outcome.ai_used,
        )

        best = candidates[0] if status == "MATCH" and candidates else None
        alternatives = []
        additional_same_id = []
        if best:
            additional_same_id = [
                candidate
                for candidate in candidates[1:]
                if candidate.article_id == best.article_id
                and best.final_score - candidate.final_score <= 0.10
            ][:2]
            alternatives = _unique_article_candidates(
                [candidate for candidate in candidates[1:] if candidate.article_id != best.article_id],
                settings.output_alternatives,
            )
        else:
            alternatives = _unique_article_candidates(candidates, settings.output_alternatives)

        no_match_reason = classification_reason
        if status == "NO_MATCH" and not no_match_reason:
            no_match_reason = outcome.no_match_reason or "No candidate passed the reliability conditions."
        elif status == "MATCH" and classification_reason:
            no_match_reason = classification_reason

        results.append(
            MatchResult(
                step=step,
                status=status,
                confidence=confidence,
                best=best,
                alternatives=alternatives,
                additional_timestamps_same_id=additional_same_id,
                no_match_reason=no_match_reason,
                elapsed_seconds=time.perf_counter() - started[step.number],
            )
        )
    return results

