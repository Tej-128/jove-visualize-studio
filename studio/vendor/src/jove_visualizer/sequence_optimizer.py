from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .config import Settings
from .models import Candidate, ScriptStep


@dataclass(slots=True)
class _Node:
    candidate: Candidate | None
    node_score: float


def _ai_tier(candidate: Candidate) -> int:
    # Scientific suitability is primary; selection only breaks ties inside that tier.
    if candidate.ai_suitability == "preferred":
        return 5 if candidate.ai_selected else 4
    if candidate.ai_suitability == "acceptable":
        return 3 if candidate.ai_selected else 2
    if candidate.ai_suitability == "rejected":
        return 0
    return 1


def _usable(candidate: Candidate, settings: Settings) -> bool:
    if candidate.critical_contradiction:
        return False
    if candidate.ai_suitability == "rejected":
        return False
    if (
        candidate.reviewer_memory_label == "rejected"
        and candidate.reviewer_memory_similarity >= settings.reviewer_memory_exact_similarity
    ):
        return False
    return candidate.final_score >= settings.sequence_min_candidate_score


def _article_coverage(
    steps: list[ScriptStep],
    candidates_by_step: dict[int, list[Candidate]],
    settings: Settings,
) -> dict[str, float]:
    """Estimate how credibly each Journal source covers this protocol.

    Coverage is only earned from candidates that are independently usable. It can therefore
    break ties and encourage continuity, but it cannot rescue a scientifically bad match.
    """
    per_article: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for position, step in enumerate(steps):
        best_for_article: dict[str, float] = {}
        for candidate in candidates_by_step.get(step.number, []):
            if not _usable(candidate, settings):
                continue
            suitability_factor = 1.0
            if candidate.ai_suitability == "acceptable":
                suitability_factor = 0.94
            elif candidate.ai_suitability is None:
                suitability_factor = 0.88
            strength = candidate.final_score * suitability_factor
            best_for_article[candidate.article_id] = max(best_for_article.get(candidate.article_id, 0.0), strength)
        for article_id, strength in best_for_article.items():
            if strength >= settings.source_coverage_min_score:
                per_article[article_id].append((position, strength))

    result: dict[str, float] = {}
    total_steps = max(1, len(steps))
    for article_id, records in per_article.items():
        positions = sorted(position for position, _ in records)
        strengths = [strength for _, strength in records]
        coverage = len(records) / total_steps
        mean_strength = sum(strengths) / max(1, len(strengths))
        adjacent_pairs = sum(1 for left, right in zip(positions, positions[1:]) if right == left + 1)
        adjacency = adjacent_pairs / max(1, len(records) - 1) if len(records) > 1 else 0.0
        # Strong source coverage matters most when it spans several protocol steps and
        # especially neighbouring steps. Cap it at 1 to keep it a bonus, never a veto.
        result[article_id] = min(1.0, 0.62 * mean_strength + 0.25 * coverage + 0.13 * adjacency)
    return result


def _transition_score(previous: Candidate | None, current: Candidate | None, settings: Settings) -> float:
    if previous is None or current is None:
        return 0.0
    if previous.article_id != current.article_id:
        return 0.0
    if not _usable(previous, settings) or not _usable(current, settings):
        return 0.0

    evidence = min(previous.final_score, current.final_score)
    chronological = 0.0
    if current.start_ms >= previous.start_ms:
        chronological = 1.0
    elif previous.start_ms - current.start_ms <= 30_000:
        chronological = 0.55
    else:
        chronological = 0.15
    return min(1.0, 0.78 * evidence + 0.22 * chronological)


def _eligible_pool(candidates: list[Candidate], settings: Settings) -> list[Candidate]:
    usable = [candidate for candidate in candidates if _usable(candidate, settings)]
    if not usable:
        return []
    usable.sort(key=lambda item: (_ai_tier(item), item.final_score), reverse=True)
    top = usable[0]
    top_tier = _ai_tier(top)
    pool: list[Candidate] = []
    for candidate in usable:
        # Do not sacrifice a clearly stronger scientific judgement for continuity.
        if top.final_score - candidate.final_score > settings.sequence_max_local_drop:
            continue
        # Continuity never lowers the scientific suitability class. It may only choose
        # another candidate from the same class (or a stronger class) within the tight
        # local-score tie window.
        if top.ai_suitability == "preferred" and candidate.ai_suitability != "preferred":
            continue
        if top.ai_suitability == "acceptable" and candidate.ai_suitability not in {"preferred", "acceptable"}:
            continue
        pool.append(candidate)
        if len(pool) >= settings.sequence_candidates_per_step:
            break
    return pool


def optimize_sequence(
    steps: list[ScriptStep],
    candidates_by_step: dict[int, list[Candidate]],
    settings: Settings,
) -> dict[int, list[Candidate]]:
    """Jointly optimize the candidate path across the whole script.

    This is intentionally conservative: local scientific correctness remains primary,
    source continuity is only a capped bonus, and rejected/contradictory candidates never
    become eligible simply because the same Journal source appears elsewhere.
    """
    if not settings.sequence_optimizer_enabled or not steps:
        return candidates_by_step

    coverage = _article_coverage(steps, candidates_by_step, settings)
    pools: list[list[Candidate | None]] = []
    for step in steps:
        candidates = candidates_by_step.get(step.number, [])
        for candidate in candidates:
            candidate.source_coverage_score = coverage.get(candidate.article_id, 0.0)
            candidate.sequence_path_selected = False
            candidate.sequence_transition_score = 0.0
            candidate.sequence_path_score = candidate.final_score
            candidate.sequence_reason = None
        pool = _eligible_pool(candidates, settings)
        pools.append(pool if pool else [None])

    # Dynamic programming over steps: node evidence + source coverage + transition continuity.
    scores: list[dict[int, float]] = []
    backpointers: list[dict[int, int | None]] = []
    for step_index, pool in enumerate(pools):
        current_scores: dict[int, float] = {}
        current_back: dict[int, int | None] = {}
        for current_index, candidate in enumerate(pool):
            if candidate is None:
                node_score = 0.0
            else:
                node_score = (
                    candidate.final_score
                    + settings.sequence_source_coverage_weight * candidate.source_coverage_score
                )
            if step_index == 0:
                current_scores[current_index] = node_score
                current_back[current_index] = None
                continue

            best_score = float("-inf")
            best_previous: int | None = None
            previous_pool = pools[step_index - 1]
            for previous_index, previous_candidate in enumerate(previous_pool):
                transition = _transition_score(previous_candidate, candidate, settings)
                total = (
                    scores[step_index - 1][previous_index]
                    + node_score
                    + settings.sequence_continuity_bonus * transition
                )
                if total > best_score:
                    best_score = total
                    best_previous = previous_index
            current_scores[current_index] = best_score
            current_back[current_index] = best_previous
        scores.append(current_scores)
        backpointers.append(current_back)

    if not scores:
        return candidates_by_step

    last_index = max(scores[-1], key=scores[-1].get)
    selected_indices = [last_index]
    for step_index in range(len(pools) - 1, 0, -1):
        previous_index = backpointers[step_index][selected_indices[-1]]
        selected_indices.append(0 if previous_index is None else previous_index)
    selected_indices.reverse()

    previous_selected: Candidate | None = None
    for step, pool, chosen_index in zip(steps, pools, selected_indices):
        chosen = pool[chosen_index]
        candidates = candidates_by_step.get(step.number, [])
        if chosen is None or chosen not in candidates:
            previous_selected = None
            continue

        transition = _transition_score(previous_selected, chosen, settings)
        chosen.sequence_transition_score = transition
        chosen.sequence_path_selected = True
        chosen.sequence_path_score = min(
            1.0,
            chosen.final_score
            + settings.sequence_source_coverage_weight * chosen.source_coverage_score
            + settings.sequence_continuity_bonus * transition,
        )
        reasons: list[str] = []
        if chosen.source_coverage_score >= 0.55:
            reasons.append("This Journal source has strong usable VTT coverage across multiple protocol steps.")
        if previous_selected is not None and previous_selected.article_id == chosen.article_id:
            reasons.append("The same source provides a coherent neighbouring-step continuation.")
        if reasons:
            chosen.sequence_reason = " ".join(reasons)

        # Move the globally selected candidate to the front without altering its scientific score.
        # The eligibility gate above guarantees it is close to the local best and not lower-tier
        # than a preferred candidate.
        candidates.remove(chosen)
        candidates.insert(0, chosen)
        previous_selected = chosen

    return candidates_by_step

