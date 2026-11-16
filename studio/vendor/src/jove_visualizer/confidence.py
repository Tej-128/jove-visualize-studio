from __future__ import annotations

from .config import Settings
from .models import Candidate


def _ai_tier(candidate: Candidate) -> int:
    # Suitability is primary; the AI-selected flag only breaks ties within a suitability tier.
    # This prevents an internally inconsistent response from placing a selected "acceptable"
    # candidate above an explicitly "preferred" candidate.
    if candidate.ai_suitability == "preferred":
        return 5 if candidate.ai_selected else 4
    if candidate.ai_suitability == "acceptable":
        return 3 if candidate.ai_selected else 2
    if candidate.ai_suitability == "rejected":
        return 0
    return 1


def finalize_scores(candidates: list[Candidate], settings: Settings, ai_was_used: bool = False) -> list[Candidate]:
    for candidate in candidates:
        pre_ai = candidate.final_score if candidate.final_score > 0 else candidate.local_score
        if ai_was_used and candidate.ai_reviewed:
            procedure = (candidate.ai_visual_score or 0) / 100.0
            context = (candidate.ai_context_score or 0) / 100.0
            conditions = (candidate.ai_condition_score or 0) / 100.0
            score = (
                0.14 * pre_ai
                + 0.07 * candidate.learned_ranker_score
                + 0.05 * candidate.reviewer_memory_score
                + 0.34 * procedure
                + 0.18 * context
                + 0.06 * conditions
                + 0.05 * candidate.relay_semantic_score
                + 0.04 * candidate.state_semantic_score
                + 0.04 * candidate.sequence_continuity_score
                + 0.05 * candidate.constraint_match_score
            )
            # The AI is the final scientific/VTT judge. Selection and suitability should
            # dominate arbitrary legacy score thresholds, while contradictions still win.
            if candidate.ai_selected and candidate.ai_suitability in {"preferred", "acceptable"}:
                score += 0.10
            if candidate.ai_suitability == "preferred":
                score += 0.08
            elif candidate.ai_suitability == "acceptable":
                score += 0.03
            elif candidate.ai_suitability == "rejected":
                score = min(score, 0.34)
            if candidate.ai_visual_uncertainty:
                score -= 0.015
        else:
            score = pre_ai

        if candidate.critical_contradiction:
            score *= 0.38
        elif candidate.contradictions:
            # Non-critical numeric/context differences are review notes, not automatic vetoes.
            score *= 0.97

        if candidate.vtt_evidence_risk == "high":
            score *= 0.92

        if (
            candidate.reviewer_memory_label == "rejected"
            and candidate.reviewer_memory_similarity >= settings.reviewer_memory_exact_similarity
        ):
            score = min(score, 0.40)

        candidate.final_score = max(0.0, min(1.0, score))

    if ai_was_used:
        candidates.sort(key=lambda item: (_ai_tier(item), item.final_score), reverse=True)
    else:
        candidates.sort(key=lambda item: item.final_score, reverse=True)
    return candidates


def classify_match(
    candidates: list[Candidate],
    settings: Settings,
    ai_no_match_recommended: bool,
    ai_no_match_reason: str,
    ai_was_used: bool,
) -> tuple[str, str, str | None]:
    if not candidates:
        return "NO_MATCH", "No reliable match", "No candidate caption passages were found."

    top = candidates[0]
    second_score = candidates[1].final_score if len(candidates) > 1 else 0.0
    gap = top.final_score - second_score
    procedure_score = top.ai_visual_score if top.ai_visual_score is not None else round(top.feature_score * 100)
    context_score = top.ai_context_score if top.ai_context_score is not None else round(top.semantic_score * 100)

    if top.critical_contradiction:
        return "NO_MATCH", "No reliable match", "The strongest VTT candidate contains a hard footage/setup contradiction."

    if ai_was_used:
        non_rejected = [candidate for candidate in candidates if candidate.ai_suitability in {"preferred", "acceptable"} and not candidate.critical_contradiction]
        # Only honor an AI no-match when the assessments themselves agree that nothing is usable.
        if not non_rejected:
            return (
                "NO_MATCH",
                "No reliable match",
                ai_no_match_reason or "The scientific reranker rejected all reviewed VTT candidates.",
            )

        # Preferred/acceptable is authoritative for the team-validation build. Legacy local
        # thresholds must not convert a human-readable preferred result into NO_MATCH.
        if top.ai_suitability == "preferred":
            strong = (procedure_score >= 82 and context_score >= 65 and not top.ai_visual_uncertainty and top.vtt_evidence_risk != "high")
            confidence = "High" if strong else "Medium"
            notes: list[str] = []
            if top.ai_visual_uncertainty or top.vtt_evidence_risk != "low":
                notes.append("VTT evidence supports the match, but some visual/setup detail remains unverified because frames are not inspected.")
            if top.sequence_continuity_score >= 0.45:
                notes.append("The source also supports neighbouring protocol context.")
            if top.sequence_reason:
                notes.append(top.sequence_reason)
            return "MATCH", confidence, " ".join(notes) or None

        if top.ai_suitability == "acceptable":
            # A usable footage match is intentionally retained despite soft numeric/reagent differences.
            if procedure_score < 50 and top.action_match < 0.40:
                return "NO_MATCH", "No reliable match", "The acceptable candidate still lacks enough VTT evidence for the requested laboratory action."
            confidence = "Medium" if procedure_score >= 70 and context_score >= 55 else "Low"
            notes = ["Usable VTT match with one or more non-critical protocol/detail differences; review the noted differences before final footage selection."]
            if top.ai_visual_uncertainty or top.vtt_evidence_risk != "low":
                notes.append("No frame-level verification was performed.")
            if top.sequence_reason:
                notes.append(top.sequence_reason)
            return "MATCH", confidence, " ".join(notes)

        return "NO_MATCH", "No reliable match", "The highest-ranked reviewed VTT candidate was rejected."

    # Local-only fallback retains conservative thresholds, with a slightly lower floor
    # than V3 because the score scale changed after context-relay features were added.
    exact_memory_rejections = [
        candidate for candidate in candidates
        if candidate.reviewer_memory_label == "rejected"
        and candidate.reviewer_memory_similarity >= settings.reviewer_memory_exact_similarity
    ]
    if exact_memory_rejections and len(exact_memory_rejections) == len(candidates):
        return "NO_MATCH", "No reliable match", "All candidates closely match reviewer-rejected examples."

    if top.final_score < settings.low_threshold:
        return "NO_MATCH", "No reliable match", "The strongest candidate is below the minimum local evidence threshold."
    if procedure_score < settings.minimum_visual_score and top.action_match < 0.45:
        return "NO_MATCH", "No reliable match", "No VTT candidate supports a sufficiently similar laboratory action."
    if top.vtt_evidence_risk == "high" and procedure_score < 65:
        return "NO_MATCH", "No reliable match", "The VTT evidence is too weak or ambiguous for the requested setup/action."

    if (
        top.final_score >= settings.high_threshold
        and gap >= settings.minimum_score_gap
        and top.vtt_evidence_risk != "high"
    ):
        return "MATCH", "High", None

    confidence = "Medium" if top.final_score >= settings.medium_threshold else "Low"
    notes: list[str] = []
    if gap < settings.minimum_score_gap:
        notes.append("The top candidates are close; alternatives should be reviewed.")
    if top.vtt_evidence_risk != "low":
        notes.append("VTT evidence does not confirm every critical setup detail; no frame-level verification was performed.")
    if top.sequence_continuity_score >= 0.45:
        notes.append("The source also has supporting evidence in neighbouring protocol steps.")
    if top.sequence_reason:
        notes.append(top.sequence_reason)
    return "MATCH", confidence, " ".join(notes) or None

