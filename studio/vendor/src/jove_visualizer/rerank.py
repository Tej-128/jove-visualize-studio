from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .models import AIRerankResponse, Candidate, RerankOutcome, ScriptStep


PROMPT_VERSION = "jove-v3.3-generalization-safe-2026-08"

SYSTEM_INSTRUCTIONS = """
You are the final VTT caption-evidence judge for JoVE Visualizer.
The goal is to find reusable Journal footage segments from VTT captions for a production-script step. You do NOT see video frames, so judge footage usability from caption evidence rather than demanding textual identity with the script.

Classify every supplied candidate as:
- preferred: strongest reusable caption evidence for the requested action/setup and sequence.
- acceptable: footage is usable for the requested action even if non-visible scientific details differ.
- rejected: wrong main action, explicitly incompatible visible/setup-defining structure, misleading extra procedure, or caption evidence too weak to support the requested footage.

Decision rules:
1. Reason over the ENTIRE protocol outline, not the current sentence in isolation.
2. Treat the previous step as the state entering the current step. The current action should logically follow from that state. Use the next step as supporting context.
3. Track persistent entities, materials, specimens, containers, instruments, and setup terms dynamically from the submitted script. Resolve references such as "it", "the sample", "the container", or "the solution" from the preceding protocol state.
4. Use the explicit experiment state transition (before -> action -> after) to interpret vague steps.
5. Prefer a scientifically suitable Journal source that can support neighbouring steps; source continuity is a tie-breaker only and can never rescue a scientifically inferior or rejected candidate.
6. HARD contradictions must be explicit and setup/action defining: incompatible primary action, explicitly different structural format/capacity when both are stated, wrong specimen/object when clearly stated, or an additional operation that would make the requested footage misleading. Do not invent a hard contradiction from missing caption detail.
7. SOFT protocol details include exact volume, concentration, pH, formulation, repetition count, duration, and temperature unless that detail defines the visible setup or is intrinsic to the requested action. Soft differences lower the condition score and belong in the reason, but do not by themselves force rejection.
8. For time/temperature-dependent actions, duration and temperature can become defining evidence. Distinguish generic reusable process footage from a truly incompatible procedure.
9. Exact reagent identity is contextual evidence. Prefer the correct scientific context, but do not reject otherwise reusable action footage solely because a different analyte/reagent is named unless that changes the visible procedure or object.
10. Use script-derived entities and state memory; do not rely on any fixed assay, organism, reagent, technique, or Journal-specific assumptions.
11. Reviewer-memory labels and learned-ranker scores are soft evidence from prior reviewed examples, never answer keys and never Journal-ID rules.
12. Because only VTT text is available, never claim frame-level verification. Set visual_uncertainty=true when a critical visible/setup detail is not explicit in the captions.
13. Select exactly one candidate when at least one is preferred or acceptable. Recommend no match ONLY when every reviewed candidate is rejected or genuinely too ambiguous to support the requested footage.
14. Use only supplied candidate keys. Never invent a Journal ID, timestamp, title, passage, entity, or condition.
15. Give concise reasons covering action fit, state/sequence fit, explicit hard conflicts, soft-detail differences, and VTT uncertainty.
""".strip()


def _schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_key": {"type": "string"},
                        "suitability": {"type": "string", "enum": ["preferred", "acceptable", "rejected"]},
                        "visual_procedure_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "scientific_context_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "conditions_match": {"type": "integer", "minimum": 0, "maximum": 100},
                        "critical_contradiction": {"type": "boolean"},
                        "contradictions": {"type": "array", "items": {"type": "string"}},
                        "visual_uncertainty": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "candidate_key", "suitability", "visual_procedure_match",
                        "scientific_context_match", "conditions_match", "critical_contradiction",
                        "contradictions", "visual_uncertainty", "reason",
                    ],
                },
            },
            "selected_candidate_key": {"type": "string"},
            "no_match_recommended": {"type": "boolean"},
            "no_match_reason": {"type": "string"},
        },
        "required": [
            "candidate_assessments", "selected_candidate_key",
            "no_match_recommended", "no_match_reason",
        ],
    }


class RerankCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rerank_cache_v3 (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, key: str) -> AIRerankResponse | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT response_json FROM rerank_cache_v3 WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        try:
            return AIRerankResponse.model_validate_json(row[0])
        except ValidationError:
            return None

    def set(self, key: str, response: AIRerankResponse) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO rerank_cache_v3(cache_key, response_json) VALUES (?, ?)",
                (key, response.model_dump_json()),
            )
            connection.commit()


def _cache_key(model: str, step: ScriptStep, candidates: list[Candidate]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "title": step.script_title,
        "step": step.text,
        "previous": step.previous_text,
        "next": step.next_text,
        "protocol_signature": step.protocol_signature,
        "protocol_phase": step.protocol_phase,
        "state_before": step.state_before,
        "state_after": step.state_after,
        "state_transition": step.state_transition,
        "experiment_state_summary": step.experiment_state_summary,
        "core_action": step.core_action,
        "core_entities": step.core_entities,
        "relay_text": step.relay_text,
        "candidates": [
            {
                "key": f"p{candidate.passage_id}",
                "text": candidate.text,
                "pre_ai": round(candidate.final_score, 6),
                "learned": round(candidate.learned_ranker_score, 6),
                "memory_label": candidate.reviewer_memory_label,
                "memory_similarity": round(candidate.reviewer_memory_similarity, 6),
            }
            for candidate in candidates
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class ScientificReranker:
    def __init__(
        self,
        settings: Settings,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.api_key = api_key
        self.cache = RerankCache(cache_path) if cache_path else None

    def _local_fallback(self, candidates: list[Candidate], reason_prefix: str = "Local assessment") -> None:
        for candidate in candidates:
            candidate.ai_visual_score = round(100 * (
                0.46 * candidate.feature_score
                + 0.24 * candidate.semantic_score
                + 0.18 * candidate.relay_semantic_score
                + 0.12 * candidate.constraint_match_score
            ))
            candidate.ai_context_score = round(100 * (
                0.34 * candidate.semantic_score
                + 0.30 * candidate.context_semantic_score
                + 0.22 * candidate.protocol_semantic_score
                + 0.14 * candidate.sequence_continuity_score
            ))
            candidate.ai_condition_score = round(100 * candidate.condition_match)
            candidate.ai_suitability = None
            candidate.ai_reason = (
                f"{reason_prefix}: action compatibility {candidate.action_match:.2f}; "
                f"contextual similarity {candidate.semantic_score:.2f}."
            )
            candidate.ai_reviewed = False

    def rerank(self, step: ScriptStep, candidates: list[Candidate]) -> RerankOutcome:
        if not candidates:
            return RerankOutcome(False, True, "No candidate passages were retrieved.")
        selected = candidates[: self.settings.ai_candidates_per_step]
        if not self.settings.openai_enabled or not self.api_key:
            self._local_fallback(selected, "OpenAI disabled or API key unavailable")
            return RerankOutcome(False, False, "Local-only scoring was used.")

        key = _cache_key(self.settings.openai_model, step, selected)
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                self._apply_response(selected, cached)
                return RerankOutcome(
                    True,
                    cached.no_match_recommended,
                    cached.no_match_reason,
                    cached.selected_candidate_key,
                    {candidate.passage_id for candidate in selected},
                )

        try:
            from openai import OpenAI
        except ImportError:
            self._local_fallback(selected, "OpenAI SDK unavailable")
            return RerankOutcome(False, False, "OpenAI SDK unavailable; local-only scoring was used.")

        client = OpenAI(
            api_key=self.api_key,
            timeout=self.settings.openai_timeout_seconds,
            # The application retry loop below owns the attempt limit.
            max_retries=0,
        )
        candidate_payload = []
        for candidate in selected:
            memory = None
            if candidate.reviewer_memory_similarity >= self.settings.reviewer_memory_min_similarity:
                memory = {
                    "label": candidate.reviewer_memory_label,
                    "similarity": round(candidate.reviewer_memory_similarity, 3),
                    "reviewer_note": candidate.reviewer_memory_note,
                }
            candidate_payload.append({
                "candidate_key": f"p{candidate.passage_id}",
                "journal_id": candidate.article_id,
                "title": candidate.article_title,
                "sections": candidate.sections,
                "caption_passage": candidate.text,
                "pre_ai_score": round(candidate.final_score, 4),
                "learned_ranker_score": round(candidate.learned_ranker_score, 4),
                "reviewer_memory": memory,
                "relay_semantic_score": round(candidate.relay_semantic_score, 4),
                "protocol_semantic_score": round(candidate.protocol_semantic_score, 4),
                "constraint_match_score": round(candidate.constraint_match_score, 4),
                "sequence_continuity_score": round(candidate.sequence_continuity_score, 4),
                "state_semantic_score": round(candidate.state_semantic_score, 4),
                "vtt_evidence_risk": candidate.vtt_evidence_risk,
                "local_detected_contradictions": candidate.contradictions,
            })
        request_payload = {
            "protocol_title": step.script_title,
            "protocol_outline": step.protocol_outline,
            "protocol_signature": step.protocol_signature,
            "protocol_phase": step.protocol_phase,
            "state_entering_current_step": step.state_before,
            "state_after_current_step": step.state_after,
            "state_transition": step.state_transition,
            "experiment_state_summary": step.experiment_state_summary,
            "core_action": step.core_action,
            "core_entities": step.core_entities,
            "active_entities": step.active_entities,
            "hard_visual_constraints": step.hard_constraints,
            "soft_protocol_details": step.soft_constraints,
            "current_step": step.text,
            "previous_step": step.previous_text,
            "next_step": step.next_text,
            "relay_context": step.relay_text,
            "candidates": candidate_payload,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.settings.openai_max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.settings.openai_model,
                    instructions=SYSTEM_INSTRUCTIONS,
                    input=json.dumps(request_payload, ensure_ascii=False),
                    store=self.settings.store_openai_responses,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "jove_visualizer_v3_vtt_context_rerank",
                            "description": "VTT-only sequence-aware scientific assessment of caption candidates.",
                            "strict": True,
                            "schema": _schema(),
                        }
                    },
                )
                parsed = AIRerankResponse.model_validate_json(response.output_text)
                self._apply_response(selected, parsed)
                if self.cache:
                    self.cache.set(key, parsed)
                return RerankOutcome(
                    True,
                    parsed.no_match_recommended,
                    parsed.no_match_reason,
                    parsed.selected_candidate_key,
                    {candidate.passage_id for candidate in selected},
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.openai_max_retries:
                    time.sleep(min(2 ** attempt, 8))

        self._local_fallback(
            selected,
            f"OpenAI fallback after error: {type(last_error).__name__ if last_error else 'unknown'}",
        )
        return RerankOutcome(False, False, "OpenAI reranking failed; local-only scoring was used.")

    @staticmethod
    def _apply_response(candidates: list[Candidate], response: AIRerankResponse) -> None:
        allowed = {f"p{candidate.passage_id}": candidate for candidate in candidates}
        seen: set[str] = set()
        for assessment in response.candidate_assessments:
            candidate = allowed.get(assessment.candidate_key)
            if not candidate:
                continue
            seen.add(assessment.candidate_key)
            candidate.ai_visual_score = assessment.visual_procedure_match
            candidate.ai_context_score = assessment.scientific_context_match
            candidate.ai_condition_score = assessment.conditions_match
            candidate.ai_suitability = assessment.suitability
            candidate.ai_visual_uncertainty = assessment.visual_uncertainty
            candidate.ai_reason = assessment.reason
            candidate.ai_reviewed = True
            candidate.critical_contradiction = candidate.critical_contradiction or assessment.critical_contradiction
            candidate.contradictions = list(dict.fromkeys(candidate.contradictions + assessment.contradictions))
            candidate.ai_selected = assessment.candidate_key == response.selected_candidate_key

        for key, candidate in allowed.items():
            if key not in seen:
                candidate.ai_visual_score = round(candidate.feature_score * 100)
                candidate.ai_context_score = round(candidate.semantic_score * 100)
                candidate.ai_condition_score = round(candidate.condition_match * 100)
                candidate.ai_suitability = "rejected"
                candidate.ai_reason = "The AI response omitted this candidate; it was not selected."
                candidate.ai_reviewed = True

