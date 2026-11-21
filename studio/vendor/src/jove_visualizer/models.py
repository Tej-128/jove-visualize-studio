from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass(slots=True)
class ArticleMetadata:
    article_id: str
    article_title: str
    sections: list[str]
    article_url: str | None = None


@dataclass(slots=True)
class CaptionCue:
    start_ms: int
    end_ms: int
    text: str


@dataclass(slots=True)
class Passage:
    passage_id: int | None
    article_id: str
    article_title: str
    sections: list[str]
    start_ms: int
    end_ms: int
    text: str
    normalized_text: str
    source_file: str


@dataclass(slots=True)
class Candidate:
    passage_id: int
    article_id: str
    article_title: str
    sections: list[str]
    start_ms: int
    end_ms: int
    text: str
    source_file: str
    semantic_score: float = 0.0
    context_semantic_score: float = 0.0
    lexical_score: float = 0.0
    feature_score: float = 0.0
    local_score: float = 0.0
    action_match: float = 0.0
    entity_match: float = 0.0
    condition_match: float = 0.0
    numeric_match: float = 0.0
    contradictions: list[str] = field(default_factory=list)
    critical_contradiction: bool = False
    retrieval_fusion: float = 0.0
    retrieval_source_coverage: float = 0.0
    retrieval_best_rank: float = 0.0
    article_support: float = 0.0
    learned_ranker_score: float = 0.5
    learned_article_score: float = 0.5
    reviewer_memory_score: float = 0.5
    reviewer_memory_similarity: float = 0.0
    reviewer_memory_label: str | None = None
    reviewer_memory_note: str | None = None

    # V3 contextual / relay signals. These are generic signals, never Journal-ID rules.
    relay_semantic_score: float = 0.0
    protocol_semantic_score: float = 0.0
    phase_semantic_score: float = 0.0
    state_semantic_score: float = 0.0
    constraint_match_score: float = 0.5
    sequence_continuity_score: float = 0.0
    vtt_evidence_risk: str = "medium"

    # V3.2 global sequence planning. These are calculated from the current run only.
    source_coverage_score: float = 0.0
    sequence_transition_score: float = 0.0
    sequence_path_score: float = 0.0
    sequence_path_selected: bool = False
    sequence_reason: str | None = None

    # Kept for backward compatibility with the V2 report/schema. In V3+ these fields
    # mean VTT-supported procedure/context evidence; they do NOT imply video inspection.
    ai_visual_score: float | None = None
    ai_context_score: float | None = None
    ai_condition_score: float | None = None
    ai_suitability: str | None = None
    ai_visual_uncertainty: bool = False
    ai_reason: str | None = None
    ai_reviewed: bool = False
    ai_selected: bool = False
    final_score: float = 0.0


@dataclass(slots=True)
class ScriptStep:
    number: int
    text: str
    previous_text: str = ""
    next_text: str = ""
    script_title: str = ""
    production_notes: list[str] = field(default_factory=list)

    # Persistent protocol context. Built generically from the submitted script; no IDs are stored.
    protocol_outline: str = ""
    protocol_signature: str = ""
    protocol_phase: str = ""
    state_before: str = ""
    state_after: str = ""
    state_transition: str = ""
    experiment_state_summary: str = ""
    relay_text: str = ""
    active_entities: list[str] = field(default_factory=list)
    core_entities: list[str] = field(default_factory=list)
    core_action: str = ""
    hard_constraints: list[str] = field(default_factory=list)
    soft_constraints: list[str] = field(default_factory=list)
    on_screen_text: list[str] = field(default_factory=list)
    narration_text: str = ""


@dataclass(slots=True)
class MatchResult:
    step: ScriptStep
    status: str
    confidence: str
    best: Candidate | None
    alternatives: list[Candidate]
    additional_timestamps_same_id: list[Candidate]
    no_match_reason: str | None
    elapsed_seconds: float


@dataclass(slots=True)
class AuditSummary:
    metadata_rows: int
    metadata_unique_ids: int
    vtt_files: int
    vtt_unique_ids: int
    matched_ids: int
    metadata_without_vtt: int
    vtt_without_metadata: int
    duplicate_vtt_ids: int
    malformed_vtts: int
    empty_vtts: int
    nonstandard_filenames: int
    output_dir: Path


class AICandidateAssessment(BaseModel):
    candidate_key: str = Field(description="Exact candidate key supplied in the request")
    suitability: Literal["preferred", "acceptable", "rejected"]
    visual_procedure_match: int = Field(ge=0, le=100)
    scientific_context_match: int = Field(ge=0, le=100)
    conditions_match: int = Field(ge=0, le=100)
    critical_contradiction: bool
    contradictions: list[str]
    visual_uncertainty: bool
    reason: str


class AIRerankResponse(BaseModel):
    candidate_assessments: list[AICandidateAssessment]
    selected_candidate_key: str
    no_match_recommended: bool
    no_match_reason: str


@dataclass(slots=True)
class RerankOutcome:
    ai_used: bool
    no_match_recommended: bool
    no_match_reason: str
    selected_candidate_key: str = ""
    reviewed_passage_ids: set[int] = field(default_factory=set)


@dataclass(slots=True)
class IndexManifest:
    version: str
    created_at: str
    passage_count: int
    embedding_dim: int
    embedding_backend: str
    embedding_model: str
    metadata_file: str
    metadata_sha256: str
    vtt_folder: str
    vtt_count: int
    settings: dict[str, Any]

