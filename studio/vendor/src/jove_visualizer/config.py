from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    embedding_backend: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 128
    embedding_device: str = "cpu"
    target_passage_words: int = 75
    max_passage_words: int = 120
    max_passage_seconds: int = 45
    overlap_seconds: int = 12

    # Broad VTT retrieval.
    semantic_top_k: int = 300
    lexical_top_k: int = 260
    article_title_top_k: int = 110
    broad_candidate_pool: int = 180
    shortlist_per_step: int = 30
    max_passages_per_article: int = 5
    ai_candidates_per_step: int = 12

    output_alternatives: int = 3
    openai_enabled: bool = True
    openai_model: str = "gpt-5-mini"
    openai_max_concurrency: int = 3
    openai_timeout_seconds: int = 120
    openai_max_retries: int = 3

    # Learned and reviewer-feedback layers remain soft signals only.
    learned_ranker_enabled: bool = True
    learned_ranker_path: str = "learning_output/learned_ranker_v3_candidate.joblib"
    confidence_calibration_path: str = "learning_output/confidence_calibration_v3_candidate.json"
    reviewer_feedback_enabled: bool = True
    reviewer_feedback_path: str = "feedback/reviewer_feedback_memory.json"
    learned_signal_weight: float = 0.10
    reviewer_memory_weight: float = 0.12
    reviewer_memory_min_similarity: float = 0.72
    reviewer_memory_exact_similarity: float = 0.88

    # V3.3 generalization-safe context relay: runtime script state + source continuity.
    context_relay_enabled: bool = True
    relay_semantic_weight: float = 0.12
    protocol_semantic_weight: float = 0.08
    phase_semantic_weight: float = 0.05
    hard_constraint_weight: float = 0.08
    adjacent_article_boost: float = 0.07
    global_article_boost: float = 0.035
    continuity_min_local_score: float = 0.48
    continuity_top_articles_per_step: int = 10

    # Conservative whole-script optimization. These are intentionally small bonuses;
    # local scientific suitability remains the primary decision signal.
    sequence_optimizer_enabled: bool = True
    sequence_candidates_per_step: int = 8
    sequence_max_local_drop: float = 0.04
    sequence_min_candidate_score: float = 0.42
    source_coverage_min_score: float = 0.48
    sequence_source_coverage_weight: float = 0.05
    sequence_continuity_bonus: float = 0.07

    high_threshold: float = 0.74
    medium_threshold: float = 0.58
    low_threshold: float = 0.46
    # Legacy field name retained for compatibility; this is VTT-supported procedure evidence, not frame inspection.
    minimum_visual_score: int = 50
    minimum_score_gap: float = 0.04
    include_unmapped_vtts: bool = False
    store_openai_responses: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    values: dict = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)

    env_overrides = {
        "openai_model": os.getenv("OPENAI_MODEL"),
        "embedding_model": os.getenv("JOVE_EMBEDDING_MODEL"),
        "embedding_device": os.getenv("JOVE_EMBEDDING_DEVICE"),
    }
    for key, value in env_overrides.items():
        if value:
            values[key] = value

    known = {field_name for field_name in Settings.__dataclass_fields__}
    clean = {key: value for key, value in values.items() if key in known}
    return Settings(**clean)

