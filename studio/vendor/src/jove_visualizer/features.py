from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .normalizer import normalize_text


# Broad procedural verb families only. These are technique-agnostic language normalizers,
# not assay/protocol rules. Unknown verbs still participate through the raw-action fallback.
ACTION_GROUPS: dict[str, tuple[str, ...]] = {
    "prepare": ("prepare", "dilute", "formulate", "reconstitute"),
    "add": ("add", "load", "pipette", "dispense", "apply", "fill", "aliquot"),
    "mix": ("mix", "vortex", "agitate", "stir", "homogenize"),
    "incubate": ("incubate", "leave", "hold", "stand", "rest"),
    "wash": ("wash", "rinse"),
    "remove": ("remove", "discard", "decant", "aspirate", "empty", "blot"),
    "separate": ("centrifuge", "spin", "pellet", "separate"),
    "filter": ("filter", "filtrate"),
    "measure": ("measure", "read", "record", "quantify", "detect"),
    "transfer": ("transfer", "move", "place"),
    "seal": ("seal", "cover", "wrap"),
    "image": ("image", "visualize", "photograph", "scan"),
    "deliver": ("inject", "deliver", "administer", "infuse"),
    "dissect": ("dissect", "excise", "isolate"),
    "fix": ("fix", "fixate"),
    "stain": ("stain", "label", "dye"),
    "analyze": ("analyze", "calculate", "plot", "fit"),
    "heat": ("heat", "warm", "boil", "bake"),
    "cool": ("cool", "chill", "freeze", "thaw"),
    "dry": ("dry", "desiccate", "evaporate"),
    "cut": ("cut", "section", "slice", "trim"),
    "mount": ("mount", "position", "secure"),
    "collect": ("collect", "harvest", "recover"),
    "resuspend": ("resuspend", "redissolve"),
}

# Grammar/utility words only. Domain nouns are intentionally NOT suppressed so novel
# materials, instruments, organisms, and containers can become state entities automatically.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "then", "this", "that", "these", "those", "each", "into", "using", "use", "used", "is", "are",
    "was", "were", "be", "been", "being", "as", "it", "its", "after", "before", "next", "following",
    "per", "all", "any", "desired", "appropriate", "finally", "first", "second", "third", "step",
    "procedure", "approximately", "about", "around", "until", "while", "during", "via", "onto", "within",
    "further", "now", "ready", "already", "still",
}

AUXILIARIES = {"be", "is", "are", "was", "were", "been", "being", "to", "then", "and"}

CONDITION_PATTERNS = {
    "temperature": re.compile(r"\b(-?\d+(?:\.\d+)?)\s*(celsius|fahrenheit|c|f)\b"),
    "ph": re.compile(r"\bph\s*(\d+(?:\.\d+)?)\b"),
    "duration": re.compile(r"\b(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?|days?)\b"),
    "concentration": re.compile(r"\b(\d+(?:\.\d+)?)\s*(ug/ml|ug/ul|mg/ml|ng/ml|ng/ul|um|nm|mm|m|%)\b"),
    "volume": re.compile(r"\b(\d+(?:\.\d+)?)\s*(ul|ml|l)\b"),
    "rate": re.compile(r"\b(\d+(?:\.\d+)?)\s*(nl/min|ul/min|ml/min|rpm|g)\b"),
}

# Generic structural format extraction: any explicitly stated N-well format is comparable.
# No particular well count is enumerated or privileged.
PLATE_FORMAT_RE = re.compile(r"\b(\d{1,4})-well\b")
REPETITION_RE = re.compile(r"\b(\d+)\s*(?:times?|x)\b")


@dataclass(slots=True)
class ScientificFeatures:
    normalized: str
    actions: set[str]
    visual_contexts: set[str]
    equipment: set[str]
    terms: set[str]
    conditions: dict[str, list[tuple[float, str]]]
    plate_formats: set[str]
    qualitative_conditions: set[str]
    repetitions: set[int]
    entity_phrases: set[str]
    canonical_actions: set[str]


def _verb_root(token: str) -> str:
    token = token.lower().strip("-./")
    if len(token) > 6 and token.endswith(("ating", "izing", "ising")):
        return token[:-3] + "e"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith(("ated", "ized", "ised")):
        return token[:-1]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _find_canonical_actions(text: str) -> set[str]:
    found: set[str] = set()
    for name, phrases in ACTION_GROUPS.items():
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}(?:s|ed|ing)?\b", text):
                found.add(name)
                break
    return found


def _raw_action_fallback(text: str) -> set[str]:
    """Find a small number of likely verbs without a protocol-specific vocabulary.

    Raw actions are soft lexical evidence only; they never create a hard contradiction by
    themselves. The heuristic supports imperative steps, passive voice, and previously unseen
    verbs without maintaining a named-technique dictionary.
    """
    scan = re.sub(r"^(?:to\s+begin|then|next|finally|first|firstly|afterward|subsequently)\s*[,;:]?\s*", "", text)
    tokens = re.findall(r"[a-z][a-z0-9-]*", scan)
    if not tokens:
        return set()

    subject_led = tokens[0] in {"the", "a", "an", "this", "that", "these", "those", "it", "they", "we"}
    first_content_index = None
    if not subject_led:
        for index, token in enumerate(tokens):
            if token in STOPWORDS or token in AUXILIARIES or token.endswith("ly") or len(token) < 3:
                continue
            first_content_index = index
            break

    raw: list[str] = []
    for index, token in enumerate(tokens):
        if len(token) < 3 or token.endswith("ly"):
            continue
        root = _verb_root(token)
        previous = tokens[index - 1] if index else ""
        likely = (
            previous in AUXILIARIES
            or token.endswith(("ing", "ed"))
            or (first_content_index is not None and index == first_content_index)
        )
        if likely and root not in STOPWORDS and root not in AUXILIARIES and len(root) >= 3:
            raw.append(f"raw:{root}")
        if len(raw) >= 4:
            break
    return set(raw)


def _content_tokens(normalized: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z][a-z0-9+./-]{2,}", normalized)
    tokens = [token.strip(".-") for token in raw_tokens]
    action_words = {
        word
        for phrases in ACTION_GROUPS.values()
        for phrase in phrases
        for word in phrase.split()
    }
    return [
        token for token in tokens
        if token
        and token not in STOPWORDS
        and token not in action_words
        and not token.isdigit()
        and not re.fullmatch(r"\d+(?:\.\d+)?", token)
    ]


def _entity_phrases(tokens: list[str]) -> set[str]:
    phrases: set[str] = set(tokens)
    for size in (2, 3):
        for index in range(0, max(0, len(tokens) - size + 1)):
            chunk = tokens[index:index + size]
            if any("/" in token for token in chunk):
                continue
            phrases.add(" ".join(chunk))
    # Bound growth for long caption passages.
    return set(sorted(phrases, key=lambda value: (len(value.split()), len(value), value), reverse=True)[:48])


def extract_scientific_features(text: str) -> ScientificFeatures:
    normalized = normalize_text(text)
    canonical_actions = _find_canonical_actions(normalized)
    actions = set(canonical_actions)
    raw_actions = _raw_action_fallback(normalized)
    canonical_roots = {
        _verb_root(word)
        for action in canonical_actions
        for phrase in ACTION_GROUPS.get(action, ())
        for word in phrase.split()
    }
    actions.update(
        action for action in raw_actions
        if action[4:] not in canonical_roots
    )

    content_tokens = _content_tokens(normalized)
    raw_action_roots = {action[4:] for action in actions if action.startswith("raw:")}
    content_tokens = [token for token in content_tokens if _verb_root(token) not in raw_action_roots]
    terms = set(content_tokens)
    entity_phrases = _entity_phrases(content_tokens)

    conditions: dict[str, list[tuple[float, str]]] = {}
    for kind, pattern in CONDITION_PATTERNS.items():
        matches: list[tuple[float, str]] = []
        for match in pattern.finditer(normalized):
            value = float(match.group(1))
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else kind
            matches.append((value, unit))
        conditions[kind] = matches

    plate_formats = {f"{match.group(1)}-well" for match in PLATE_FORMAT_RE.finditer(normalized)}
    qualitative_conditions: set[str] = set()
    if "room-temperature" in normalized or "ambient temperature" in normalized:
        qualitative_conditions.add("room-temperature")
    if "overnight" in normalized:
        qualitative_conditions.add("overnight")
    if re.search(r"\bon ice\b|\bice-cold\b", normalized):
        qualitative_conditions.add("on-ice")
    if re.search(r"\bin the dark\b|\bdark conditions\b", normalized):
        qualitative_conditions.add("dark")

    repetitions = {int(match.group(1)) for match in REPETITION_RE.finditer(normalized)}

    # Kept for schema compatibility. They are derived dynamically rather than from named
    # assay/equipment dictionaries.
    visual_contexts = set(sorted(entity_phrases, key=lambda value: (len(value.split()), len(value)), reverse=True)[:12])
    equipment = set()

    return ScientificFeatures(
        normalized=normalized,
        actions=actions,
        visual_contexts=visual_contexts,
        equipment=equipment,
        terms=terms,
        conditions=conditions,
        plate_formats=plate_formats,
        qualitative_conditions=qualitative_conditions,
        repetitions=repetitions,
        entity_phrases=entity_phrases,
        canonical_actions=canonical_actions,
    )


def _main_clause(text: str) -> str:
    raw = (text or "").strip()
    lower = raw.lower()
    if "," in raw and re.match(r"^(after|following|once|afterward|next|then|when)\b", lower):
        return raw.split(",", 1)[1]
    return raw


def _primary_actions(text: str) -> set[str]:
    features = extract_scientific_features(_main_clause(text))
    return features.actions


def _primary_canonical_actions(text: str) -> set[str]:
    features = extract_scientific_features(_main_clause(text))
    return features.canonical_actions


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _normalize_condition(kind: str, value: float, unit: str) -> tuple[float, str]:
    unit = unit.lower()
    if kind == "temperature":
        if unit in {"f", "fahrenheit"}:
            return (value - 32.0) * 5.0 / 9.0, "celsius"
        return value, "celsius"
    if kind == "duration":
        scale = {"second": 1.0, "seconds": 1.0, "minute": 60.0, "minutes": 60.0,
                 "hour": 3600.0, "hours": 3600.0, "day": 86400.0, "days": 86400.0}
        return value * scale.get(unit, 1.0), "seconds"
    if kind == "volume":
        scale = {"ul": 1.0, "ml": 1000.0, "l": 1_000_000.0}
        return value * scale.get(unit, 1.0), "ul"
    return value, unit


def _condition_score(
    query: ScientificFeatures,
    candidate: ScientificFeatures,
    query_canonical_actions: set[str],
) -> tuple[float, float, list[str], bool]:
    score_parts: list[float] = []
    contradictions: list[str] = []
    critical = False
    condition_defining = bool(query_canonical_actions & {"incubate", "heat", "cool", "dry"})

    for kind in ("temperature", "ph", "duration", "concentration", "volume", "rate"):
        q_values = query.conditions.get(kind, [])
        c_values = candidate.conditions.get(kind, [])
        if not q_values:
            continue
        if not c_values:
            score_parts.append(0.35)
            continue

        best = 0.0
        compared = False
        for q_value, q_unit in q_values:
            q_norm, q_norm_unit = _normalize_condition(kind, q_value, q_unit)
            for c_value, c_unit in c_values:
                c_norm, c_norm_unit = _normalize_condition(kind, c_value, c_unit)
                if q_norm_unit != c_norm_unit:
                    continue
                compared = True
                relative = abs(q_norm - c_norm) / max(abs(q_norm), 1.0)
                pair_score = max(0.0, 1.0 - relative)
                best = max(best, pair_score)
                if relative >= 0.75:
                    contradictions.append(
                        f"{kind.capitalize()} differs substantially ({q_value:g} {q_unit} vs {c_value:g} {c_unit})."
                    )
                    if condition_defining and kind in {"temperature", "duration"}:
                        critical = True
        score_parts.append(best if compared else 0.2)

    if "overnight" in query.qualitative_conditions and condition_defining:
        if "overnight" in candidate.qualitative_conditions:
            score_parts.append(1.0)
        elif candidate.conditions.get("duration"):
            score_parts.append(0.1)
            contradictions.append("Requested an overnight hold/incubation but candidate states a bounded duration.")
            critical = True
        else:
            score_parts.append(0.35)

    if "room-temperature" in query.qualitative_conditions and condition_defining:
        if "room-temperature" in candidate.qualitative_conditions:
            score_parts.append(1.0)
        elif candidate.conditions.get("temperature"):
            temps = [_normalize_condition("temperature", value, unit)[0] for value, unit in candidate.conditions["temperature"]]
            if any(value <= 8 or value >= 35 for value in temps):
                score_parts.append(0.1)
                contradictions.append("Requested room-temperature conditions but candidate states a substantially different temperature.")
                critical = True
            else:
                score_parts.append(0.6)
        else:
            score_parts.append(0.35)

    if query.repetitions:
        if candidate.repetitions:
            if query.repetitions & candidate.repetitions:
                score_parts.append(1.0)
            else:
                score_parts.append(0.25)
                contradictions.append(
                    f"Repetition count differs ({sorted(query.repetitions)} vs {sorted(candidate.repetitions)})."
                )
        else:
            score_parts.append(0.35)

    condition_score = sum(score_parts) / len(score_parts) if score_parts else 0.5
    numeric_query = sum(len(values) for values in query.conditions.values()) + len(query.repetitions)
    numeric_candidate = sum(len(values) for values in candidate.conditions.values()) + len(candidate.repetitions)
    numeric_score = condition_score if numeric_query and numeric_candidate else (0.35 if numeric_query else 0.5)
    return condition_score, numeric_score, contradictions, critical


def score_feature_compatibility(query_text: str, candidate_text: str) -> dict[str, object]:
    query = extract_scientific_features(query_text)
    candidate = extract_scientific_features(candidate_text)
    query_actions = _primary_actions(query_text) or query.actions
    query_canonical = _primary_canonical_actions(query_text)
    candidate_actions = candidate.actions
    candidate_canonical = candidate.canonical_actions

    if query_actions and candidate_actions:
        action_match = _containment(query_actions, candidate_actions)
    elif not query_actions:
        action_match = 0.45
    else:
        action_match = 0.0

    # Dynamic context/entity overlap: no named assay categories or equipment vocabularies.
    context_match = _containment(query.entity_phrases, candidate.entity_phrases)
    if query.entity_phrases and not candidate.entity_phrases:
        context_match = 0.1
    elif not query.entity_phrases:
        context_match = 0.4

    term_overlap = _containment(query.terms, candidate.terms)
    phrase_overlap = _containment(
        {value for value in query.entity_phrases if " " in value},
        {value for value in candidate.entity_phrases if " " in value},
    )
    equipment_match = phrase_overlap if query.entity_phrases else 0.4

    condition_match, numeric_match, contradictions, critical = _condition_score(
        query, candidate, query_canonical
    )

    # Hard action contradiction only when BOTH sides contain recognized canonical procedural
    # actions. Raw/unknown verbs remain soft evidence so new techniques are not rejected merely
    # because the local vocabulary has never seen their action verb.
    if query_canonical and candidate_canonical and not (query_canonical & candidate_canonical):
        contradictions.append(
            "The primary narrated procedural action differs from the requested step "
            f"({', '.join(sorted(query_canonical))} vs {', '.join(sorted(candidate_canonical))})."
        )
        critical = True

    # Generic explicit structural mismatch. Any N-well counts are compared; none are hardcoded.
    if query.plate_formats and candidate.plate_formats and not (query.plate_formats & candidate.plate_formats):
        contradictions.append(
            f"Explicit well-format mismatch ({', '.join(sorted(query.plate_formats))} vs "
            f"{', '.join(sorted(candidate.plate_formats))})."
        )
        critical = True

    # Extra narrated actions are a bounded soft penalty. We do not maintain a protocol-specific
    # list of 'disruptive' operations; the final scientific judge receives the full action sets.
    extra_canonical = candidate_canonical - query_canonical if query_canonical else set()
    extra_action_penalty = min(0.16, 0.04 * len(extra_canonical))
    if len(extra_canonical) >= 2:
        contradictions.append("Candidate narrates additional procedural actions: " + ", ".join(sorted(extra_canonical)) + ".")

    score = (
        0.42 * action_match
        + 0.16 * context_match
        + 0.10 * equipment_match
        + 0.16 * term_overlap
        + 0.06 * phrase_overlap
        + 0.07 * condition_match
        + 0.03 * numeric_match
    ) - extra_action_penalty

    if critical:
        score *= 0.45
    elif contradictions:
        score *= 0.92

    evidence_risk = "low"
    if critical:
        evidence_risk = "high"
    elif query.plate_formats and not candidate.plate_formats:
        evidence_risk = "medium"
    elif query_canonical and action_match < 0.5:
        evidence_risk = "high"
    elif query.terms and term_overlap < 0.15:
        evidence_risk = "medium"

    constraint_match = max(0.0, min(1.0, 0.50 * condition_match + 0.30 * action_match + 0.20 * term_overlap))
    return {
        "action_match": float(action_match),
        "context_match": float(context_match),
        "equipment_match": float(equipment_match),
        "entity_match": float(term_overlap),
        "condition_match": float(condition_match),
        "numeric_match": float(numeric_match),
        "constraint_match": float(constraint_match),
        "feature_score": float(max(0.0, min(1.0, score))),
        "contradictions": list(dict.fromkeys(contradictions)),
        "critical_contradiction": critical,
        "vtt_evidence_risk": evidence_risk,
    }


def build_fts_query(text: str, max_terms: int = 18) -> str:
    features = extract_scientific_features(text)
    prioritized: list[str] = []
    prioritized.extend(sorted(features.canonical_actions))
    prioritized.extend(sorted(features.plate_formats))

    # Prefer multiword entities learned directly from this script, then informative tokens.
    prioritized.extend(sorted(
        (value for value in features.entity_phrases if " " in value),
        key=lambda value: (len(value.split()), len(value), value),
        reverse=True,
    ))
    prioritized.extend(sorted(features.terms, key=lambda term: (len(term), term), reverse=True))

    seen: set[str] = set()
    safe: list[str] = []
    for token in prioritized:
        token = re.sub(r"[^a-z0-9./+ -]", "", token.lower()).strip()
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        safe.append(f'"{token}"')
        if len(safe) >= max_terms:
            break
    return " OR ".join(safe)


def sigmoid_score(value: float, midpoint: float = 0.5, steepness: float = 8.0) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))

