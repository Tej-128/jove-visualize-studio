from __future__ import annotations

import re
from collections import Counter, deque

from .features import ACTION_GROUPS, extract_scientific_features
from .models import ScriptStep
from .normalizer import normalize_text


# Grammar/measurement words only. Novel scientific nouns are intentionally retained.
GENERIC_TERMS = {
    "first", "then", "next", "following", "final", "using", "used", "desired", "appropriate",
    "procedure", "step", "amount", "time", "times", "minute", "minutes", "hour", "hours",
    "second", "seconds", "day", "days", "onscreen", "scientific", "detail", "degrees",
    "celsius", "fahrenheit", "ug", "ml", "ul", "nm", "um", "mm", "rpm",
}


def _action_display(action: str) -> str:
    return action[4:] if action.startswith("raw:") else action


def _ordered_actions(text: str) -> list[str]:
    raw = (text or "").strip()
    lower = raw.lower()
    if "," in raw and re.match(r"^(after|following|once|afterward|next|then|when)\b", lower):
        raw = raw.split(",", 1)[1]

    features = extract_scientific_features(raw)
    normalized = normalize_text(raw)
    found: list[tuple[int, int, str]] = []
    for action in features.actions:
        if action.startswith("raw:"):
            root = action[4:]
            pos = normalized.find(root)
            found.append((pos if pos >= 0 else 10_000, 1, action))
            continue
        positions = [
            normalized.find(phrase)
            for phrase in ACTION_GROUPS.get(action, (action,))
            if normalized.find(phrase) >= 0
        ]
        pos = min(positions) if positions else 10_000
        # Canonical action families win ties over raw fallback tokens.
        found.append((pos, 0, action))

    ordered: list[str] = []
    for _, _, action in sorted(found):
        if action not in ordered:
            ordered.append(action)
    return ordered


def _phase_for_step(step: ScriptStep) -> str:
    actions = _ordered_actions(step.text)
    if not actions:
        return "procedure"
    return f"{_action_display(actions[0])} phase"


def _constraint_groups(text: str) -> tuple[list[str], list[str]]:
    features = extract_scientific_features(text)
    hard: list[str] = []
    soft: list[str] = []

    # Only explicit structural formats are local hard constraints. This is generic: any
    # N-well format can be compared; no particular count or technique is encoded.
    for item in sorted(features.plate_formats):
        hard.append(f"explicit_structure={item}")

    for item in sorted(features.qualitative_conditions):
        soft.append(f"condition={item}")
    for kind, values in features.conditions.items():
        for value, unit in values:
            soft.append(f"{kind}={value:g}{unit}")
    for repetition in sorted(features.repetitions):
        soft.append(f"repetitions={repetition}")

    return list(dict.fromkeys(hard)), list(dict.fromkeys(soft))


def _important_terms(text: str) -> list[str]:
    features = extract_scientific_features(text)
    canonical_words = {
        word
        for phrases in ACTION_GROUPS.values()
        for phrase in phrases
        for word in phrase.split()
    }

    multiword = [
        value for value in features.entity_phrases
        if " " in value and value not in GENERIC_TERMS
    ]
    single = [
        value for value in features.terms
        if value not in GENERIC_TERMS and value not in canonical_words and len(value) >= 3
    ]

    ordered = sorted(multiword, key=lambda value: (len(value.split()), len(value), value), reverse=True)
    ordered.extend(sorted(single, key=lambda value: (len(value), value), reverse=True))
    ordered.extend(sorted(features.plate_formats))
    return list(dict.fromkeys(ordered))[:24]


def _compact_event(step_number: int, action: str, phase: str, entities: list[str]) -> str:
    entity_text = ", ".join(entities[:5]) if entities else "current material/setup"
    action_text = _action_display(action) if action else phase
    return f"Step {step_number}: {action_text} [{entity_text}]"


def build_protocol_context(steps: list[ScriptStep]) -> list[ScriptStep]:
    """Build technique-agnostic full-protocol context and persistent experiment state.

    Everything is derived from the submitted script at runtime. The production logic contains
    no approved Journal IDs, timestamps, named training protocols, assay-specific shortcuts, or
    source-specific answer rules.
    """
    if not steps:
        return steps

    title = next((step.script_title.strip() for step in steps if step.script_title.strip()), "")
    outline = "\n".join(f"{step.number}. {step.text}" for step in steps)

    term_counter: Counter[str] = Counter()
    action_sequence: list[str] = []
    per_step_terms: list[list[str]] = []
    per_step_actions: list[list[str]] = []

    for step in steps:
        terms = _important_terms(step.text)
        actions = _ordered_actions(step.text)
        per_step_terms.append(terms)
        per_step_actions.append(actions)
        term_counter.update(terms)
        action_sequence.extend(actions)

    key_entities = [
        term for term, _ in sorted(
            term_counter.items(),
            key=lambda item: (item[1], len(item[0].split()), len(item[0]), item[0]),
            reverse=True,
        )[:28]
    ]

    compact_actions: list[str] = []
    for action in action_sequence:
        display = _action_display(action)
        if not compact_actions or compact_actions[-1] != display:
            compact_actions.append(display)

    signature_parts: list[str] = []
    if title:
        signature_parts.append(f"Protocol: {title}")
    if compact_actions:
        signature_parts.append("Procedure sequence: " + " -> ".join(compact_actions[:32]))
    if key_entities:
        signature_parts.append("Script-derived entities/materials/setup: " + ", ".join(key_entities))
    protocol_signature = ". ".join(signature_parts)

    recent_entities: deque[str] = deque(maxlen=24)
    event_history: deque[str] = deque(maxlen=5)

    for index, step in enumerate(steps):
        previous = steps[index - 1].text if index > 0 else ""
        previous_two = steps[index - 2].text if index > 1 else ""
        following = steps[index + 1].text if index + 1 < len(steps) else ""
        actions = per_step_actions[index]
        current_terms = per_step_terms[index]
        phase = _phase_for_step(step)
        core_action = _action_display(actions[0]) if actions else "procedure"

        before_entities = list(dict.fromkeys(recent_entities))[-14:]
        before_events = list(event_history)[-3:]
        state_before_parts: list[str] = []
        if before_events:
            state_before_parts.append("Recent state: " + " -> ".join(before_events))
        if before_entities:
            state_before_parts.append("Persistent script-derived entities/setup: " + ", ".join(before_entities))
        if not state_before_parts and previous:
            state_before_parts.append(f"Immediately previous: {previous}")
        state_before = " | ".join(state_before_parts)

        for entity in current_terms:
            if entity in recent_entities:
                try:
                    recent_entities.remove(entity)
                except ValueError:
                    pass
            recent_entities.append(entity)

        active_entities = list(dict.fromkeys(list(recent_entities)[-14:]))
        core_entities = current_terms[:10] or active_entities[:6]
        event = _compact_event(step.number, actions[0] if actions else core_action, phase, core_entities)
        event_history.append(event)

        state_after_parts = ["Recent state: " + " -> ".join(list(event_history)[-3:])]
        if active_entities:
            state_after_parts.append("Active script-derived entities/setup: " + ", ".join(active_entities))
        state_after = " | ".join(state_after_parts)

        transition_parts: list[str] = []
        if state_before:
            transition_parts.append(f"Before: {state_before}")
        transition_parts.append(f"Action now: {step.text}")
        if state_after:
            transition_parts.append(f"After: {state_after}")
        state_transition = " | ".join(transition_parts)

        relay_parts: list[str] = []
        if previous:
            relay_parts.append(f"After: {previous}")
        relay_parts.append(f"Now: {step.text}")
        if following:
            relay_parts.append(f"Then: {following}")

        experiment_summary_parts = [f"Current phase: {phase}"]
        if title:
            experiment_summary_parts.append(f"Protocol: {title}")
        if active_entities:
            experiment_summary_parts.append("Tracked script-derived entities/setup: " + ", ".join(active_entities))
        if event_history:
            experiment_summary_parts.append("Recent transitions: " + " -> ".join(list(event_history)))

        step.protocol_outline = outline
        step.protocol_signature = protocol_signature
        step.protocol_phase = phase
        step.state_before = state_before
        step.state_after = state_after
        step.state_transition = state_transition
        step.experiment_state_summary = " | ".join(experiment_summary_parts)
        step.relay_text = " | ".join(relay_parts)
        step.active_entities = active_entities
        step.core_entities = core_entities
        step.core_action = core_action
        step.hard_constraints, step.soft_constraints = _constraint_groups(step.text)

        if previous_two and not step.state_before:
            step.state_before = f"Two steps earlier: {previous_two} | Immediately previous: {previous}"

    return steps

