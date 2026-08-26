"""Deterministic caption-SFX event planning for the experimental v1 contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


GLOBAL_COOLDOWN_MS = 1800
ROLLING_WINDOW_MS = 10_000
MAX_EVENTS_PER_WINDOW = 2
FOREIGN_SFX_CLEARANCE_MS = 650
COOLDOWN_GROUP = "caption_sfx"

_LEVEL_POLICY = {
    "strong": {"event": "emphasis", "lead_ms": 115, "strength": "strong"},
    "medium": {"event": "keyword_pop", "lead_ms": 95, "strength": "soft"},
}


class CaptionSFXPlanningError(ValueError):
    """Raised when an enabled caption-SFX plan receives malformed input."""


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptionSFXPlanningError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptionSFXPlanningError(f"{field} must be a finite number >= {minimum}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise CaptionSFXPlanningError(f"{field} must be a finite number >= {minimum}")
    return number


def _foreign_event_times_ms(existing_sfx_events: Sequence[Mapping[str, Any]]) -> list[int]:
    times: list[int] = []
    for index, event in enumerate(existing_sfx_events):
        if not isinstance(event, Mapping):
            raise CaptionSFXPlanningError(f"existing_sfx_events[{index}] must be an object")
        seconds = _finite_number(event.get("time_sec"), f"existing_sfx_events[{index}].time_sec")
        times.append(int(round(seconds * 1000.0)))
    return sorted(times)


def _eligible_candidates(
    contract: Mapping[str, Any], body_duration_ms: int
) -> list[dict[str, Any]]:
    timeline = contract.get("timeline")
    policy = contract.get("render_policy")
    phrases = contract.get("phrases")
    if not isinstance(timeline, Mapping) or not isinstance(policy, Mapping):
        raise CaptionSFXPlanningError("timeline and render_policy must be objects")
    if not isinstance(phrases, list):
        raise CaptionSFXPlanningError("phrases must be an array")

    final_offset_ms = _integer(
        timeline.get("final_assembly_offset_ms"), "timeline.final_assembly_offset_ms"
    )
    if final_offset_ms != 1500:
        raise CaptionSFXPlanningError("timeline.final_assembly_offset_ms must equal 1500")
    candidates: list[dict[str, Any]] = []
    phrase_ids: set[str] = set()
    for phrase_index, phrase in enumerate(phrases):
        field = f"phrases[{phrase_index}]"
        if not isinstance(phrase, Mapping):
            raise CaptionSFXPlanningError(f"{field} must be an object")
        phrase_id = phrase.get("id")
        if not isinstance(phrase_id, str) or not phrase_id.strip():
            raise CaptionSFXPlanningError(f"{field}.id must be a non-empty string")
        if phrase_id in phrase_ids:
            raise CaptionSFXPlanningError(f"{field}.id is duplicated")
        phrase_ids.add(phrase_id)
        sfx_allowed = phrase.get("sfx_allowed")
        if not isinstance(sfx_allowed, bool):
            raise CaptionSFXPlanningError(f"{field}.sfx_allowed must be boolean")
        if not sfx_allowed:
            continue
        if not isinstance(phrase.get("has_synthetic_timing"), bool):
            raise CaptionSFXPlanningError(f"{field}.has_synthetic_timing must be boolean")
        if phrase.get("has_synthetic_timing") or phrase.get("timing_source") != "measured":
            continue
        wpm = _finite_number(phrase.get("wpm"), f"{field}.wpm")
        if wpm <= 0.0:
            raise CaptionSFXPlanningError(f"{field}.wpm must be positive")
        if wpm >= 150.0:
            raise CaptionSFXPlanningError(f"{field} cannot allow SFX at >=150 WPM")
        level = phrase.get("balloon_level")
        expected_level = "strong" if wpm < 90.0 else "medium"
        if level != expected_level:
            raise CaptionSFXPlanningError(
                f"{field}.balloon_level must equal {expected_level} for {wpm:g} WPM"
            )
        words = phrase.get("words")
        if not isinstance(words, list):
            raise CaptionSFXPlanningError(f"{field}.words must be an array")
        for word_index, item in enumerate(words):
            if not isinstance(item, Mapping):
                raise CaptionSFXPlanningError(f"{field}.words[{word_index}] must be an object")
            if not isinstance(item.get("synthetic"), bool):
                raise CaptionSFXPlanningError(
                    f"{field}.words[{word_index}].synthetic must be boolean"
                )
            if not isinstance(item.get("emphasis"), bool):
                raise CaptionSFXPlanningError(
                    f"{field}.words[{word_index}].emphasis must be boolean"
                )
        word = next((
            item for item in words
            if item.get("emphasis") is True
            and item.get("emphasis_source") == "llm"
            and item.get("timing_source") == "measured"
            and item.get("synthetic") is False
        ), None)
        if word is None:
            continue
        level_policy = _LEVEL_POLICY[level]
        spoken_start_ms = _integer(
            word.get("spoken_start_ms"), f"{field}.words.spoken_start_ms"
        )
        if spoken_start_ms >= body_duration_ms:
            raise CaptionSFXPlanningError(
                f"{field}.words.spoken_start_ms must be before body end"
            )
        trigger_ms = spoken_start_ms + final_offset_ms - level_policy["lead_ms"]
        if not 0 <= trigger_ms <= body_duration_ms + final_offset_ms:
            raise CaptionSFXPlanningError(f"{field} produces an out-of-range SFX trigger")
        word_id = word.get("id")
        if not isinstance(word_id, str) or not word_id.strip():
            raise CaptionSFXPlanningError(f"{field}.words.id must be a non-empty string")
        candidates.append(
            {
                "trigger_ms": trigger_ms,
                "phrase_index": _integer(phrase.get("index"), f"{field}.index"),
                "phrase_id": phrase_id,
                "word_id": word_id,
                "word_start_ms": spoken_start_ms,
                **level_policy,
            }
        )
    return sorted(candidates, key=lambda item: (item["trigger_ms"], item["phrase_index"]))


def plan_caption_sfx_events(
    contract: Mapping[str, Any],
    *,
    existing_sfx_events: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return sparse mixer events; disabled/accessibility policies always return empty."""

    if not isinstance(contract, Mapping):
        raise CaptionSFXPlanningError("contract must be an object")
    policy = contract.get("render_policy")
    if not isinstance(policy, Mapping):
        raise CaptionSFXPlanningError("render_policy must be an object")
    if (
        policy.get("stage") != "full"
        or policy.get("audio_profile") != "standard"
        or policy.get("caption_sfx") != "aligned"
    ):
        return []

    if contract.get("schema_version") != "caption-contract.v1":
        raise CaptionSFXPlanningError("unsupported schema_version")
    timeline = contract.get("timeline")
    if not isinstance(timeline, Mapping):
        raise CaptionSFXPlanningError("timeline must be an object")
    body_duration_ms = _integer(
        timeline.get("body_duration_ms"), "timeline.body_duration_ms", minimum=1
    )
    total_limit = math.ceil(body_duration_ms / 10_000)
    foreign_times_ms = _foreign_event_times_ms(existing_sfx_events)

    accepted: list[dict[str, Any]] = []
    accepted_times: list[int] = []
    for candidate in _eligible_candidates(contract, body_duration_ms):
        trigger_ms = candidate["trigger_ms"]
        if any(abs(trigger_ms - foreign_ms) <= FOREIGN_SFX_CLEARANCE_MS for foreign_ms in foreign_times_ms):
            continue
        if accepted_times and trigger_ms - accepted_times[-1] < GLOBAL_COOLDOWN_MS:
            continue
        recent = [time_ms for time_ms in accepted_times if trigger_ms - time_ms < ROLLING_WINDOW_MS]
        if len(recent) >= MAX_EVENTS_PER_WINDOW or len(accepted) >= total_limit:
            continue

        event_index = len(accepted)
        accepted.append(
            {
                "id": f"csfx{event_index:04d}",
                "event": candidate["event"],
                "time_sec": trigger_ms / 1000.0,
                "strength": candidate["strength"],
                "cooldown_group": COOLDOWN_GROUP,
                "meta": {
                    "phrase_id": candidate["phrase_id"],
                    "word_id": candidate["word_id"],
                    "word_start_ms": candidate["word_start_ms"],
                    "alignment_lead_ms": candidate["lead_ms"],
                },
            }
        )
        accepted_times.append(trigger_ms)
    return accepted


def canonical_caption_sfx_json(events: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize a detached event plan reproducibly for manifests and hashing."""

    return json.dumps(
        deepcopy(list(events)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
