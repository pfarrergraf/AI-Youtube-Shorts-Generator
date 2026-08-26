"""Safe, deterministic ASS override cues for caption effect events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


SUPPORTED_PRESETS = {"none", "soft_pop", "shine", "impact_hit"}
MAX_SCALE_PERCENT = 124

_SCALE_PRESETS = {
    "soft_pop": {"kinds": {"phrase_in", "keyword_pop", "emphasis"}, "peak": 112},
    "impact_hit": {"kinds": {"emphasis"}, "peak": 124},
}
_SHINE_KINDS = {"keyword_pop", "emphasis"}


class CaptionFXError(ValueError):
    """Raised when an enabled visual plan violates the v1 effect contract."""


@dataclass(frozen=True)
class CaptionFXCue:
    id: str
    event_id: str
    preset: str
    phrase_id: str
    word_id: str | None
    start_ms: int
    peak_ms: int
    settle_end_ms: int
    dialogue_start_ms: int
    dialogue_end_ms: int
    relative_start_ms: int
    relative_peak_ms: int
    relative_settle_end_ms: int
    transform_scope: str
    peak_scale_percent: int
    ass_override: str


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptionFXError(f"{field} must be an integer")
    return value


def _scale_override(start: int, peak: int, settle: int, peak_scale: int) -> str:
    return (
        r"{\fscx100\fscy100"
        f"\\t({start},{peak},\\fscx{peak_scale}\\fscy{peak_scale})"
        f"\\t({peak},{settle},\\fscx100\\fscy100)"
        "}"
    )


def _shine_override(start: int, peak: int, settle: int) -> str:
    return (
        r"{\3c&H00FFFFFF&\3a&H00&"
        f"\\t({start},{peak},\\3c&H0000FFFF&\\3a&H10&)"
        f"\\t({peak},{settle},\\3c&H00FFFFFF&\\3a&H00&)"
        "}"
    )


def _validate_effect_plan(
    phrase_by_id: Mapping[str, Mapping[str, Any]],
    events: list[Any],
    policy: Mapping[str, Any],
) -> None:
    """Validate every event before preset filtering so execution is atomic."""

    for event_index, event in enumerate(events):
        field = f"effect_events[{event_index}]"
        if not isinstance(event, Mapping):
            raise CaptionFXError(f"{field} must be an object")
        if event.get("id") != f"e{event_index:04d}" or event.get("index") != event_index:
            raise CaptionFXError(f"{field} id/index must be contiguous")
        kind = event.get("kind")
        if kind not in {"phrase_in", "keyword_pop", "emphasis"}:
            raise CaptionFXError(f"{field}.kind is unknown")
        channels = event.get("channels")
        if (
            not isinstance(channels, list)
            or not channels
            or any(not isinstance(channel, str) for channel in channels)
            or len(channels) != len(set(channels))
            or any(channel not in {"visual", "sfx"} for channel in channels)
        ):
            raise CaptionFXError(f"{field}.channels are invalid")
        if event.get("strength") not in {"strong", "medium", "soft"}:
            raise CaptionFXError(f"{field}.strength is invalid")
        event_seed = event.get("seed")
        if (
            isinstance(event_seed, bool)
            or not isinstance(event_seed, int)
            or event_seed < 0
            or event_seed != policy.get("seed")
        ):
            raise CaptionFXError(f"{field}.seed does not match render policy")
        phrase = phrase_by_id.get(event.get("phrase_id"))
        if phrase is None:
            raise CaptionFXError(f"{field}.phrase_id is unknown")
        if "sfx" in channels and (
            policy.get("audio_profile") == "accessible"
            or policy.get("caption_sfx") == "off"
            or phrase.get("sfx_allowed") is not True
        ):
            raise CaptionFXError(f"{field}.channels violate the SFX policy")
        words = phrase.get("words")
        if not isinstance(words, list) or any(not isinstance(word, Mapping) for word in words):
            raise CaptionFXError(f"{field}.words must be objects")
        if phrase.get("has_synthetic_timing") is not False or any(
            word.get("synthetic") is not False for word in words
        ):
            raise CaptionFXError(f"{field} targets synthetic timing")
        if phrase.get("motion_allowed") is not True:
            raise CaptionFXError(f"{field} targets a motion-disabled phrase")
        word_id = event.get("word_id")
        word_matches = [word for word in words if word.get("id") == word_id]
        if kind == "phrase_in" and word_id is not None:
            raise CaptionFXError(f"{field}.word_id must be null for phrase_in")
        if kind != "phrase_in" and len(word_matches) != 1:
            raise CaptionFXError(f"{field}.word_id must reference one phrase word")
        scope = event.get("transform_scope")
        if scope not in {"whole_phrase", "solo_unit", "none"}:
            raise CaptionFXError(f"{field}.transform_scope is unknown")
        if "visual" in channels and scope == "none":
            raise CaptionFXError(f"{field}.transform_scope is not visual")
        if "visual" not in channels and scope != "none":
            raise CaptionFXError(f"{field}.transform_scope requires the visual channel")
        if phrase.get("kind") == "shared_line" and scope not in {"whole_phrase", "none"}:
            raise CaptionFXError(f"{field}.transform_scope would reflow a shared line")
        start_ms = _integer(event.get("start_ms"), f"{field}.start_ms")
        peak_ms = _integer(event.get("peak_ms"), f"{field}.peak_ms")
        settle_ms = _integer(event.get("settle_end_ms"), f"{field}.settle_end_ms")
        phrase_start = _integer(phrase.get("display_start_ms"), f"{field}.display_start_ms")
        phrase_end = _integer(phrase.get("display_end_ms"), f"{field}.display_end_ms")
        if not phrase_start <= start_ms <= peak_ms <= settle_ms <= phrase_end or start_ms == settle_ms:
            raise CaptionFXError(f"{field} lies outside its phrase or has invalid timing")
        if kind == "phrase_in":
            first_word_start = _integer(words[0].get("display_start_ms"), f"{field}.first_word.display_start_ms")
            first_word_end = _integer(words[0].get("display_end_ms"), f"{field}.first_word.display_end_ms")
            if not first_word_start <= start_ms < settle_ms <= first_word_end:
                raise CaptionFXError(f"{field} lies outside its first Dialogue interval")
        else:
            target_word = word_matches[0]
            word_start = _integer(target_word.get("display_start_ms"), f"{field}.word.display_start_ms")
            word_end = _integer(target_word.get("display_end_ms"), f"{field}.word.display_end_ms")
            if not word_start <= start_ms < settle_ms <= word_end:
                raise CaptionFXError(f"{field} lies outside its target Dialogue interval")
        scale = event.get("motion_peak_scale")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or not 1.0 <= float(scale) <= 1.24
        ):
            raise CaptionFXError(f"{field}.motion_peak_scale violates the 1.24 clamp")
        if scope == "none" and float(scale) != 1.0:
            raise CaptionFXError(f"{field}.motion_peak_scale must equal 1.0 without geometry")


def build_caption_fx_cues(
    contract: Mapping[str, Any],
    preset: str,
) -> tuple[CaptionFXCue, ...]:
    """Convert validated v1 events into layout-safe, phrase-relative ASS cues."""

    if preset not in SUPPORTED_PRESETS:
        raise CaptionFXError(f"unsupported caption FX preset: {preset!r}")
    if preset == "none":
        return ()
    if not isinstance(contract, Mapping) or contract.get("schema_version") != "caption-contract.v1":
        raise CaptionFXError("caption-contract.v1 is required")
    phrases = contract.get("phrases")
    events = contract.get("effect_events")
    policy = contract.get("render_policy")
    if not isinstance(phrases, list) or not isinstance(events, list) or not isinstance(policy, Mapping):
        raise CaptionFXError("phrases, effect_events and render_policy are required")
    if (
        isinstance(policy.get("seed"), bool)
        or not isinstance(policy.get("seed"), int)
        or policy.get("seed") < 0
    ):
        raise CaptionFXError("render_policy.seed must be a non-negative integer")

    phrase_by_id: dict[str, Mapping[str, Any]] = {}
    for index, phrase in enumerate(phrases):
        if not isinstance(phrase, Mapping):
            raise CaptionFXError(f"phrases[{index}] must be an object")
        phrase_id = phrase.get("id")
        if not isinstance(phrase_id, str) or not phrase_id.strip() or phrase_id in phrase_by_id:
            raise CaptionFXError(f"phrases[{index}].id must be unique and non-empty")
        phrase_by_id[phrase_id] = phrase

    _validate_effect_plan(phrase_by_id, events, policy)

    cues: list[CaptionFXCue] = []
    event_ids: set[str] = set()
    for event_index, event in enumerate(events):
        field = f"effect_events[{event_index}]"
        event_id = event.get("id")
        if event_id != f"e{event_index:04d}" or event_id in event_ids:
            raise CaptionFXError(f"{field}.id must be contiguous and unique")
        event_ids.add(event_id)
        if event.get("index") != event_index:
            raise CaptionFXError(f"{field}.index must equal its array position")
        if "visual" not in event.get("channels"):
            continue
        kind = event.get("kind")
        if preset in _SCALE_PRESETS and kind not in _SCALE_PRESETS[preset]["kinds"]:
            continue
        if preset == "impact_hit" and event.get("strength") != "strong":
            continue
        if preset == "shine" and kind not in _SHINE_KINDS:
            continue
        phrase_id = event.get("phrase_id")
        phrase = phrase_by_id.get(phrase_id)
        if phrase is None:
            raise CaptionFXError(f"{field}.phrase_id is unknown")
        scope = event.get("transform_scope")
        if scope not in {"whole_phrase", "solo_unit"}:
            raise CaptionFXError(f"{field}.transform_scope is not layout-safe")
        start_ms = _integer(event.get("start_ms"), f"{field}.start_ms")
        peak_ms = _integer(event.get("peak_ms"), f"{field}.peak_ms")
        settle_ms = _integer(event.get("settle_end_ms"), f"{field}.settle_end_ms")
        phrase_start = _integer(phrase.get("display_start_ms"), f"{field}.display_start_ms")
        phrase_end = _integer(phrase.get("display_end_ms"), f"{field}.display_end_ms")
        if not phrase_start <= start_ms <= peak_ms <= settle_ms <= phrase_end or start_ms == settle_ms:
            raise CaptionFXError(f"{field} lies outside its phrase or has invalid timing")
        word_id = event.get("word_id")
        if kind == "phrase_in":
            if word_id is not None:
                raise CaptionFXError(f"{field}.word_id must be null for phrase_in")
            first_word = phrase["words"][0]
            dialogue_start_ms = _integer(
                first_word.get("display_start_ms"), f"{field}.first_word.display_start_ms"
            )
            dialogue_end_ms = _integer(
                first_word.get("display_end_ms"), f"{field}.first_word.display_end_ms"
            )
        else:
            words = phrase.get("words")
            if not isinstance(words, list):
                raise CaptionFXError(f"{field}.words must be an array")
            matching_words = [
                word for word in words
                if isinstance(word, Mapping) and word.get("id") == word_id
            ]
            if len(matching_words) != 1:
                raise CaptionFXError(f"{field}.word_id must reference one phrase word")
            target_word = matching_words[0]
            dialogue_start_ms = _integer(
                target_word.get("display_start_ms"), f"{field}.word.display_start_ms"
            )
            dialogue_end_ms = _integer(
                target_word.get("display_end_ms"), f"{field}.word.display_end_ms"
            )
            if not dialogue_start_ms <= start_ms < settle_ms <= dialogue_end_ms:
                raise CaptionFXError(f"{field} lies outside its target Dialogue interval")
        relative = (
            start_ms - dialogue_start_ms,
            peak_ms - dialogue_start_ms,
            settle_ms - dialogue_start_ms,
        )

        if preset == "shine":
            peak_scale = 100
            override = _shine_override(*relative)
        else:
            declared_scale = event.get("motion_peak_scale")
            if isinstance(declared_scale, bool) or not isinstance(declared_scale, (int, float)):
                raise CaptionFXError(f"{field}.motion_peak_scale must be numeric")
            if not math.isfinite(float(declared_scale)) or not 1.0 <= float(declared_scale) <= 1.24:
                raise CaptionFXError(f"{field}.motion_peak_scale violates the 1.24 clamp")
            peak_scale = min(
                MAX_SCALE_PERCENT,
                _SCALE_PRESETS[preset]["peak"],
                int(round(float(declared_scale) * 100)),
            )
            override = _scale_override(*relative, peak_scale)

        cues.append(
            CaptionFXCue(
                id=f"cfx{len(cues):04d}",
                event_id=event_id,
                preset=preset,
                phrase_id=phrase_id,
                word_id=word_id,
                start_ms=start_ms,
                peak_ms=peak_ms,
                settle_end_ms=settle_ms,
                dialogue_start_ms=dialogue_start_ms,
                dialogue_end_ms=dialogue_end_ms,
                relative_start_ms=relative[0],
                relative_peak_ms=relative[1],
                relative_settle_end_ms=relative[2],
                transform_scope=scope,
                peak_scale_percent=peak_scale,
                ass_override=override,
            )
        )
    return tuple(cues)


def apply_caption_fx_to_ass_text(ass_text: str, cue: CaptionFXCue) -> str:
    """Insert the cue after the leading ASS block so it overrides Balloon scale tags."""

    if not isinstance(ass_text, str):
        raise CaptionFXError("ASS text must be a string")
    if not ass_text.startswith("{") or "}" not in ass_text:
        raise CaptionFXError("ASS text must start with an override block")
    block_end = ass_text.index("}") + 1
    return ass_text[:block_end] + cue.ass_override + ass_text[block_end:]
