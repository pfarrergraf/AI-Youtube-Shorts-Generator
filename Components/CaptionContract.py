"""MAT-47: materializes ``caption-contract.v1`` from Subtitles.py's own,
already-normalised render state.

Where this sits: ``_write_ass_file`` computes ``word_events`` through Balloon/
solo eligibility marking and ``_normalise_phrase_timings`` before it ever
emits an ASS ``Dialogue`` line. That post-normalisation list is exactly the
state ``docs/caption_effect_event_contract.md`` calls "before ASS emission" --
this module turns it into the contract dict, reusing the real per-word sizing
functions (``_word_scales``, ``_solo_word_font_size_px``) instead of
re-deriving them by hand, so the contract can never drift from what actually
renders.

Deliberately import-free of ``cli.caption_effect_events``: that module lives
in the outer repo and is the validator/planner, not a dependency of the
nested submodule -- the same separation ``CaptionFX.py``/``CaptionSFX.py``
already use. Validate/plan the returned dict with that module from the
caller's side (see ``tests/test_caption_contract_producer.py``).

Two real render behaviours were originally not representable in
caption-contract.v1, both found by this producer against real material rather
than assumed. **Both are now representable** through the ``size_reason`` /
``min_word_scale`` / ``max_base_scale`` fields (MAT-48, see
``docs/caption_effect_event_contract.md``), so neither raises any more:

1. A word whose width-fit pass (SAFE-25/BAL-25) shrinks it *below* the
   style's base size without it being the emphasis pick. Measured on 15 real
   ICF highlights this is ordinary German-compound width fitting, not a
   pathology: shrink ratios 0.68-0.98 in 9 of 15 clips. Such a word has no
   line-mates by construction (it triggered its own wrap), so there is no
   reflow fix -- it is reported as ``size_reason="width_fit"``.
2. A solo/word-by-word unit (``impact`` style's slow-speech mode) always
   renders at ``max(1.25, preset["emphasis_scale"])`` -- there is no
   unenlarged solo state, the word is always alone on screen. It is reported
   as ``size_reason="emphasis"`` with ``emphasis_layout=True`` and
   ``emphasis=False``: enlarged layout, no semantic emphasis signal.

``size_reason`` is derived from the resolved pixel size against
``geometry.font_size_px``, never from the scale factor that produced it --
``int(round(font_size * 1.002))`` can land back on the base size, and the
contract must describe what renders, not what was intended.
"""

from __future__ import annotations

CAPTION_CONTRACT_SCHEMA_VERSION = "caption-contract.v1"
CAPTION_EVENT_POLICY = "caption-events.v1"
SAFE_AREA_MASK_SHA256 = (
    "19f858bf36f0f9b8855f19da91ca4e13fd0a322baea3ea78fb06f3e81e6ff436"
)

_STYLE_FONT_STYLE = {
    "Arial Black": "Black",
    "Barlow Semi Condensed Black": "Black",
}


class CaptionContractError(ValueError):
    """The real render state for this word_events list cannot be expressed
    in caption-contract.v1. Raised instead of emitting a contract that would
    fail EVT-20's own validator or misrepresent what actually renders."""


def _ms(seconds):
    return int(round(float(seconds) * 1000.0))


def _size_reason(size, font_size):
    """Which of the three legal sizing outcomes this resolved size is.

    Derived from the pixel size, not from the scale factor: an emphasis word
    the width-fit pass stepped back to exactly the base size renders *as* the
    base size and must say so.
    """
    if size == font_size:
        return "base"
    return "emphasis" if size > font_size else "width_fit"


def build_caption_contract(
    word_events,
    *,
    preset,
    font_size,
    max_chars,
    layout,
    balloon,
    caption_style,
    caption_pop,
    video_width,
    video_height,
    contract_id,
    video_key,
    highlight_rank,
    highlight_start_ms,
    highlight_end_ms,
    source_start_ms,
    source_end_ms,
    body_duration_ms,
    caption_cutoff_ms,
    outer_revision,
    generator_revision,
    dirty_paths=(),
    final_assembly_offset_ms=1500,
    stage="full",
    audio_profile="accessible",
    caption_sfx_policy="off",
    preset_key="default",
    seed=0,
):
    """Build a ``caption-contract.v1`` dict (``effect_events`` left empty --
    that is EVT-20's planner's job) from Subtitles.py's post-normalisation
    ``word_events``.

    All of ``preset``/``font_size``/``max_chars``/``layout``/``balloon`` are
    exactly what ``_write_ass_file`` already has in scope at the point it
    calls ``_normalise_phrase_timings`` -- pass those same objects, not
    recomputed copies, so the contract can only ever describe what that call
    actually produced.
    """
    from Components.Subtitles import (
        BALLOON_MAX_WPM,
        BALLOON_MEDIUM_MAX_WPM,
        BALLOON_MOTION_PROFILES,
        MAX_WORDS_PER_PHRASE,
        _MIN_WORD_SCALE,
        _build_phrase_layout_metadata,
        _pick_emphasis_indices,
        _solo_word_font_size_px,
        _word_scales,
    )

    # Every scale the sizing passes can legally reach for this style/pop:
    # the style's own emphasis scale, the solo path's max(1.25, ...) floor, and
    # -- in balloon mode -- the per-level emphasis scales that override it.
    _scale_candidates = [1.0, float(preset.get("emphasis_scale", 1.0))]
    if preset.get("solo_slow"):
        _scale_candidates.append(max(1.25, float(preset.get("emphasis_scale", 1.0))))
    if balloon:
        _scale_candidates.extend(
            float(profile["emphasis_scale"]) for profile in BALLOON_MOTION_PROFILES.values()
        )
    max_base_scale = max(_scale_candidates)

    source_start_ms = int(round(source_start_ms))
    source_end_ms = int(round(source_end_ms))
    body_duration_ms = int(body_duration_ms)
    if source_end_ms - source_start_ms != body_duration_ms:
        raise CaptionContractError(
            "source_end_ms - source_start_ms must equal body_duration_ms"
        )
    if not (source_start_ms <= highlight_start_ms < highlight_end_ms <= source_end_ms):
        raise CaptionContractError(
            "highlight bounds must lie inside the padded source window"
        )

    fontname = preset["fontname"]
    font_style = _STYLE_FONT_STYLE.get(fontname, "Bold" if preset["bold"] else "Regular")
    outline_px = max(3, int(video_height * preset["outline_ratio"]))
    shadow_px = max(1, int(video_height * preset["shadow_ratio"]))
    bold = bool(preset["bold"])

    phrases = []
    word_index_counter = 0
    max_spoken_end_ms = 0

    for phrase_index, phrase in enumerate(word_events):
        if not phrase:
            continue
        phrase = phrase[:MAX_WORDS_PER_PHRASE]
        is_solo = bool(phrase[0].get("solo"))
        kind = "solo_unit" if is_solo else "shared_line"
        phrase_id = f"p{phrase_index:04d}"

        has_synthetic = any(w.get("synthetic") for w in phrase)
        timing_source = "mixed_or_synthetic" if has_synthetic else "measured"

        spoken_start_ms = _ms(min(w["spoken_start"] for w in phrase))
        spoken_end_ms = _ms(max(w["spoken_end"] for w in phrase))
        display_start_ms = _ms(min(w["start"] for w in phrase))
        display_end_ms = _ms(max(w["end"] for w in phrase))
        max_spoken_end_ms = max(max_spoken_end_ms, spoken_end_ms)

        balloon_level = phrase[0].get("balloon_level") if balloon else None
        wpm = None
        pacing_wpm = None
        if not has_synthetic:
            duration_ms = spoken_end_ms - spoken_start_ms
            if duration_ms <= 0:
                raise CaptionContractError(
                    f"{phrase_id}: non-synthetic phrase has non-positive spoken duration"
                )
            wpm = len(phrase) * 60000.0 / duration_ms
            # The rate that actually decided balloon_level/reveal_mode, stamped
            # by _mark_balloon_eligibility (and carried onto solo units by
            # split_slow_phrases) on the *parent* phrase before any solo split.
            # A solo unit's own `wpm` measures a ~350ms fragment of merged
            # tokens and would cross the 90/150 thresholds on the wrong side.
            pacing_wpm = phrase[0].get("pacing_wpm")
            if pacing_wpm is None or pacing_wpm == float("inf"):
                # No marking pass ran (caption_pop="none"), so this phrase's
                # own measured rate *is* the pacing rate.
                pacing_wpm = wpm
            pacing_wpm = float(pacing_wpm)
            # Only caption_pop=balloon phrases carry a resolved balloon_level
            # at all (_mark_balloon_eligibility only runs then); reproduce
            # the exact same threshold rule for phrases that reached this
            # producer without it (e.g. a caller materialising a contract for
            # caption_pop="none").
            if balloon and balloon_level is None:
                if pacing_wpm < BALLOON_MAX_WPM:
                    balloon_level = "strong"
                elif pacing_wpm < BALLOON_MEDIUM_MAX_WPM:
                    balloon_level = "medium"

        reveal_mode = "progressive_prefix" if (balloon_level and not has_synthetic) else "full_phrase"
        motion_allowed = bool(balloon_level) and not has_synthetic
        sfx_allowed = False  # SFX-45 owns audio selection; not this producer's call.
        if audio_profile == "accessible" or caption_sfx_policy == "off":
            sfx_allowed = False

        words_out = []
        word_ids = []
        line_word_ids_flat_scales = []

        if is_solo:
            word = phrase[0]
            wid = f"w{word_index_counter:04d}"
            word_index_counter += 1
            word_ids.append(wid)
            emphasis_source = "llm" if word.get("emphasis") else "none"
            emphasis = emphasis_source != "none"
            size = _solo_word_font_size_px(
                word, base_font_size=font_size, preset=preset, max_chars=max_chars,
                balloon_level=balloon_level, usable_width=layout.get("isolated_word_usable_width"),
            )
            size_reason = _size_reason(size, font_size)
            emphasis_layout = size_reason == "emphasis"
            words_out.append({
                "id": wid,
                "text": word["text"],
                "spoken_start_ms": _ms(word["spoken_start"]),
                "spoken_end_ms": _ms(word["spoken_end"]),
                "display_start_ms": _ms(word["start"]),
                "display_end_ms": _ms(word["end"]),
                "timing_source": "interpolated" if word.get("synthetic") else "measured",
                "synthetic": bool(word.get("synthetic")),
                "emphasis": emphasis,
                "emphasis_source": emphasis_source,
                "emphasis_layout": emphasis_layout,
                "base_font_size_px": size,
                "size_reason": size_reason,
            })
            line_word_ids = [[wid]]
            geometry_alignment = "an5"
            geometry_x = int(layout["anchor_x"] if layout.get("anchor_x") is not None else video_width // 2)
            geometry_y = int(
                layout["anchor_y"] if layout.get("anchor_y") is not None
                else video_height - layout["margin_v"]
            )
        else:
            scale_override = (
                BALLOON_MOTION_PROFILES[balloon_level]["emphasis_scale"] if balloon_level else None
            )
            peak_multiplier = (
                BALLOON_MOTION_PROFILES[balloon_level]["overshoot_scale"] / 100.0
                if balloon_level else 1.0
            )
            scales = _word_scales(
                phrase, preset, max_chars=max_chars, scale_override=scale_override,
                font_size=font_size, peak_multiplier=peak_multiplier,
                isolated_usable_width_px=layout.get("isolated_word_usable_width"),
            )
            picked = _pick_emphasis_indices(phrase)
            has_explicit = any(w.get("emphasis") for w in phrase)

            for word_idx, word in enumerate(phrase):
                wid = f"w{word_index_counter:04d}"
                word_index_counter += 1
                word_ids.append(wid)

                if word.get("emphasis"):
                    emphasis_source = "llm"
                elif (not has_explicit) and word_idx in picked:
                    emphasis_source = "heuristic"
                else:
                    emphasis_source = "none"
                emphasis = emphasis_source != "none"

                size = int(round(font_size * scales[word_idx]))
                size_reason = _size_reason(size, font_size)
                emphasis_layout = size_reason == "emphasis"

                words_out.append({
                    "id": wid,
                    "text": word["text"],
                    "spoken_start_ms": _ms(word["spoken_start"]),
                    "spoken_end_ms": _ms(word["spoken_end"]),
                    "display_start_ms": _ms(word["start"]),
                    "display_end_ms": _ms(word["end"]),
                    "timing_source": "interpolated" if word.get("synthetic") else "measured",
                    "synthetic": bool(word.get("synthetic")),
                    "emphasis": emphasis,
                    "emphasis_source": emphasis_source,
                    "emphasis_layout": emphasis_layout,
                    "base_font_size_px": size,
                    "size_reason": size_reason,
                })

            _wrapped_lines, line_ranges = _build_phrase_layout_metadata(
                phrase, scales=scales, max_chars=max_chars,
            )
            line_word_ids = [word_ids[start:end] for start, end in line_ranges]
            geometry_alignment = "an2"
            geometry_x = int(layout["anchor_x"])
            geometry_y = int(layout["anchor_y"])

        phrases.append({
            "id": phrase_id,
            "index": len(phrases),
            "kind": kind,
            "text": " ".join(w["text"] for w in phrase),
            "word_ids": word_ids,
            "line_word_ids": line_word_ids,
            "layout_key": f"{phrase_id}-layout",
            "reveal_mode": reveal_mode,
            "geometry": {
                "font_family": fontname,
                "font_style": font_style,
                "bold": bold,
                "font_size_px": font_size,
                "outline_px": outline_px,
                "shadow_px": shadow_px,
                "alignment": geometry_alignment,
                "x_px": geometry_x,
                "y_px": geometry_y,
            },
            "spoken_start_ms": spoken_start_ms,
            "spoken_end_ms": spoken_end_ms,
            "display_start_ms": display_start_ms,
            "display_end_ms": display_end_ms,
            "timing_source": timing_source,
            "has_synthetic_timing": has_synthetic,
            "balloon_level": balloon_level,
            "wpm": wpm,
            "pacing_wpm": pacing_wpm,
            "motion_allowed": motion_allowed,
            "sfx_allowed": sfx_allowed,
            "words": words_out,
        })

    # INT-51: the cutoff is the one _normalise_phrase_timings actually clamped
    # the display intervals to (handed over through contract_sink), not a
    # recomputation. The equality below is kept as a fail-closed self-check --
    # it now holds by construction, and a violation means the renderer and this
    # producer were fed different word sets.
    caption_cutoff_ms = int(caption_cutoff_ms)
    expected_cutoff_ms = min(body_duration_ms, max_spoken_end_ms)
    if caption_cutoff_ms != expected_cutoff_ms:
        raise CaptionContractError(
            f"caption_cutoff_ms {caption_cutoff_ms} != min(body_duration_ms, "
            f"max spoken word end) {expected_cutoff_ms} -- the renderer and this "
            "producer disagree about which words are in the clip"
        )
    for phrase in phrases:
        if phrase["display_end_ms"] > caption_cutoff_ms:
            raise CaptionContractError(
                f"{phrase['id']}: display_end_ms {phrase['display_end_ms']} exceeds "
                f"caption_cutoff_ms {caption_cutoff_ms} -- the caption_cutoff passed "
                "into _normalise_phrase_timings must equal min(body_duration_ms, "
                "max spoken word end), not an unrelated clip-length cutoff"
            )

    return {
        "schema_version": CAPTION_CONTRACT_SCHEMA_VERSION,
        "contract_id": contract_id,
        "producer": {
            "name": "parakeet_uv",
            "outer_revision": outer_revision,
            "generator_revision": generator_revision,
            "dirty_paths": list(dirty_paths),
        },
        "source": {
            "video_key": video_key,
            "highlight_rank": highlight_rank,
            "highlight_start_ms": highlight_start_ms,
            "highlight_end_ms": highlight_end_ms,
        },
        "timeline": {
            "unit": "milliseconds",
            "origin": "body_start",
            "source_start_ms": source_start_ms,
            "source_end_ms": source_end_ms,
            "body_duration_ms": body_duration_ms,
            "caption_cutoff_ms": caption_cutoff_ms,
            "final_assembly_offset_ms": final_assembly_offset_ms,
        },
        "canvas": {
            "width_px": video_width,
            "height_px": video_height,
            "pixel_aspect_ratio": "1:1",
            "coordinate_system": "top_left_x_right_y_down",
            "safe_area_schema_version": 1,
            "safe_area_profile": "social_vertical",
            "safe_area_mask_sha256": SAFE_AREA_MASK_SHA256,
            "anchor": {
                "x_px": int(layout["anchor_x"]),
                "y_px": int(layout["anchor_y"]),
                "alignment": "bottom_center",
            },
            "max_base_caption_width_px": 630,
            "interior_padding_px": 24,
            "border_style": 1,
            "max_lines": 2,
            "max_words_per_phrase": MAX_WORDS_PER_PHRASE,
            "max_motion_scale": 1.24,
            # Declared from the sizing *constants* the width-fit and emphasis
            # passes can reach for this style/pop combination -- deliberately
            # not from the sizes this clip happened to produce, which would be
            # a tautology. _MIN_WORD_SCALE is _word_scales'/
            # _solo_word_font_size_px' shared safety floor.
            "min_word_scale": float(_MIN_WORD_SCALE),
            "max_base_scale": max_base_scale,
        },
        "render_policy": {
            "stage": stage,
            "caption_style": caption_style,
            "caption_pop": caption_pop,
            "caption_safe_area": "social_vertical",
            "audio_profile": audio_profile,
            "caption_sfx": caption_sfx_policy,
            "event_policy": CAPTION_EVENT_POLICY,
            "preset_key": preset_key,
            "seed": seed,
        },
        "phrases": phrases,
        "effect_events": [],
        "extensions": {},
    }
