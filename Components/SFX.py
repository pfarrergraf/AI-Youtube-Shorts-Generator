import os
import shlex
import subprocess
from typing import Any


SFX_TYPES = {
    "pop": [
        "organic-pop-roy-s-noise-1-00-02.mp3",
    ],
    "click": [
        "mixkit-retro-game-notification-212.wav",
    ],
    "whoosh_soft": [
        "mixkit-fast-whoosh-transition-1490.wav",
    ],
    "whoosh_strong": [
        "mixkit-air-woosh-1489.wav",
    ],
    "whoosh_heavy": [
        "Swoosh.mp3",
    ],
    "whoosh_transition": [
        "mixkit-cinematic-whoosh-fast-transition-1492.wav",
    ],
    "vacuum_turn": [
        "mixkit-air-zoom-vacuum-2608.wav",
    ],
    "vacuum_transition": [
        "mixkit-vacuum-swoosh-transition-1465.wav",
    ],
    "impact_soft": [
        "mixkit-cool-impact-movie-trailer-2909.wav",
    ],
    "impact_strong": [
        "mixkit-movie-trailer-epic-impact-2908.wav",
    ],
    "punch": [
        "mixkit-martial-arts-fast-punch-2047.wav",
    ],
}

EVENT_TO_SFX = {
    "zoom_in": "whoosh_soft",
    "zoom_out": "whoosh_soft",
    "reframe": "vacuum_turn",
    "jump_cut": "click",
    "scene_transition": "whoosh_strong",
    "hard_transition": "whoosh_soft",
    "hook_in": "whoosh_transition",
    "hook_hit": "impact_soft",
    "keyword_pop": "pop",
    "emphasis": "punch",
    "dramatic_hit": "impact_strong",
    "title_to_video_transition": "vacuum_transition",
}

SFX_BEHAVIOR = {
    "pop": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 80},
    "click": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 60},
    "whoosh_soft": {"offset_ms": -40, "fade_in_ms": 5, "fade_out_ms": 120},
    "whoosh_strong": {"offset_ms": -60, "fade_in_ms": 5, "fade_out_ms": 140},
    "whoosh_heavy": {"offset_ms": -120, "fade_in_ms": 5, "fade_out_ms": 200},
    "whoosh_transition": {"offset_ms": -300, "fade_in_ms": 5, "fade_out_ms": 180},
    "vacuum_turn": {"offset_ms": -30, "fade_in_ms": 5, "fade_out_ms": 120},
    "vacuum_transition": {"offset_ms": -80, "fade_in_ms": 5, "fade_out_ms": 160},
    "impact_soft": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 180},
    "impact_strong": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 220},
    "punch": {"offset_ms": -10, "fade_in_ms": 0, "fade_out_ms": 120},
}

SFX_GAIN = {
    "pop": -18.0,
    "click": -20.0,
    "whoosh_soft": -18.0,
    "whoosh_strong": -16.0,
    "whoosh_heavy": -14.0,
    "whoosh_transition": -20.0,
    "vacuum_turn": -17.0,
    "vacuum_transition": -15.0,
    "impact_soft": -18.0,
    "impact_strong": -14.0,
    "punch": -16.0,
}

SFX_PROFILE_OVERRIDES = {
    "clean": {
        "enabled_events": {
            "keyword_pop",
            "hook_hit",
            "hard_transition",
        },
        "gain_adjust_db": -2.0,
    },
    "social": {
        "enabled_events": {
            "zoom_in",
            "zoom_out",
            "reframe",
            "jump_cut",
            "scene_transition",
            "hard_transition",
            "hook_in",
            "hook_hit",
            "keyword_pop",
            "emphasis",
            "title_to_video_transition",
        },
        "gain_adjust_db": 0.0,
    },
    "dramatic": {
        "enabled_events": {
            "zoom_in",
            "zoom_out",
            "reframe",
            "jump_cut",
            "scene_transition",
            "hard_transition",
            "hook_in",
            "hook_hit",
            "keyword_pop",
            "emphasis",
            "dramatic_hit",
            "title_to_video_transition",
        },
        "gain_adjust_db": 1.5,
    },
}

EVENT_COOLDOWNS_SEC = {
    "zoom_in": 1.0,
    "zoom_out": 1.0,
    "reframe": 0.8,
    "jump_cut": 0.33,
    "scene_transition": 2.0,
    "hard_transition": 1.0,
    "hook_in": 2.0,
    "hook_hit": 1.2,
    "keyword_pop": 0.5,
    "emphasis": 0.7,
    "dramatic_hit": 2.5,
    "title_to_video_transition": 3.0,
}

DEFAULT_SFX_DIR_CANDIDATES = [
    "sound_effects",
    "assets/sfx",
]


def _log(message: str) -> None:
    print(f"[SFX] {message}")


def _run_ffmpeg(command: list[str], description: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"{description} failed: {stderr}")


def resolve_sfx_dir(explicit_sfx_dir: str | None = None) -> str | None:
    if explicit_sfx_dir:
        explicit_sfx_dir = os.path.abspath(explicit_sfx_dir)
        if os.path.isdir(explicit_sfx_dir):
            return explicit_sfx_dir
        _log(f"Configured SFX dir not found: {explicit_sfx_dir}")
        return None

    for candidate in DEFAULT_SFX_DIR_CANDIDATES:
        abs_candidate = os.path.abspath(candidate)
        if os.path.isdir(abs_candidate):
            return abs_candidate

    return None


def resolve_sfx_path(event_name: str, sfx_dir: str | None) -> str | None:
    if not sfx_dir:
        return None

    sfx_type = EVENT_TO_SFX.get(event_name)
    if not sfx_type:
        return None

    filenames = SFX_TYPES.get(sfx_type, [])
    for filename in filenames:
        path = os.path.join(sfx_dir, filename)
        if os.path.isfile(path):
            return path

    return None


def _clamp_nonnegative_ms(value_ms: int) -> int:
    return max(0, int(value_ms))


def _get_profile_config(profile: str) -> dict[str, Any]:
    return SFX_PROFILE_OVERRIDES.get(profile, SFX_PROFILE_OVERRIDES["social"])


def _is_event_enabled(event_name: str, profile: str) -> bool:
    enabled = _get_profile_config(profile)["enabled_events"]
    return event_name in enabled


def _get_effective_gain_db(event_name: str, profile: str) -> float:
    sfx_type = EVENT_TO_SFX[event_name]
    base_gain = SFX_GAIN.get(sfx_type, -18.0)
    profile_adjust = float(_get_profile_config(profile).get("gain_adjust_db", 0.0))
    return base_gain + profile_adjust


def _get_behavior(event_name: str) -> dict[str, int]:
    sfx_type = EVENT_TO_SFX[event_name]
    return SFX_BEHAVIOR.get(
        sfx_type,
        {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 80},
    )


def _normalize_event(raw_event: dict[str, Any]) -> dict[str, Any] | None:
    event_name = raw_event.get("event")
    time_sec = raw_event.get("time_sec")

    if not event_name or event_name not in EVENT_TO_SFX:
        return None
    if time_sec is None:
        return None

    return {
        "event": event_name,
        "time_sec": float(time_sec),
        "strength": raw_event.get("strength"),
        "meta": raw_event.get("meta", {}),
    }


def build_sfx_events(
    raw_events: list[dict[str, Any]],
    sfx_dir: str | None = None,
    profile: str = "social",
) -> list[dict[str, Any]]:
    """
    Convert raw timeline events into concrete SFX playback events.

    Expected raw event format:
    {
        "event": "zoom_in",
        "time_sec": 12.34,
        "strength": "soft",   # optional
        "meta": {...},        # optional
    }
    """
    sfx_dir = resolve_sfx_dir(sfx_dir)
    if not sfx_dir:
        _log("No SFX directory found. Continuing without SFX.")
        return []

    normalized: list[dict[str, Any]] = []
    for raw in raw_events:
        item = _normalize_event(raw)
        if item is not None:
            normalized.append(item)

    normalized.sort(key=lambda x: x["time_sec"])

    last_trigger_time_by_event: dict[str, float] = {}
    built: list[dict[str, Any]] = []

    for item in normalized:
        event_name = item["event"]
        time_sec = item["time_sec"]

        if not _is_event_enabled(event_name, profile):
            continue

        cooldown = EVENT_COOLDOWNS_SEC.get(event_name, 0.0)
        last_time = last_trigger_time_by_event.get(event_name)
        if last_time is not None and (time_sec - last_time) < cooldown:
            continue

        sfx_path = resolve_sfx_path(event_name, sfx_dir)
        if not sfx_path:
            _log(f"Missing SFX file for event '{event_name}' in {sfx_dir}")
            continue

        behavior = _get_behavior(event_name)
        offset_ms = int(behavior.get("offset_ms", 0))
        start_ms = max(0, int(round(time_sec * 1000.0)) + offset_ms)

        built.append(
            {
                "event": event_name,
                "time_sec": time_sec,
                "start_ms": start_ms,
                "sfx_path": sfx_path,
                "sfx_type": EVENT_TO_SFX[event_name],
                "gain_db": _get_effective_gain_db(event_name, profile),
                "fade_in_ms": int(behavior.get("fade_in_ms", 0)),
                "fade_out_ms": int(behavior.get("fade_out_ms", 80)),
                "meta": item.get("meta", {}),
            }
        )
        last_trigger_time_by_event[event_name] = time_sec

    return built


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.6f}"


def _build_single_sfx_filter(
    input_index: int,
    output_label: str,
    start_ms: int,
    gain_db: float,
    fade_in_ms: int,
    fade_out_ms: int,
) -> str:
    """
    Builds a filter chain for one SFX input.
    """
    start_sec = start_ms / 1000.0
    fade_in_sec = max(0.0, fade_in_ms / 1000.0)
    fade_out_sec = max(0.0, fade_out_ms / 1000.0)

    parts = [
        f"[{input_index}:a]aresample=48000",
        f"volume={gain_db}dB",
    ]

    if fade_in_sec > 0:
        parts.append(f"afade=t=in:st=0:d={_format_seconds(fade_in_sec)}")

    if fade_out_sec > 0:
        parts.append(f"afade=t=out:st=0:d={_format_seconds(fade_out_sec)}")

    parts.append(f"adelay={start_ms}|{start_ms}")
    parts.append(f"{output_label}")

    return ",".join(parts)


def _build_filter_complex_for_sfx(
    sfx_events: list[dict[str, Any]],
    speech_input_index: int = 1,
) -> tuple[str, str]:
    """
    Returns (filter_complex, final_audio_label).
    Input 0 = video
    Input 1 = base speech/audio
    Inputs 2..N = sfx files
    """
    filter_parts: list[str] = []
    mix_inputs = [f"[{speech_input_index}:a]"]

    for idx, event in enumerate(sfx_events):
        input_index = 2 + idx
        label = f"[sfx{idx}]"
        filter_parts.append(
            _build_single_sfx_filter(
                input_index=input_index,
                output_label=label,
                start_ms=int(event["start_ms"]),
                gain_db=float(event["gain_db"]),
                fade_in_ms=int(event["fade_in_ms"]),
                fade_out_ms=int(event["fade_out_ms"]),
            )
        )
        mix_inputs.append(label)

    final_label = "[aout]"
    amix = (
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:normalize=0:dropout_transition=0,"
          "alimiter=limit=0.95"
        + final_label
    )
    filter_parts.append(amix)

    return ";".join(filter_parts), final_label


def mix_sfx_with_audio_ffmpeg(
    video_input_path: str,
    audio_input_path: str,
    output_path: str,
    raw_events: list[dict[str, Any]],
    sfx_dir: str | None = None,
    profile: str = "social",
    video_codec_flags: list[str] | None = None,
    keep_video_copy: bool = True,
) -> str:
    """
    Mix SFX onto an existing video + audio pair.

    Assumptions:
    - `video_input_path` provides the video stream.
    - `audio_input_path` provides the base speech/audio stream.
    - raw_events use absolute times relative to the start of the final clip.

    Example raw_events:
    [
        {"event": "hook_in", "time_sec": 0.0},
        {"event": "hook_hit", "time_sec": 0.55},
        {"event": "zoom_in", "time_sec": 3.20},
        {"event": "keyword_pop", "time_sec": 7.10},
    ]
    """
    video_input_path = os.path.abspath(video_input_path)
    audio_input_path = os.path.abspath(audio_input_path)
    output_path = os.path.abspath(output_path)

    sfx_events = build_sfx_events(raw_events, sfx_dir=sfx_dir, profile=profile)
    if not sfx_events:
        _log("No usable SFX events found. Falling back to base video+audio mux.")
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_input_path,
            "-i",
            audio_input_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
        ]
        if keep_video_copy:
            command += ["-c:v", "copy"]
        elif video_codec_flags:
            command += video_codec_flags
        else:
            command += ["-c:v", "copy"]

        command += [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            output_path,
        ]
        _run_ffmpeg(command, "fallback mux without SFX")
        return output_path

    command: list[str] = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_input_path,
        "-i",
        audio_input_path,
    ]

    for event in sfx_events:
        command += ["-i", event["sfx_path"]]

    filter_complex, audio_label = _build_filter_complex_for_sfx(
        sfx_events,
        speech_input_index=1,
    )

    command += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        audio_label,
    ]

    if keep_video_copy:
        command += ["-c:v", "copy"]
    elif video_codec_flags:
        command += video_codec_flags
    else:
        command += ["-c:v", "copy"]

    command += [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        output_path,
    ]

    _run_ffmpeg(command, "SFX audio mix")
    return output_path


def debug_print_sfx_events(
    raw_events: list[dict[str, Any]],
    sfx_dir: str | None = None,
    profile: str = "social",
) -> None:
    events = build_sfx_events(raw_events, sfx_dir=sfx_dir, profile=profile)
    if not events:
        _log("No SFX events built.")
        return

    _log("Resolved SFX events:")
    for event in events:
        _log(
            f"{event['event']:>24}  "
            f"time={event['time_sec']:.3f}s  "
            f"start_ms={event['start_ms']}  "
            f"type={event['sfx_type']}  "
            f"gain={event['gain_db']}dB  "
            f"file={os.path.basename(event['sfx_path'])}"
        )


def build_example_events() -> list[dict[str, Any]]:
    """
    Example timeline for manual testing.
    """
    return [
        {"event": "hook_in", "time_sec": 0.00},
        {"event": "hook_hit", "time_sec": 0.55},
        {"event": "title_to_video_transition", "time_sec": 0.95},
        {"event": "zoom_in", "time_sec": 3.20},
        {"event": "jump_cut", "time_sec": 6.00},
        {"event": "keyword_pop", "time_sec": 7.15},
        {"event": "reframe", "time_sec": 9.40},
        {"event": "scene_transition", "time_sec": 13.00},
    ]


if __name__ == "__main__":
    # Simple manual debug example:
    # python -m Components.SFX
    debug_print_sfx_events(build_example_events())