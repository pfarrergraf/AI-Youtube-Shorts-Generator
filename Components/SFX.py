import os
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
    "lower_third_in": "whoosh_soft",
    "lower_third_out": "whoosh_soft",
}

SFX_BEHAVIOR = {
    "pop": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 80, "trim_ms": 220},
    "click": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 60, "trim_ms": 180},
    "whoosh_soft": {"offset_ms": -40, "fade_in_ms": 5, "fade_out_ms": 120, "trim_ms": 550},
    "whoosh_strong": {"offset_ms": -60, "fade_in_ms": 5, "fade_out_ms": 140, "trim_ms": 800},
    "whoosh_heavy": {"offset_ms": -120, "fade_in_ms": 5, "fade_out_ms": 220, "trim_ms": 1100},
    "whoosh_transition": {"offset_ms": -300, "fade_in_ms": 5, "fade_out_ms": 180, "trim_ms": 1200},
    "vacuum_turn": {"offset_ms": -30, "fade_in_ms": 5, "fade_out_ms": 120, "trim_ms": 500},
    "vacuum_transition": {"offset_ms": -80, "fade_in_ms": 5, "fade_out_ms": 180, "trim_ms": 900},
    "impact_soft": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 180, "trim_ms": 700},
    "impact_strong": {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 220, "trim_ms": 1000},
    "punch": {"offset_ms": -10, "fade_in_ms": 0, "fade_out_ms": 120, "trim_ms": 400},
}

SFX_GAIN = {
    "pop": -16.0,
    "click": -16.0,
    "whoosh_soft": -14.0,
    "whoosh_strong": -14.0,
    "whoosh_heavy": -14.0,
    "whoosh_transition": -14.0,
    "vacuum_turn": -15.0,
    "vacuum_transition": -15.0,
    "impact_soft": -14.0,
    "impact_strong": -11.0,
    "punch": -14.0,
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
            "lower_third_in",
            "lower_third_out",
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
    "hard_transition": 0.0,
    "hook_in": 2.0,
    "hook_hit": 1.2,
    "keyword_pop": 0.5,
    "emphasis": 0.7,
    "dramatic_hit": 2.5,
    "title_to_video_transition": 3.0,
    "lower_third_in": 2.0,
    "lower_third_out": 2.0,
}

# Per-event gain overrides (dB). Takes precedence over SFX_GAIN for the
# resolved SFX type whenever these events are rendered.
EVENT_GAIN_OVERRIDE = {
    "lower_third_in": -7.0,
    "lower_third_out": -7.0,
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


def _get_profile_config(profile: str) -> dict[str, Any]:
    return SFX_PROFILE_OVERRIDES.get(profile, SFX_PROFILE_OVERRIDES["social"])


def _is_event_enabled(event_name: str, profile: str) -> bool:
    enabled = _get_profile_config(profile)["enabled_events"]
    return event_name in enabled


def _get_effective_gain_db(event_name: str, profile: str) -> float:
    # Per-event override takes precedence (e.g. lower_third_in at -7dB)
    if event_name in EVENT_GAIN_OVERRIDE:
        base_gain = EVENT_GAIN_OVERRIDE[event_name]
    else:
        sfx_type = EVENT_TO_SFX[event_name]
        base_gain = SFX_GAIN.get(sfx_type, -18.0)
    profile_adjust = float(_get_profile_config(profile).get("gain_adjust_db", 0.0))
    return base_gain + profile_adjust


def _get_behavior(event_name: str) -> dict[str, int]:
    sfx_type = EVENT_TO_SFX[event_name]
    return SFX_BEHAVIOR.get(
        sfx_type,
        {"offset_ms": 0, "fade_in_ms": 0, "fade_out_ms": 80, "trim_ms": 500},
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
                "trim_ms": int(behavior.get("trim_ms", 500)),
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
    trim_ms: int = 500,
) -> str:
    """
    Builds a filter chain for one SFX input.
    """
    trim_sec = max(0.05, trim_ms / 1000.0)
    fade_in_sec = max(0.0, fade_in_ms / 1000.0)
    fade_out_sec = max(0.0, fade_out_ms / 1000.0)

    # fade-out must not exceed the trimmed duration
    if fade_out_sec >= trim_sec:
        fade_out_sec = max(0.02, trim_sec * 0.35)

    fade_out_start_sec = max(0.0, trim_sec - fade_out_sec)

    parts = [
        f"[{input_index}:a]aresample=48000",
        f"atrim=0:{_format_seconds(trim_sec)}",
        "asetpts=PTS-STARTPTS",
        f"volume={gain_db}dB",
    ]

    if fade_in_sec > 0:
        parts.append(f"afade=t=in:st=0:d={_format_seconds(fade_in_sec)}")

    if fade_out_sec > 0:
        parts.append(
            f"afade=t=out:st={_format_seconds(fade_out_start_sec)}:d={_format_seconds(fade_out_sec)}"
        )

    parts.append(f"adelay={start_ms}|{start_ms}")

    return ",".join(parts) + output_label


def _build_filter_complex_for_sfx(
    sfx_events: list[dict[str, Any]],
    speech_input_index: int = 1,
    duck_sfx_under_speech: bool = True,
    duck_threshold: float = 0.035,
    duck_ratio: float = 8.0,
    duck_attack_ms: int = 15,
    duck_release_ms: int = 180,
) -> tuple[str, str]:
    """
    Returns (filter_complex, final_audio_label).
    Input 0 = video
    Input 1 = base speech/audio
    Inputs 2..N = sfx files
    """
    filter_parts: list[str] = []

    # Resample speech for consistent mixing
    speech_label = "[speech]"
    filter_parts.append(f"[{speech_input_index}:a]aresample=48000{speech_label}")

    mix_inputs: list[str] = []
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
                trim_ms=int(event.get("trim_ms", 500)),
            )
        )
        mix_inputs.append(label)

    # Mix all SFX into a single bus
    if mix_inputs:
        sfx_bus_label = "[sfxbus]"
        filter_parts.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:normalize=0:dropout_transition=0"
            + sfx_bus_label
        )
    else:
        sfx_bus_label = "[sfxbus]"
        filter_parts.append(
            "anullsrc=r=48000:cl=stereo,atrim=0:0.05,asetpts=PTS-STARTPTS"
            + sfx_bus_label
        )

    final_label = "[aout]"

    if duck_sfx_under_speech:
        # Duck the SFX bus using speech as sidechain
        ducked_label = "[sfxduck]"
        filter_parts.append(
            f"{sfx_bus_label}{speech_label}sidechaincompress="
            f"threshold={duck_threshold}:ratio={duck_ratio}:"
            f"attack={duck_attack_ms}:release={duck_release_ms}"
            f"{ducked_label}"
        )
        # Re-read speech for final mix (sidechaincompress consumed it)
        speech_label2 = "[speech2]"
        filter_parts.append(
            f"[{speech_input_index}:a]aresample=48000{speech_label2}"
        )
        filter_parts.append(
            f"{speech_label2}{ducked_label}amix=inputs=2:normalize=0:dropout_transition=0,"
            f"alimiter=limit=0.95{final_label}"
        )
    else:
        filter_parts.append(
            f"{speech_label}{sfx_bus_label}amix=inputs=2:normalize=0:dropout_transition=0,"
            f"alimiter=limit=0.95{final_label}"
        )

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
    duck_sfx_under_speech: bool = True,
    duck_threshold: float = 0.035,
    duck_ratio: float = 8.0,
    duck_attack_ms: int = 15,
    duck_release_ms: int = 180,
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
        sfx_events=sfx_events,
        speech_input_index=1,
        duck_sfx_under_speech=duck_sfx_under_speech,
        duck_threshold=duck_threshold,
        duck_ratio=duck_ratio,
        duck_attack_ms=duck_attack_ms,
        duck_release_ms=duck_release_ms,
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
            f"trim_ms={event['trim_ms']}  "
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
