from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from Components.ComfyUIBackground import _fetch_history, _queue_prompt

_DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
_DEFAULT_MODEL = "acestep_v1.5_turbo.safetensors"
_DEFAULT_TEXT_ENCODER = "qwen_0.6b_ace15.safetensors"
_DEFAULT_LM = "qwen_4b_ace15.safetensors"
_DEFAULT_VAE = "ace_1.5_vae.safetensors"


def _clean_filename_prefix(value: str) -> str:
    parts = []
    for part in str(value or "").replace("\\", "/").split("/"):
        part = re.sub(r"[^A-Za-z0-9_.-]+", "_", part).strip("._")
        if part:
            parts.append(part)
    return "/".join(parts) or "audio/parakeet_music"


def build_ace_step_workflow(
    *,
    tags: str,
    duration_sec: float,
    bpm: int,
    seed: int,
    keyscale: str = "D major",
    language: str = "en",
    lyrics: str = "[Instrumental]",
    filename_prefix: str = "audio/parakeet_music",
    audio_format: str = "flac",
    generate_audio_codes: bool = False,
) -> dict[str, Any]:
    """Build a native ComfyUI API graph for ACE-Step 1.5 Turbo."""
    prompt = " ".join(str(tags or "").split())
    if not prompt:
        raise ValueError("tags must not be empty")
    if not 10.0 <= float(duration_sec) <= 600.0:
        raise ValueError("duration_sec must be between 10 and 600 seconds")
    if not 10 <= int(bpm) <= 300:
        raise ValueError("bpm must be between 10 and 300")
    if audio_format not in {"flac", "mp3"}:
        raise ValueError("audio_format must be 'flac' or 'mp3'")

    save_node: dict[str, Any] = {
        "class_type": "SaveAudio" if audio_format == "flac" else "SaveAudioMP3",
        "inputs": {
            "audio": ["level", 0],
            "filename_prefix": _clean_filename_prefix(filename_prefix),
        },
    }
    if audio_format == "mp3":
        save_node["inputs"]["quality"] = "320k"

    return {
        "model": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": _DEFAULT_MODEL, "weight_dtype": "default"},
        },
        "sampling": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["model", 0], "shift": 3.0},
        },
        "clip": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": _DEFAULT_TEXT_ENCODER,
                "clip_name2": _DEFAULT_LM,
                "type": "ace",
                "device": "default",
            },
        },
        "positive": {
            "class_type": "TextEncodeAceStepAudio1.5",
            "inputs": {
                "clip": ["clip", 0],
                "tags": prompt,
                "lyrics": str(lyrics or "[Instrumental]"),
                "seed": int(seed),
                "bpm": int(bpm),
                "duration": float(duration_sec),
                "timesignature": "4",
                "language": str(language),
                "keyscale": str(keyscale),
                "generate_audio_codes": bool(generate_audio_codes),
                "cfg_scale": 2.0,
                "temperature": 0.85,
                "top_p": 0.9,
                "top_k": 0,
                "min_p": 0.0,
            },
        },
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["positive", 0]},
        },
        "latent": {
            "class_type": "EmptyAceStep1.5LatentAudio",
            "inputs": {"seconds": float(duration_sec), "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["sampling", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
                "seed": int(seed),
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": _DEFAULT_VAE},
        },
        "decode": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
        },
        "level": {
            "class_type": "AudioAdjustVolume",
            "inputs": {"audio": ["decode", 0], "volume": -3},
        },
        "save": save_node,
    }


def _extract_output_audio(history_entry: dict[str, Any]) -> dict[str, str] | None:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        audio_items = node_output.get("audio")
        if not isinstance(audio_items, list):
            continue
        for audio in audio_items:
            if not isinstance(audio, dict):
                continue
            filename = str(audio.get("filename") or "").strip()
            if filename:
                return {
                    "filename": filename,
                    "subfolder": str(audio.get("subfolder") or "").strip(),
                    "type": str(audio.get("type") or "output").strip(),
                }
    return None


def _history_error(history_entry: dict[str, Any]) -> str | None:
    status = history_entry.get("status")
    if not isinstance(status, dict) or status.get("status_str") != "error":
        return None
    messages = status.get("messages")
    return json.dumps(messages, ensure_ascii=False, default=str)


def _download_audio(audio_ref: dict[str, str], base_url: str) -> bytes:
    query = urllib.parse.urlencode(audio_ref)
    request = urllib.request.Request(f"{base_url.rstrip('/')}/view?{query}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _probe_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    value = result.stdout.strip()
    if value and value != "N/A":
        return float(value)

    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    times = re.findall(r"^out_time_us=(\d+)$", decoded.stdout, re.MULTILINE)
    if not times:
        raise RuntimeError(f"Could not determine audio duration: {audio_path}")
    return int(times[-1]) / 1_000_000.0


def prepare_loopable_audio(audio_path: str | Path, *, crossfade_sec: float = 1.5) -> dict[str, float]:
    """Trim a quiet ending and make the file wrap with a tail-to-head crossfade."""
    target = Path(audio_path).expanduser().resolve(strict=True)
    if target.suffix.lower() not in {".flac", ".mp3"}:
        raise ValueError("audio_path must end in .flac or .mp3")
    original_duration = _probe_duration(target)
    temp_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}_trim_",
            suffix=".flac",
            dir=target.parent,
            delete=False,
        ) as handle:
            trimmed = Path(handle.name)
        temp_paths.append(trimmed)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(target),
                "-af",
                (
                    "areverse,"
                    "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-45dB,"
                    "areverse"
                ),
                "-c:a",
                "flac",
                str(trimmed),
            ],
            check=True,
        )
        trimmed_duration = _probe_duration(trimmed)
        fade = min(float(crossfade_sec), max(0.0, trimmed_duration / 4.0))
        if fade < 0.1:
            os.replace(trimmed, target)
            temp_paths.remove(trimmed)
            return {
                "original_duration_sec": original_duration,
                "trimmed_duration_sec": trimmed_duration,
                "output_duration_sec": trimmed_duration,
                "crossfade_sec": 0.0,
            }

        output_duration = trimmed_duration - fade
        middle_end = trimmed_duration - fade
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}_loop_",
            suffix=target.suffix,
            dir=target.parent,
            delete=False,
        ) as handle:
            looped = Path(handle.name)
        temp_paths.append(looped)
        codec_args = ["-c:a", "flac"] if target.suffix.lower() == ".flac" else [
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
        ]
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(trimmed),
                "-filter_complex",
                (
                    f"[0:a]asplit=3[tail_in][head_in][middle_in];"
                    f"[tail_in]atrim=start={middle_end:.6f}:end={trimmed_duration:.6f},"
                    f"asetpts=PTS-STARTPTS,afade=t=out:st=0:d={fade:.6f}[tail];"
                    f"[head_in]atrim=start=0:end={fade:.6f},"
                    f"asetpts=PTS-STARTPTS,afade=t=in:st=0:d={fade:.6f}[head];"
                    f"[tail][head]amix=inputs=2:duration=longest:normalize=0[seam];"
                    f"[middle_in]atrim=start={fade:.6f}:end={middle_end:.6f},"
                    f"asetpts=PTS-STARTPTS[middle];"
                    f"[seam][middle]concat=n=2:v=0:a=1,"
                    f"alimiter=limit=0.707:attack=5:release=50:level=false:latency=true[out]"
                ),
                "-map",
                "[out]",
                *codec_args,
                str(looped),
            ],
            check=True,
        )
        os.replace(looped, target)
        temp_paths.remove(looped)
        return {
            "original_duration_sec": original_duration,
            "trimmed_duration_sec": trimmed_duration,
            "output_duration_sec": _probe_duration(target),
            "crossfade_sec": fade,
        }
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)


def generate_music(
    *,
    tags: str,
    output_path: str | Path,
    duration_sec: float = 45.0,
    bpm: int = 78,
    seed: int | None = None,
    keyscale: str = "D major",
    language: str = "en",
    lyrics: str = "[Instrumental]",
    generate_audio_codes: bool = False,
    prepare_loop: bool = True,
    crossfade_sec: float = 1.5,
    base_url: str = _DEFAULT_COMFYUI_URL,
    timeout_sec: int = 600,
) -> tuple[Path, dict[str, Any]]:
    """Generate ACE-Step music through a running ComfyUI server."""
    target = Path(output_path).expanduser().resolve(strict=False)
    audio_format = target.suffix.lower().lstrip(".")
    if audio_format not in {"flac", "mp3"}:
        raise ValueError("output_path must end in .flac or .mp3")

    if seed is None:
        seed_payload = f"{tags}|{duration_sec}|{bpm}|{keyscale}|{language}|{lyrics}"
        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)

    workflow = build_ace_step_workflow(
        tags=tags,
        duration_sec=duration_sec,
        bpm=bpm,
        seed=seed,
        keyscale=keyscale,
        language=language,
        lyrics=lyrics,
        filename_prefix=f"audio/parakeet_music_{seed}",
        audio_format=audio_format,
        generate_audio_codes=generate_audio_codes,
    )
    prompt_id = _queue_prompt(workflow, base_url=base_url)
    deadline = time.time() + float(timeout_sec)
    audio_ref: dict[str, str] | None = None
    while time.time() < deadline:
        history = _fetch_history(prompt_id, base_url=base_url)
        history_entry = history.get(prompt_id)
        if isinstance(history_entry, dict):
            error = _history_error(history_entry)
            if error:
                raise RuntimeError(f"ComfyUI music generation failed: {error}")
            audio_ref = _extract_output_audio(history_entry)
            if audio_ref is not None:
                break
        time.sleep(1.0)
    if audio_ref is None:
        raise TimeoutError(f"ComfyUI music generation timed out after {timeout_sec}s")

    audio_bytes = _download_audio(audio_ref, base_url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(audio_bytes)
    loop_info = (
        prepare_loopable_audio(target, crossfade_sec=crossfade_sec)
        if prepare_loop
        else None
    )
    return target, {
        "backend": "comfyui",
        "model": "ACE-Step 1.5 Turbo",
        "prompt_id": prompt_id,
        "audio_ref": audio_ref,
        "tags": " ".join(str(tags).split()),
        "lyrics": lyrics,
        "generate_audio_codes": bool(generate_audio_codes),
        "loop_preparation": loop_info,
        "duration_sec": float(duration_sec),
        "bpm": int(bpm),
        "keyscale": keyscale,
        "language": language,
        "seed": int(seed),
        "output_path": str(target),
    }
