from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from Components.ThumbnailStudioBridge import resolve_comfyui_root

_DEFAULT_WORKFLOW = "user/default/workflows/sdxlturbo_example.json"
_DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
_DEFAULT_PROMPT_WIDTH = 832
_DEFAULT_PROMPT_HEIGHT = 1472
_DEFAULT_EDIT_CHECKPOINT = "sd_xl_turbo_1.0_fp16.safetensors"
_DEFAULT_EDIT_NEGATIVE = (
    "text, letters, caption, watermark, logo, subtitles, screenshot, UI, "
    "extra limbs, deformed, low quality, blurry, distorted, artifacts"
)
_DEFAULT_BACKGROUND_NEGATIVE = (
    "text, letters, caption, watermark, logo, subtitles, screenshot, UI, "
    "people, face, hands, phone, smartphone, camera, blurry, distorted, clutter"
)

_TEMPLATE_MOODS: dict[str, str] = {
    "navy_dark": "deep navy, orange rim light, restrained stage atmosphere",
    "energy_orange": "orange-violet energy, dramatic contrast, dark stage atmosphere",
    "warm_gold": "warm gold glow, soft radiance, reverent atmosphere",
    "cinematic_dark": "moody black stage, subtle bloom, editorial contrast",
    "fire_red": "red-black intensity, dramatic ember glow, sharp contrast",
    "heaven_blue": "blue heaven light, airy highlights, calm depth",
    "bold_minimal": "dark cinematic stage, strong negative space, calm background structure",
    "sunset_warm": "sunset amber and violet, soft bokeh, warm depth",
}

_TITLE_SCENE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("endzeit", "apokalypse", "apocalypse", "revelation"), "stormy horizon, broken stone, distant dawn light"),
    (("loslassen", "loslass", "letting go", "vertrauen"), "open path, soft dawn, calm horizon"),
    (("treue", "standhaft", "steadfast", "faithful"), "solid rock, mountain ridge, sunrise glow"),
    (("spannung", "spannungen", "conflict", "konflikt", "irritation"), "layered shadows, cracked surface, dramatic contrast"),
    (("gnade", "grace", "segen", "segnen", "blessing"), "warm radiant light, gentle haze, subtle bloom"),
    (("bibel", "wort", "word", "truth"), "quiet chapel light, subtle depth, clean focal plane"),
    (("iran", "israel", "krieg", "war"), "dark sky, distant glow, tense horizon"),
)


def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text


def _extract_theme_words(title: str, *, limit: int = 6) -> str:
    words = []
    for token in _clean_text(title).split():
        token = token.strip(".,;:!?()[]{}\"'").lower()
        if len(token) < 4:
            continue
        if token in {
            "move", "church", "antonio", "weil", "olaf", "latzel", "predigt",
            "highlight", "clip", "short", "shorts", "video", "sermon",
        }:
            continue
        if token not in words:
            words.append(token)
    return " ".join(words[:limit])


def _scene_hint_for_title(title: str) -> str:
    lowered = _clean_text(title).lower()
    if not lowered:
        return ""
    hints: list[str] = []
    for markers, phrase in _TITLE_SCENE_HINTS:
        if any(marker in lowered for marker in markers):
            hints.append(phrase)
    # Keep the prompt focused: one or two concrete scene cues are enough.
    unique_hints: list[str] = []
    for hint in hints:
        if hint not in unique_hints:
            unique_hints.append(hint)
    return ", ".join(unique_hints[:2])


def build_background_prompt(
    title: str,
    *,
    template: str = "bold_minimal",
    speaker_name: str | None = None,
    brand_label: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> tuple[str, str]:
    explicit_prompt = _clean_text(prompt)
    explicit_negative = _clean_text(negative_prompt)
    if explicit_prompt:
        return explicit_prompt, explicit_negative or _DEFAULT_BACKGROUND_NEGATIVE

    theme = _extract_theme_words(title)
    scene_hint = _scene_hint_for_title(title)
    mood = _TEMPLATE_MOODS.get(str(template or "").strip().lower(), _TEMPLATE_MOODS["bold_minimal"])
    subject_hint = f" inspired by the message theme {theme}" if theme else ""
    scene_hint_text = f", {scene_hint}" if scene_hint else ""
    speaker_hint = f" for {speaker_name}" if speaker_name else ""
    brand_hint = f" branded for {brand_label}" if brand_label else ""
    positive = (
        "cinematic sermon background, portrait orientation, "
        f"{mood}{subject_hint}{scene_hint_text}{speaker_hint}{brand_hint}, "
        "no people, no text, no watermark, no logo, no subtitles, "
        "subtle depth, soft atmospheric light, mobile thumbnail background"
    )
    negative = (
        "people, face, hands, smartphone, camera, text, letters, caption, watermark, logo, "
        "subtitles, screenshot, UI, busy foreground, clutter, low quality, blurry, distorted"
    )
    return positive, negative


def _workflow_path(comfyui_root: Path | None, workflow_name: str) -> Path | None:
    root = comfyui_root or resolve_comfyui_root()
    if root is None:
        return None
    candidate = root / workflow_name
    return candidate if candidate.exists() else None


def _load_workflow_template(workflow_path: Path) -> dict[str, Any]:
    payload = json.loads(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow JSON must be an object: {workflow_path}")
    return payload


def _ordered_nodes(workflow: dict[str, Any], class_type: str) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") != class_type:
            continue
        items.append((str(node_id), node))
    items.sort(key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0])
    return items


def _prepare_workflow(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    prepared = json.loads(json.dumps(workflow))
    clip_nodes = _ordered_nodes(prepared, "CLIPTextEncode")
    if clip_nodes:
        clip_nodes[0][1].setdefault("inputs", {})["text"] = prompt
    if len(clip_nodes) > 1:
        clip_nodes[1][1].setdefault("inputs", {})["text"] = negative_prompt

    latent_nodes = _ordered_nodes(prepared, "EmptyLatentImage")
    if latent_nodes:
        latent_inputs = latent_nodes[0][1].setdefault("inputs", {})
        latent_inputs["width"] = int(width)
        latent_inputs["height"] = int(height)
        latent_inputs["batch_size"] = 1

    sampler_nodes = _ordered_nodes(prepared, "SamplerCustom") or _ordered_nodes(prepared, "KSampler")
    if sampler_nodes:
        sampler_inputs = sampler_nodes[0][1].setdefault("inputs", {})
        if "noise_seed" in sampler_inputs:
            sampler_inputs["noise_seed"] = int(seed)
        if "seed" in sampler_inputs:
            sampler_inputs["seed"] = int(seed)

    scheduler_nodes = _ordered_nodes(prepared, "SDTurboScheduler")
    if scheduler_nodes:
        scheduler_inputs = scheduler_nodes[0][1].setdefault("inputs", {})
        scheduler_inputs["steps"] = int(scheduler_inputs.get("steps") or 1)
        scheduler_inputs["denoise"] = 1

    save_nodes = _ordered_nodes(prepared, "SaveImage")
    if save_nodes:
        save_nodes[0][1].setdefault("inputs", {})["filename_prefix"] = filename_prefix

    return prepared


def _queue_prompt(prompt: dict[str, Any], base_url: str = _DEFAULT_COMFYUI_URL) -> str:
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    prompt_id = str(data.get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return a prompt_id")
    return prompt_id


def _fetch_history(prompt_id: str, base_url: str = _DEFAULT_COMFYUI_URL) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/history/{prompt_id}")
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _extract_output_image(history_entry: dict[str, Any]) -> dict[str, str] | None:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            if str(image.get("type") or "") != "output":
                continue
            filename = _clean_text(image.get("filename"))
            subfolder = _clean_text(image.get("subfolder"))
            folder_type = _clean_text(image.get("type")) or "output"
            if filename:
                return {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": folder_type,
                }
    return None


def _download_image(image_ref: dict[str, str], base_url: str = _DEFAULT_COMFYUI_URL) -> bytes:
    query = urllib.parse.urlencode(image_ref)
    request = urllib.request.Request(f"{base_url.rstrip('/')}/view?{query}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _procedural_fallback(width: int, height: int, *, template: str, title: str) -> Image.Image:
    from PIL import Image as _Image
    import numpy as np

    palette = {
        "bold_minimal": ((8, 10, 16), (20, 28, 44)),
        "warm_gold": ((20, 16, 12), (62, 42, 20)),
        "sunset_warm": ((26, 10, 40), (60, 24, 6)),
    }.get(template, ((10, 12, 18), (32, 42, 68)))
    top, bottom = palette
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    grad = ((1.0 - ramp) * np.array(top, dtype=np.float32) + ramp * np.array(bottom, dtype=np.float32)).astype(np.uint8)
    canvas = _Image.fromarray(np.repeat(grad[:, None, :], width, axis=1), mode="RGB").convert("RGBA")
    overlay = _Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(
        [int(width * 0.55), int(height * 0.02), int(width * 1.05), int(height * 0.48)],
        fill=(210, 150, 60, 48),
    )
    draw.ellipse(
        [int(width * -0.10), int(height * 0.50), int(width * 0.70), int(height * 1.04)],
        fill=(255, 255, 255, 18),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(24, width // 24)))
    canvas = _Image.alpha_composite(canvas, overlay)
    scrim = _Image.new("RGBA", (width, height), (0, 0, 0, 0))
    scrim_draw = ImageDraw.Draw(scrim)
    for y_pos in range(height):
        frac = y_pos / float(max(1, height - 1))
        alpha = int(32 + 72 * frac)
        scrim_draw.line([(0, y_pos), (width, y_pos)], fill=(6, 8, 14, alpha))
    canvas = _Image.alpha_composite(canvas, scrim)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.05)
    canvas = ImageEnhance.Brightness(canvas).enhance(0.88)
    return canvas


def generate_background_image(
    *,
    title: str,
    template: str = "bold_minimal",
    speaker_name: str | None = None,
    brand_label: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    width: int = _DEFAULT_PROMPT_WIDTH,
    height: int = _DEFAULT_PROMPT_HEIGHT,
    seed: int | None = None,
    cache_dir: str | Path | None = None,
    comfyui_root: str | Path | None = None,
    workflow_name: str = _DEFAULT_WORKFLOW,
    base_url: str = _DEFAULT_COMFYUI_URL,
    timeout_sec: int = 120,
) -> tuple[Image.Image, dict[str, Any]]:
    prompt, negative = build_background_prompt(
        title,
        template=template,
        speaker_name=speaker_name,
        brand_label=brand_label,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    seed_value = int(seed if seed is not None else int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16))
    cache_root = Path(cache_dir).expanduser() if cache_dir else None
    cache_path = None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative": negative,
                    "template": template,
                    "width": int(width),
                    "height": int(height),
                    "seed": seed_value,
                    "workflow": workflow_name,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = cache_root / f"{cache_key}_{template}_{width}x{height}.png"
        if cache_path.exists():
            with Image.open(cache_path) as cached:
                image = cached.convert("RGBA")
            return image, {
                "backend": "cache",
                "cache_path": str(cache_path),
                "prompt": prompt,
                "negative_prompt": negative,
                "seed": seed_value,
                "workflow": workflow_name,
                "dimensions": [int(width), int(height)],
            }

    comfyui_root_path = resolve_comfyui_root(comfyui_root)
    workflow_path = _workflow_path(comfyui_root_path, workflow_name) if comfyui_root_path else None
    if workflow_path is None:
        image = _procedural_fallback(width, height, template=template, title=title)
        if cache_path is not None:
            image.save(cache_path, "PNG")
        return image, {
            "backend": "fallback",
            "reason": "workflow_not_found",
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": seed_value,
            "workflow": workflow_name,
            "dimensions": [int(width), int(height)],
        }

    try:
        workflow = _load_workflow_template(workflow_path)
        filename_prefix = f"parakeet_bg_{template}_{seed_value}"
        prepared = _prepare_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            seed=seed_value,
            filename_prefix=filename_prefix,
        )
        prompt_id = _queue_prompt(prepared, base_url=base_url)
        deadline = time.time() + float(timeout_sec)
        image_ref: dict[str, str] | None = None
        while time.time() < deadline:
            history = _fetch_history(prompt_id, base_url=base_url)
            history_entry = history.get(prompt_id)
            if isinstance(history_entry, dict):
                image_ref = _extract_output_image(history_entry)
                if image_ref is not None:
                    break
            time.sleep(1.0)
        if image_ref is None:
            raise TimeoutError(f"ComfyUI background generation timed out after {timeout_sec}s")
        image_bytes = _download_image(image_ref, base_url=base_url)
        with Image.open(BytesIO(image_bytes)) as generated:
            image = generated.convert("RGBA")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image = ImageEnhance.Color(image).enhance(0.92)
        image = ImageEnhance.Contrast(image).enhance(1.04)
        image = ImageEnhance.Brightness(image).enhance(0.92)
        if cache_path is not None:
            image.save(cache_path, "PNG")
        return image, {
            "backend": "comfyui",
            "cache_path": str(cache_path) if cache_path else None,
            "prompt_id": prompt_id,
            "image_ref": image_ref,
            "workflow_path": str(workflow_path),
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": seed_value,
            "workflow": workflow_name,
            "dimensions": [int(width), int(height)],
        }
    except Exception as exc:
        image = _procedural_fallback(width, height, template=template, title=title)
        if cache_path is not None:
            image.save(cache_path, "PNG")
        return image, {
            "backend": "fallback",
            "reason": f"{type(exc).__name__}: {exc}",
            "cache_path": str(cache_path) if cache_path else None,
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": seed_value,
            "workflow": workflow_name,
            "dimensions": [int(width), int(height)],
        }


def check_server(base_url: str = _DEFAULT_COMFYUI_URL, *, timeout_sec: int = 5) -> dict[str, Any]:
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/system_stats")
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
        system = data.get("system") if isinstance(data, dict) else {}
        system = system if isinstance(system, dict) else {}
        return {
            "up": True,
            "base_url": base_url,
            "comfyui_version": system.get("comfyui_version"),
            "os": system.get("os"),
        }
    except Exception as exc:
        return {"up": False, "base_url": base_url, "reason": f"{type(exc).__name__}: {exc}"}


def detect_workflow_format(path: str | Path) -> str:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return "other"
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return "ui"
    if isinstance(data, dict) and data and all(isinstance(value, dict) and "class_type" in value for value in data.values()):
        return "api"
    return "other"


def run_api_workflow(
    workflow: dict[str, Any],
    *,
    base_url: str = _DEFAULT_COMFYUI_URL,
    timeout_sec: int = 300,
) -> tuple[Image.Image, dict[str, Any]]:
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("workflow must be a non-empty API-format dict")
    if not all(isinstance(value, dict) and "class_type" in value for value in workflow.values()):
        raise ValueError("workflow is not in API format (expected flat {node_id: {class_type, inputs}})")
    prompt_id = _queue_prompt(workflow, base_url=base_url)
    deadline = time.time() + float(timeout_sec)
    image_ref: dict[str, str] | None = None
    while time.time() < deadline:
        history = _fetch_history(prompt_id, base_url=base_url)
        history_entry = history.get(prompt_id)
        if isinstance(history_entry, dict):
            image_ref = _extract_output_image(history_entry)
            if image_ref is not None:
                break
        time.sleep(1.0)
    if image_ref is None:
        raise TimeoutError(f"ComfyUI workflow timed out after {timeout_sec}s")
    image_bytes = _download_image(image_ref, base_url=base_url)
    with Image.open(BytesIO(image_bytes)) as generated:
        image = generated.convert("RGBA")
    return image, {"backend": "comfyui", "prompt_id": prompt_id, "image_ref": image_ref}


def _upload_image(image_path: str | Path, base_url: str = _DEFAULT_COMFYUI_URL) -> str:
    source = Path(image_path).expanduser().resolve(strict=False)
    data = source.read_bytes()
    boundary = "----parakeetcomfy" + hashlib.md5(data[:64] + source.name.encode("utf-8")).hexdigest()[:16]
    body = BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        f'Content-Disposition: form-data; name="image"; filename="{source.name}"\r\n'.encode("utf-8")
    )
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(data)
    body.write(f"\r\n--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n')
    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/upload/image",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.loads(response.read().decode("utf-8"))
    name = _clean_text(info.get("name"))
    subfolder = _clean_text(info.get("subfolder"))
    if not name:
        raise RuntimeError("ComfyUI did not return an uploaded image name")
    return f"{subfolder}/{name}" if subfolder else name


def _build_img2img_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    input_filename: str,
    denoise: float,
    steps: int,
    cfg: float,
    seed: int,
    checkpoint: str,
    filename_prefix: str,
) -> dict[str, Any]:
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "load": {"class_type": "LoadImage", "inputs": {"image": input_filename}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["ckpt", 1]}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["load", 0], "vae": ["ckpt", 2]}},
        "samp": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": float(denoise),
                "model": ["ckpt", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["enc", 0],
            },
        },
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["dec", 0]}},
    }


def generate_edited_image(
    *,
    input_image: str | Path,
    prompt: str,
    negative_prompt: str | None = None,
    denoise: float = 0.6,
    steps: int = 6,
    cfg: float = 1.0,
    seed: int | None = None,
    checkpoint: str = _DEFAULT_EDIT_CHECKPOINT,
    cache_dir: str | Path | None = None,
    base_url: str = _DEFAULT_COMFYUI_URL,
    timeout_sec: int = 180,
) -> tuple[Image.Image, dict[str, Any]]:
    source = Path(input_image).expanduser().resolve(strict=False)
    if not source.exists():
        raise FileNotFoundError(f"Input image not found: {source}")
    prompt_text = _clean_text(prompt)
    negative = _clean_text(negative_prompt) if negative_prompt else _DEFAULT_EDIT_NEGATIVE
    seed_value = int(
        seed
        if seed is not None
        else int(hashlib.md5((prompt_text + source.name).encode("utf-8")).hexdigest()[:8], 16)
    )

    cache_root = Path(cache_dir).expanduser() if cache_dir else None
    cache_path = None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(
            json.dumps(
                {
                    "prompt": prompt_text,
                    "negative": negative,
                    "source": str(source),
                    "source_mtime": source.stat().st_mtime,
                    "denoise": float(denoise),
                    "steps": int(steps),
                    "cfg": float(cfg),
                    "seed": seed_value,
                    "checkpoint": checkpoint,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = cache_root / f"edit_{cache_key}.png"
        if cache_path.exists():
            with Image.open(cache_path) as cached:
                image = cached.convert("RGBA")
            return image, {
                "backend": "cache",
                "cache_path": str(cache_path),
                "prompt": prompt_text,
                "negative_prompt": negative,
                "seed": seed_value,
                "source_image": str(source),
            }

    try:
        uploaded = _upload_image(source, base_url=base_url)
        filename_prefix = f"parakeet_edit_{seed_value}"
        workflow = _build_img2img_workflow(
            prompt=prompt_text,
            negative_prompt=negative,
            input_filename=uploaded,
            denoise=denoise,
            steps=steps,
            cfg=cfg,
            seed=seed_value,
            checkpoint=checkpoint,
            filename_prefix=filename_prefix,
        )
        prompt_id = _queue_prompt(workflow, base_url=base_url)
        deadline = time.time() + float(timeout_sec)
        image_ref: dict[str, str] | None = None
        while time.time() < deadline:
            history = _fetch_history(prompt_id, base_url=base_url)
            history_entry = history.get(prompt_id)
            if isinstance(history_entry, dict):
                image_ref = _extract_output_image(history_entry)
                if image_ref is not None:
                    break
            time.sleep(1.0)
        if image_ref is None:
            raise TimeoutError(f"ComfyUI image edit timed out after {timeout_sec}s")
        image_bytes = _download_image(image_ref, base_url=base_url)
        with Image.open(BytesIO(image_bytes)) as generated:
            image = generated.convert("RGBA")
        if cache_path is not None:
            image.save(cache_path, "PNG")
        return image, {
            "backend": "comfyui",
            "cache_path": str(cache_path) if cache_path else None,
            "prompt_id": prompt_id,
            "image_ref": image_ref,
            "uploaded_as": uploaded,
            "prompt": prompt_text,
            "negative_prompt": negative,
            "denoise": float(denoise),
            "steps": int(steps),
            "cfg": float(cfg),
            "seed": seed_value,
            "checkpoint": checkpoint,
            "source_image": str(source),
        }
    except Exception as exc:
        with Image.open(source) as original:
            image = original.convert("RGBA")
        return image, {
            "backend": "fallback",
            "reason": f"{type(exc).__name__}: {exc}",
            "prompt": prompt_text,
            "negative_prompt": negative,
            "seed": seed_value,
            "source_image": str(source),
        }
