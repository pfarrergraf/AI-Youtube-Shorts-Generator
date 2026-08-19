"""
High-quality speaker matting via ComfyUI, with an offline fallback.
===================================================================

rembg leaves two visible defects on sermon frames:

* a bright fringe of leftover background along the subject edge, and
* everything person-shaped in the frame, including a second speaker and the
  podium, because it has no notion of "the primary subject".

``LayerMask: PersonMaskUltra V2`` from ComfyUI_LayerStyle_Advance fixes the
first (VITMatte detail refinement) and the second is handled here, by keeping
only the connected component that contains the primary face.

Every entry point degrades rather than raises: if ComfyUI is down the caller
gets the rembg result and a reason string, so the Sunday pipeline never blocks
on a service being off.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from Components.ComfyUIBackground import _DEFAULT_COMFYUI_URL, check_server

_PERSON_PARTS = ("face", "hair", "body", "clothes")


@dataclass
class MattingResult:
    subject_rgba: Image.Image
    provider: str
    coverage: float
    face_box: tuple[int, int, int, int] | None = None
    info: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# ComfyUI plumbing
# ────────────────────────────────────────────────────────────────────────────

def build_person_mask_workflow(
    image_name: str,
    *,
    confidence: float = 0.4,
    detail_method: str = "VITMatte",
    detail_erode: int = 6,
    detail_dilate: int = 6,
    black_point: float = 0.15,
    white_point: float = 0.99,
    filename_prefix: str = "parakeet_mask",
) -> dict:
    """API-format graph: LoadImage -> PersonMaskUltra V2 -> MaskToImage -> SaveImage.

    The mask travels back as a greyscale PNG rather than as an RGBA image,
    because SaveImage writes RGB and would silently drop the alpha channel.
    """
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "LayerMask: PersonMaskUltra V2",
            "inputs": {
                "images": ["1", 0],
                "face": True,
                "hair": True,
                "body": True,
                "clothes": True,
                # Keep the handheld microphone — it reads as part of the speaker
                # and looks like a bite out of the jacket when dropped.
                "accessories": True,
                "background": False,
                "confidence": confidence,
                "detail_method": detail_method,
                "detail_erode": detail_erode,
                "detail_dilate": detail_dilate,
                "black_point": black_point,
                "white_point": white_point,
                "process_detail": True,
                "device": "cuda",
                "max_megapixels": 4.0,
            },
        },
        "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 1]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": filename_prefix}},
    }


def _upload_image(image: Image.Image, base_url: str, name: str, timeout: int) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    payload = buf.getvalue()

    boundary = f"----parakeet{uuid.uuid4().hex}"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n",
        payload,
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{base_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()).get("name", name)


def _run_workflow(workflow: dict, base_url: str, timeout: int) -> Image.Image:
    req = urllib.request.Request(
        f"{base_url}/prompt",
        data=json.dumps({"prompt": workflow}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        prompt_id = json.loads(resp.read().decode())["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{base_url}/history/{prompt_id}", timeout=15) as resp:
            history = json.loads(resp.read().decode())
        entry = history.get(prompt_id)
        if entry and entry.get("outputs"):
            for out in entry["outputs"].values():
                for meta in out.get("images", []):
                    query = urllib.parse.urlencode(
                        {"filename": meta["filename"], "subfolder": meta.get("subfolder", ""),
                         "type": meta.get("type", "output")}
                    )
                    with urllib.request.urlopen(f"{base_url}/view?{query}", timeout=60) as img_resp:
                        return Image.open(io.BytesIO(img_resp.read())).copy()
        time.sleep(1.0)
    raise TimeoutError(f"ComfyUI did not return a mask within {timeout}s")


import urllib.parse  # noqa: E402  (used inside _run_workflow)


# ────────────────────────────────────────────────────────────────────────────
# Primary-subject isolation
# ────────────────────────────────────────────────────────────────────────────

def keep_primary_component(
    alpha: np.ndarray,
    face_box: tuple[int, int, int, int] | None = None,
    *,
    min_area_ratio: float = 0.004,
) -> np.ndarray:
    """Drop everything except the person the thumbnail is about.

    Sermon frames routinely contain a second speaker, the band, or a lit
    podium; a person-segmentation model keeps all of them. Selection prefers the
    component under `face_box`, falling back to the largest one.
    """
    from scipy import ndimage  # noqa: PLC0415

    binary = alpha > 96
    labels, n = ndimage.label(binary)

    keep = binary
    if n > 1:
        sizes = ndimage.sum(binary, labels, index=range(1, n + 1))
        total = float(binary.size)

        chosen = None
        if face_box is not None:
            fx, fy, fw, fh = face_box
            cy, cx = int(fy + fh / 2), int(fx + fw / 2)
            if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
                label_at_face = int(labels[cy, cx])
                if label_at_face > 0:
                    chosen = label_at_face
        if chosen is None:
            chosen = int(np.argmax(sizes)) + 1

        if sizes[chosen - 1] / total >= min_area_ratio:
            keep = labels == chosen

    # Close interior holes. Where a second person's arm crossed in front of the
    # subject, the segmenter assigns those pixels to the other person and drops
    # them here, punching a hole straight through the torso. This runs even for
    # a single component — a lone subject with a hole in it is the common case,
    # and returning early on `n == 1` would skip the repair entirely.
    filled = ndimage.binary_fill_holes(keep)
    holes = filled & ~keep

    out = np.where(filled, alpha, 0).astype(alpha.dtype)
    # Inside a filled hole the source alpha is 0, so masking alone leaves the
    # hole exactly as it was. The pixels have to be made opaque explicitly.
    out[holes] = 255
    return out


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────

def extract_subject(
    frame_bgr: np.ndarray,
    *,
    face_box: tuple[int, int, int, int] | None = None,
    isolate_primary: bool = True,
    base_url: str = _DEFAULT_COMFYUI_URL,
    timeout_sec: int = 180,
    allow_fallback: bool = True,
) -> MattingResult:
    """Matte the speaker. Prefers ComfyUI PersonMaskUltra, falls back to rembg."""
    rgb = Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGB")

    status = check_server(base_url)
    if status.get("up"):
        try:
            name = _upload_image(rgb, base_url, f"parakeet_src_{uuid.uuid4().hex[:8]}.png", timeout_sec)
            workflow = build_person_mask_workflow(name)
            mask_img = _run_workflow(workflow, base_url, timeout_sec)
            alpha = np.asarray(mask_img.convert("L").resize(rgb.size), dtype=np.uint8)
            if isolate_primary:
                alpha = keep_primary_component(alpha, face_box)
            subject = rgb.convert("RGBA")
            subject.putalpha(Image.fromarray(alpha))
            coverage = float((alpha > 96).mean())
            return MattingResult(
                subject_rgba=_crop_to_alpha(subject),
                provider="comfyui_person_mask_ultra_v2",
                coverage=coverage,
                face_box=face_box,
                info={"detail_method": "VITMatte", "isolated_primary": isolate_primary},
            )
        except (urllib.error.URLError, TimeoutError, KeyError, OSError, ImportError) as exc:
            if not allow_fallback:
                raise
            reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = f"comfyui_down: {status.get('reason', 'unknown')}"

    from Components.ThumbnailMoveChurch import _extract_speaker_cutout  # noqa: PLC0415

    legacy = _extract_speaker_cutout(frame_bgr, bg_removal_provider="auto")
    return MattingResult(
        subject_rgba=legacy["speaker_rgba"],
        provider=f"fallback_{legacy.get('provider_used')}",
        coverage=float(legacy.get("coverage") or 0.0),
        face_box=legacy.get("face_box"),
        info={"fallback_reason": reason},
    )


def _crop_to_alpha(rgba: Image.Image) -> Image.Image:
    bbox = rgba.split()[-1].getbbox()
    return rgba.crop(bbox) if bbox else rgba


def upscale_subject(
    subject_rgba: Image.Image,
    *,
    base_url: str = _DEFAULT_COMFYUI_URL,
    model_name: str = "RealESRGAN_x4plus.safetensors",
    timeout_sec: int = 240,
    min_width: int = 900,
) -> tuple[Image.Image, str]:
    """4x the cutout through RealESRGAN when it is too soft to hold up at 1080 wide.

    The alpha is carried across separately and re-attached, because the upscale
    graph is RGB-only.
    """
    if subject_rgba.width >= min_width:
        return subject_rgba, "skipped_already_sharp"
    if not check_server(base_url).get("up"):
        return subject_rgba, "skipped_comfyui_down"

    try:
        rgb = subject_rgba.convert("RGB")
        name = _upload_image(rgb, base_url, f"parakeet_up_{uuid.uuid4().hex[:8]}.png", timeout_sec)
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": name}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
            "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "parakeet_upscaled"}},
        }
        big = _run_workflow(workflow, base_url, timeout_sec).convert("RGB")
        alpha = subject_rgba.split()[-1].resize(big.size, Image.LANCZOS)
        out = big.convert("RGBA")
        out.putalpha(alpha)
        return out, f"upscaled_{model_name}"
    except (urllib.error.URLError, TimeoutError, KeyError, OSError) as exc:
        return subject_rgba, f"upscale_failed: {type(exc).__name__}: {exc}"
