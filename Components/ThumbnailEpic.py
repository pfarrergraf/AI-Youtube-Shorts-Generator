"""
The "epic" thumbnail composer.
==============================

Puts the reference look together: dark stage, one light source, a real speaker
cutout, and reference-scale typography that overlaps the subject for depth.

Z-order, mirroring the references::

    background plate
    haze
    god rays / light shaft
    back glow
    BACK text lines        (the upper lines, behind the speaker)
    speaker cutout + rim light
    FRONT text lines       (the lower lines, in front — this is the depth cue)
    bloom / vignette / grain / finish

Speaker-render variants are selected with `speaker_render`; the ones that need
ComfyUI degrade to the offline path when the server is down, so a render never
fails just because a service is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from PIL import Image

from Components import ThumbnailAtmosphere as atmo
from Components.ThumbnailReferenceGate import GateResult, load_bands, run_gate
from Components.ThumbnailTypeEngine import TypeLayout, layout_and_render

CANVAS_9X16 = (1080, 1920)

SPEAKER_RENDERS = (
    "frame_cinematic",   # real frame darkened as the stage, cutout on top — no server
    "real_procedural",   # matted cutout + procedural rim/rays — matting only
    "real_relight",      # cutout relit through img2img — ComfyUI
    "ai_plate",          # model generates the light environment, real cutout on top
    "ai_repertoire",     # approved text-free speaker hero from assets/ — no server
)

# The references are portrait-led: the face must still read at phone size.
# Keep the crop recipes explicit so a design change cannot silently turn the
# speaker back into a full figure.
SPEAKER_LAYOUTS = {
    "balanced": {"min_face_ratio": 0.22, "waist_factor": 4.6, "height": 0.66},
    "closeup": {"min_face_ratio": 0.34, "waist_factor": 3.2, "height": 0.72},
    "portrait": {"min_face_ratio": 0.40, "waist_factor": 2.8, "height": 0.76},
}

# Named light setups, each a (light colour, accent colour, rays vs shaft) recipe.
MOODS: dict[str, dict] = {
    "warm_shaft":  {"light": "warm",    "accent": "red",    "shape": "rays"},
    "gold_burst":  {"light": "gold",    "accent": "yellow", "shape": "rays"},
    "cool_door":   {"light": "cool",    "accent": "cyan",   "shape": "shaft"},
    "cyan_split":  {"light": "cyan",    "accent": "red",    "shape": "shaft"},
    "red_alert":   {"light": "red",     "accent": "white",  "shape": "rays"},
    "white_stage": {"light": "white",   "accent": "red",    "shape": "shaft"},
}
DEFAULT_MOOD = "warm_shaft"


def _speaker_key(speaker_name: str | None) -> str:
    """Resolve common speaker labels to the repertoire key."""
    value = " ".join(str(speaker_name or "").lower().split())
    aliases = {
        "antonio": "antonio_weil",
        "antonio weil": "antonio_weil",
        "pastor antonio weil": "antonio_weil",
        "olaf": "olaf_latzel",
        "olaf latzel": "olaf_latzel",
        "pastor olaf latzel": "olaf_latzel",
        "leo bigger": "leo_bigger_icf",
    }
    return aliases.get(value, value.replace(" ", "_")) if value else ""


def load_speaker_hero(
    speaker_name: str | None,
    *,
    variant: str | None = None,
    seed: int = 0,
) -> tuple[Image.Image | None, dict]:
    """Load an approved, text-free hero from the local repertoire.

    This function intentionally never generates an image. Generation is an
    explicit offline/job step; normal rendering must be deterministic and
    cost-free when a hero already exists.
    """
    key = _speaker_key(speaker_name)
    if not key:
        return None, {"backend": "repertoire", "status": "no_speaker"}
    root = Path(__file__).resolve().parents[1] / "assets" / "speaker_references"
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("speakers", {}).get(key, {}).get("heroes", [])
    except (OSError, ValueError, TypeError):
        return None, {"backend": "repertoire", "status": "manifest_unavailable", "speaker": key}
    candidates = [
        item for item in entries
        if isinstance(item, dict) and item.get("approved") and item.get("text_free")
    ]
    if variant:
        preferred = [item for item in candidates if str(item.get("variant")) == variant]
        if preferred:
            candidates = preferred
    if not candidates:
        return None, {"backend": "repertoire", "status": "no_approved_hero", "speaker": key}
    item = candidates[int(seed) % len(candidates)]
    path = root / str(item.get("path") or "")
    if not path.is_file():
        return None, {"backend": "repertoire", "status": "asset_missing", "path": str(path)}
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None, {"backend": "repertoire", "status": "asset_unreadable", "path": str(path)}
    return image, {
        "backend": "repertoire",
        "status": "loaded",
        "speaker": key,
        "variant": item.get("variant"),
        "path": str(path),
        "provider": item.get("provider"),
        "approved": bool(item.get("approved")),
        "text_free": bool(item.get("text_free")),
    }


@dataclass
class EpicResult:
    image: Image.Image
    hook: str
    mood: str
    speaker_render: str
    type_layout: TypeLayout | None = None
    gate: GateResult | None = None
    info: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        out = {
            "hook": self.hook,
            "mood": self.mood,
            "speaker_render": self.speaker_render,
            **self.info,
        }
        if self.type_layout is not None:
            out["type"] = self.type_layout.metrics()
        if self.gate is not None:
            out["gate"] = self.gate.as_dict()
        return out


# ────────────────────────────────────────────────────────────────────────────
# Stage
# ────────────────────────────────────────────────────────────────────────────

def build_stage(
    size: tuple[int, int],
    mood: dict,
    *,
    frame_bgr: np.ndarray | None = None,
    light_center: tuple[float, float] = (0.5, 0.40),
    seed: int = 0,
    darken: float = 0.30,
) -> Image.Image:
    """Near-black stage with exactly one light source."""
    w, h = size
    if frame_bgr is not None:
        base = Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGB")
        base = _cover_resize(base, size)
        arr = np.asarray(base, dtype=np.float32) / 255.0
        arr = arr * darken
        canvas = Image.fromarray((arr * 255).astype(np.uint8), "RGB").filter(
            __import__("PIL.ImageFilter", fromlist=["ImageFilter"]).GaussianBlur(max(6, w // 90))
        )
    else:
        canvas = Image.new("RGB", size, (7, 8, 11))

    layers = [atmo.atmospheric_haze(size, color=mood["light"], density=0.09, seed=seed)]
    if mood["shape"] == "shaft":
        layers.append(
            atmo.light_shaft(
                size,
                apex=(light_center[0], 0.0),
                base_width=0.62,
                color=mood["light"],
                intensity=0.85,
            )
        )
    else:
        layers.append(
            atmo.god_rays(
                size,
                origin=(light_center[0], max(0.02, light_center[1] - 0.30)),
                color=mood["light"],
                intensity=0.70,
                seed=seed,
            )
        )
    layers.append(
        atmo.back_glow(size, center=light_center, radius=0.40, color=mood["light"], intensity=0.95)
    )
    # Wide, weak ambient fill. Without it the frame is ~72% near-black and lands
    # outside the reference dark_fraction band (0.34-0.67): the references are
    # high contrast, not uniformly black.
    layers.append(
        atmo.back_glow(size, center=(0.5, 0.55), radius=1.15, color=mood["light"], power=1.15, intensity=0.30)
    )
    return atmo.apply_light_stack(canvas, layers)


def build_ai_plate(
    size: tuple[int, int],
    mood: dict,
    *,
    theme: str = "",
    seed: int = 0,
) -> tuple[Image.Image | None, dict]:
    """Generate only the *light environment* with SDXL — no people, no text.

    The sleeper option among the speaker variants: the model supplies the
    cinematic lighting the references have, while the face stays the real
    speaker's, composited on top. Returns (None, info) when ComfyUI is down.
    """
    from Components.ComfyUIBackground import generate_background_image  # noqa: PLC0415

    prompt = (
        f"dark empty stage, single dramatic {mood['light']} light beam from above, "
        f"volumetric haze, deep black background, cinematic rim lighting, "
        f"hyper contrast, no people, no text, portrait orientation"
    )
    if theme:
        prompt = f"{prompt}, mood of {theme}"
    try:
        image, info = generate_background_image(
            title=theme or "sermon",
            prompt=prompt,
            negative_prompt="people, person, face, text, watermark, logo, letters, subtitles",
            width=832,
            height=1472,
            seed=seed,
        )
        if info.get("backend") == "procedural":
            return None, info
        return _cover_resize(image.convert("RGB"), size), info
    except (OSError, ValueError, RuntimeError) as exc:
        return None, {"backend": "error", "reason": f"{type(exc).__name__}: {exc}"}


def relight_subject(
    subject_rgba: Image.Image,
    mood: dict,
    *,
    denoise: float = 0.30,
    seed: int = 0,
) -> tuple[Image.Image, dict]:
    """Relight the real cutout through img2img at low denoise.

    Low denoise on purpose: the face has to stay recognisably the same person.
    The alpha is preserved from the original, since the edit path is RGB-only.
    """
    import tempfile  # noqa: PLC0415

    from Components.ComfyUIBackground import generate_edited_image  # noqa: PLC0415

    # generate_edited_image takes a path, not an Image.
    tmp = Path(tempfile.gettempdir()) / f"parakeet_relight_{seed}.png"
    subject_rgba.convert("RGB").save(tmp)

    try:
        edited, info = generate_edited_image(
            input_image=str(tmp),
            prompt=(
                f"dramatic {mood['light']} cinematic rim lighting on the person, "
                f"studio key light, deep shadows, high contrast portrait, sharp detail"
            ),
            negative_prompt="blurry, distorted face, extra limbs, text, watermark",
            denoise=denoise,
            seed=seed,
        )
        if info.get("backend") in {"fallback", "error"}:
            return subject_rgba, info
        out = edited.convert("RGB").resize(subject_rgba.size, Image.LANCZOS).convert("RGBA")
        out.putalpha(subject_rgba.split()[-1])
        return out, info
    except (OSError, ValueError, RuntimeError) as exc:
        return subject_rgba, {"backend": "error", "reason": f"{type(exc).__name__}: {exc}"}


def build_ai_hero(
    size: tuple[int, int],
    mood: dict,
    *,
    speaker_hint: str = "a middle-aged male preacher",
    theme: str = "",
    seed: int = 0,
) -> tuple[Image.Image | None, dict]:
    """Fully synthetic hero portrait — no real likeness is used as input.

    Included because it was asked for as a comparison point. Note the person in
    the result is invented, so it should not be presented as a photograph of the
    actual speaker.
    """
    from Components.ComfyUIBackground import generate_background_image  # noqa: PLC0415

    prompt = (
        f"{speaker_hint}, dramatic {mood['light']} backlight, dark stage, "
        f"hyper saturated, high contrast, emotionally expressive, cinematic portrait, "
        f"vertical composition, no text"
    )
    if theme:
        prompt = f"{prompt}, theme of {theme}"
    try:
        image, info = generate_background_image(
            title=theme or "portrait",
            prompt=prompt,
            negative_prompt="text, watermark, logo, letters, deformed, extra limbs",
            width=832,
            height=1472,
            seed=seed,
        )
        if info.get("backend") == "procedural":
            return None, info
        return _cover_resize(image.convert("RGB"), size), info
    except (OSError, ValueError, RuntimeError) as exc:
        return None, {"backend": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize preserving aspect so the image covers `size`, then centre-crop."""
    tw, th = size
    scale = max(tw / image.width, th / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
    left = (resized.width - tw) // 2
    top = (resized.height - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _detect_largest_face_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect the largest face on the selected, final-size hero plate."""
    try:
        import cv2  # noqa: PLC0415

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        model_root = Path(__file__).resolve().parents[1] / "models"
        prototxt = model_root / "deploy.prototxt"
        weights = model_root / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        if prototxt.is_file() and weights.is_file():
            net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
            blob = cv2.dnn.blobFromImage(
                cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            net.setInput(blob)
            detections = net.forward()[0, 0]
            candidates = []
            for detection in detections:
                if float(detection[2]) < 0.55:
                    continue
                x1, y1, x2, y2 = detection[3:7] * np.array(
                    [image.width, image.height, image.width, image.height]
                )
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
                if x2 > x1 and y2 > y1:
                    candidates.append((x1, y1, x2 - x1, y2 - y1))
            if candidates:
                return max(candidates, key=lambda box: box[2] * box[3])
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        min_side = max(48, min(image.size) // 12)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
        if len(faces):
            return tuple(int(value) for value in max(faces, key=lambda box: box[2] * box[3]))
    except Exception:
        pass
    return None


def _padded_face_box(
    face_box: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand an ``x,y,w,h`` face into a strict no-type ``x1,y1,x2,y2`` zone."""
    x, y, fw, fh = face_box
    w, h = size
    pad_x = max(24, int(round(fw * 0.12)))
    pad_y = max(32, int(round(fh * 0.18)))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(w, x + fw + pad_x),
        min(h, y + fh + pad_y),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _layout_title_outside_face(
    hook: str,
    *,
    size: tuple[int, int],
    accent_line: int | None,
    accent_color: str,
    font: str,
    text_anchor: str,
    seed: int,
    face_box: tuple[int, int, int, int] | None,
) -> tuple[TypeLayout, tuple[int, int, int, int] | None, bool]:
    """Render the title in the first lane that does not touch the face zone."""
    preferred = 0.045 if text_anchor == "top" else 0.53
    safe_box = _padded_face_box(face_box, size) if face_box is not None else None
    candidates = [preferred]
    if face_box is not None:
        candidates.extend([
            0.53 if text_anchor == "top" else 0.045,
            min(0.78, (safe_box[3] + 2) / max(1, size[1])),
            0.02,
        ])

    first = None
    for max_cap_ratio in (0.152, 0.145, 0.135, 0.125):
        for block_top in dict.fromkeys(round(value, 5) for value in candidates):
            layout = layout_and_render(
                hook,
                canvas_size=size,
                accent_line=accent_line,
                accent_color=accent_color,
                font=font,
                block_top_ratio=block_top,
                max_cap_ratio=max_cap_ratio,
                seed=seed,
            )
            first = first or layout
            x1, y1, x2, y2 = layout.block_box
            within_canvas = x1 >= 0 and y1 >= 0 and x2 <= size[0] and y2 <= size[1]
            if within_canvas and (safe_box is None or not _boxes_overlap(layout.block_box, safe_box)):
                return layout, safe_box, block_top != preferred
    return first, safe_box, False


def frame_subject(
    subject_rgba: Image.Image,
    face_box: tuple[int, int, int, int] | None,
    *,
    min_face_ratio: float = 0.22,
    waist_factor: float = 4.6,
) -> Image.Image:
    """Crop a full standing figure down to waist-up so the face reads large.

    A whole-body cutout scaled to fit 9:16 leaves a face maybe 5% of the frame;
    the references put the head at 20-35%. The crop is top-anchored, so any face
    coordinates measured on the original stay valid.
    """
    if face_box is None:
        return subject_rgba
    _, fy, _, fh = face_box
    if fh / max(1, subject_rgba.height) >= min_face_ratio:
        return subject_rgba
    waist = int(fy + fh * waist_factor)
    if waist >= subject_rgba.height:
        return subject_rgba
    cropped = subject_rgba.crop((0, 0, subject_rgba.width, waist))
    bbox = cropped.split()[-1].getbbox()
    return cropped.crop(bbox) if bbox else cropped


def _speaker_placement_geometry(
    canvas_size: tuple[int, int],
    subject_size: tuple[int, int],
    *,
    target_height_ratio: float,
    anchor_x: float,
    bottom_ratio: float,
) -> tuple[int, int, int, float]:
    """Where a subject cutout lands on canvas, before any pixels are touched.

    Pure geometry (no canvas content, no resampling) so callers can learn the
    speaker's future footprint — in particular the face band — before deciding
    what else to composite underneath it.
    """
    w, h = canvas_size
    subject_w, subject_h = subject_size
    target_h = int(h * target_height_ratio)
    scale = target_h / max(1, subject_h)
    scaled_w = max(1, int(subject_w * scale))
    scaled_h = max(1, int(subject_h * scale))
    x = int(anchor_x * w - scaled_w / 2)
    y = int(bottom_ratio * h - scaled_h)
    return x, y, scaled_w, scale


def place_speaker(
    canvas: Image.Image,
    subject_rgba: Image.Image,
    *,
    mood: dict,
    target_height_ratio: float = 0.62,
    anchor_x: float = 0.5,
    bottom_ratio: float = 1.0,
    rim: bool = True,
) -> tuple[Image.Image, tuple[int, int, int, int], Image.Image]:
    """Composite the cutout, bottom-anchored. Returns canvas, box, placed alpha."""
    w, h = canvas.size
    subject = subject_rgba.convert("RGBA")
    x, y, scaled_w, scale = _speaker_placement_geometry(
        (w, h),
        subject.size,
        target_height_ratio=target_height_ratio,
        anchor_x=anchor_x,
        bottom_ratio=bottom_ratio,
    )
    subject = subject.resize(
        (scaled_w, max(1, int(subject.height * scale))), Image.LANCZOS
    )
    if rim:
        subject = atmo.rim_light_from_alpha(subject, color=mood["light"], width=max(3, w // 260))

    out = canvas.convert("RGBA")
    out.alpha_composite(subject, (x, y))

    # Full-canvas alpha at the final position. The halo metric has to compare
    # the composite against *this*, not against the raw cutout — those differ in
    # both scale and position, and comparing them measures nothing.
    placed_alpha = Image.new("L", (w, h), 0)
    placed_alpha.paste(subject.split()[-1], (x, y))

    return out.convert("RGB"), (x, y, x + subject.width, y + subject.height), placed_alpha


def compose(
    hook: str,
    *,
    subject_rgba: Image.Image | None = None,
    frame_bgr: np.ndarray | None = None,
    size: tuple[int, int] = CANVAS_9X16,
    mood: str = DEFAULT_MOOD,
    speaker_render: str = "real_procedural",
    speaker_layout: str = "closeup",
    accent_line: int | None = None,
    font: str = "anton",
    text_anchor: str = "top",
    seed: int = 0,
    gate_tier: str = "normal",
    face_box: tuple[int, int, int, int] | None = None,
    subject_face_box: tuple[int, int, int, int] | None = None,
    subject_height_ratio: float | None = None,
    speaker_bottom_ratio: float | None = None,
    speaker_name: str | None = None,
    hero_variant: str | None = None,
    theme: str = "",
    story_asset_ids: tuple[str, ...] | list[str] = (),
    light_direction: str = "upper_left",
) -> EpicResult:
    """Render one thumbnail and measure it."""
    w, h = size
    recipe = MOODS.get(mood, MOODS[DEFAULT_MOOD])
    if speaker_layout not in SPEAKER_LAYOUTS:
        raise ValueError(f"unknown speaker layout: {speaker_layout!r}")
    layout_recipe = SPEAKER_LAYOUTS[speaker_layout]

    light_center = (0.5, 0.38)
    render_info: dict = {}

    from Components.ThumbnailStoryAssets import add_foreground_assets, resolve_story_assets

    story_layers = resolve_story_assets(story_asset_ids, size)
    if story_layers["background"] is not None:
        render_info["story_background"] = story_layers["background_id"]
        render_info["story_focus_box"] = story_layers["story_focus_box"]
        layout_recipe = SPEAKER_LAYOUTS["balanced"]
    if story_layers["rejected"]:
        render_info["story_assets_rejected"] = story_layers["rejected"]

    plate = None
    if speaker_render == "ai_repertoire":
        plate, repertoire_info = load_speaker_hero(
            speaker_name, variant=hero_variant, seed=seed
        )
        render_info["repertoire"] = repertoire_info
        if plate is None:
            render_info["repertoire_fallback"] = "real_procedural"
            speaker_render = "real_procedural"
        else:
            plate = _cover_resize(plate, size)
    if speaker_render in {"ai_plate", "ai_hero"}:
        builder = build_ai_plate if speaker_render == "ai_plate" else build_ai_hero
        plate, plate_info = builder(size, recipe, theme=theme, seed=seed)
        render_info["plate"] = plate_info
        if plate is None:
            # Degrade instead of failing: a missing ComfyUI must not cost us a
            # thumbnail in the Sunday pipeline.
            render_info["plate_fallback"] = "procedural_stage"

    if story_layers["background"] is not None:
        canvas = story_layers["background"].convert("RGB")
    elif plate is not None:
        canvas = plate
    else:
        canvas = build_stage(
            size,
            recipe,
            frame_bgr=frame_bgr if speaker_render == "frame_cinematic" else None,
            light_center=light_center,
            seed=seed,
        )

    if speaker_render == "real_relight" and subject_rgba is not None:
        subject_rgba, relight_info = relight_subject(subject_rgba, recipe, seed=seed)
        render_info["relight"] = relight_info
    if speaker_render in {"ai_hero", "ai_repertoire"}:
        # The generated plate already contains the (synthetic) person.
        subject_rgba = None

    # Learn where the speaker's face will land on canvas *before* compositing
    # anything, so the title split can be decided by real face geometry
    # instead of an anchor-only guess. The face — never the rest of the title
    # block — is what must always end up in front.
    framed = None
    canvas_face_band = None
    canvas_face_box = None
    speaker_placement = None
    if subject_rgba is not None:
        framed = frame_subject(
            subject_rgba,
            subject_face_box,
            min_face_ratio=layout_recipe["min_face_ratio"],
            waist_factor=layout_recipe["waist_factor"],
        )
        target_height = (
            float(subject_height_ratio)
            if subject_height_ratio is not None
            else (0.56 if story_layers["background"] is not None else float(layout_recipe["height"]))
        )
        bottom_ratio = (
            float(speaker_bottom_ratio)
            if speaker_bottom_ratio is not None
            else (0.86 if text_anchor != "top" else 1.0)
        )
        anchor_x = float(story_layers["speaker_anchor_x"] or 0.5)
        speaker_placement = (target_height, bottom_ratio, anchor_x)
        sx, sy, _, scale = _speaker_placement_geometry(
            size,
            framed.size,
            target_height_ratio=target_height,
            anchor_x=anchor_x,
            bottom_ratio=bottom_ratio,
        )
        if subject_face_box is not None:
            fx, fy, fw, fh = subject_face_box
            canvas_face_band = (sy + fy * scale, sy + (fy + fh) * scale)
            canvas_face_box = (
                int(round(sx + fx * scale)),
                int(round(sy + fy * scale)),
                max(1, int(round(fw * scale))),
                max(1, int(round(fh * scale))),
            )
    elif speaker_render in {"ai_hero", "ai_repertoire"}:
        canvas_face_box = _detect_largest_face_box(canvas)
        if canvas_face_box is not None:
            _, fy, _, fh = canvas_face_box
            canvas_face_band = (fy, fy + fh)

    type_layout, face_safe_box, title_relocated = _layout_title_outside_face(
        hook,
        size=size,
        accent_line=accent_line,
        accent_color=recipe["accent"],
        font=font,
        text_anchor=text_anchor,
        seed=seed,
        face_box=canvas_face_box,
    )

    back_flags = _face_aware_back_flags(
        type_layout.lines,
        text_anchor=text_anchor,
        canvas_face_band=canvas_face_band,
        margin_px=max(4, h // 200),
    )
    back_layer, front_layer = _split_type_layer(type_layout, back_flags)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), back_layer).convert("RGB")

    subject_box = None
    placed_alpha = None
    if framed is not None and speaker_placement is not None:
        target_height, bottom_ratio, anchor_x = speaker_placement
        canvas, subject_box, placed_alpha = place_speaker(
            canvas,
            framed,
            mood=recipe,
            target_height_ratio=target_height,
            anchor_x=anchor_x,
            bottom_ratio=bottom_ratio,
        )

    canvas, foreground_report = add_foreground_assets(
        canvas,
        story_layers["foreground"],
        light_direction=light_direction,
    )
    if foreground_report:
        render_info["story_foreground"] = foreground_report

    canvas = Image.alpha_composite(canvas.convert("RGBA"), front_layer).convert("RGB")

    canvas = atmo.bloom(canvas, threshold=0.74, radius=max(24, w // 18), strength=0.5)
    canvas = atmo.vignette(canvas, strength=0.30)
    canvas = atmo.cinematic_finish(canvas)
    canvas = atmo.film_grain(canvas, opacity=0.028, seed=seed)

    gate = run_gate(
        canvas,
        tier=gate_tier,
        title=hook,
        rendered_lines=type_layout.texts,
        text_alpha=type_layout.alpha,
        text_block_box=type_layout.block_box,
        cap_height_px=type_layout.mean_cap_height,
        face_box=canvas_face_box,
        subject_alpha=placed_alpha,
        line_fill_ratio=max((ln.fill_ratio for ln in type_layout.lines), default=0.0),
        accent_present=accent_line is not None,
        bands=load_bands(),
    )
    if face_safe_box is not None and _boxes_overlap(type_layout.block_box, face_safe_box):
        if "face_not_covered" not in gate.hard_failures:
            gate.hard_failures.append("face_not_covered")
        gate.passed = False

    story_focus_box = story_layers.get("story_focus_box")
    if story_focus_box and placed_alpha is not None:
        x1, y1, x2, y2 = (int(value) for value in story_focus_box)
        subject_arr = np.asarray(placed_alpha, dtype=np.uint8)[y1:y2, x1:x2] > 8
        text_arr = np.asarray(type_layout.alpha, dtype=np.uint8)[y1:y2, x1:x2] > 8
        if subject_arr.size:
            occupied = np.logical_or(subject_arr, text_arr)
            story_occlusion_ratio = float(occupied.mean())
            gate.metrics["story_focus_occlusion_ratio"] = round(story_occlusion_ratio, 5)
            if story_occlusion_ratio > 0.62:
                gate.hard_failures.append("story_focus_occluded")
                gate.passed = False

    return EpicResult(
        image=canvas,
        hook=hook,
        mood=mood,
        speaker_render=speaker_render,
        type_layout=type_layout,
        gate=gate,
        info={
            "subject_box": list(subject_box) if subject_box else None,
            "speaker_layout": speaker_layout,
            "text_anchor": text_anchor,
            "title_relocated_for_face": title_relocated,
            "face_box": list(canvas_face_box) if canvas_face_box else None,
            "face_safe_box": list(face_safe_box) if face_safe_box else None,
            "seed": seed,
            **render_info,
        },
    )


def _split_type_layer(layout: TypeLayout, back_flags: list[bool]) -> tuple[Image.Image, Image.Image]:
    """Cut the rendered type layer into a back and a front image, per line.

    ``back_flags[i]`` says whether line ``i`` is drawn to the back image
    (composited *before* the speaker, so the speaker paints over it) or the
    front image (composited *after*, so the line stays fully visible). Unlike
    a single top/bottom boundary, this allows any subset of lines — in
    particular exactly the lines that overlap the speaker's face — to sit
    behind, regardless of anchor.
    """
    w, h = layout.image.size
    back = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    front = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for line, is_back in zip(layout.lines, back_flags):
        x1, y1, x2, y2 = line.box
        y1, y2 = max(0, y1), min(h, y2)
        if y2 <= y1:
            continue
        strip = layout.image.crop((0, y1, w, y2))
        (back if is_back else front).paste(strip, (0, y1))
    return back, front


def _face_aware_back_flags(
    lines,
    *,
    text_anchor: str,
    canvas_face_band: tuple[int, int] | None,
    margin_px: int = 0,
) -> list[bool]:
    """Decide, per title line, whether the speaker's face should sit in front of it.

    With a known face band (the speaker's face position on canvas, computed
    *before* the speaker is pasted — see ``_speaker_placement_geometry``), a
    line goes behind the speaker exactly when it vertically overlaps that
    band. This is what lets a title legitimately run partially behind the
    head instead of covering it: only the overlapping line(s) move back,
    every other line stays in front and fully readable.

    Falls back to the previous anchor-only heuristic when no face band is
    known (e.g. an AI-plate render with no real subject cutout to measure).
    """
    if canvas_face_band is not None:
        face_top, face_bottom = canvas_face_band
        face_top -= margin_px
        face_bottom += margin_px
        return [
            not (line.box[3] <= face_top or line.box[1] >= face_bottom)
            for line in lines
        ]
    # Old anchor-based fallback: for a low title block the speaker stands
    # deliberately above the type (no known face position to protect), for a
    # top-anchored block all but the last line may sit behind.
    if text_anchor != "top":
        return [False] * len(lines)
    split = max(1, len(lines) - 1)
    return [i < split for i in range(len(lines))]


def save(result: EpicResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(path)
    return path
