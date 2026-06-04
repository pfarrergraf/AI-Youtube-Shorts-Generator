from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


V2_VARIANTS = ("clean_editorial", "strong_contrast", "emotion_first")
_DEFAULT_HOOK = "GOTT BEWEGT"
_DEFAULT_BRAND = "PREDIGTCLIP"
_IVORY = (248, 241, 230, 255)
_SHADOW = (8, 12, 18, 210)
_NVENC_FLAGS = [
    "-c:v", "h264_nvenc",
    "-preset", "p7",
    "-rc", "constqp",
    "-qp", "18",
    "-b:v", "0",
    "-gpu", "0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
]
_X264_FALLBACK_FLAGS = [
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "16",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
]
_PALETTES = {
    "clean_editorial": {
        "bg_a": (22, 31, 43),
        "bg_b": (48, 78, 102),
        "accent": (244, 183, 64, 255),
        "chip_fg": (19, 28, 35, 255),
        "chip_bg": (244, 183, 64, 240),
    },
    "strong_contrast": {
        "bg_a": (8, 13, 25),
        "bg_b": (18, 82, 118),
        "accent": (110, 216, 255, 255),
        "chip_fg": (10, 16, 26, 255),
        "chip_bg": (110, 216, 255, 235),
    },
    "emotion_first": {
        "bg_a": (28, 18, 33),
        "bg_b": (142, 92, 54),
        "accent": (255, 209, 122, 255),
        "chip_fg": (20, 14, 20, 255),
        "chip_bg": (255, 209, 122, 235),
    },
}
_MOTION_BY_VARIANT = {
    "clean_editorial": "push_in",
    "strong_contrast": "zoom_pulse",
    "emotion_first": "drift",
}


def _find_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _find_font()
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = [word for word in str(text or "").split() if word]
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]


def _sanitize_hook(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    cleaned = cleaned.upper()[:34].strip()
    return cleaned or _DEFAULT_HOOK


def _sanitize_keyword(hook_text: str, accent_keyword: str) -> str:
    hook_words = [word for word in hook_text.split() if word]
    candidate = str(accent_keyword or "").strip().upper()
    if candidate and candidate in hook_text.upper():
        return candidate
    return max(hook_words, key=len) if hook_words else ""


def _detect_face_box(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_bgr.shape[:2]
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(w // 10, h // 10))
        if len(faces) > 0:
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    except Exception:
        pass
    return None


def _subject_rect(width: int, height: int, face_box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    if face_box is None:
        return (max(0, width // 6), max(0, height // 12), max(40, width * 2 // 3), max(40, height * 5 // 6))

    fx, fy, fw, fh = face_box
    x = max(0, int(fx - fw * 0.9))
    y = max(0, int(fy - fh * 0.55))
    rect_w = min(width - x, int(fw * 2.8))
    rect_h = min(height - y, int(fh * 4.4))
    return (x, y, max(40, rect_w), max(40, rect_h))


def _extract_subject_rgba(frame_bgr: np.ndarray, face_box: tuple[int, int, int, int] | None) -> tuple[Image.Image, float]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    rect = _subject_rect(width, height, face_box)
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        alpha = np.zeros((height, width), dtype=np.uint8)

    coverage = float(np.count_nonzero(alpha)) / float(max(1, width * height))
    if coverage < 0.05:
        x, y, rect_w, rect_h = rect
        alpha = np.zeros((height, width), dtype=np.uint8)
        alpha[y:y + rect_h, x:x + rect_w] = 255
        coverage = float(np.count_nonzero(alpha)) / float(max(1, width * height))

    binary = (alpha > 24).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count > 1:
        chosen_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if face_box is not None:
            fx, fy, fw, fh = face_box
            sample_x = min(width - 1, max(0, int(fx + fw / 2)))
            sample_y = min(height - 1, max(0, int(fy + fh / 2)))
            face_label = int(labels[sample_y, sample_x])
            if face_label > 0:
                chosen_label = face_label
        alpha = np.where(labels == chosen_label, 255, 0).astype(np.uint8)
        coverage = float(np.count_nonzero(alpha)) / float(max(1, width * height))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.6, sigmaY=1.6)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA"), coverage


def _background(width: int, height: int, variant: str) -> Image.Image:
    palette = _PALETTES[variant]
    top = np.array(palette["bg_a"], dtype=np.float32)
    bottom = np.array(palette["bg_b"], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    grad = ((1.0 - ramp) * top + ramp * bottom).astype(np.uint8)
    img = np.repeat(grad[:, None, :], width, axis=1)
    canvas = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    accent = palette["accent"]
    draw.ellipse([
        int(width * 0.58),
        int(height * 0.05),
        int(width * 1.04),
        int(height * 0.48),
    ], fill=(accent[0], accent[1], accent[2], 64))
    draw.ellipse([
        int(width * -0.18),
        int(height * 0.48),
        int(width * 0.52),
        int(height * 1.05),
    ], fill=(255, 255, 255, 30 if variant == "clean_editorial" else 18))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=80 if variant != "strong_contrast" else 58))
    canvas = Image.alpha_composite(canvas, overlay)

    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    scrim_draw = ImageDraw.Draw(scrim)
    for y_pos in range(height):
        frac = y_pos / float(max(1, height - 1))
        alpha = int((40 + 65 * frac) if variant != "strong_contrast" else (54 + 76 * frac))
        scrim_draw.line([(0, y_pos), (width, y_pos)], fill=(7, 10, 16, alpha))
    return Image.alpha_composite(canvas, scrim)


def _frame_background(frame_bgr: np.ndarray, variant: str) -> Image.Image:
    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb, mode="RGB").convert("RGBA")

    contrast_by_variant = {
        "clean_editorial": 1.06,
        "strong_contrast": 1.16,
        "emotion_first": 1.10,
    }
    brightness_by_variant = {
        "clean_editorial": 0.92,
        "strong_contrast": 0.88,
        "emotion_first": 0.90,
    }
    color_by_variant = {
        "clean_editorial": 0.94,
        "strong_contrast": 1.02,
        "emotion_first": 1.00,
    }
    canvas = ImageEnhance.Contrast(canvas).enhance(contrast_by_variant[variant])
    canvas = ImageEnhance.Color(canvas).enhance(color_by_variant[variant])
    canvas = ImageEnhance.Brightness(canvas).enhance(brightness_by_variant[variant])

    tint = _background(width, height, variant)
    return Image.blend(canvas, tint, 0.28 if variant != "strong_contrast" else 0.24)


def _crop_to_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").getbbox()
    return image.crop(bbox) if bbox else image


def _subject_coverage_estimate(width: int, height: int, face_box: tuple[int, int, int, int] | None) -> float:
    if face_box is None:
        return 0.28
    _x, _y, rect_w, rect_h = _subject_rect(width, height, face_box)
    return min(0.72, max(0.12, (rect_w * rect_h) / float(max(1, width * height))))


def _measure_line_width(line: str, font, draw: ImageDraw.ImageDraw, accent: str, font_size: int) -> int:
    bbox = draw.textbbox((0, 0), line, font=font)
    width = bbox[2] - bbox[0]
    if accent and accent in line:
        chip_pad_x = max(16, font_size // 6)
        width += chip_pad_x * 2
    return int(width)


def _text_side(width: int, face_box: tuple[int, int, int, int] | None, brief: dict) -> str:
    requested = str((brief or {}).get("speaker_side") or "").strip().lower()
    if requested in {"left", "right"}:
        return "right" if requested == "left" else "left"
    if face_box is None:
        return "right"
    fx, _fy, fw, _fh = face_box
    center_x = fx + fw / 2.0
    return "left" if center_x > width * 0.5 else "right"


def _compose_variant(
    frame_bgr: np.ndarray,
    *,
    variant: str,
    hook_text: str,
    accent_keyword: str,
    brief: dict,
) -> tuple[Image.Image, dict]:
    height, width = frame_bgr.shape[:2]
    face_box = _detect_face_box(frame_bgr)
    subject_coverage = _subject_coverage_estimate(width, height, face_box)
    canvas = _frame_background(frame_bgr, variant)
    text_side = _text_side(width, face_box, brief)
    palette = _PALETTES[variant]

    draw = ImageDraw.Draw(canvas)
    hook = _sanitize_hook(hook_text)
    accent = _sanitize_keyword(hook, accent_keyword)
    brand_label = str((brief or {}).get("brand_label") or _DEFAULT_BRAND).strip().upper()[:28]

    safe_margin_x = int(width * 0.06)
    font_size = int(height * (0.088 if variant != "emotion_first" else 0.094))
    font = _load_font(font_size)
    max_width = int(width * (0.43 if variant != "emotion_first" else 0.50))
    tmp = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    allowed_lines = 3 if variant != "emotion_first" else 2
    max_text_height = int(height * (0.22 if variant != "emotion_first" else 0.18))
    while font_size > 34:
        lines = _wrap_text(hook, font, max_width, tmp_draw)
        line_height = int(font_size * 1.12)
        rendered_width = max(
            (_measure_line_width(line, font, tmp_draw, accent, font_size) for line in lines),
            default=0,
        )
        total_height = line_height * len(lines)
        if len(lines) <= allowed_lines and rendered_width <= max_width and total_height <= max_text_height:
            break
        font_size -= 4
        font = _load_font(font_size)

    lines = _wrap_text(hook, font, max_width, tmp_draw)
    line_height = int(font_size * 1.12)
    total_height = line_height * len(lines)
    rendered_width = max(
        (_measure_line_width(line, font, tmp_draw, accent, font_size) for line in lines),
        default=max_width,
    )
    box_y = int(height * 0.09)
    if variant == "emotion_first":
        box_y = int(height * 0.07)
        max_width = int(width * 0.50)

    brand_font_size = max(24, font_size // 3)
    brand_font = _load_font(brand_font_size)
    brand_bbox = tmp_draw.textbbox((0, 0), brand_label, font=brand_font)
    while brand_font_size > 18 and (brand_bbox[2] - brand_bbox[0]) > max_width:
        brand_font_size -= 2
        brand_font = _load_font(brand_font_size)
        brand_bbox = tmp_draw.textbbox((0, 0), brand_label, font=brand_font)

    content_width = max(rendered_width, (brand_bbox[2] - brand_bbox[0]) + 34)
    box_x = safe_margin_x if text_side == "left" else width - safe_margin_x - content_width
    box_x = max(safe_margin_x, min(width - safe_margin_x - content_width, box_x))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    panel = [
        box_x - 22,
        box_y - 22,
        min(width - safe_margin_x, box_x + content_width + 22),
        box_y + total_height + 98,
    ]
    if variant == "emotion_first":
        panel[2] = min(width - safe_margin_x, panel[2])
    panel_fill = (6, 10, 16, 112 if variant == "clean_editorial" else 138)
    overlay_draw.rounded_rectangle(panel, radius=34, fill=panel_fill)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.4))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    text_bboxes: list[tuple[int, int, int, int]] = []
    y_pos = box_y
    for line in lines:
        fill = _IVORY
        if accent and accent in line:
            accent_bbox = draw.textbbox((box_x, y_pos), line, font=font)
            text_bboxes.append((accent_bbox[0], accent_bbox[1], accent_bbox[2], accent_bbox[3]))
            draw.text((box_x + 3, y_pos + 5), line, font=font, fill=_SHADOW)
            parts = line.split(accent, 1)
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
            prefix_w = draw.textbbox((0, 0), prefix, font=font)[2]
            accent_w = draw.textbbox((0, 0), accent, font=font)[2]
            draw.text((box_x, y_pos), prefix, font=font, fill=fill)
            chip_pad_x = max(16, font_size // 6)
            chip_pad_y = max(8, font_size // 10)
            chip_rect = [
                box_x + prefix_w - chip_pad_x,
                y_pos - chip_pad_y,
                box_x + prefix_w + accent_w + chip_pad_x,
                y_pos + line_height - chip_pad_y,
            ]
            draw.rounded_rectangle(chip_rect, radius=22, fill=palette["chip_bg"])
            draw.text((box_x + prefix_w, y_pos), accent, font=font, fill=palette["chip_fg"])
            draw.text((box_x + prefix_w + accent_w, y_pos), suffix, font=font, fill=fill)
        else:
            bbox = draw.textbbox((box_x, y_pos), line, font=font)
            text_bboxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
            draw.text((box_x + 3, y_pos + 5), line, font=font, fill=_SHADOW)
            draw.text((box_x, y_pos), line, font=font, fill=fill)
        y_pos += line_height

    brand_bbox = draw.textbbox((0, 0), brand_label, font=brand_font)
    brand_rect = [
        box_x,
        panel[3] - 54,
        box_x + (brand_bbox[2] - brand_bbox[0]) + 34,
        panel[3] - 14,
    ]
    draw.rounded_rectangle(brand_rect, radius=18, fill=(8, 13, 22, 215))
    draw.text((brand_rect[0] + 17, brand_rect[1] + 7), brand_label, font=brand_font, fill=(244, 241, 233, 240))

    text_bounds = text_bboxes[0] if text_bboxes else (box_x, box_y, box_x + max_width, box_y + total_height)
    for bbox in text_bboxes[1:]:
        text_bounds = (
            min(text_bounds[0], bbox[0]),
            min(text_bounds[1], bbox[1]),
            max(text_bounds[2], bbox[2]),
            max(text_bounds[3], bbox[3]),
        )

    metadata = {
        "variant": variant,
        "speaker_side": "left" if text_side == "right" else "right",
        "text_side": text_side,
        "hook_text": hook,
        "accent_keyword": accent,
        "subject_coverage": round(subject_coverage, 4),
        "face_box": list(face_box) if face_box else None,
        "text_bounds": [int(v) for v in text_bounds],
        "panel": [int(v) for v in panel],
        "brand_label": brand_label,
    }
    return canvas, metadata


def _score_variant(image: Image.Image, metadata: dict) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = metadata.get("text_bounds") or [0, 0, w, h // 3]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    patch = arr[y1:y2, x1:x2]
    luminance = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
    patch_std = float(luminance.std())
    patch_mean = float(luminance.mean())
    contrast_score = min(1.0, abs(240.0 - patch_mean) / 140.0 + patch_std / 110.0)

    face_score = 0.22
    face_box = metadata.get("face_box")
    if face_box:
        fw = float(face_box[2])
        fh = float(face_box[3])
        face_ratio = (fw * fh) / float(max(1.0, w * h))
        face_score = min(1.0, face_ratio * 14.0)

    coverage = float(metadata.get("subject_coverage") or 0.0)
    subject_score = max(0.0, 1.0 - min(1.0, abs(coverage - 0.26) / 0.24))
    word_count = len(str(metadata.get("hook_text") or "").split())
    brevity_score = max(0.0, 1.0 - max(0, word_count - 4) * 0.22)

    variant = metadata.get("variant")
    prior = {
        "clean_editorial": 0.08,
        "strong_contrast": 0.06,
        "emotion_first": 0.10 if face_score > 0.48 else 0.03,
    }.get(variant, 0.0)

    score = contrast_score * 0.36 + face_score * 0.30 + subject_score * 0.18 + brevity_score * 0.10 + prior
    return round(float(score), 4)


def _write_video_from_image(image_path: str, output_path: str, fps: float, duration: float, motion_style: str) -> None:
    with Image.open(image_path) as opened:
        width, height = opened.size
    total_frames = max(1, int(fps * duration))
    if motion_style == "zoom_pulse":
        zoom_expr = (
            f"zoompan=z='1+0.08*(1-pow(1-on/{total_frames},3))':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )
    elif motion_style == "drift":
        zoom_expr = (
            f"zoompan=z='1.045':"
            f"x='iw/2-(iw/zoom/2)-18*on/{total_frames}':"
            f"y='ih/2-(ih/zoom/2)+14*on/{total_frames}':"
            f"d={total_frames}:s={width}x{height}:fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )
    else:
        zoom_expr = (
            f"zoompan=z='1+0.014*on/{total_frames}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )

    base_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", f"[0:v]{zoom_expr}[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-t", str(duration),
        "-c:a", "aac", "-b:a", "128k",
    ]
    last_stderr = ""
    for codec_flags in (_NVENC_FLAGS, _X264_FALLBACK_FLAGS):
        cmd = [*base_cmd, *codec_flags, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return
        last_stderr = result.stderr
    raise RuntimeError(f"FFmpeg thumbnail-v2 generation failed:\n{last_stderr}")


def render_thumbnail_v2_assets(
    frame_bgr: np.ndarray,
    *,
    hook_text: str,
    accent_keyword: str,
    output_video_path: str,
    thumbnail_image_path: str,
    duration: float,
    fps: float,
    brief: dict | None = None,
    variants_dir: str | None = None,
) -> tuple[str, str, str]:
    brief = dict(brief or {})
    thumb_path = Path(thumbnail_image_path)
    variant_root = Path(variants_dir) if variants_dir else thumb_path.parent / "_thumbnail_v2" / thumb_path.stem
    variant_root.mkdir(parents=True, exist_ok=True)
    preferred_variant = {
        "clean_gradient": "clean_editorial",
        "strong_contrast": "strong_contrast",
        "emotion_focus": "emotion_first",
    }.get(str(brief.get("background_style") or "").strip().lower())

    candidates: list[dict] = []
    for variant in V2_VARIANTS:
        rendered, metadata = _compose_variant(
            frame_bgr,
            variant=variant,
            hook_text=hook_text,
            accent_keyword=accent_keyword,
            brief=brief,
        )
        variant_path = variant_root / f"{variant}.jpg"
        rendered.convert("RGB").save(variant_path, "JPEG", quality=94)
        metadata["image_path"] = str(variant_path)
        metadata["score"] = _score_variant(rendered, metadata)
        if variant == preferred_variant:
            metadata["score"] = round(float(metadata["score"]) + 0.08, 4)
        if brief.get("emotion_target") and variant == "emotion_first":
            metadata["score"] = round(float(metadata["score"]) + 0.02, 4)
        candidates.append(metadata)

    best = max(candidates, key=lambda item: float(item.get("score") or 0.0))
    shutil.copy2(best["image_path"], thumb_path)
    selected_path = variant_root / "selected.jpg"
    shutil.copy2(best["image_path"], selected_path)

    report = {
        "selected_variant": best["variant"],
        "best_image": str(selected_path),
        "hook_text": hook_text,
        "accent_keyword": accent_keyword,
        "brief": brief,
        "variants": candidates,
    }
    with (variant_root / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _write_video_from_image(
        image_path=str(selected_path),
        output_path=output_video_path,
        fps=fps,
        duration=duration,
        motion_style=_MOTION_BY_VARIANT.get(best["variant"], "push_in"),
    )
    return output_video_path, str(thumb_path), f"v2_{best['variant']}"