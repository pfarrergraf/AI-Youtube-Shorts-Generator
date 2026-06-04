from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from Components.ThumbnailEffects import add_speaker_outline, crop_to_alpha, get_background_removal_provider
from Components.ThumbnailV2 import (
    _DEFAULT_BRAND,
    _IVORY,
    _PALETTES,
    _SHADOW,
    _frame_background,
    _load_font,
    _sanitize_hook,
    _sanitize_keyword,
    _score_variant,
    _wrap_text,
    _write_video_from_image,
)


V2_TEST_VARIANTS = ("outlined_sermon", "stacked_keyword", "echo_cutout")
_BASE_VARIANT_BY_TEST = {
    "outlined_sermon": "emotion_first",
    "stacked_keyword": "strong_contrast",
    "echo_cutout": "clean_editorial",
}
_OUTLINE_PRESET_BY_TEST = {
    "outlined_sermon": "creator_white",
    "stacked_keyword": "creator_blue",
    "echo_cutout": "sermon_gold",
}
_MOTION_BY_TEST_VARIANT = {
    "outlined_sermon": "drift",
    "stacked_keyword": "zoom_pulse",
    "echo_cutout": "push_in",
}


def _resize_image(image: Image.Image, *, target_height: int) -> Image.Image:
    ratio = target_height / float(max(1, image.height))
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _draw_title_panel(
    canvas: Image.Image,
    *,
    hook: str,
    accent: str,
    brand_label: str,
    panel: list[int],
    palette: dict,
    font_ratio: float,
) -> tuple[list[int], list[int]]:
    draw = ImageDraw.Draw(canvas)
    width = panel[2] - panel[0]
    height = panel[3] - panel[1]
    font_size = int(canvas.height * font_ratio)
    font = _load_font(font_size)
    max_width = max(120, width - 52)
    allowed_lines = 3
    max_text_height = max(120, height - 108)
    lines = _wrap_text(hook, font, max_width, draw)
    line_height = int(font_size * 1.1)
    total_height = line_height * len(lines)

    while (len(lines) > allowed_lines or total_height > max_text_height) and font_size > 44:
        font_size -= 4
        font = _load_font(font_size)
        lines = _wrap_text(hook, font, max_width, draw)
        line_height = int(font_size * 1.1)
        total_height = line_height * len(lines)

    panel_fill = (6, 10, 16, 178)
    shadow_fill = (0, 0, 0, 114)
    shadow_panel = [panel[0] + 8, panel[1] + 12, panel[2] + 8, panel[3] + 12]
    draw.rounded_rectangle(shadow_panel, radius=34, fill=shadow_fill)
    draw.rounded_rectangle(panel, radius=34, fill=panel_fill, outline=(255, 255, 255, 22), width=2)

    x_pos = panel[0] + 28
    y_pos = panel[1] + 24
    text_bounds: list[int] | None = None
    for line in lines:
        fill = _IVORY
        draw.text((x_pos + 4, y_pos + 6), line, font=font, fill=_SHADOW)
        if accent and accent in line:
            prefix, _, suffix = line.partition(accent)
            prefix_w = draw.textbbox((0, 0), prefix, font=font)[2]
            accent_w = draw.textbbox((0, 0), accent, font=font)[2]
            chip_pad_x = max(14, font_size // 6)
            chip_pad_y = max(8, font_size // 10)
            chip_rect = [
                x_pos + prefix_w - chip_pad_x,
                y_pos - chip_pad_y,
                x_pos + prefix_w + accent_w + chip_pad_x,
                y_pos + line_height - chip_pad_y,
            ]
            draw.text((x_pos, y_pos), prefix, font=font, fill=fill)
            draw.rounded_rectangle(chip_rect, radius=22, fill=palette["chip_bg"])
            draw.text((x_pos + prefix_w, y_pos), accent, font=font, fill=palette["chip_fg"])
            draw.text((x_pos + prefix_w + accent_w, y_pos), suffix, font=font, fill=fill)
            bbox = [x_pos, y_pos, x_pos + prefix_w + accent_w + draw.textbbox((0, 0), suffix, font=font)[2], y_pos + line_height]
        else:
            draw.text((x_pos, y_pos), line, font=font, fill=fill)
            bbox_raw = draw.textbbox((x_pos, y_pos), line, font=font)
            bbox = [bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]]
        if text_bounds is None:
            text_bounds = bbox
        else:
            text_bounds = [
                min(text_bounds[0], bbox[0]),
                min(text_bounds[1], bbox[1]),
                max(text_bounds[2], bbox[2]),
                max(text_bounds[3], bbox[3]),
            ]
        y_pos += line_height

    brand_font = _load_font(max(24, int(font_size * 0.34)))
    brand_bbox = draw.textbbox((0, 0), brand_label, font=brand_font)
    brand_rect = [
        panel[0] + 28,
        panel[3] - 56,
        panel[0] + 28 + (brand_bbox[2] - brand_bbox[0]) + 34,
        panel[3] - 16,
    ]
    draw.rounded_rectangle(brand_rect, radius=18, fill=(8, 13, 22, 224))
    draw.text((brand_rect[0] + 17, brand_rect[1] + 7), brand_label, font=brand_font, fill=(244, 241, 233, 245))
    return panel, text_bounds or [panel[0], panel[1], panel[2], panel[1] + 80]


def _subject_layout(canvas: Image.Image, variant: str, speaker: Image.Image) -> tuple[Image.Image, list[int]]:
    width, height = canvas.size
    target_height = int(height * {"outlined_sermon": 0.70, "stacked_keyword": 0.68, "echo_cutout": 0.66}[variant])
    fitted = _resize_image(speaker, target_height=target_height)

    if variant == "outlined_sermon":
        x_pos = int(width * 0.52)
        y_pos = height - fitted.height - int(height * 0.04)
    elif variant == "stacked_keyword":
        x_pos = int(width * 0.04)
        y_pos = height - fitted.height - int(height * 0.04)
    else:
        x_pos = int(width * 0.55)
        y_pos = height - fitted.height - int(height * 0.05)

    if variant == "echo_cutout":
        ghost = _resize_image(speaker, target_height=int(height * 0.86))
        ghost_alpha = ghost.getchannel("A").filter(ImageFilter.GaussianBlur(radius=10))
        ghost_fill = ImageOps.colorize(ghost_alpha.convert("L"), black=(18, 22, 32), white=(245, 212, 160)).convert("RGBA")
        ghost_fill.putalpha(ghost_alpha.point(lambda value: min(255, int(value * 0.32))))
        canvas.alpha_composite(ghost_fill, (int(width * -0.10), height - ghost.height - int(height * 0.01)))

    canvas.alpha_composite(fitted, (x_pos, y_pos))
    return canvas, [x_pos, y_pos, x_pos + fitted.width, y_pos + fitted.height]


def _compose_variant(
    frame_bgr,
    *,
    variant: str,
    hook_text: str,
    accent_keyword: str,
    brief: dict,
    speaker_rgba: Image.Image,
    face_box,
    coverage: float,
) -> tuple[Image.Image, dict]:
    hook = _sanitize_hook(hook_text)
    accent = _sanitize_keyword(hook, accent_keyword)
    brand_label = str(brief.get("brand_label") or _DEFAULT_BRAND).strip().upper()[:28] or _DEFAULT_BRAND
    base_variant = _BASE_VARIANT_BY_TEST[variant]
    palette = _PALETTES[base_variant]

    canvas = _frame_background(frame_bgr, base_variant)
    speaker = crop_to_alpha(speaker_rgba)
    speaker = add_speaker_outline(speaker, _OUTLINE_PRESET_BY_TEST[variant])
    canvas, subject_bounds = _subject_layout(canvas, variant, speaker)

    width, height = canvas.size
    if variant == "outlined_sermon":
        panel = [int(width * 0.05), int(height * 0.60), int(width * 0.65), int(height * 0.89)]
        font_ratio = 0.088
    elif variant == "stacked_keyword":
        panel = [int(width * 0.42), int(height * 0.26), int(width * 0.94), int(height * 0.66)]
        font_ratio = 0.082
    else:
        panel = [int(width * 0.08), int(height * 0.66), int(width * 0.92), int(height * 0.92)]
        font_ratio = 0.084

    panel, text_bounds = _draw_title_panel(
        canvas,
        hook=hook,
        accent=accent,
        brand_label=brand_label,
        panel=panel,
        palette=palette,
        font_ratio=font_ratio,
    )

    metadata = {
        "variant": variant,
        "layout": variant,
        "hook_text": hook,
        "accent_keyword": accent,
        "face_box": list(face_box) if face_box else None,
        "speaker_bounds": subject_bounds,
        "subject_coverage": round(float(coverage), 4),
        "panel": [int(value) for value in panel],
        "text_bounds": [int(value) for value in text_bounds],
        "brand_label": brand_label,
        "outline_preset": _OUTLINE_PRESET_BY_TEST[variant],
    }
    return canvas, metadata


def render_thumbnail_v2_test_assets(
    frame_bgr,
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
    variant_root = Path(variants_dir) if variants_dir else thumb_path.parent / "_thumbnail_v2_test" / thumb_path.stem
    variant_root.mkdir(parents=True, exist_ok=True)

    provider = get_background_removal_provider("grabcut_local")
    speaker_rgba, face_box, coverage = provider.extract_subject(frame_bgr)
    raw_subject_path = variant_root / "speaker_cutout.png"
    crop_to_alpha(speaker_rgba).save(raw_subject_path)

    preferred_variant = {
        "clean_gradient": "echo_cutout",
        "strong_contrast": "stacked_keyword",
        "emotion_focus": "outlined_sermon",
    }.get(str(brief.get("background_style") or "").strip().lower())

    candidates: list[dict] = []
    for variant in V2_TEST_VARIANTS:
        rendered, metadata = _compose_variant(
            frame_bgr,
            variant=variant,
            hook_text=hook_text,
            accent_keyword=accent_keyword,
            brief=brief,
            speaker_rgba=speaker_rgba,
            face_box=face_box,
            coverage=coverage,
        )
        variant_path = variant_root / f"{variant}.jpg"
        rendered.convert("RGB").save(variant_path, "JPEG", quality=94)
        metadata["image_path"] = str(variant_path)
        metadata["score"] = _score_variant(rendered, metadata)
        if variant == preferred_variant:
            metadata["score"] = round(float(metadata["score"]) + 0.08, 4)
        if brief.get("emotion_target") and variant == "outlined_sermon":
            metadata["score"] = round(float(metadata["score"]) + 0.03, 4)
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
        "provider": provider.name,
        "speaker_cutout": str(raw_subject_path),
        "variants": candidates,
    }
    with (variant_root / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _write_video_from_image(
        image_path=str(selected_path),
        output_path=output_video_path,
        fps=fps,
        duration=duration,
        motion_style=_MOTION_BY_TEST_VARIANT.get(best["variant"], "push_in"),
    )
    return output_video_path, str(thumb_path), f"v2_test_{best['variant']}"
