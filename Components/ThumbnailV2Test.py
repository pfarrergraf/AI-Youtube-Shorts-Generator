from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

from Components.ThumbnailEffects import (
    add_speaker_outline,
    brighten_face_region,
    crop_to_alpha,
    feather_alpha,
    get_background_removal_provider,
    save_outline_debug_layers,
)
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


V2_TEST_VARIANTS = (
    "outlined_sermon",
    "stacked_keyword",
    "echo_cutout",
    "depth_typography",
    "split_depth_text",
)
_BASE_VARIANT_BY_TEST = {
    "outlined_sermon": "emotion_first",
    "stacked_keyword": "strong_contrast",
    "echo_cutout": "clean_editorial",
    "depth_typography": "emotion_first",
    "split_depth_text": "strong_contrast",
}
_OUTLINE_PRESET_BY_TEST = {
    "outlined_sermon": "creator_white",
    "stacked_keyword": "creator_white",
    "echo_cutout": "sermon_gold",
    "depth_typography": "sermon_gold",
    "split_depth_text": "creator_blue",
}
_MOTION_BY_TEST_VARIANT = {
    "outlined_sermon": "drift",
    "stacked_keyword": "zoom_pulse",
    "echo_cutout": "push_in",
    "depth_typography": "drift",
    "split_depth_text": "zoom_pulse",
}
TEXT_PANEL_MODES = {
    "full_panel": "full_panel",
    "partial_panel": "partial_panel",
    "no_panel_with_shadow": "no_panel_with_shadow",
}
FONT_ROLES = {
    "bold": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ),
    "condensed": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ),
    "script": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf",
    ),
    "serif": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    ),
}
_CONNECTOR_WORDS = {
    "AM",
    "AN",
    "AUF",
    "AUS",
    "BEI",
    "DAS",
    "DEM",
    "DEN",
    "DER",
    "DES",
    "DIE",
    "EIN",
    "EINE",
    "FÜR",
    "IM",
    "IN",
    "MIT",
    "UND",
    "VOM",
    "VON",
    "ZU",
    "ZUM",
    "ZUR",
}


def _load_role_font(font_size: int, role: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_ROLES.get(role, ()):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return _load_font(font_size)


def _resize_image(image: Image.Image, *, target_height: int) -> Image.Image:
    ratio = target_height / float(max(1, image.height))
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _apply_opacity(layer: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 0.999:
        return layer
    adjusted = layer.copy()
    alpha = adjusted.getchannel("A").point(lambda value: max(0, min(255, int(value * opacity))))
    adjusted.putalpha(alpha)
    return adjusted


def add_layer(
    canvas: Image.Image,
    layer: Image.Image,
    *,
    opacity: float = 1.0,
    blend_mode: str = "normal",
    position: tuple[int, int] = (0, 0),
) -> Image.Image:
    if layer.size != canvas.size or position != (0, 0):
        placed = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        placed.alpha_composite(layer, position)
    else:
        placed = layer.copy()
    placed = _apply_opacity(placed, opacity)

    if blend_mode == "screen":
        screened = ImageChops.screen(canvas.convert("RGB"), placed.convert("RGB")).convert("RGBA")
        return Image.composite(screened, Image.alpha_composite(canvas, placed), placed.getchannel("A"))
    if blend_mode == "add":
        added = ImageChops.add(canvas.convert("RGB"), placed.convert("RGB"), scale=1.0, offset=0).convert("RGBA")
        return Image.composite(added, Image.alpha_composite(canvas, placed), placed.getchannel("A"))
    return Image.alpha_composite(canvas, placed)


def compose_layers(canvas: Image.Image, layers: list[dict]) -> Image.Image:
    composed = canvas.copy().convert("RGBA")
    for layer in layers:
        image = layer.get("image")
        if image is None:
            continue
        composed = add_layer(
            composed,
            image.convert("RGBA"),
            opacity=float(layer.get("opacity", 1.0)),
            blend_mode=str(layer.get("blend_mode") or "normal"),
            position=tuple(layer.get("position") or (0, 0)),
        )
    return composed


def _clip_rect(bounds: list[int] | tuple[int, int, int, int], size: tuple[int, int]) -> list[int]:
    width, height = size
    x1, y1, x2, y2 = [int(v) for v in bounds]
    return [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]


def _rect_union(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    if a is None:
        return b[:] if b else None
    if b is None:
        return a[:]
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def _rect_overlap_ratio(a: list[int] | tuple[int, int, int, int] | None, b: list[int] | tuple[int, int, int, int] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    overlap = float((ix2 - ix1) * (iy2 - iy1))
    base_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    return overlap / base_area


def _speaker_face_proxy(subject_bounds: list[int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in subject_bounds]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return [
        x1 + int(width * 0.18),
        y1 + int(height * 0.04),
        x1 + int(width * 0.76),
        y1 + int(height * 0.30),
    ]


def _build_echo_layer(
    canvas_size: tuple[int, int],
    source_image: Image.Image,
    *,
    target_height: int,
    position: tuple[int, int],
) -> tuple[Image.Image, Image.Image, list[int]]:
    ghost_base = crop_to_alpha(source_image)
    ghost = _resize_image(ghost_base, target_height=target_height)
    ghost_alpha = ghost.getchannel("A")
    ghost_alpha = feather_alpha(ghost_alpha, blur_radius=5.0, low=8, high=230)
    ghost_alpha = ghost_alpha.filter(ImageFilter.GaussianBlur(radius=8))
    ghost_fill = Image.new("RGBA", ghost.size, (235, 228, 214, 0))
    ghost_fill.putalpha(ghost_alpha.point(lambda value: min(255, int(value * 0.10))))

    echo_overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    echo_overlay.alpha_composite(ghost_fill, position)
    bounds = [position[0], position[1], position[0] + ghost.width, position[1] + ghost.height]
    return ghost_alpha, echo_overlay, bounds


def draw_background_word(
    canvas: Image.Image,
    word: str,
    *,
    position: tuple[int, int],
    font_size: int,
    fill: tuple[int, int, int, int],
    rotate_degrees: float = 0.0,
    tracking: int = 0,
    role: str = "condensed",
) -> Image.Image:
    word = str(word or "").strip().upper()
    if not word:
        return Image.new("RGBA", canvas.size, (0, 0, 0, 0))

    font = _load_role_font(font_size, role)
    probe = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    widths = []
    max_height = 0
    for char in word:
        bbox = probe_draw.textbbox((0, 0), char, font=font)
        widths.append(max(1, bbox[2] - bbox[0]))
        max_height = max(max_height, bbox[3] - bbox[1])
    total_width = sum(widths) + max(0, len(widths) - 1) * max(0, tracking) + 40
    layer = Image.new("RGBA", (total_width, max_height + 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x_pos = 20
    for index, char in enumerate(word):
        draw.text((x_pos, 14), char, font=font, fill=fill)
        x_pos += widths[index] + max(0, tracking)
    if abs(rotate_degrees) > 0.01:
        layer = layer.rotate(rotate_degrees, resample=Image.Resampling.BICUBIC, expand=True)

    target = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    target.alpha_composite(layer, position)
    return target


def choose_text_color_for_region(canvas: Image.Image, bbox: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    clipped = _clip_rect(list(bbox), canvas.size)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return _IVORY
    crop = canvas.crop(tuple(clipped)).convert("RGB")
    stat = ImageStat.Stat(crop)
    r_mean, g_mean, b_mean = stat.mean[:3]
    luminance = 0.2126 * r_mean + 0.7152 * g_mean + 0.0722 * b_mean
    if luminance >= 156.0:
        return (16, 22, 30, 255)
    return _IVORY


def _split_hook_layout(hook: str, accent: str) -> dict[str, str]:
    words = [word for word in hook.split() if word]
    accent = str(accent or "").strip().upper()
    accent_index = next((index for index, word in enumerate(words) if word == accent), -1)
    chip_text = accent if accent_index >= 0 else (words[-1] if words else "")
    background_word = accent if len(accent) >= 4 else max(words, key=len, default=chip_text or hook)

    remaining = words[:]
    if accent_index >= 0:
        remaining.pop(accent_index)
    if not remaining and chip_text:
        remaining = [chip_text]

    script_text = ""
    if len(remaining) >= 3 and len(remaining[0]) <= 7 and remaining[0] not in {background_word, chip_text}:
        script_text = remaining[0].title()
        remaining = remaining[1:]

    if len(remaining) >= 4:
        split_index = max(1, len(remaining) // 2)
        while split_index > 1 and remaining[split_index - 1] in _CONNECTOR_WORDS:
            split_index -= 1
    elif len(remaining) == 3:
        split_index = 2
    else:
        split_index = len(remaining)

    primary_text = " ".join(remaining[:split_index]).strip()
    secondary_text = " ".join(remaining[split_index:]).strip()
    if not primary_text and chip_text:
        primary_text = chip_text
    if not background_word and primary_text:
        background_word = max(primary_text.split(), key=len, default=primary_text)

    return {
        "background_word": background_word or chip_text or hook,
        "primary_text": primary_text,
        "secondary_text": secondary_text,
        "chip_text": chip_text,
        "script_text": script_text,
    }


def _render_text_with_shadow(
    layer: Image.Image,
    text: str,
    *,
    position: tuple[int, int],
    font,
    fill: tuple[int, int, int, int],
    shadow_fill: tuple[int, int, int, int] = _SHADOW,
) -> list[int]:
    draw = ImageDraw.Draw(layer)
    x_pos, y_pos = position
    draw.text((x_pos + 4, y_pos + 5), text, font=font, fill=shadow_fill)
    draw.text((x_pos, y_pos), text, font=font, fill=fill)
    bbox = draw.textbbox((x_pos, y_pos), text, font=font)
    return [bbox[0], bbox[1], bbox[2], bbox[3]]


def _render_chip_layer(
    canvas_size: tuple[int, int],
    *,
    text: str,
    position: tuple[int, int],
    palette: dict,
    font_size: int,
) -> tuple[Image.Image, list[int]]:
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    font = _load_role_font(font_size, "bold")
    draw = ImageDraw.Draw(layer)
    bbox = draw.textbbox((0, 0), text, font=font)
    pad_x = max(16, font_size // 6)
    pad_y = max(8, font_size // 10)
    chip_rect = [
        position[0],
        position[1],
        position[0] + (bbox[2] - bbox[0]) + pad_x * 2,
        position[1] + (bbox[3] - bbox[1]) + pad_y * 2,
    ]
    draw.rounded_rectangle(chip_rect, radius=max(18, font_size // 3), fill=palette["chip_bg"])
    draw.text((position[0] + pad_x, position[1] + pad_y - 2), text, font=font, fill=palette["chip_fg"])
    return layer, chip_rect


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
    max_width = max(120, width - 42)
    allowed_lines = 3
    max_text_height = max(120, height - 88)
    lines = _wrap_text(hook, font, max_width, draw)
    line_height = int(font_size * 1.1)
    total_height = line_height * len(lines)

    while (len(lines) > allowed_lines or total_height > max_text_height) and font_size > 44:
        font_size -= 4
        font = _load_font(font_size)
        lines = _wrap_text(hook, font, max_width, draw)
        line_height = int(font_size * 1.1)
        total_height = line_height * len(lines)

    panel_fill = (6, 10, 16, 170)
    shadow_fill = (0, 0, 0, 88)
    shadow_panel = [panel[0] + 6, panel[1] + 9, panel[2] + 6, panel[3] + 9]
    draw.rounded_rectangle(shadow_panel, radius=28, fill=shadow_fill)
    draw.rounded_rectangle(panel, radius=28, fill=panel_fill, outline=(255, 255, 255, 18), width=2)

    x_pos = panel[0] + 22
    y_pos = panel[1] + 18
    text_bounds: list[int] | None = None
    for line in lines:
        fill = _IVORY
        draw.text((x_pos + 4, y_pos + 6), line, font=font, fill=_SHADOW)
        if accent and accent in line:
            prefix, _, suffix = line.partition(accent)
            prefix_w = draw.textbbox((0, 0), prefix, font=font)[2]
            accent_w = draw.textbbox((0, 0), accent, font=font)[2]
            chip_pad_x = max(12, font_size // 7)
            chip_pad_y = max(6, font_size // 11)
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
        text_bounds = _rect_union(text_bounds, bbox)
        y_pos += line_height

    brand_font = _load_font(max(24, int(font_size * 0.34)))
    brand_bbox = draw.textbbox((0, 0), brand_label, font=brand_font)
    brand_rect = [
        panel[0] + 22,
        panel[3] - 48,
        panel[0] + 22 + (brand_bbox[2] - brand_bbox[0]) + 28,
        panel[3] - 14,
    ]
    draw.rounded_rectangle(brand_rect, radius=15, fill=(8, 13, 22, 210))
    draw.text((brand_rect[0] + 14, brand_rect[1] + 6), brand_label, font=brand_font, fill=(244, 241, 233, 238))
    return panel, text_bounds or [panel[0], panel[1], panel[2], panel[1] + 80]


def _subject_layers(
    canvas_size: tuple[int, int],
    variant: str,
    speaker: Image.Image,
    *,
    echo_source: Image.Image | None = None,
) -> dict:
    width, height = canvas_size
    target_height = int(height * {
        "outlined_sermon": 0.70,
        "stacked_keyword": 0.68,
        "echo_cutout": 0.66,
        "depth_typography": 0.68,
        "split_depth_text": 0.67,
    }[variant])
    fitted = _resize_image(speaker, target_height=target_height)

    if variant == "outlined_sermon":
        x_pos = int(width * 0.52)
        y_pos = height - fitted.height - int(height * 0.04)
    elif variant == "stacked_keyword":
        x_pos = int(width * 0.04)
        y_pos = height - fitted.height - int(height * 0.04)
    elif variant == "split_depth_text":
        x_pos = int(width * 0.33)
        y_pos = height - fitted.height - int(height * 0.05)
    else:
        x_pos = int(width * 0.54)
        y_pos = height - fitted.height - int(height * 0.05)

    speaker_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    speaker_layer.alpha_composite(fitted, (x_pos, y_pos))
    subject_bounds = [x_pos, y_pos, x_pos + fitted.width, y_pos + fitted.height]

    echo_overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    echo_debug = None
    echo_bounds = None
    if variant in {"echo_cutout", "depth_typography", "split_depth_text"}:
        if variant == "split_depth_text":
            ghost_position = (int(width * 0.02), int(height * 0.20))
            ghost_height = int(height * 0.72)
        else:
            ghost_position = (int(width * 0.02), int(height * 0.16))
            ghost_height = int(height * 0.78)
        ghost_alpha, echo_overlay, echo_bounds = _build_echo_layer(
            canvas_size,
            echo_source if echo_source is not None else speaker,
            target_height=ghost_height,
            position=ghost_position,
        )
        echo_debug = {
            "echo_alpha": ghost_alpha,
            "echo_layer": echo_overlay,
            "echo_bounds": echo_bounds,
        }

    return {
        "speaker_layer": speaker_layer,
        "subject_bounds": subject_bounds,
        "echo_layer": echo_overlay,
        "echo_debug": echo_debug,
        "echo_bounds": echo_bounds,
    }


def _subject_layout(
    canvas: Image.Image,
    variant: str,
    speaker: Image.Image,
    *,
    echo_source: Image.Image | None = None,
) -> tuple[Image.Image, list[int], dict | None]:
    layers = _subject_layers(canvas.size, variant, speaker, echo_source=echo_source)
    composed = compose_layers(
        canvas,
        [
            {"image": layers["echo_layer"], "opacity": 1.0},
            {"image": layers["speaker_layer"], "opacity": 1.0},
        ],
    )
    return composed, layers["subject_bounds"], layers["echo_debug"]


def _render_brand_layer(canvas_size: tuple[int, int], *, brand_label: str, anchor: tuple[int, int]) -> tuple[Image.Image, list[int]]:
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    font = _load_role_font(28, "bold")
    draw = ImageDraw.Draw(layer)
    brand_bbox = draw.textbbox((0, 0), brand_label, font=font)
    rect = [
        anchor[0],
        anchor[1],
        anchor[0] + (brand_bbox[2] - brand_bbox[0]) + 30,
        anchor[1] + (brand_bbox[3] - brand_bbox[1]) + 18,
    ]
    draw.rounded_rectangle(rect, radius=15, fill=(8, 13, 22, 214))
    draw.text((rect[0] + 15, rect[1] + 7), brand_label, font=font, fill=(244, 241, 233, 238))
    return layer, rect


def _build_partial_panel_layer(
    canvas_size: tuple[int, int],
    line_bounds: list[list[int]],
    *,
    fill: tuple[int, int, int, int],
) -> Image.Image:
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for bounds in line_bounds:
        rect = [bounds[0] - 18, bounds[1] - 12, bounds[2] + 18, bounds[3] + 10]
        draw.rounded_rectangle(rect, radius=22, fill=fill)
    return layer.filter(ImageFilter.GaussianBlur(radius=0.8))


def _render_depth_text_system(
    base_canvas: Image.Image,
    *,
    variant: str,
    hook: str,
    accent: str,
    palette: dict,
    brand_label: str,
    subject_bounds: list[int],
    echo_bounds: list[int] | None,
) -> tuple[Image.Image, dict, dict]:
    width, height = base_canvas.size
    pieces = _split_hook_layout(hook, accent)
    panel_mode = (
        TEXT_PANEL_MODES["partial_panel"]
        if variant == "depth_typography"
        else TEXT_PANEL_MODES["no_panel_with_shadow"]
    )

    background_word_layer = draw_background_word(
        base_canvas,
        pieces["background_word"],
        position=(int(width * 0.02), int(height * 0.08)),
        font_size=int(height * (0.18 if variant == "depth_typography" else 0.15)),
        fill=(245, 232, 206, 58 if variant == "depth_typography" else 48),
        rotate_degrees=-7.0 if variant == "depth_typography" else -3.5,
        tracking=max(2, width // 240),
        role="condensed",
    )

    speaker_face_proxy = _speaker_face_proxy(subject_bounds)
    midground_shapes = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
    shape_draw = ImageDraw.Draw(midground_shapes)
    shape_draw.rounded_rectangle(
        [int(width * 0.05), int(height * 0.50), int(width * 0.47), int(height * 0.90)],
        radius=34,
        fill=(8, 10, 16, 34 if variant == "depth_typography" else 24),
    )
    midground_shapes = midground_shapes.filter(ImageFilter.GaussianBlur(radius=18))

    stage_canvas = compose_layers(
        base_canvas,
        [
            {"image": background_word_layer, "blend_mode": "screen", "opacity": 0.92},
            {"image": midground_shapes, "opacity": 0.90},
        ],
    )

    text_back = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
    text_front = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
    primary_font = _load_role_font(int(height * (0.078 if variant == "depth_typography" else 0.086)), "bold")
    secondary_font = _load_role_font(int(height * 0.046), "serif")
    script_font = _load_role_font(int(height * 0.042), "script")
    back_word_font = _load_role_font(int(height * 0.102), "condensed")

    primary_origin = (int(width * 0.07), int(height * 0.58 if variant == "depth_typography" else 0.49))
    secondary_origin = (primary_origin[0], primary_origin[1] + int(height * 0.15))
    script_origin = (primary_origin[0], primary_origin[1] - int(height * 0.07))
    max_text_width = int(width * (0.36 if variant == "depth_typography" else 0.28))

    background_for_color = stage_canvas
    primary_fill = choose_text_color_for_region(
        background_for_color,
        [primary_origin[0], primary_origin[1], primary_origin[0] + max_text_width, primary_origin[1] + int(height * 0.16)],
    )
    secondary_fill = choose_text_color_for_region(
        background_for_color,
        [secondary_origin[0], secondary_origin[1], secondary_origin[0] + max_text_width, secondary_origin[1] + int(height * 0.10)],
    )

    line_bounds: list[list[int]] = []
    text_bounds: list[int] | None = None
    back_text_bounds: list[int] | None = None

    if pieces["script_text"]:
        script_bounds = _render_text_with_shadow(
            text_back,
            pieces["script_text"],
            position=script_origin,
            font=script_font,
            fill=(245, 225, 188, 218),
            shadow_fill=(0, 0, 0, 160),
        )
        text_bounds = _rect_union(text_bounds, script_bounds)

    primary_lines = _wrap_text(pieces["primary_text"] or hook, primary_font, max_text_width, ImageDraw.Draw(text_back))
    y_pos = primary_origin[1]
    for line in primary_lines[:2]:
        bounds = _render_text_with_shadow(
            text_back,
            line,
            position=(primary_origin[0], y_pos),
            font=primary_font,
            fill=primary_fill,
        )
        line_bounds.append(bounds)
        text_bounds = _rect_union(text_bounds, bounds)
        y_pos = bounds[3] + max(10, int(height * 0.012))

    if pieces["secondary_text"]:
        secondary_lines = _wrap_text(pieces["secondary_text"], secondary_font, max_text_width, ImageDraw.Draw(text_front))
        y_pos = max(secondary_origin[1], y_pos + int(height * 0.01))
        for line in secondary_lines[:2]:
            bounds = _render_text_with_shadow(
                text_front,
                line,
                position=(secondary_origin[0], y_pos),
                font=secondary_font,
                fill=secondary_fill,
                shadow_fill=(0, 0, 0, 168),
            )
            line_bounds.append(bounds)
            text_bounds = _rect_union(text_bounds, bounds)
            y_pos = bounds[3] + max(8, int(height * 0.010))

    if panel_mode == TEXT_PANEL_MODES["partial_panel"] and line_bounds:
        partial_panel = _build_partial_panel_layer(base_canvas.size, line_bounds, fill=(6, 10, 16, 128))
        text_back = compose_layers(text_back, [{"image": partial_panel, "opacity": 1.0}, {"image": text_back, "opacity": 1.0}])

    split_back_layer = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
    if variant == "split_depth_text":
        back_phrase = " ".join((pieces["primary_text"] or hook).split()[:2]).strip() or (pieces["primary_text"] or hook)
        back_phrase_bounds = _render_text_with_shadow(
            split_back_layer,
            back_phrase,
            position=(int(width * 0.04), int(height * 0.56)),
            font=back_word_font,
            fill=(244, 239, 228, 164),
            shadow_fill=(0, 0, 0, 110),
        )
        back_text_bounds = back_phrase_bounds
        text_bounds = _rect_union(text_bounds, back_phrase_bounds)

    chip_layer, chip_bounds = _render_chip_layer(
        base_canvas.size,
        text=pieces["chip_text"] or accent or pieces["background_word"],
        position=(primary_origin[0], int(height * (0.80 if variant == "depth_typography" else 0.70))),
        palette=palette,
        font_size=int(height * 0.045),
    )
    text_front = compose_layers(text_front, [{"image": chip_layer, "opacity": 1.0}])
    text_bounds = _rect_union(text_bounds, chip_bounds)

    brand_layer, brand_bounds = _render_brand_layer(
        base_canvas.size,
        brand_label=brand_label,
        anchor=(primary_origin[0], height - int(height * 0.10)),
    )

    depth_overlap = _rect_overlap_ratio(back_text_bounds, subject_bounds)
    face_collision = max(
        _rect_overlap_ratio(text_bounds, speaker_face_proxy),
        _rect_overlap_ratio(chip_bounds, speaker_face_proxy),
    )
    background_word_penalty = 0.0
    if _rect_overlap_ratio(text_bounds, _clip_rect([int(width * 0.02), int(height * 0.08), int(width * 0.70), int(height * 0.42)], base_canvas.size)) > 0.74:
        background_word_penalty = 0.04

    text_occlusion_penalty = 0.0
    depth_bonus = 0.0
    if variant == "split_depth_text":
        if depth_overlap > 0.42:
            text_occlusion_penalty = min(0.12, (depth_overlap - 0.42) * 0.36)
        elif 0.12 <= depth_overlap <= 0.36:
            depth_bonus = 0.04
    elif 0.06 <= depth_overlap <= 0.22:
        depth_bonus = 0.03

    echo_rect_penalty = 0.0
    if echo_bounds:
        echo_area_ratio = ((echo_bounds[2] - echo_bounds[0]) * (echo_bounds[3] - echo_bounds[1])) / float(max(1, width * height))
        if echo_area_ratio > 0.22:
            echo_rect_penalty += 0.03
        if _rect_overlap_ratio(echo_bounds, subject_bounds) < 0.10:
            echo_rect_penalty += 0.02

    negative_space_penalty = 0.0
    if text_bounds:
        text_area_ratio = ((text_bounds[2] - text_bounds[0]) * (text_bounds[3] - text_bounds[1])) / float(max(1, width * height))
        if text_area_ratio < 0.032:
            negative_space_penalty = 0.03

    clipped_text_bounds = _clip_rect(
        text_bounds or [primary_origin[0], primary_origin[1], primary_origin[0] + max_text_width, primary_origin[1] + 120],
        base_canvas.size,
    )
    metadata = {
        "panel_mode": panel_mode,
        "text_bounds": clipped_text_bounds,
        "background_word": pieces["background_word"],
        "speaker_face_proxy": speaker_face_proxy,
        "background_word_penalty": round(background_word_penalty, 4),
        "face_collision_penalty": round(face_collision * 0.22, 4),
        "text_occlusion_penalty": round(text_occlusion_penalty, 4),
        "negative_space_penalty": round(negative_space_penalty, 4),
        "depth_bonus": round(depth_bonus, 4),
        "keyword_visibility_bonus": 0.03 if _rect_overlap_ratio(chip_bounds, subject_bounds) < 0.24 else 0.0,
        "echo_rect_penalty": round(echo_rect_penalty, 4),
        "chip_bounds": chip_bounds,
        "brand_bounds": brand_bounds,
    }
    debug = {
        "background_typography": background_word_layer,
        "midground_shapes": midground_shapes,
        "text_back": compose_layers(split_back_layer, [{"image": text_back, "opacity": 1.0}]) if variant == "split_depth_text" else text_back,
        "text_front": compose_layers(text_front, [{"image": brand_layer, "opacity": 1.0}]),
        "split_back": split_back_layer,
        "brand_layer": brand_layer,
    }
    if variant == "split_depth_text":
        debug["text_back"] = compose_layers(split_back_layer, [{"image": text_back, "opacity": 1.0}])
    return brand_layer, metadata, debug


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
    provider_metrics: dict | None = None,
) -> tuple[Image.Image, dict, dict | None]:
    hook = _sanitize_hook(hook_text)
    accent = _sanitize_keyword(hook, accent_keyword)
    brand_label = str(brief.get("brand_label") or _DEFAULT_BRAND).strip().upper()[:28] or _DEFAULT_BRAND
    base_variant = _BASE_VARIANT_BY_TEST[variant]
    palette = _PALETTES[base_variant]
    provider_metrics = dict(provider_metrics or {})

    canvas = _frame_background(frame_bgr, base_variant)
    prepared_speaker = brighten_face_region(speaker_rgba, face_box, brightness=1.12, color=1.05)
    prepared_speaker = crop_to_alpha(prepared_speaker)
    prepared_speaker = ImageEnhance.Brightness(prepared_speaker).enhance(1.02)
    echo_source = prepared_speaker.copy()
    speaker = add_speaker_outline(prepared_speaker, _OUTLINE_PRESET_BY_TEST[variant])

    layer_debug = None
    if variant in {"depth_typography", "split_depth_text"}:
        subject_layers = _subject_layers(canvas.size, variant, speaker, echo_source=echo_source)
        background_base = canvas.copy()
        background_typography = draw_background_word(
            background_base,
            _split_hook_layout(hook, accent)["background_word"],
            position=(int(background_base.width * 0.02), int(background_base.height * 0.08)),
            font_size=int(background_base.height * (0.18 if variant == "depth_typography" else 0.15)),
            fill=(245, 232, 206, 58 if variant == "depth_typography" else 48),
            rotate_degrees=-7.0 if variant == "depth_typography" else -3.5,
            tracking=max(2, background_base.width // 240),
            role="condensed",
        )
        background_canvas = compose_layers(
            background_base,
            [
                {"image": background_typography, "blend_mode": "screen", "opacity": 0.92},
                {"image": subject_layers["echo_layer"], "opacity": 1.0},
            ],
        )
        brand_layer, depth_meta, depth_debug = _render_depth_text_system(
            background_canvas,
            variant=variant,
            hook=hook,
            accent=accent,
            palette=palette,
            brand_label=brand_label,
            subject_bounds=subject_layers["subject_bounds"],
            echo_bounds=subject_layers["echo_bounds"],
        )
        final_canvas = compose_layers(
            background_canvas,
            [
                {"image": depth_debug["text_back"], "opacity": 1.0},
                {"image": subject_layers["speaker_layer"], "opacity": 1.0},
                {"image": depth_debug["text_front"], "opacity": 1.0},
            ],
        )
        final_grade = Image.new("RGBA", final_canvas.size, (166, 104, 58, 18 if variant == "depth_typography" else 12))
        final_canvas = Image.alpha_composite(final_canvas, final_grade)

        metadata = {
            "variant": variant,
            "layout": variant,
            "hook_text": hook,
            "accent_keyword": accent,
            "face_box": list(face_box) if face_box else None,
            "speaker_bounds": subject_layers["subject_bounds"],
            "subject_coverage": round(float(coverage), 4),
            "panel": [0, 0, 0, 0],
            "text_bounds": [int(value) for value in depth_meta["text_bounds"]],
            "brand_label": brand_label,
            "outline_preset": _OUTLINE_PRESET_BY_TEST[variant],
            "background_word": depth_meta["background_word"],
            "panel_mode": depth_meta["panel_mode"],
            "background_word_penalty": depth_meta["background_word_penalty"],
            "face_collision_penalty": depth_meta["face_collision_penalty"],
            "text_occlusion_penalty": depth_meta["text_occlusion_penalty"],
            "negative_space_penalty": depth_meta["negative_space_penalty"],
            "depth_bonus": depth_meta["depth_bonus"],
            "keyword_visibility_bonus": depth_meta["keyword_visibility_bonus"],
            "echo_rect_penalty": depth_meta["echo_rect_penalty"],
            "edge_touch_penalty": float(provider_metrics.get("edge_touch_penalty") or 0.0),
            "thin_component_penalty": float(provider_metrics.get("thin_component_penalty") or 0.0),
        }
        layer_debug = {
            "layers": {
                "debug_layers_01_background.png": background_base,
                "debug_layers_02_background_typography.png": compose_layers(background_base, [{"image": background_typography, "opacity": 1.0}]),
                "debug_layers_03_echo.png": subject_layers["echo_layer"],
                "debug_layers_04_speaker.png": subject_layers["speaker_layer"],
                "debug_layers_05_text_back.png": depth_debug["text_back"],
                "debug_layers_06_text_front.png": depth_debug["text_front"],
                "debug_layers_07_final.png": final_canvas,
            },
            "echo_debug": subject_layers["echo_debug"],
        }
        return final_canvas, metadata, layer_debug

    canvas, subject_bounds, echo_debug = _subject_layout(canvas, variant, speaker, echo_source=echo_source)

    width, height = canvas.size
    if variant == "outlined_sermon":
        panel = [int(width * 0.06), int(height * 0.64), int(width * 0.60), int(height * 0.85)]
        font_ratio = 0.082
    elif variant == "stacked_keyword":
        panel = [int(width * 0.48), int(height * 0.29), int(width * 0.93), int(height * 0.58)]
        font_ratio = 0.078
    else:
        panel = [int(width * 0.09), int(height * 0.70), int(width * 0.88), int(height * 0.87)]
        font_ratio = 0.080

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
        "edge_touch_penalty": float(provider_metrics.get("edge_touch_penalty") or 0.0),
        "thin_component_penalty": float(provider_metrics.get("thin_component_penalty") or 0.0),
        "echo_rect_penalty": 0.04 if variant == "echo_cutout" and echo_debug is not None and echo_debug.get("echo_bounds") else 0.0,
    }
    return canvas, metadata, {"echo_debug": echo_debug} if echo_debug is not None else None


def _extract_provider_metrics(provider) -> dict:
    raw = getattr(provider, "last_debug", None) or getattr(provider, "last_component_debug", None) or {}
    edge_ratio = float(
        raw.get("edge_touch_ratio")
        or raw.get("border_touch_ratio")
        or raw.get("remaining_edge_ratio")
        or 0.0
    )
    thin_ratio = float(
        raw.get("thin_component_ratio")
        or raw.get("remaining_thin_ratio")
        or raw.get("removed_thin_ratio")
        or 0.0
    )
    return {
        "edge_touch_penalty": round(min(0.16, edge_ratio * 0.18), 4),
        "thin_component_penalty": round(min(0.16, thin_ratio * 0.22), 4),
    }


def _save_layer_debug(output_dir: Path, layer_debug: dict | None) -> bool:
    if not layer_debug:
        return False
    layers = layer_debug.get("layers") or {}
    if not layers:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, image in layers.items():
        image.convert("RGBA").save(output_dir / file_name)
    return True


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

    provider = get_background_removal_provider(None)
    speaker_rgba, face_box, coverage = provider.extract_subject(frame_bgr)
    provider_metrics = _extract_provider_metrics(provider)
    raw_subject_path = variant_root / "speaker_cutout.png"
    crop_to_alpha(speaker_rgba).save(raw_subject_path)

    preferred_variant = {
        "clean_gradient": "echo_cutout",
        "strong_contrast": "echo_cutout",
        "emotion_focus": "outlined_sermon",
    }.get(str(brief.get("background_style") or "").strip().lower())

    candidates: list[dict] = []
    best_debug = None
    best_variant_name = None
    best_outline_preset = "creator_white"
    fallback_echo_debug = None
    for variant in V2_TEST_VARIANTS:
        rendered, metadata, variant_debug = _compose_variant(
            frame_bgr,
            variant=variant,
            hook_text=hook_text,
            accent_keyword=accent_keyword,
            brief=brief,
            speaker_rgba=speaker_rgba,
            face_box=face_box,
            coverage=coverage,
            provider_metrics=provider_metrics,
        )
        variant_path = variant_root / f"{variant}.jpg"
        rendered.convert("RGB").save(variant_path, "JPEG", quality=94)
        metadata["image_path"] = str(variant_path)
        metadata["score"] = _score_variant(rendered, metadata)
        if variant == preferred_variant:
            metadata["score"] = round(float(metadata["score"]) + 0.16, 4)
        if brief.get("emotion_target") and variant == "outlined_sermon":
            metadata["score"] = round(float(metadata["score"]) + 0.03, 4)
        if brief.get("emotion_target") and variant == "depth_typography":
            metadata["score"] = round(float(metadata["score"]) + 0.02, 4)
        if preferred_variant == "echo_cutout" and variant in {"depth_typography", "split_depth_text"}:
            metadata["score"] = round(float(metadata["score"]) - 0.03, 4)
        if variant in {"depth_typography", "split_depth_text"} and _save_layer_debug(variant_root / f"{variant}_layers", variant_debug):
            metadata["layer_debug_dir"] = str(variant_root / f"{variant}_layers")
        candidates.append(metadata)
        if fallback_echo_debug is None and variant_debug and variant_debug.get("echo_debug"):
            fallback_echo_debug = variant_debug["echo_debug"]
        if best_variant_name is None or float(metadata["score"]) > float(next(item["score"] for item in candidates if item["variant"] == best_variant_name)):
            best_debug = variant_debug
            best_variant_name = variant
            best_outline_preset = str(metadata.get("outline_preset") or "creator_white")

    best = max(candidates, key=lambda item: float(item.get("score") or 0.0))
    if best["variant"] != best_variant_name:
        for variant in V2_TEST_VARIANTS:
            if variant == best["variant"]:
                _, _, best_debug = _compose_variant(
                    frame_bgr,
                    variant=variant,
                    hook_text=hook_text,
                    accent_keyword=accent_keyword,
                    brief=brief,
                    speaker_rgba=speaker_rgba,
                    face_box=face_box,
                    coverage=coverage,
                    provider_metrics=provider_metrics,
                )
                best_outline_preset = str(best.get("outline_preset") or "creator_white")
                break

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

    debug_dir = variant_root / "outline_debug"
    save_outline_debug_layers(
        speaker_rgba,
        debug_dir,
        best_outline_preset,
        component_debug=getattr(provider, "last_debug", None),
    )
    echo_debug = None
    if best_debug and best_debug.get("echo_debug"):
        echo_debug = best_debug["echo_debug"]
    elif fallback_echo_debug is not None:
        echo_debug = fallback_echo_debug
    if echo_debug:
        if echo_debug.get("echo_alpha") is not None:
            echo_debug["echo_alpha"].save(debug_dir / "debug_07_echo_alpha.png")
        if echo_debug.get("echo_layer") is not None:
            echo_debug["echo_layer"].save(debug_dir / "debug_08_echo_layer.png")
    _save_layer_debug(variant_root / "layer_debug", best_debug)

    _write_video_from_image(
        image_path=str(selected_path),
        output_path=output_video_path,
        fps=fps,
        duration=duration,
        motion_style=_MOTION_BY_TEST_VARIANT.get(best["variant"], "push_in"),
    )
    return output_video_path, str(thumb_path), f"v2_test_{best['variant']}"
