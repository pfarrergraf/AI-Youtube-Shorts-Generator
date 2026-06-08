from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from Components.ThumbnailV2 import _detect_face_box, _extract_subject_rgba


@dataclass(frozen=True)
class OutlinePreset:
    stroke_color: tuple[int, int, int, int]
    stroke_width: int
    stroke_feather: float
    glow_color: tuple[int, int, int, int]
    glow_radius: int
    glow_feather: float
    shadow_color: tuple[int, int, int, int]
    shadow_blur: int
    shadow_offset: tuple[int, int]
    edge_feather: float


OUTLINE_PRESETS: dict[str, OutlinePreset] = {
    "creator_white": OutlinePreset(
        stroke_color=(250, 248, 242, 225),
        stroke_width=6,
        stroke_feather=3.0,
        glow_color=(255, 245, 225, 40),
        glow_radius=14,
        glow_feather=9.0,
        shadow_color=(0, 0, 0, 115),
        shadow_blur=30,
        shadow_offset=(6, 12),
        edge_feather=1.25,
    ),
    "creator_blue": OutlinePreset(
        stroke_color=(214, 224, 228, 160),
        stroke_width=7,
        stroke_feather=2.6,
        glow_color=(142, 176, 198, 48),
        glow_radius=16,
        glow_feather=4.0,
        shadow_color=(4, 10, 18, 136),
        shadow_blur=26,
        shadow_offset=(6, 13),
        edge_feather=1.5,
    ),
    "sermon_gold": OutlinePreset(
        stroke_color=(255, 245, 225, 225),
        stroke_width=6,
        stroke_feather=3.2,
        glow_color=(255, 205, 120, 46),
        glow_radius=16,
        glow_feather=10.0,
        shadow_color=(0, 0, 0, 105),
        shadow_blur=32,
        shadow_offset=(6, 12),
        edge_feather=1.35,
    ),
}


def _clamp_byte(value: float | int) -> int:
    return max(0, min(255, int(round(float(value)))))


def _clamp_opacity(opacity: float) -> float:
    return max(0.0, min(1.0, float(opacity)))


def _normalise_color(color: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(color) == 4:
        red, green, blue, alpha = color
        return (_clamp_byte(red), _clamp_byte(green), _clamp_byte(blue), _clamp_byte(alpha))
    if len(color) == 3:
        red, green, blue = color
        return (_clamp_byte(red), _clamp_byte(green), _clamp_byte(blue), 255)
    raise ValueError(f"Expected RGB or RGBA color tuple, got {color!r}")


def _normalise_point(point: tuple[float, float], size: tuple[int, int]) -> tuple[float, float]:
    width, height = size
    x_pos, y_pos = float(point[0]), float(point[1])
    if 0.0 <= x_pos <= 1.0:
        x_pos *= max(1, width - 1)
    if 0.0 <= y_pos <= 1.0:
        y_pos *= max(1, height - 1)
    return x_pos, y_pos


def _normalise_radius(value: float, extent: int) -> float:
    resolved = float(value)
    if 0.0 < resolved <= 1.0:
        resolved *= max(1, extent)
    return max(1.0, resolved)


def _normalise_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    alpha = mask.convert("L")
    if alpha.size != size:
        alpha = alpha.resize(size, Image.Resampling.LANCZOS)
    return alpha


def _normalise_box(size: tuple[int, int], box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    width, height = size
    if box is None:
        return (0, 0, width, height)
    left, top, right, bottom = [int(round(float(value))) for value in box]
    left = max(0, min(width, left))
    top = max(0, min(height, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return (left, top, right, bottom)


def build_linear_gradient_mask(
    size: tuple[int, int],
    *,
    start: tuple[float, float] = (0.0, 0.0),
    end: tuple[float, float] = (0.0, 1.0),
    start_alpha: int = 0,
    end_alpha: int = 255,
    power: float = 1.0,
    box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    left, top, right, bottom = _normalise_box(size, box)
    region_width = max(1, right - left)
    region_height = max(1, bottom - top)
    start_x, start_y = _normalise_point(start, (region_width, region_height))
    end_x, end_y = _normalise_point(end, (region_width, region_height))
    vector_x = end_x - start_x
    vector_y = end_y - start_y
    denom = max(1e-6, vector_x * vector_x + vector_y * vector_y)

    xs = np.arange(region_width, dtype=np.float32)[None, :]
    ys = np.arange(region_height, dtype=np.float32)[:, None]
    progress = ((xs - start_x) * vector_x + (ys - start_y) * vector_y) / denom
    progress = np.clip(progress, 0.0, 1.0)
    if power != 1.0:
        progress = np.power(progress, max(1e-3, float(power)))

    start_value = float(_clamp_byte(start_alpha))
    end_value = float(_clamp_byte(end_alpha))
    alpha = np.clip(start_value + (end_value - start_value) * progress, 0.0, 255.0).astype(np.uint8)
    mask[top:bottom, left:right] = alpha
    return Image.fromarray(mask, mode="L")


def build_radial_gradient_mask(
    size: tuple[int, int],
    *,
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.5,
    radius_y: float | None = None,
    inner_alpha: int = 255,
    outer_alpha: int = 0,
    power: float = 1.0,
    box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    left, top, right, bottom = _normalise_box(size, box)
    region_width = max(1, right - left)
    region_height = max(1, bottom - top)
    center_x, center_y = _normalise_point(center, (region_width, region_height))
    radius_x = _normalise_radius(radius, region_width)
    radius_vertical = _normalise_radius(radius if radius_y is None else radius_y, region_height)

    xs = np.arange(region_width, dtype=np.float32)[None, :]
    ys = np.arange(region_height, dtype=np.float32)[:, None]
    distance = np.sqrt(((xs - center_x) / radius_x) ** 2 + ((ys - center_y) / radius_vertical) ** 2)
    progress = np.clip(distance, 0.0, 1.0)
    if power != 1.0:
        progress = np.power(progress, max(1e-3, float(power)))

    inner_value = float(_clamp_byte(inner_alpha))
    outer_value = float(_clamp_byte(outer_alpha))
    alpha = np.clip(inner_value + (outer_value - inner_value) * progress, 0.0, 255.0).astype(np.uint8)
    mask[top:bottom, left:right] = alpha
    return Image.fromarray(mask, mode="L")


def build_shape_mask(
    size: tuple[int, int],
    *,
    shape: str = "rounded_rect",
    box: tuple[int, int, int, int] | None = None,
    radius: int | None = None,
    feather: float = 0.0,
    invert: bool = False,
    alpha: int = 255,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    left, top, right, bottom = _normalise_box(size, box)
    fill_alpha = _clamp_byte(alpha)
    resolved_shape = str(shape or "rounded_rect").strip().lower()
    width = max(1, right - left)
    height = max(1, bottom - top)
    rounded_radius = max(0, int(radius if radius is not None else min(width, height) * 0.12))

    if resolved_shape in {"rect", "rectangle"}:
        draw.rectangle([left, top, right, bottom], fill=fill_alpha)
    elif resolved_shape in {"rounded_rect", "rounded_rectangle"}:
        draw.rounded_rectangle([left, top, right, bottom], radius=rounded_radius, fill=fill_alpha)
    elif resolved_shape in {"pill", "capsule"}:
        draw.rounded_rectangle([left, top, right, bottom], radius=max(1, min(width, height) // 2), fill=fill_alpha)
    elif resolved_shape in {"ellipse", "oval"}:
        draw.ellipse([left, top, right, bottom], fill=fill_alpha)
    elif resolved_shape == "circle":
        diameter = min(width, height)
        circle_left = left + (width - diameter) // 2
        circle_top = top + (height - diameter) // 2
        draw.ellipse([circle_left, circle_top, circle_left + diameter, circle_top + diameter], fill=fill_alpha)
    elif resolved_shape == "diamond":
        center_x = left + width / 2.0
        center_y = top + height / 2.0
        draw.polygon(
            [
                (center_x, top),
                (right, center_y),
                (center_x, bottom),
                (left, center_y),
            ],
            fill=fill_alpha,
        )
    else:
        raise ValueError(f"Unsupported mask shape: {shape}")

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    if invert:
        mask = ImageChops.invert(mask)
    return mask


def combine_masks(*masks: Image.Image, mode: str = "multiply") -> Image.Image:
    if not masks:
        raise ValueError("combine_masks requires at least one mask")

    result = masks[0].convert("L")
    resolved_mode = str(mode or "multiply").strip().lower()
    for mask in masks[1:]:
        other = _normalise_mask(mask, result.size)
        if resolved_mode == "multiply":
            result = ImageChops.multiply(result, other)
        elif resolved_mode == "screen":
            result = ImageChops.screen(result, other)
        elif resolved_mode in {"add", "lighter"}:
            result = ImageChops.lighter(result, other) if resolved_mode == "lighter" else ImageChops.add(result, other)
        elif resolved_mode in {"min", "darker"}:
            result = ImageChops.darker(result, other)
        elif resolved_mode == "subtract":
            result = ImageChops.subtract(result, other)
        else:
            raise ValueError(f"Unsupported mask combine mode: {mode}")
    return result


def render_mask_layer(
    size: tuple[int, int],
    color: tuple[int, ...],
    mask: Image.Image,
    *,
    opacity: float = 1.0,
    blur_radius: float = 0.0,
) -> Image.Image:
    red, green, blue, alpha = _normalise_color(color)
    alpha_mask = _normalise_mask(mask, size)
    if blur_radius > 0:
        alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    opacity_value = _clamp_opacity(opacity)
    alpha_scale = float(alpha) * opacity_value / 255.0
    alpha_mask = alpha_mask.point(lambda value: _clamp_byte(value * alpha_scale))
    layer = Image.new("RGBA", size, (red, green, blue, 0))
    layer.putalpha(alpha_mask)
    return layer


def composite_layer(
    base: Image.Image,
    layer: Image.Image,
    *,
    mask: Image.Image | None = None,
    offset: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    blur_radius: float = 0.0,
    blend_mode: str = "alpha",
) -> Image.Image:
    base_rgba = base.convert("RGBA")
    overlay = layer.convert("RGBA")
    alpha = overlay.getchannel("A")
    if mask is not None:
        alpha = combine_masks(alpha, _normalise_mask(mask, overlay.size), mode="multiply")
    if blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    if opacity != 1.0:
        opacity_value = _clamp_opacity(opacity)
        alpha = alpha.point(lambda value: _clamp_byte(value * opacity_value))
    overlay.putalpha(alpha)

    prepared = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    prepared.paste(overlay, tuple(int(value) for value in offset), overlay)
    resolved_mode = str(blend_mode or "alpha").strip().lower()
    if resolved_mode == "alpha":
        return Image.alpha_composite(base_rgba, prepared)

    base_rgb = base_rgba.convert("RGB")
    prepared_rgb = prepared.convert("RGB")
    if resolved_mode == "screen":
        blended_rgb = ImageChops.screen(base_rgb, prepared_rgb)
    elif resolved_mode == "multiply":
        blended_rgb = ImageChops.multiply(base_rgb, prepared_rgb)
    elif resolved_mode == "add":
        blended_rgb = ImageChops.add(base_rgb, prepared_rgb)
    elif resolved_mode in {"overlay", "hard_light", "soft_light", "darken", "lighten", "difference", "subtract"}:
        base_arr = np.asarray(base_rgb, dtype=np.float32) / 255.0
        layer_arr = np.asarray(prepared_rgb, dtype=np.float32) / 255.0
        if resolved_mode == "overlay":
            blended_arr = np.where(base_arr <= 0.5, 2.0 * base_arr * layer_arr, 1.0 - 2.0 * (1.0 - base_arr) * (1.0 - layer_arr))
        elif resolved_mode == "hard_light":
            blended_arr = np.where(layer_arr <= 0.5, 2.0 * base_arr * layer_arr, 1.0 - 2.0 * (1.0 - base_arr) * (1.0 - layer_arr))
        elif resolved_mode == "soft_light":
            d = np.where(base_arr <= 0.25, ((16.0 * base_arr - 12.0) * base_arr + 4.0) * base_arr, np.sqrt(np.clip(base_arr, 0.0, 1.0)))
            blended_arr = np.where(
                layer_arr <= 0.5,
                base_arr - (1.0 - 2.0 * layer_arr) * base_arr * (1.0 - base_arr),
                base_arr + (2.0 * layer_arr - 1.0) * (d - base_arr),
            )
        elif resolved_mode == "darken":
            blended_arr = np.minimum(base_arr, layer_arr)
        elif resolved_mode == "lighten":
            blended_arr = np.maximum(base_arr, layer_arr)
        elif resolved_mode == "difference":
            blended_arr = np.abs(base_arr - layer_arr)
        else:  # subtract
            blended_arr = np.clip(base_arr - layer_arr, 0.0, 1.0)
        blended_rgb = Image.fromarray(np.clip(blended_arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")

    overlay_alpha = prepared.getchannel("A")
    result_rgb = Image.composite(blended_rgb, base_rgb, overlay_alpha)
    result = result_rgb.convert("RGBA")
    result.putalpha(ImageChops.lighter(base_rgba.getchannel("A"), overlay_alpha))
    return result


def add_layer(
    canvas: Image.Image,
    layer: Image.Image,
    *,
    opacity: float = 1.0,
    blend_mode: str = "normal",
    mask: Image.Image | None = None,
    offset: tuple[int, int] = (0, 0),
    blur_radius: float = 0.0,
) -> Image.Image:
    resolved_blend = "alpha" if str(blend_mode or "normal").strip().lower() in {"normal", "alpha"} else str(blend_mode)
    return composite_layer(
        canvas,
        layer,
        mask=mask,
        offset=offset,
        opacity=opacity,
        blur_radius=blur_radius,
        blend_mode=resolved_blend,
    )


def compose_layers(canvas: Image.Image, layers: list[dict]) -> tuple[Image.Image, dict[str, Image.Image]]:
    result = canvas.convert("RGBA")
    debug_layers: dict[str, Image.Image] = {}
    for index, layer_spec in enumerate(layers):
        if not layer_spec:
            continue
        layer = layer_spec.get("layer")
        if not isinstance(layer, Image.Image):
            continue
        result = add_layer(
            result,
            layer,
            opacity=float(layer_spec.get("opacity", 1.0)),
            blend_mode=str(layer_spec.get("blend_mode", "normal")),
            mask=layer_spec.get("mask"),
            offset=tuple(layer_spec.get("offset", (0, 0))),
            blur_radius=float(layer_spec.get("blur_radius", 0.0)),
        )
        debug_key = str(layer_spec.get("debug_key") or "").strip()
        if debug_key:
            debug_image = layer_spec.get("debug_image")
            debug_layers[debug_key] = (debug_image if isinstance(debug_image, Image.Image) else layer).convert("RGBA")
    return result, debug_layers


def get_configured_background_removal_provider_name(config_path: str = "config.yaml") -> str:
    try:
        from utils.config_utils import load_config
    except Exception:
        return "grabcut_local"

    cfg = load_config(config_path) or {}
    thumbnail_cfg = cfg.get("thumbnail") or {}
    provider_name = str(thumbnail_cfg.get("background_removal_provider") or "grabcut_local").strip().lower()
    return provider_name or "grabcut_local"


def _soft_threshold_alpha(alpha: Image.Image, low: int = 18, high: int = 180) -> Image.Image:
    """
    Converts a rough alpha mask into a smoother alpha ramp.
    Values below low become 0.
    Values above high become 255.
    Values in between are interpolated smoothly.
    """
    alpha = alpha.convert("L")
    low = max(0, min(254, int(low)))
    high = max(low + 1, min(255, int(high)))

    def remap(value: int) -> int:
        if value <= low:
            return 0
        if value >= high:
            return 255
        t_value = (value - low) / float(high - low)
        t_value = t_value * t_value * (3.0 - 2.0 * t_value)
        return int(t_value * 255)

    return alpha.point(remap)


def feather_alpha(alpha: Image.Image, *, blur_radius: float = 2.0, low: int = 12, high: int = 210) -> Image.Image:
    """
    Creates a feathered alpha mask with smoother edges.
    Useful for outline/glow masks.
    """
    alpha = alpha.convert("L")
    alpha = alpha.filter(ImageFilter.MedianFilter(3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    alpha = _soft_threshold_alpha(alpha, low=low, high=high)
    return alpha


def _filter_size_from_radius(radius: int) -> int:
    size = max(3, int(radius) * 2 + 1)
    return size if size % 2 == 1 else size + 1


def expand_alpha(alpha: Image.Image, radius: int, *, feather: float = 1.5) -> Image.Image:
    """
    Expands alpha outward and feathers the expanded edge.
    """
    radius = max(0, int(radius))
    alpha = alpha.convert("L")
    if radius <= 0:
        return feather_alpha(alpha, blur_radius=max(0.0, feather), low=6, high=245)

    expanded = alpha.filter(ImageFilter.MaxFilter(_filter_size_from_radius(radius)))
    if feather > 0:
        expanded = expanded.filter(ImageFilter.GaussianBlur(radius=feather))
    return _soft_threshold_alpha(expanded, low=6, high=245)


def contract_alpha(alpha: Image.Image, radius: int, *, feather: float = 1.0) -> Image.Image:
    """
    Contracts alpha inward. Useful to prevent the outline from bleeding into the subject.
    """
    radius = max(0, int(radius))
    alpha = alpha.convert("L")
    if radius <= 0:
        return feather_alpha(alpha, blur_radius=max(0.0, feather), low=8, high=245)

    contracted = alpha.filter(ImageFilter.MinFilter(_filter_size_from_radius(radius)))
    if feather > 0:
        contracted = contracted.filter(ImageFilter.GaussianBlur(radius=feather))
    return _soft_threshold_alpha(contracted, low=8, high=245)


def estimate_face_box(subject_rgba: Image.Image):
    alpha = subject_rgba.convert("RGBA").getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = [int(value) for value in bbox]
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    face_w = max(40, int(box_w * 0.42))
    face_h = max(40, int(box_h * 0.26))
    face_x = left + int(box_w * 0.28)
    face_y = top + int(box_h * 0.02)
    return (face_x, face_y, face_w, face_h)


def refine_cutout_edges(
    subject_rgba: Image.Image,
    *,
    threshold: int = 18,
    crop: bool = True,
    soft: bool = True,
) -> Image.Image:
    subject = subject_rgba.convert("RGBA")
    alpha = subject.getchannel("A")
    alpha = alpha.filter(ImageFilter.MedianFilter(3))

    if soft:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.9))
        alpha = _soft_threshold_alpha(alpha, low=threshold, high=210)
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.45))
    else:
        alpha = alpha.filter(ImageFilter.MaxFilter(5))
        alpha = alpha.filter(ImageFilter.MinFilter(3))
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.2))
        alpha = alpha.point(
            lambda value: 0 if value < threshold else min(255, int((value - threshold) * 255 / max(1, 255 - threshold)))
        )

    subject.putalpha(alpha)
    return crop_to_alpha(subject) if crop else subject


def brighten_face_region(
    subject_rgba: Image.Image,
    face_box,
    *,
    brightness: float = 1.16,
    color: float = 1.04,
) -> Image.Image:
    subject = subject_rgba.convert("RGBA")
    resolved_face_box = face_box or estimate_face_box(subject)
    if resolved_face_box is None:
        return subject

    base_rgb = subject.convert("RGB")
    enhanced_rgb = ImageEnhance.Brightness(base_rgb).enhance(brightness)
    enhanced_rgb = ImageEnhance.Color(enhanced_rgb).enhance(color)
    enhanced = enhanced_rgb.convert("RGBA")
    enhanced.putalpha(subject.getchannel("A"))

    x_pos, y_pos, box_w, box_h = [int(value) for value in resolved_face_box]
    mask = Image.new("L", subject.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [
            int(x_pos - box_w * 0.18),
            int(y_pos - box_h * 0.16),
            int(x_pos + box_w * 1.18),
            int(y_pos + box_h * 1.08),
        ],
        fill=192,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(22, int(box_h * 0.22))))
    mask = ImageChops.multiply(mask, subject.getchannel("A"))
    return Image.composite(enhanced, subject, mask)


def _component_metrics_from_alpha(alpha: Image.Image) -> dict:
    alpha_l = alpha.convert("L")
    arr = np.array(alpha_l, dtype=np.uint8)
    binary = (arr > 8).astype(np.uint8) * 255
    height, width = binary.shape
    image_area = float(max(1, width * height))
    edge_touch_pixels = 0
    thin_component_pixels = 0

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for label in range(1, component_count):
        x_pos, y_pos, box_w, box_h, area = [int(value) for value in stats[label]]
        if area <= 0:
            continue
        touches_border = x_pos <= 1 or y_pos <= 1 or (x_pos + box_w) >= width - 1 or (y_pos + box_h) >= height - 1
        aspect = max(box_w / float(max(1, box_h)), box_h / float(max(1, box_w)))
        if touches_border:
            edge_touch_pixels += area
        if aspect > 6.0:
            thin_component_pixels += area

    return {
        "edge_touch_ratio": round(edge_touch_pixels / image_area, 4),
        "thin_component_ratio": round(thin_component_pixels / image_area, 4),
    }


def _build_removed_components_preview(
    raw_alpha: Image.Image,
    kept_alpha: Image.Image,
) -> Image.Image:
    raw = raw_alpha.convert("L")
    kept = kept_alpha.convert("L")
    removed = ImageChops.subtract(raw, kept)
    preview = Image.new("RGBA", raw.size, (0, 0, 0, 0))
    preview.putalpha(removed.point(lambda value: _clamp_byte(value * 0.82)))
    tint = Image.new("RGBA", raw.size, (230, 54, 54, 0))
    tint.putalpha(preview.getchannel("A"))
    return tint


def remove_non_person_components(
    alpha: Image.Image,
    *,
    face_box: tuple[int, int, int, int] | None = None,
    min_area_ratio: float = 0.008,
    max_thinness_ratio: float = 9.0,
    keep_largest_if_no_face: bool = True,
) -> Image.Image:
    """
    Removes background artifacts from an alpha mask.

    Keeps components that plausibly belong to the person and removes stage strips,
    border artifacts and other thin non-person elements before outline generation.
    """
    alpha_l = alpha.convert("L")
    arr = np.array(alpha_l, dtype=np.uint8)
    binary = (arr > 8).astype(np.uint8) * 255
    height, width = binary.shape
    image_area = float(max(1, width * height))

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    face_center = None
    if face_box is not None:
        fx, fy, fw, fh = [int(value) for value in face_box]
        face_center = (fx + fw / 2.0, fy + fh / 2.0)
    else:
        fx = fy = fw = fh = 0

    candidates: list[dict] = []
    for label in range(1, component_count):
        x_pos, y_pos, box_w, box_h, area = [int(value) for value in stats[label]]
        if area <= 0:
            continue

        area_ratio = area / image_area
        cx, cy = centroids[label]
        touches_border = x_pos <= 1 or y_pos <= 1 or (x_pos + box_w) >= width - 1 or (y_pos + box_h) >= height - 1
        vertical_ratio = box_h / float(max(1, box_w))
        horizontal_ratio = box_w / float(max(1, box_h))
        aspect = max(vertical_ratio, horizontal_ratio)
        overlaps_face = False
        contains_face_center = False
        normalized_dist = 0.0

        if face_center is not None:
            face_x, face_y = face_center
            overlaps_face = not (
                x_pos > fx + fw
                or (x_pos + box_w) < fx
                or y_pos > fy + fh
                or (y_pos + box_h) < fy
            )
            contains_face_center = x_pos <= face_x <= (x_pos + box_w) and y_pos <= face_y <= (y_pos + box_h)
            normalized_dist = (((cx - face_x) ** 2 + (cy - face_y) ** 2) ** 0.5) / float(max(width, height))

        too_small = area_ratio < min_area_ratio
        vertical_strip = vertical_ratio > max_thinness_ratio
        horizontal_strip = horizontal_ratio > max_thinness_ratio
        too_thin = vertical_strip or horizontal_strip
        suspicious_border_strip = touches_border and too_thin
        lower_caption_like = False
        if face_center is not None and fh > 0:
            lower_caption_like = (
                y_pos > (face_center[1] + fh * 2.05)
                and horizontal_ratio > 1.75
                and box_h < max(28, int(fh * 0.85))
                and box_w > max(80, int(fw * 0.95))
            )

        if too_small:
            continue
        if lower_caption_like:
            continue
        if suspicious_border_strip and not overlaps_face:
            continue
        if too_thin and not overlaps_face:
            continue

        score = area_ratio
        if face_center is not None:
            score += max(0.0, 0.28 - normalized_dist)
            if contains_face_center:
                score += 0.38
            if x_pos <= face_center[0] <= x_pos + box_w:
                score += 0.18
            if y_pos <= face_center[1] <= y_pos + box_h:
                score += 0.15
            if y_pos <= face_center[1] <= y_pos + box_h or (y_pos + box_h) >= face_center[1]:
                score += 0.10
            if touches_border:
                score -= 0.08
            if vertical_strip or horizontal_strip:
                score -= 0.10
            if fw > 0 and box_w < max(18, int(fw * 0.18)) and box_h > int(fh * 1.7):
                score -= 0.16
        elif touches_border:
            score -= 0.04

        candidates.append(
            {
                "label": label,
                "score": score,
                "area_ratio": area_ratio,
                "bbox": (x_pos, y_pos, box_w, box_h),
                "touches_border": touches_border,
                "too_thin": too_thin,
            }
        )

    if not candidates:
        if keep_largest_if_no_face and component_count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep[labels == largest] = 255
        return Image.fromarray(keep, mode="L")

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best = candidates[0]
    keep_labels = {int(best["label"])}

    best_x, best_y, best_w, best_h = [int(value) for value in best["bbox"]]
    margin_x = max(18, int(best_w * 0.20))
    margin_y = max(18, int(best_h * 0.15))
    expanded_left = max(0, best_x - margin_x)
    expanded_top = max(0, best_y - margin_y)
    expanded_right = min(width, best_x + best_w + margin_x)
    expanded_bottom = min(height, best_y + best_h + margin_y)

    for item in candidates[1:]:
        if item["too_thin"]:
            continue
        if item["area_ratio"] < min_area_ratio * 1.35:
            continue
        x_pos, y_pos, box_w, box_h = [int(value) for value in item["bbox"]]
        overlaps_best_zone = not (
            (x_pos + box_w) < expanded_left
            or x_pos > expanded_right
            or (y_pos + box_h) < expanded_top
            or y_pos > expanded_bottom
        )
        if face_center is not None:
            below_face = (y_pos + box_h) >= face_center[1]
            aligned_with_face = abs((x_pos + box_w / 2.0) - face_center[0]) <= max(best_w * 0.55, fw * 0.9)
            if overlaps_best_zone or (below_face and aligned_with_face and not item["touches_border"]):
                keep_labels.add(int(item["label"]))
        elif overlaps_best_zone and not item["touches_border"]:
            keep_labels.add(int(item["label"]))

    for label in keep_labels:
        keep[labels == label] = 255
    return Image.fromarray(keep, mode="L")


class BackgroundRemovalProvider:
    name = "base"

    def extract_subject(self, frame_bgr):
        raise NotImplementedError


def _coverage_from_alpha(alpha: Image.Image) -> float:
    arr = np.array(alpha.convert("L"), dtype=np.uint8)
    return float(np.count_nonzero(arr > 0)) / float(max(1, arr.size))


def _remove_caption_bars_from_alpha(
    alpha: Image.Image,
    frame_bgr: np.ndarray | None,
    face_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    if frame_bgr is None or face_box is None:
        return alpha
    arr = np.array(alpha.convert("L"), dtype=np.uint8)
    height, width = arr.shape
    fx, fy, fw, fh = [int(value) for value in face_box]
    lower_start = min(height - 1, max(0, int(fy + fh * 1.05)))
    if lower_start >= height - 2:
        return alpha

    lower = frame_bgr[lower_start:, :, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    dark = (gray < 46).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    clear_mask = np.zeros_like(arr, dtype=np.uint8)
    for label in range(1, component_count):
        x_pos, y_pos, box_w, box_h, area = [int(value) for value in stats[label]]
        if area <= 0:
            continue
        horizontal = box_w > max(120, int(fw * 1.05)) and box_w / float(max(1, box_h)) > 1.8
        plausible_height = 18 <= box_h <= max(180, int(fh * 0.95))
        if not (horizontal and plausible_height):
            continue
        y_abs = lower_start + y_pos
        pad_x = max(16, int(box_w * 0.04))
        pad_y = max(12, int(box_h * 0.35))
        x1 = max(0, x_pos - pad_x)
        y1 = max(0, y_abs - pad_y)
        x2 = min(width, x_pos + box_w + pad_x)
        y2 = min(height, y_abs + box_h + pad_y)
        clear_mask[y1:y2, x1:x2] = 255

    if not np.any(clear_mask):
        return alpha
    arr[clear_mask > 0] = 0
    return Image.fromarray(arr, mode="L")


def _postprocess_subject_alpha(
    subject_rgba: Image.Image,
    face_box: tuple[int, int, int, int] | None,
    *,
    raw_coverage: float | None = None,
    frame_bgr: np.ndarray | None = None,
) -> tuple[Image.Image, tuple[int, int, int, int] | None, float, dict[str, Any]]:
    raw_alpha_before = subject_rgba.getchannel("A")
    raw_alpha_before = _remove_caption_bars_from_alpha(raw_alpha_before, frame_bgr, face_box)
    person_alpha = remove_non_person_components(raw_alpha_before, face_box=face_box)
    subject_rgba.putalpha(person_alpha)
    filtered_coverage = _coverage_from_alpha(person_alpha)
    debug = {
        "raw_alpha_before_component_filter": raw_alpha_before.copy(),
        "person_component_alpha": person_alpha.copy(),
        "removed_components_preview": _build_removed_components_preview(raw_alpha_before, person_alpha),
        "coverage_after": round(filtered_coverage, 4),
        **_component_metrics_from_alpha(person_alpha),
    }
    if raw_coverage is not None:
        debug["coverage_before"] = round(float(raw_coverage), 4)

    subject_rgba = refine_cutout_edges(subject_rgba, crop=False, soft=True)
    subject_rgba = brighten_face_region(subject_rgba, face_box)
    subject_rgba, crop_bbox = crop_to_alpha_with_bbox(subject_rgba)
    face_box = translate_face_box(face_box, crop_bbox)
    face_box = face_box or estimate_face_box(subject_rgba)
    return subject_rgba, face_box, filtered_coverage, debug


class GrabCutBackgroundRemovalProvider(BackgroundRemovalProvider):
    name = "grabcut_local"

    def extract_subject(self, frame_bgr):
        self.last_debug = {}
        face_box = _detect_face_box(frame_bgr)
        speaker_rgba, coverage = _extract_subject_rgba(frame_bgr, face_box)
        speaker_rgba, face_box, filtered_coverage, self.last_debug = _postprocess_subject_alpha(
            speaker_rgba,
            face_box,
            raw_coverage=coverage,
            frame_bgr=frame_bgr,
        )
        return speaker_rgba, face_box, filtered_coverage


class RemBgBackgroundRemovalProvider(BackgroundRemovalProvider):
    name = "rembg"

    def extract_subject(self, frame_bgr):
        self.last_debug = {}
        try:
            from rembg import remove
        except Exception as exc:
            raise RuntimeError("rembg provider requested, but rembg is not installed yet.") from exc

        frame_rgb = Image.fromarray(frame_bgr[:, :, ::-1], mode="RGB")
        face_box = _detect_face_box(frame_bgr)
        subject_rgba = remove(frame_rgb).convert("RGBA")
        subject_rgba, face_box, filtered_coverage, self.last_debug = _postprocess_subject_alpha(
            subject_rgba,
            face_box,
            frame_bgr=frame_bgr,
        )
        return subject_rgba, face_box, filtered_coverage


class BiRefNetBackgroundRemovalProvider(BackgroundRemovalProvider):
    name = "birefnet_rmbg2"
    model_id = "briaai/RMBG-2.0"
    _model = None
    _device = None
    _dtype = None

    def extract_subject(self, frame_bgr):
        self.last_debug = {}
        try:
            import torch
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation
        except Exception as exc:
            raise RuntimeError(
                "birefnet provider requested, but torch/torchvision/transformers are not installed."
            ) from exc

        if BiRefNetBackgroundRemovalProvider._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            model = model.to(device=device, dtype=dtype).eval()
            BiRefNetBackgroundRemovalProvider._model = model
            BiRefNetBackgroundRemovalProvider._device = device
            BiRefNetBackgroundRemovalProvider._dtype = dtype

        frame_rgb = Image.fromarray(frame_bgr[:, :, ::-1], mode="RGB")
        face_box = _detect_face_box(frame_bgr)
        input_size = (1024, 1024)
        preprocess = transforms.Compose(
            [
                transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = preprocess(frame_rgb).unsqueeze(0).to(
            device=BiRefNetBackgroundRemovalProvider._device,
            dtype=BiRefNetBackgroundRemovalProvider._dtype,
        )

        model = BiRefNetBackgroundRemovalProvider._model
        with torch.inference_mode():
            output = model(tensor)
            if isinstance(output, dict):
                pred = output.get("logits")
                if pred is None:
                    pred = output.get("out")
                if pred is None:
                    pred = next(iter(output.values()))
            elif isinstance(output, (list, tuple)):
                pred = output[-1]
            else:
                pred = output
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            mask_tensor = pred.sigmoid().detach().float().cpu()[0]
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.squeeze(0)

        mask_arr = np.clip(mask_tensor.numpy() * 255.0, 0, 255).astype(np.uint8)
        alpha = Image.fromarray(mask_arr, mode="L").resize(frame_rgb.size, Image.Resampling.LANCZOS)
        subject_rgba = frame_rgb.convert("RGBA")
        subject_rgba.putalpha(alpha)
        subject_rgba, face_box, coverage, self.last_debug = _postprocess_subject_alpha(
            subject_rgba,
            face_box,
            raw_coverage=_coverage_from_alpha(alpha),
            frame_bgr=frame_bgr,
        )
        self.last_debug["model_id"] = self.model_id
        self.last_debug["device"] = str(BiRefNetBackgroundRemovalProvider._device)
        self.last_debug["dtype"] = str(BiRefNetBackgroundRemovalProvider._dtype)
        return subject_rgba, face_box, coverage


def get_background_removal_provider(name: str | None = None) -> BackgroundRemovalProvider:
    provider_name = str(name or get_configured_background_removal_provider_name()).strip().lower()
    if provider_name in {"grabcut", "grabcut_local"}:
        return GrabCutBackgroundRemovalProvider()
    if provider_name == "rembg":
        return RemBgBackgroundRemovalProvider()
    if provider_name in {"birefnet", "rmbg", "rmbg2", "rmbg-2.0", "birefnet_rmbg2"}:
        return BiRefNetBackgroundRemovalProvider()
    raise ValueError(f"Unsupported background removal provider: {provider_name}")


def crop_to_alpha_with_bbox(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return (image.crop(bbox), bbox) if bbox else (image, None)


def crop_to_alpha(image: Image.Image) -> Image.Image:
    cropped, _bbox = crop_to_alpha_with_bbox(image)
    return cropped


def translate_face_box(face_box, crop_bbox):
    if face_box is None or crop_bbox is None:
        return face_box
    x_pos, y_pos, box_w, box_h = [int(value) for value in face_box]
    left, top, _right, _bottom = [int(value) for value in crop_bbox]
    return (x_pos - left, y_pos - top, box_w, box_h)


def _build_outline_layers(subject_rgba: Image.Image, preset_name: str):
    preset = OUTLINE_PRESETS[preset_name]

    subject = refine_cutout_edges(subject_rgba, soft=True)
    raw_alpha = subject.getchannel("A")
    subject_alpha = feather_alpha(
        raw_alpha,
        blur_radius=preset.edge_feather,
        low=8,
        high=235,
    )
    subject = subject.copy()
    subject.putalpha(subject_alpha)

    expanded_alpha = expand_alpha(
        subject_alpha,
        preset.stroke_width,
        feather=preset.stroke_feather,
    )

    inner_alpha = contract_alpha(
        subject_alpha,
        max(1, preset.stroke_width // 5),
        feather=1.0,
    )

    outline_mask = ImageChops.subtract(expanded_alpha, inner_alpha)
    outline_mask = outline_mask.filter(ImageFilter.GaussianBlur(radius=preset.stroke_feather))
    outline_mask = _soft_threshold_alpha(outline_mask, low=4, high=190)

    glow_base = expand_alpha(
        subject_alpha,
        preset.stroke_width + max(4, preset.glow_radius // 3),
        feather=preset.glow_feather,
    )
    glow_mask = glow_base.filter(ImageFilter.GaussianBlur(radius=preset.glow_radius))

    shadow_base = expand_alpha(
        subject_alpha,
        preset.stroke_width + 4,
        feather=3.0,
    )
    shadow_mask = shadow_base.filter(ImageFilter.GaussianBlur(radius=preset.shadow_blur))

    return {
        "preset": preset,
        "subject": subject,
        "raw_alpha": raw_alpha,
        "subject_alpha": subject_alpha,
        "expanded_alpha": expanded_alpha,
        "outline_mask": outline_mask,
        "glow_mask": glow_mask,
        "shadow_mask": shadow_mask,
    }


def _compose_outline_canvas(subject: Image.Image, preset: OutlinePreset, outline_mask: Image.Image, glow_mask: Image.Image, shadow_mask: Image.Image) -> Image.Image:
    pad = max(
        preset.stroke_width * 4,
        preset.glow_radius * 4,
        preset.shadow_blur * 3,
        abs(preset.shadow_offset[0]) + abs(preset.shadow_offset[1]) + 24,
    )

    canvas = Image.new("RGBA", (subject.width + pad * 2, subject.height + pad * 2), (0, 0, 0, 0))
    subject_pos = (pad, pad)

    shadow_layer = render_mask_layer(
        subject.size,
        preset.shadow_color,
        shadow_mask,
    )
    canvas = composite_layer(
        canvas,
        shadow_layer,
        offset=(subject_pos[0] + preset.shadow_offset[0], subject_pos[1] + preset.shadow_offset[1]),
    )

    glow_layer = render_mask_layer(
        subject.size,
        preset.glow_color,
        glow_mask,
    )
    canvas = composite_layer(canvas, glow_layer, offset=subject_pos)

    outline_layer = render_mask_layer(
        subject.size,
        preset.stroke_color,
        outline_mask,
    )
    canvas = composite_layer(canvas, outline_layer, offset=subject_pos)
    canvas = composite_layer(canvas, subject, offset=subject_pos)
    return crop_to_alpha(canvas)


def save_outline_debug_layers(
    subject_rgba: Image.Image,
    output_dir: Path,
    preset_name: str = "creator_white",
    component_debug: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = _build_outline_layers(subject_rgba, preset_name)
    final_outline = _compose_outline_canvas(
        layers["subject"],
        layers["preset"],
        layers["outline_mask"],
        layers["glow_mask"],
        layers["shadow_mask"],
    )

    layers["raw_alpha"].save(output_dir / "debug_01_subject_alpha.png")
    layers["subject_alpha"].save(output_dir / "debug_02_subject_alpha_feathered.png")
    layers["expanded_alpha"].save(output_dir / "debug_03_expanded_alpha.png")
    layers["outline_mask"].save(output_dir / "debug_04_outline_mask.png")
    layers["glow_mask"].save(output_dir / "debug_05_glow_mask.png")
    final_outline.save(output_dir / "debug_06_final_outline.png")
    if component_debug:
        raw_alpha = component_debug.get("raw_alpha_before_component_filter")
        person_alpha = component_debug.get("person_component_alpha")
        removed_preview = component_debug.get("removed_components_preview")
        if isinstance(raw_alpha, Image.Image):
            raw_alpha.save(output_dir / "debug_12_raw_alpha_before_component_filter.png")
        if isinstance(person_alpha, Image.Image):
            person_alpha.save(output_dir / "debug_13_person_component_alpha.png")
        if isinstance(removed_preview, Image.Image):
            removed_preview.save(output_dir / "debug_14_removed_components_preview.png")


def add_speaker_outline(subject_rgba: Image.Image, preset_name: str = "creator_white") -> Image.Image:
    layers = _build_outline_layers(subject_rgba, preset_name)
    return _compose_outline_canvas(
        layers["subject"],
        layers["preset"],
        layers["outline_mask"],
        layers["glow_mask"],
        layers["shadow_mask"],
    )


def defringe_subject(subject_rgba: Image.Image, *, band_width: int = 4) -> Image.Image:
    """Remove the bright halo a backlit stage bakes into the cutout edge.

    Replaces edge-band colors with colors bled outward from the subject core,
    and trims one pixel of alpha so the blown-out fringe doesn't survive into
    the rim/outline pass.
    """
    subject = subject_rgba.convert("RGBA")
    rgb = np.asarray(subject, dtype=np.float32)[:, :, :3]
    alpha = np.asarray(subject.getchannel("A"), dtype=np.float32) / 255.0

    core_img = contract_alpha(subject.getchannel("A"), band_width, feather=1.0)
    core = np.asarray(core_img, dtype=np.float32) / 255.0

    sigma = max(3.0, band_width * 2.0)
    weight = cv2.GaussianBlur(core, (0, 0), sigma)
    color_acc = cv2.GaussianBlur(rgb * core[:, :, None], (0, 0), sigma)
    inward = color_acc / np.maximum(weight, 1e-4)[:, :, None]

    band = np.clip(alpha - core, 0.0, 1.0)[:, :, None]
    blended = rgb * (1.0 - band * 0.85) + inward * (band * 0.85)

    out = np.dstack([np.clip(blended, 0, 255).astype(np.uint8),
                     (alpha * 255).astype(np.uint8)])
    result = Image.fromarray(out, mode="RGBA")
    trimmed = contract_alpha(result.getchannel("A"), 1, feather=0.8)
    result.putalpha(trimmed)
    return result


def grade_subject(subject_rgba: Image.Image) -> Image.Image:
    """Lift a flat, backlit cutout: local contrast, highlight rolloff,
    shadow lift, vibrance, gentle sharpening. Operates on the subject only."""
    subject = subject_rgba.convert("RGBA")
    alpha = subject.getchannel("A")
    rgb = cv2.cvtColor(np.asarray(subject.convert("RGB")), cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(rgb)

    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)

    lut = np.arange(256, dtype=np.float32)
    lut = np.where(lut > 225, 225 + (lut - 225) * 0.55, lut)          # highlight rolloff
    lut = np.where(lut < 60, lut + (60 - lut) * 0.18, lut)            # shadow lift
    l_chan = cv2.LUT(l_chan, np.clip(lut, 0, 255).astype(np.uint8))

    graded = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    blurred = cv2.GaussianBlur(graded, (0, 0), 2.2)
    sharpened = cv2.addWeighted(graded, 1.45, blurred, -0.45, 0)

    out = Image.fromarray(sharpened, mode="RGB")
    out = ImageEnhance.Color(out).enhance(1.08)
    out = out.convert("RGBA")
    out.putalpha(alpha)
    return out


def add_speaker_rim_light(
    subject_rgba: Image.Image,
    *,
    rim_color: tuple[int, int, int],
    glow_color: tuple[int, int, int] | None = None,
    rim_width: int = 11,
    glow_radius: int = 44,
    shadow_color: tuple[int, int, int, int] = (0, 0, 0, 125),
    shadow_blur: int = 30,
    shadow_offset: tuple[int, int] = (8, 14),
) -> Image.Image:
    """Palette-colored rim light instead of the white sticker stroke.

    Builds an inner edge band (inside the silhouette), weighted toward the
    top so it reads as stage light from above/behind, plus a soft colored
    outer glow and a drop shadow for separation.
    """
    subject = refine_cutout_edges(subject_rgba, soft=True)
    alpha = feather_alpha(subject.getchannel("A"), blur_radius=1.1, low=8, high=235)
    subject = subject.copy()
    subject.putalpha(alpha)

    alpha_np = np.asarray(alpha, dtype=np.float32) / 255.0
    inner = np.asarray(contract_alpha(alpha, rim_width, feather=2.2), dtype=np.float32) / 255.0
    rim_band = np.clip(alpha_np - inner, 0.0, 1.0)

    height = rim_band.shape[0]
    vertical = np.linspace(1.0, 0.30, height, dtype=np.float32)[:, None]
    rim_band *= vertical

    rim_mask = Image.fromarray((rim_band * 255).astype(np.uint8), mode="L")
    rim_mask = rim_mask.filter(ImageFilter.GaussianBlur(radius=1.4))

    bright_rim = tuple(min(255, int(c + (255 - c) * 0.72)) for c in rim_color)
    rim_layer = render_mask_layer(subject.size, (*bright_rim, 240), rim_mask)
    lit_subject = Image.alpha_composite(subject, rim_layer)

    pad = max(glow_radius * 4, shadow_blur * 3, abs(shadow_offset[0]) + abs(shadow_offset[1]) + 24)
    canvas = Image.new("RGBA", (subject.width + pad * 2, subject.height + pad * 2), (0, 0, 0, 0))
    pos = (pad, pad)

    shadow_mask = expand_alpha(alpha, 4, feather=3.0).filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    shadow_layer = render_mask_layer(subject.size, shadow_color, shadow_mask)
    canvas = composite_layer(canvas, shadow_layer, offset=(pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))

    glow = glow_color or rim_color
    glow_mask = expand_alpha(alpha, max(8, rim_width * 2), feather=6.0).filter(
        ImageFilter.GaussianBlur(radius=glow_radius)
    )
    glow_layer = render_mask_layer(subject.size, (*glow, 130), glow_mask)
    canvas = composite_layer(canvas, glow_layer, offset=pos)

    # Hot near-white core pass hugging the silhouette — keylight edge burn
    hot = tuple(min(255, int(c + (255 - c) * 0.65)) for c in glow)
    hot_mask = expand_alpha(alpha, rim_width, feather=4.0).filter(
        ImageFilter.GaussianBlur(radius=max(8, glow_radius // 3))
    )
    hot_layer = render_mask_layer(subject.size, (*hot, 110), hot_mask)
    canvas = composite_layer(canvas, hot_layer, offset=pos)

    canvas = composite_layer(canvas, lit_subject, offset=pos)
    return crop_to_alpha(canvas)
