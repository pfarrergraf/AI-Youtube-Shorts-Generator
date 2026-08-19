"""Resolve and compose approved, reusable narrative thumbnail layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_PATH = _PROJECT_ROOT / "assets" / "thumbnail_strategy" / "story_assets.json"


def load_catalog() -> dict[str, dict]:
    payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return {
        str(item["id"]): item
        for item in payload.get("assets", [])
        if isinstance(item, dict) and item.get("id")
    }


def _asset_path(item: dict) -> Path | None:
    value = str(item.get("candidate_path") or "").strip()
    return (_CATALOG_PATH.parent / value).resolve() if value else None


def resolve_story_assets(asset_ids: Iterable[str], size: tuple[int, int]) -> dict:
    """Load approved story assets and normalize them to renderer layers."""
    catalog = load_catalog()
    background = None
    background_id = None
    speaker_anchor_x = None
    story_focus_box = None
    foreground: list[tuple[str, Image.Image]] = []
    rejected: list[str] = []
    for raw_id in asset_ids or []:
        asset_id = str(raw_id or "").strip().lower()
        item = catalog.get(asset_id)
        if not item or item.get("review_status") != "approved":
            rejected.append(asset_id)
            continue
        path = _asset_path(item)
        if path is None or not path.is_file():
            rejected.append(asset_id)
            continue
        image = Image.open(path).convert("RGBA")
        if item.get("type") == "background_plate" and background is None:
            background = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
            background_id = asset_id
            try:
                speaker_anchor_x = float(item.get("speaker_anchor_x"))
            except (TypeError, ValueError):
                speaker_anchor_x = None
            raw_focus = item.get("story_focus_box")
            if isinstance(raw_focus, list) and len(raw_focus) == 4:
                story_focus_box = [
                    int(float(raw_focus[0]) * size[0]),
                    int(float(raw_focus[1]) * size[1]),
                    int(float(raw_focus[2]) * size[0]),
                    int(float(raw_focus[3]) * size[1]),
                ]
        elif item.get("type") in {"foreground_overlay", "depth_overlay"}:
            foreground.append((asset_id, image))
    return {
        "background": background,
        "background_id": background_id,
        "speaker_anchor_x": speaker_anchor_x,
        "story_focus_box": story_focus_box,
        "foreground": foreground,
        "rejected": rejected,
    }


def _foreground_position(
    canvas_size: tuple[int, int], overlay_size: tuple[int, int], *, index: int
) -> tuple[int, int]:
    width, height = canvas_size
    overlay_w, overlay_h = overlay_size
    margin_x = int(width * 0.035)
    bottom = int(height * 0.97) - overlay_h
    return (width - overlay_w - margin_x, bottom) if index % 2 else (margin_x, bottom)


def add_foreground_assets(
    canvas: Image.Image,
    assets: list[tuple[str, Image.Image]],
    *,
    light_direction: str = "upper_left",
) -> tuple[Image.Image, list[dict]]:
    """Add physically readable shadow/light treatment to foreground overlays."""
    out = canvas.convert("RGBA")
    width, height = out.size
    reports: list[dict] = []
    for index, (asset_id, source) in enumerate(assets):
        alpha_bbox = source.getchannel("A").getbbox()
        if not alpha_bbox:
            continue
        overlay = source.crop(alpha_bbox)
        target_w = int(width * (0.34 if index == 0 else 0.27))
        target_h = max(1, int(overlay.height * (target_w / float(max(1, overlay.width)))))
        if target_h > int(height * 0.34):
            target_h = int(height * 0.34)
            target_w = max(1, int(overlay.width * (target_h / float(max(1, overlay.height)))))
        overlay = overlay.resize((target_w, target_h), Image.Resampling.LANCZOS)
        overlay = ImageEnhance.Contrast(overlay).enhance(1.06)
        overlay = ImageEnhance.Color(overlay).enhance(1.05)
        x, y = _foreground_position(out.size, overlay.size, index=index)

        shadow_alpha = overlay.getchannel("A").filter(ImageFilter.GaussianBlur(max(8, width // 90)))
        shadow = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.58)))
        shadow_offset = (18, 20) if "left" in light_direction else (-18, 20)
        out.alpha_composite(shadow, (x + shadow_offset[0], y + shadow_offset[1]))

        rim_colour = (255, 190, 92, 72) if any(
            token in light_direction for token in ("left", "gold", "sunset", "lamp")
        ) else (104, 202, 255, 64)
        glow_alpha = overlay.getchannel("A").filter(ImageFilter.GaussianBlur(max(5, width // 150)))
        glow = Image.new("RGBA", overlay.size, rim_colour)
        glow.putalpha(glow_alpha.point(lambda value: int(value * (rim_colour[3] / 255.0))))
        glow_x = x - 5 if "left" in light_direction else x + 5
        out.alpha_composite(glow, (glow_x, y - 3))
        out.alpha_composite(overlay, (x, y))
        reports.append({"asset_id": asset_id, "box": [x, y, x + target_w, y + target_h]})
    return out.convert("RGB"), reports
