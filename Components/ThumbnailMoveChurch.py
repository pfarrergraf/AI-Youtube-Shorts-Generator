"""
Move Church Thumbnail Generator
================================
Builds on ThumbnailEffects.py and ThumbnailV2.py.

Key features:
- 9:16 (1080×1920) and 16:9 (1280×720) formats
- Depth layering: text BEHIND speaker + text IN FRONT of speaker
- 3 brand templates: navy_dark, energy_orange, warm_gold
- Glow, outline, shadow effects via ThumbnailEffects
- Logo optional, symbols, decorative frames/arrows
- Font downloader (Barlow Condensed Black – closest to Move Church typography)

Usage (API):
    from Components.ThumbnailMoveChurch import generate_move_church_thumbnail
    img = generate_move_church_thumbnail(
        "path/to/video.mp4",
        title_back="EINE WIE",
        title_front="KEINE",
        template="navy_dark",
        fmt="9x16",
        show_logo=False,
        output_path="thumbnail.png",
    )

Usage (CLI):
    python -m Components.ThumbnailMoveChurch \\
        --source video.mp4 \\
        --title "EINE WIE KEINE" \\
        --template navy_dark \\
        --fmt 9x16 \\
        --output thumb.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ── re-use existing infrastructure ──────────────────────────────────────────
from Components.ThumbnailEffects import (
    OUTLINE_PRESETS,
    build_linear_gradient_mask,
    build_radial_gradient_mask,
    build_shape_mask,
    combine_masks,
    add_speaker_outline,
    add_speaker_rim_light,
    brighten_face_region,
    composite_layer,
    compose_layers,
    crop_to_alpha,
    defringe_subject,
    estimate_face_box,
    get_background_removal_provider,
    grade_subject,
    render_mask_layer,
)
from Components.TextOnPath import arc_text_layer, circle_text_layer, text_on_svg_path_layer
from Components.ComfyUIBackground import generate_background_image
from Components.ThumbnailV2 import _detect_face_box, _score_variant

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

FORMATS: dict[str, tuple[int, int]] = {
    "9x16": (1080, 1920),
    "16x9": (1280, 720),
}

TEMPLATES = (
    "navy_dark",
    "energy_orange",
    "warm_gold",
    "cinematic_dark",
    "fire_red",
    "heaven_blue",
    "bold_minimal",
    "sunset_warm",
)
EFFECT_PROFILES = (
    "classic",
    "editorial",
    "premium",
    "halo",
    "poster",
)
_DEFAULT_EFFECT_PROFILE = "classic"
_LAST_FRAME_SELECTION: dict = {}
_LAST_TOP_FRAME_CANDIDATES: list[dict] = []
_LAST_LAYOUT_METADATA: dict = {}
_LAST_FACE_SCORER_INFO: dict = {}
_MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Move Church brand colour
_MC_ORANGE = (224, 92, 32, 255)
_MC_GOLD   = (212, 165, 32, 255)
_MC_WHITE  = (255, 255, 255, 255)
_MC_NAVY   = (6, 11, 24, 255)

PALETTES: dict[str, dict] = {
    "navy_dark": {
        "bg_top":    (6, 11, 24),
        "bg_bottom": (12, 22, 48),
        "glow_a":    (20, 50, 120),
        "glow_b":    (10, 30, 80),
        "accent":    _MC_ORANGE,
        "text":      _MC_WHITE,
        "ray_color": (100, 160, 255, 100),
        "rings":     True,
        "italic":    False,
        "logo_color": _MC_ORANGE,
    },
    "energy_orange": {
        "bg_top":    (13, 4, 8),
        "bg_bottom": (8, 4, 14),
        "glow_a":    (200, 60, 10),
        "glow_b":    (60, 20, 130),
        "accent":    _MC_ORANGE,
        "text":      _MC_WHITE,
        "ray_color": (255, 80, 0, 60),
        "rings":     False,
        "italic":    False,  # real italic font carries the slant now
        "logo_color": _MC_WHITE,
    },
    "warm_gold": {
        "bg_top":    (8, 6, 16),
        "bg_bottom": (12, 8, 20),
        "glow_a":    (230, 170, 40),
        "glow_b":    (100, 20, 150),
        "accent":    _MC_GOLD,
        "text":      _MC_WHITE,
        "ray_color": (230, 180, 50, 140),
        "rings":     True,
        "italic":    False,
        "logo_color": _MC_GOLD,
    },
    "cinematic_dark": {
        "bg_top":    (5, 5, 8),
        "bg_bottom": (5, 5, 8),
        "glow_a":    (190, 190, 205),
        "glow_b":    (90, 90, 110),
        "accent":    _MC_GOLD,
        "text":      _MC_WHITE,
        "ray_color": (255, 255, 255, 70),
        "rings":     False,
        "italic":    False,
        "logo_color": _MC_GOLD,
        "separator_color": (212, 165, 32, 210),
        "grain":     True,
    },
    "fire_red": {
        "bg_top":    (26, 0, 0),      # radial centre
        "bg_bottom": (0, 0, 0),       # radial edges
        "bg_radial": True,
        "glow_a":    (255, 80, 0),
        "glow_b":    (200, 30, 10),
        "accent":    (255, 34, 0, 255),
        "text":      _MC_WHITE,
        "ray_color": (255, 60, 0, 70),
        "rings":     False,
        "italic":    True,
        "logo_color": _MC_WHITE,
    },
    "heaven_blue": {
        "bg_top":    (3, 6, 18),
        "bg_bottom": (3, 6, 18),
        "glow_a":    (110, 170, 255),
        "glow_b":    (50, 90, 180),
        "accent":    (150, 200, 255, 255),
        "text":      _MC_WHITE,
        "ray_color": (140, 190, 255, 80),
        "rings":     True,
        "rings_center": (0.5, -0.15),
        "italic":    False,
        "logo_color": _MC_WHITE,
        "bar_color": (255, 255, 255, 230),
        "bar_height": 3,
    },
    "bold_minimal": {
        "bg_top":    (10, 10, 10),
        "bg_bottom": (10, 10, 10),
        "glow_a":    (40, 40, 40),
        "glow_b":    (30, 30, 30),
        "accent":    _MC_ORANGE,
        "text":      _MC_WHITE,
        "ray_color": (0, 0, 0, 0),
        "rings":     False,
        "italic":    False,
        "logo_color": _MC_WHITE,
        "minimal":   True,
        "text_glow": False,
        "back_scale": 1.15,
        "front_accent": True,
        "ai_background": True,
    },
    "sunset_warm": {
        "bg_top":    (26, 10, 46),
        "bg_bottom": (45, 16, 0),
        "glow_a":    (255, 170, 60),
        "glow_b":    (120, 50, 140),
        "accent":    _MC_GOLD,
        "text":      (255, 245, 230, 255),
        "ray_color": (255, 180, 80, 110),
        "rings":     True,
        "rings_center": (1.15, -0.13),
        "italic":    False,
        "logo_color": _MC_GOLD,
    },
}

SYMBOLS: dict[str, str] = {
    "cross":  "✝",
    "bible":  "📖",
    "fire":   "🔥",
    "dove":   "🕊",
    "star":   "✦",
    "heart":  "♡",
    "anchor": "⚓",
    "crown":  "♛",
    "arrow":  "→",
}

# ════════════════════════════════════════════════════════════════════════════
# FONT
# ════════════════════════════════════════════════════════════════════════════

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

_MC_FONT_DIR = Path.home() / ".local/share/fonts/mc_thumbnails"

# Per-template font assignment (first existing file wins, falls back to
# BarlowCondensed-Black via _FONT_CANDIDATES_BOLD_CONDENSED).
TEMPLATE_FONTS: dict[str, tuple[str, ...]] = {
    # Unified preview: use a condensed, energetic italicized Barlow
    # variant across all templates so the impact of the chosen
    # energetic/slanted aesthetic is visible consistently.
    "navy_dark":      ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "energy_orange":  ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "warm_gold":      ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "cinematic_dark": ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "fire_red":       ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "heaven_blue":    ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "bold_minimal":   ("BarlowCondensed-ExtraBoldItalic.ttf",),
    "sunset_warm":    ("BarlowCondensed-ExtraBoldItalic.ttf",),
}

_TEMPLATE_FONT_PATH_CACHE: dict[str, str | None] = {}

_EFFECT_PROFILE_CONFIG: dict[str, dict] = {
    "classic": {
        "decorations": True,
        "finish": {
            "contrast": 1.00,
            "color": 1.00,
            "brightness": 1.00,
            "shadow_lift": 0.0,
            "vignette": 0.00,
            "grain": 0.00,
        },
        "text": {
            "stroke_scale": 1.00,
            "shadow_scale": 1.00,
            "glow_scale": 1.00,
            "extrude_scale": 1.00,
            "tracking_scale": 1.00,
            "gradient": None,
        },
        "badge": None,
    },
    "editorial": {
        "decorations": False,
        "finish": {
            "contrast": 1.10,
            "color": 1.06,
            "brightness": 0.98,
            "shadow_lift": 0.04,
            "vignette": 0.10,
            "grain": 0.015,
        },
        "text": {
            "stroke_scale": 1.10,
            "shadow_scale": 0.80,
            "glow_scale": 0.65,
            "extrude_scale": 0.70,
            "tracking_scale": 0.94,
            "gradient": ((255, 255, 255), (236, 239, 245)),
        },
        "badge": {
            "text": "MOVE CHURCH",
            "mode": "arc",
            "position": "top_right",
        },
    },
    "premium": {
        "decorations": True,
        "finish": {
            "contrast": 1.14,
            "color": 1.10,
            "brightness": 0.96,
            "shadow_lift": 0.05,
            "vignette": 0.16,
            "grain": 0.028,
        },
        "text": {
            "stroke_scale": 1.18,
            "shadow_scale": 1.05,
            "glow_scale": 1.18,
            "extrude_scale": 1.08,
            "tracking_scale": 0.90,
            "gradient": ((255, 255, 255), (251, 210, 147)),
        },
        "badge": {
            "text": "MOVE CHURCH",
            "mode": "circle",
            "position": "top_right",
        },
    },
    "halo": {
        "decorations": True,
        "finish": {
            "contrast": 1.08,
            "color": 1.04,
            "brightness": 0.98,
            "shadow_lift": 0.03,
            "vignette": 0.12,
            "grain": 0.020,
        },
        "text": {
            "stroke_scale": 1.08,
            "shadow_scale": 0.96,
            "glow_scale": 1.10,
            "extrude_scale": 1.00,
            "tracking_scale": 0.96,
            "gradient": ((255, 255, 255), (214, 235, 255)),
        },
        "badge": {
            "text": "SERMON HIGHLIGHT",
            "mode": "arc",
            "position": "upper_left",
        },
    },
    "poster": {
        "decorations": True,
        "finish": {
            "contrast": 1.18,
            "color": 1.12,
            "brightness": 0.92,
            "shadow_lift": 0.02,
            "vignette": 0.22,
            "grain": 0.035,
        },
        "text": {
            "stroke_scale": 1.28,
            "shadow_scale": 1.18,
            "glow_scale": 1.05,
            "extrude_scale": 1.30,
            "tracking_scale": 0.92,
            "gradient": ((255, 255, 255), (255, 232, 184)),
        },
        "badge": {
            "text": "MOMENT",
            "mode": "circle",
            "position": "bottom_left",
        },
    },
}


def _template_font_path(template: str | None) -> str | None:
    if not template or template not in TEMPLATE_FONTS:
        return None
    if template in _TEMPLATE_FONT_PATH_CACHE:
        return _TEMPLATE_FONT_PATH_CACHE[template]
    resolved: str | None = None
    for filename in TEMPLATE_FONTS[template]:
        matches = sorted(_MC_FONT_DIR.rglob(filename)) if _MC_FONT_DIR.exists() else []
        if matches:
            resolved = str(matches[0])
            break
    _TEMPLATE_FONT_PATH_CACHE[template] = resolved
    return resolved

_FONT_CANDIDATES_BOLD_CONDENSED = [
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-condensed/BarlowCondensed-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-condensed/BarlowCondensed-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-condensed/BarlowCondensed-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/teko/Teko-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/oswald/Oswald-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-semi-condensed/BarlowSemiCondensed-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-semi-condensed/BarlowSemiCondensed-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-semi-condensed/BarlowSemiCondensed-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/saira-condensed/SairaCondensed-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/saira-condensed/SairaCondensed-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/saira-condensed/SairaCondensed-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/rajdhani/Rajdhani-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/Anton-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/BarlowCondensed-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/BarlowCondensed-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/BarlowSemiCondensed-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/ChakraPetch-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/PaytoneOne-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/Rajdhani-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/anton/Anton-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/bangers/Bangers-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-condensed/BarlowCondensed-BlackItalic.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-condensed/BarlowCondensed-ExtraBoldItalic.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/bebas-neue/BebasNeue-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/chakra-petch/ChakraPetch-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/chakra-petch/ChakraPetch-BoldItalic.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/covered-by-your-grace/CoveredByYourGrace-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/exo-2/Exo2-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/exo-2/Exo2-BlackItalic.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/exo-2/Exo2-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/holtwood-one-sc/HoltwoodOneSc-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/merriweather/Merriweather-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/merriweather/Merriweather-Bold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/montserrat/Montserrat-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/montserrat/Montserrat-BlackItalic.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/montserrat/Montserrat-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/montserrat/Montserrat-ExtraBoldItalic.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/nunito/Nunito-Black.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/nunito/Nunito-ExtraBold.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/oswald/Oswald-ExtraLight.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/oswald/Oswald-Regular.ttf",
    r"/home/benjamin_graf/.local/share/fonts/mc_thumbnails/oswald/Oswald-SemiBold.ttf",
    # System fonts (apt)
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-BoldOblique.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Oblique.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Bold.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-BoldItalic.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    r"/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    r"/usr/share/fonts/truetype/freefont/FreeMonoBoldOblique.ttf",
    r"/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    r"/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf",
    r"/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    r"/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def ensure_font() -> str | None:
    target = Path.home() / ".local/share/fonts/BarlowCondensed-Black.ttf"
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    # 1) Try apt (Ubuntu has fonts-barlow in repos)
    if shutil.which("apt-get"):
        print("[ThumbnailMoveChurch] Installing fonts-barlow via apt...")
        r = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "fonts-barlow"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            # apt installs to /usr/share/fonts/truetype/barlow/
            apt_candidates = list(Path("/usr/share/fonts").rglob("BarlowCondensed-Black.ttf"))
            if apt_candidates:
                import shutil as _shutil
                _shutil.copy2(str(apt_candidates[0]), str(target))
                subprocess.run(["fc-cache", "-f"], capture_output=True, check=False)
                print(f"[ThumbnailMoveChurch] Font installed via apt → {target}")
                return str(target)

    # 2) GitHub google/fonts direct download (reliable, no CDN URL changes)
    urls = [
        "https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Black.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-Black.ttf",
    ]
    for url in urls:
        try:
            print(f"[ThumbnailMoveChurch] Downloading Barlow Condensed Black...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) > 10_000:  # sanity check: real font file
                target.write_bytes(data)
                subprocess.run(["fc-cache", "-f"], capture_output=True, check=False)
                print(f"[ThumbnailMoveChurch] Font saved → {target}")
                return str(target)
        except Exception as exc:
            print(f"[ThumbnailMoveChurch] Download failed ({exc}), trying next source...")

    print("[ThumbnailMoveChurch] Could not install Barlow Condensed. Using system fallback.")
    return None


def _load_mc_font(size: int, template: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (template if template in TEMPLATE_FONTS else "mc", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    template_path = _template_font_path(template)
    if template_path:
        try:
            font = ImageFont.truetype(template_path, size)
            _FONT_CACHE[key] = font
            return font
        except Exception:
            pass
    for path in _FONT_CANDIDATES_BOLD_CONDENSED:
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                continue
    # Last resort: try to download
    downloaded = ensure_font()
    if downloaded and os.path.isfile(downloaded):
        try:
            font = ImageFont.truetype(downloaded, size)
            _FONT_CACHE[key] = font
            return font
        except Exception:
            pass
    return ImageFont.load_default()


def _resolve_mc_font_path(template: str | None = None) -> str | None:
    """Return a usable font path for path-text helpers."""
    template_path = _template_font_path(template)
    if template_path and os.path.isfile(template_path):
        return template_path
    for path in _FONT_CANDIDATES_BOLD_CONDENSED:
        if os.path.isfile(path):
            return path
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ):
        if os.path.isfile(path):
            return path
    return None


def _resolve_effect_profile(effect_profile: str | None) -> tuple[str, dict]:
    resolved = str(effect_profile or _DEFAULT_EFFECT_PROFILE).strip().lower()
    if resolved == "auto":
        resolved = _DEFAULT_EFFECT_PROFILE
    if resolved not in EFFECT_PROFILES:
        raise ValueError(f"Unknown effect profile '{effect_profile}'. Choose: {EFFECT_PROFILES}")
    return resolved, dict(_EFFECT_PROFILE_CONFIG[resolved])


def _build_linear_gradient(
    size: tuple[int, int],
    top_color: tuple[int, int, int] = (255, 255, 255),
    bottom_color: tuple[int, int, int] = (200, 200, 200),
) -> Image.Image:
    width, height = size
    top = np.array(top_color, dtype=np.float32)
    bottom = np.array(bottom_color, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    grad = ((1.0 - ramp) * top + ramp * bottom).astype(np.uint8)
    arr = np.repeat(grad[:, None, :], width, axis=1)
    return Image.fromarray(arr, mode="RGB").convert("RGBA")


def _apply_canvas_finish(
    canvas: Image.Image,
    *,
    effect_profile: str,
    template: str,
) -> Image.Image:
    config = _EFFECT_PROFILE_CONFIG[effect_profile]["finish"]
    result = canvas.convert("RGBA")
    result = ImageEnhance.Contrast(result).enhance(float(config.get("contrast", 1.0)))
    result = ImageEnhance.Color(result).enhance(float(config.get("color", 1.0)))
    result = ImageEnhance.Brightness(result).enhance(float(config.get("brightness", 1.0)))

    shadow_lift = float(config.get("shadow_lift", 0.0))
    if shadow_lift > 0:
        lift = Image.new("RGBA", result.size, (18, 24, 35, int(255 * shadow_lift)))
        result = Image.alpha_composite(result, lift)

    vignette_strength = float(config.get("vignette", 0.0))
    if vignette_strength > 0:
        w, h = result.size
        vignette = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(vignette)
        steps = 28
        cx, cy = w // 2, h // 2
        max_r = math.hypot(cx, cy)
        for i in range(steps, 0, -1):
            frac = i / steps
            alpha = int(255 * vignette_strength * frac * frac)
            r = int(max_r * frac)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)
        vignette_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        vignette_rgba.putalpha(vignette.filter(ImageFilter.GaussianBlur(radius=max(8, w // 70))))
        result = Image.alpha_composite(result, vignette_rgba)

    grain_opacity = float(config.get("grain", 0.0))
    if grain_opacity > 0:
        w, h = result.size
        noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        alpha = np.full((h, w), int(255 * grain_opacity), dtype=np.uint8)
        grain = Image.fromarray(np.dstack([noise, noise, noise, alpha]), mode="RGBA")
        result = Image.alpha_composite(result, grain)

    gradient = config.get("gradient")
    if gradient:
        top_color, bottom_color = gradient
        tint = _build_linear_gradient(result.size, top_color, bottom_color)
        result = Image.blend(result, tint, 0.10)

    if template == "cinematic_dark":
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))
    return result


def _apply_profile_overlay_stack(
    canvas: Image.Image,
    *,
    template: str,
    effect_profile: str,
    face_box: tuple[int, int, int, int] | None,
) -> tuple[Image.Image, list[dict]]:
    """Apply optional high-end overlay layers for the chosen render profile."""
    if effect_profile == "classic":
        return canvas, []

    w, h = canvas.size
    palette = PALETTES[template]
    layers_meta: list[dict] = []
    face_center = (0.50, 0.32)
    if face_box is not None:
        fx, fy, fw, fh = face_box
        face_center = ((fx + fw / 2.0) / float(max(1, w)), (fy + fh / 2.0) / float(max(1, h)))

    # Gentle subject-centric bloom. This is the equivalent of a stacked glow/
    # adjustment layer group in a PSD-style workflow.
    if effect_profile in {"editorial", "premium", "halo", "poster"}:
        halo_mask = build_radial_gradient_mask(
            (w, h),
            center=face_center,
            radius=0.26 if effect_profile in {"premium", "halo"} else 0.22,
            radius_y=0.30 if effect_profile != "poster" else 0.25,
            inner_alpha=220,
            outer_alpha=0,
            power=1.55,
        )
        halo_layer = render_mask_layer(
            (w, h),
            (*palette["glow_a"][:3], 180),
            halo_mask,
            opacity=0.45 if effect_profile != "poster" else 0.34,
            blur_radius=max(12, w // 100),
        )
        canvas = composite_layer(
            canvas,
            halo_layer,
            blend_mode="screen" if effect_profile != "editorial" else "soft_light",
            opacity=0.92,
        )
        layers_meta.append({
            "name": "subject_halo",
            "kind": "glow",
            "blend_mode": "screen" if effect_profile != "editorial" else "soft_light",
        })

    # Diagonal light sheet for premium/poster looks.
    if effect_profile in {"premium", "poster"}:
        beam_mask = build_linear_gradient_mask(
            (w, h),
            start=(0.05, 0.0),
            end=(0.62, 0.92),
            start_alpha=0,
            end_alpha=180 if effect_profile == "premium" else 210,
            power=1.25,
        )
        beam_mask = combine_masks(
            beam_mask,
            build_shape_mask(
                (w, h),
                shape="ellipse",
                box=(int(w * -0.05), int(h * -0.15), int(w * 0.92), int(h * 0.78)),
                feather=28,
                alpha=190,
            ),
            mode="multiply",
        )
        beam_layer = render_mask_layer(
            (w, h),
            (*palette["glow_a"][:3], 120),
            beam_mask,
            opacity=0.72,
            blur_radius=18,
        )
        canvas = composite_layer(
            canvas,
            beam_layer,
            blend_mode="overlay",
            opacity=0.85,
        )
        layers_meta.append({
            "name": "diagonal_light_sheet",
            "kind": "adjustment_layer",
            "blend_mode": "overlay",
        })

    # Text-friendly dark plate. Keeps the premium stack readable when a bright
    # stage is used and gives the layout a more designed, poster-like finish.
    if effect_profile in {"poster", "premium"}:
        plate_mask = build_shape_mask(
            (w, h),
            shape="rounded_rect",
            box=(int(w * 0.03), int(h * 0.50), int(w * 0.97), int(h * 0.94)),
            radius=int(min(w, h) * 0.035),
            feather=24.0,
            alpha=160 if effect_profile == "premium" else 130,
        )
        plate_tint = render_mask_layer(
            (w, h),
            (10, 14, 22, 170),
            plate_mask,
            opacity=0.82,
        )
        canvas = composite_layer(
            canvas,
            plate_tint,
            blend_mode="multiply",
            opacity=0.75,
        )
        layers_meta.append({
            "name": "lower_text_plate",
            "kind": "mask",
            "blend_mode": "multiply",
        })

    # Tiny corner accent wash to emulate a "grouped layer" highlight.
    if effect_profile in {"editorial", "halo"}:
        corner_mask = build_shape_mask(
            (w, h),
            shape="diamond",
            box=(int(w * 0.66), int(h * 0.02), int(w * 0.98), int(h * 0.26)),
            feather=20.0,
            alpha=130,
        )
        corner_tint = render_mask_layer(
            (w, h),
            (*palette["accent"][:3], 110),
            corner_mask,
            opacity=0.65,
        )
        canvas = composite_layer(
            canvas,
            corner_tint,
            blend_mode="screen",
            opacity=0.72,
        )
        layers_meta.append({
            "name": "corner_accent",
            "kind": "mask",
            "blend_mode": "screen",
        })

    return canvas, layers_meta


def _apply_text_gradient(word_img: Image.Image, top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]) -> Image.Image:
    alpha = word_img.getchannel("A")
    gradient = _build_linear_gradient(word_img.size, top_color, bottom_color)
    gradient.putalpha(alpha)
    return Image.alpha_composite(word_img.convert("RGBA"), gradient)


def _add_path_badge(
    canvas: Image.Image,
    *,
    template: str,
    text: str,
    mode: str,
    position: str,
) -> Image.Image:
    if not text:
        return canvas
    font_path = _resolve_mc_font_path(template)
    if not font_path:
        return canvas

    w, h = canvas.size
    mode = str(mode or "arc").strip().lower()
    position = str(position or "top_right").strip().lower()

    if position == "top_left":
        center = (int(w * 0.22), int(h * 0.17))
    elif position == "bottom_left":
        center = (int(w * 0.22), int(h * 0.82))
    elif position == "upper_left":
        center = (int(w * 0.24), int(h * 0.22))
    else:
        center = (int(w * 0.78), int(h * 0.22))

    radius = int(min(w, h) * (0.15 if mode == "circle" else 0.18))
    badge_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    badge_draw = ImageDraw.Draw(badge_layer)
    palette = PALETTES[template]
    badge_draw.ellipse(
        [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
        outline=(*palette["accent"][:3], 140),
        width=max(3, radius // 14),
        fill=(0, 0, 0, 24 if template != "bold_minimal" else 0),
    )
    badge_layer = badge_layer.filter(ImageFilter.GaussianBlur(radius=max(0, radius // 18)))

    font_size = max(22, int(radius * 0.58))
    if mode == "circle":
        text_layer = circle_text_layer(
            text.upper(),
            font_path,
            font_size,
            center=center,
            radius=max(20, int(radius * 1.06)),
            image_size=(w, h),
            outside=True,
            preset="badge_top",
            fill=(*palette["text"][:3], 255),
            stroke=(*palette["accent"][:3], 210),
        )
    else:
        text_layer = arc_text_layer(
            text.upper(),
            font_path,
            font_size,
            center=center,
            radius=max(20, int(radius * 1.08)),
            start_angle_deg=-160.0,
            end_angle_deg=-20.0,
            image_size=(w, h),
            preset="badge_top",
            fill=(*palette["text"][:3], 255),
            stroke=(*palette["accent"][:3], 210),
        )

    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    canvas = Image.alpha_composite(canvas, badge_layer)
    return Image.alpha_composite(canvas, text_layer)


def _load_spec_image(value: object) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.exists():
            return Image.open(path).convert("RGBA")
    return None


def _materialize_layer_spec(
    spec: dict,
    *,
    template: str,
    canvas_size: tuple[int, int],
    default_text: str = "",
) -> dict:
    """Convert a JSON-like layer spec into a compose_layers-compatible layer."""
    if not isinstance(spec, dict):
        raise TypeError("layer spec must be a dict")

    kind = str(spec.get("kind") or "shape").strip().lower()
    blend_mode = str(spec.get("blend_mode") or "alpha").strip().lower()
    opacity = float(spec.get("opacity", 1.0))
    offset = tuple(int(round(float(value))) for value in spec.get("offset", (0, 0)))
    blur_radius = float(spec.get("blur_radius", 0.0))
    name = str(spec.get("name") or kind).strip()
    layer: Image.Image | None = None

    if kind == "image":
        layer = _load_spec_image(spec.get("image") or spec.get("source"))
    elif kind == "shape":
        shape = str(spec.get("shape") or "rounded_rect")
        box = spec.get("box")
        radius = spec.get("radius")
        feather = float(spec.get("feather", 0.0))
        alpha = int(spec.get("alpha", 255))
        color = tuple(spec.get("color") or PALETTES[template]["accent"])
        mask = build_shape_mask(canvas_size, shape=shape, box=box, radius=radius, feather=feather, alpha=alpha)
        layer = render_mask_layer(canvas_size, color, mask)
    elif kind == "gradient":
        gradient_kind = str(spec.get("gradient_kind") or ("radial" if spec.get("radial") else "linear")).strip().lower()
        color = tuple(spec.get("color") or PALETTES[template]["glow_a"])
        if gradient_kind == "radial":
            mask = build_radial_gradient_mask(
                canvas_size,
                center=tuple(spec.get("center", (0.5, 0.5))),
                radius=float(spec.get("radius", 0.5)),
                radius_y=spec.get("radius_y"),
                inner_alpha=int(spec.get("inner_alpha", 255)),
                outer_alpha=int(spec.get("outer_alpha", 0)),
                power=float(spec.get("power", 1.0)),
                box=spec.get("box"),
            )
        else:
            mask = build_linear_gradient_mask(
                canvas_size,
                start=tuple(spec.get("start", (0.0, 0.0))),
                end=tuple(spec.get("end", (0.0, 1.0))),
                start_alpha=int(spec.get("start_alpha", 0)),
                end_alpha=int(spec.get("end_alpha", 255)),
                power=float(spec.get("power", 1.0)),
                box=spec.get("box"),
            )
        layer = render_mask_layer(canvas_size, color, mask, opacity=float(spec.get("mask_opacity", 1.0)))
    elif kind in {"text_path", "badge"}:
        font_path = _resolve_mc_font_path(template)
        if not font_path:
            raise RuntimeError("No usable font available for text_path layer")
        text = str(spec.get("text") or default_text or "").strip()
        if not text:
            raise ValueError("text_path layer requires text")
        font_size = int(spec.get("font_size", max(22, int(min(canvas_size) * 0.055))))
        fill = tuple(spec.get("fill") or PALETTES[template]["text"])
        stroke = tuple(spec.get("stroke") or PALETTES[template]["accent"])
        mode = str(spec.get("mode") or "arc").strip().lower()
        if spec.get("path"):
            layer = text_on_svg_path_layer(
                text,
                font_path,
                font_size,
                str(spec.get("path")),
                canvas_size,
                preset=str(spec.get("preset") or "swoosh_headline"),
                fill=(*fill[:3], 255),
                stroke=(*stroke[:3], 210),
            )
        elif mode == "circle":
            center = tuple(spec.get("center", (canvas_size[0] * 0.8, canvas_size[1] * 0.2)))
            radius = float(spec.get("radius", min(canvas_size) * 0.16))
            layer = circle_text_layer(
                text,
                font_path,
                font_size,
                center=center,
                radius=radius,
                image_size=canvas_size,
                outside=bool(spec.get("outside", True)),
                preset=str(spec.get("preset") or "badge_top"),
                fill=(*fill[:3], 255),
                stroke=(*stroke[:3], 210),
            )
        else:
            center = tuple(spec.get("center", (canvas_size[0] * 0.8, canvas_size[1] * 0.2)))
            radius = float(spec.get("radius", min(canvas_size) * 0.16))
            layer = arc_text_layer(
                text,
                font_path,
                font_size,
                center=center,
                radius=radius,
                start_angle_deg=float(spec.get("start_angle_deg", -160.0)),
                end_angle_deg=float(spec.get("end_angle_deg", -20.0)),
                image_size=canvas_size,
                preset=str(spec.get("preset") or "badge_top"),
                fill=(*fill[:3], 255),
                stroke=(*stroke[:3], 210),
            )
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    if layer is None:
        raise ValueError(f"Could not materialize layer spec: {spec}")
    return {
        "name": name,
        "kind": kind,
        "layer": layer,
        "blend_mode": blend_mode,
        "opacity": opacity,
        "offset": offset,
        "blur_radius": blur_radius,
        "debug_key": name,
    }


def apply_custom_layer_stack(
    canvas: Image.Image,
    layer_specs: list[dict],
    *,
    template: str,
    default_text: str = "",
) -> tuple[Image.Image, list[dict]]:
    """Apply a declarative layer stack to a canvas.

    Supports a small, practical subset of PSD-style layer operations:
    shape/gradient/image/path-text layers with blend modes and opacity.
    """
    if not layer_specs:
        return canvas, []
    materialized: list[dict] = []
    layer_meta: list[dict] = []
    for spec in layer_specs:
        materialized_spec = _materialize_layer_spec(
            spec,
            template=template,
            canvas_size=canvas.size,
            default_text=default_text,
        )
        materialized.append(materialized_spec)
        layer_meta.append(
            {
                "name": materialized_spec["name"],
                "kind": materialized_spec["kind"],
                "blend_mode": materialized_spec["blend_mode"],
                "opacity": round(float(materialized_spec["opacity"]), 4),
            }
        )
    result, _debug_layers = compose_layers(canvas, materialized)
    return result, layer_meta



# ════════════════════════════════════════════════════════════════════════════
# TITLE SPLITTING
# ════════════════════════════════════════════════════════════════════════════

def auto_split_title(title: str) -> tuple[str, str]:
    """
    Auto-splits a sermon title into (back_words, front_word).
    Last word goes in front of the speaker, rest behind.

    Examples:
        "EINE WIE KEINE"       → ("EINE WIE", "KEINE")
        "GOTT NUTZT WEN ER WILL" → ("GOTT NUTZT WEN ER", "WILL")
        "ES WIRD ZEIT"          → ("ES WIRD", "ZEIT")
    """
    words = title.strip().upper().split()
    if len(words) <= 1:
        return ("", words[0] if words else "")
    if len(words) >= 5:
        return (" ".join(words[:-2]), " ".join(words[-2:]))
    return (" ".join(words[:-1]), words[-1])


# ════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def _ffmpeg_extract_frame(video_path: str, timestamp: float, out_path: str) -> bool:
    """Extract a single frame at timestamp via ffmpeg (handles AV1, HEVC, etc.)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and Path(out_path).exists()


def _get_video_duration_ffmpeg(video_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 300.0  # fallback: assume 5 minutes


def _candidate_timestamps(
    duration: float,
    n_candidates: int,
    prefer_timestamp: float | None,
) -> list[float]:
    start, end = duration * 0.20, duration * 0.80
    count = max(3, int(n_candidates))
    timestamps = [start + (end - start) * i / max(1, count - 1) for i in range(count)]
    if prefer_timestamp is not None:
        timestamps.insert(0, max(0.0, min(float(duration), float(prefer_timestamp))))
    return timestamps


def _mediapipe_model_cache_dir() -> Path:
    return Path(os.environ.get("PARAKEET_MODEL_CACHE", "~/.cache/parakeet_uv")).expanduser() / "mediapipe"


def _resolve_mediapipe_face_model() -> Path | None:
    configured = os.environ.get("PARAKEET_MEDIAPIPE_FACE_MODEL")
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None

    cache_path = _mediapipe_model_cache_dir() / "face_landmarker.task"
    if cache_path.exists() and cache_path.stat().st_size > 1024 * 1024:
        return cache_path

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".task.tmp")
        urllib.request.urlretrieve(_MEDIAPIPE_FACE_LANDMARKER_URL, tmp_path)
        if tmp_path.stat().st_size <= 1024 * 1024:
            tmp_path.unlink(missing_ok=True)
            return None
        tmp_path.replace(cache_path)
        return cache_path
    except Exception:
        return None


class _MediaPipeTasksFaceLandmarker:
    def __init__(self, landmarker, mp_module, model_path: Path) -> None:
        self.landmarker = landmarker
        self.mp = mp_module
        self.model_path = model_path
        self._parakeet_source = "mediapipe_tasks"

    def detect(self, mp_image):
        return self.landmarker.detect(mp_image)

    def close(self) -> None:
        close = getattr(self.landmarker, "close", None)
        if close:
            close()


def _load_face_mesh():
    global _LAST_FACE_SCORER_INFO
    try:
        import mediapipe as mp
    except Exception as exc:
        _LAST_FACE_SCORER_INFO = {
            "requested": "mediapipe",
            "active": "haar",
            "reason": f"mediapipe_import_failed: {type(exc).__name__}: {exc}",
        }
        return None
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _LAST_FACE_SCORER_INFO = {
            "requested": "mediapipe",
            "active": "mediapipe_solutions",
            "reason": "",
        }
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
        )
    try:
        from mediapipe.tasks.python.core import base_options as base_options_module
        from mediapipe.tasks.python.vision import face_landmarker
        from mediapipe.tasks.python.vision.core import vision_task_running_mode

        model_path = _resolve_mediapipe_face_model()
        if model_path is None:
            _LAST_FACE_SCORER_INFO = {
                "requested": "mediapipe_tasks",
                "active": "haar",
                "reason": "face_landmarker_model_unavailable",
                "model_url": _MEDIAPIPE_FACE_LANDMARKER_URL,
            }
            return None
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            output_face_blendshapes=True,
        )
        landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
        _LAST_FACE_SCORER_INFO = {
            "requested": "mediapipe_tasks",
            "active": "mediapipe_tasks",
            "reason": "",
            "model_path": str(model_path),
        }
        return _MediaPipeTasksFaceLandmarker(landmarker, mp, model_path)
    except Exception as exc:
        _LAST_FACE_SCORER_INFO = {
            "requested": "mediapipe_tasks",
            "active": "haar",
            "reason": f"mediapipe_tasks_init_failed: {type(exc).__name__}: {exc}",
            "model_path": str(_resolve_mediapipe_face_model() or ""),
            "system_hint": "Install libgles2 on WSL/Ubuntu to enable MediaPipe Tasks.",
        }
        return None


def _blendshape_scores(result) -> dict[str, float]:
    scores: dict[str, float] = {}
    blendshapes = getattr(result, "face_blendshapes", None) or []
    if not blendshapes:
        return scores
    for category in blendshapes[0]:
        name = str(getattr(category, "category_name", "") or "")
        scores[name] = float(getattr(category, "score", 0.0) or 0.0)
    return scores


def _face_mesh_metrics(frame_bgr: np.ndarray, face_mesh) -> dict | None:
    if face_mesh is None:
        return None
    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    source = "mediapipe"
    blendshapes: dict[str, float] = {}
    if hasattr(face_mesh, "detect"):
        try:
            mp = getattr(face_mesh, "mp", None)
            if mp is None:
                import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_mesh.detect(mp_image)
        except Exception:
            return None
        if not getattr(result, "face_landmarks", None):
            return None
        points = result.face_landmarks[0]
        source = "mediapipe_tasks"
        blendshapes = _blendshape_scores(result)
    else:
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        points = result.multi_face_landmarks[0].landmark
    xs = np.array([pt.x * width for pt in points], dtype=np.float32)
    ys = np.array([pt.y * height for pt in points], dtype=np.float32)
    x1, x2 = float(np.clip(xs.min(), 0, width - 1)), float(np.clip(xs.max(), 0, width - 1))
    y1, y2 = float(np.clip(ys.min(), 0, height - 1)), float(np.clip(ys.max(), 0, height - 1))
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    def dist(a: int, b: int) -> float:
        return float(((xs[a] - xs[b]) ** 2 + (ys[a] - ys[b]) ** 2) ** 0.5)

    eye_line = max(1.0, dist(33, 263))
    mouth_open = max(dist(13, 14) / eye_line, blendshapes.get("jawOpen", 0.0) * 0.22)
    left_eye = dist(159, 145) / eye_line
    right_eye = dist(386, 374) / eye_line
    if blendshapes:
        left_eye = max(0.0, left_eye * (1.0 - blendshapes.get("eyeBlinkLeft", 0.0)))
        right_eye = max(0.0, right_eye * (1.0 - blendshapes.get("eyeBlinkRight", 0.0)))
    tilt = abs(math.degrees(math.atan2(ys[263] - ys[33], xs[263] - xs[33])))
    return {
        "face_box": (int(x1), int(y1), int(fw), int(fh)),
        "face_width_ratio": fw / float(max(1, width)),
        "face_area_ratio": (fw * fh) / float(max(1, width * height)),
        "mouth_open": mouth_open,
        "eye_open": (left_eye + right_eye) / 2.0,
        "head_tilt": tilt,
        "source": source,
    }


def _haar_face_metrics(frame_bgr: np.ndarray) -> dict | None:
    face_box = _detect_face_box(frame_bgr)
    if not face_box:
        return None
    height, width = frame_bgr.shape[:2]
    x_pos, y_pos, box_w, box_h = face_box
    return {
        "face_box": face_box,
        "face_width_ratio": box_w / float(max(1, width)),
        "face_area_ratio": (box_w * box_h) / float(max(1, width * height)),
        "mouth_open": 0.10,
        "eye_open": 0.08,
        "head_tilt": 0.0,
        "source": "haar",
    }


def _backlight_metrics(gray: np.ndarray, face_box: tuple[int, int, int, int]) -> dict:
    """Measure how strongly the area around/behind the face is blown out.

    A speaker in front of a stage light produces a bright halo around the
    head/torso: high hot-pixel ratio and ring luminance far above face
    luminance. Both feed a 0..1 backlight score (1 = severe backlight).
    """
    height, width = gray.shape[:2]
    fx, fy, fw, fh = face_box
    x1 = max(0, int(fx - fw * 1.2))
    x2 = min(width, int(fx + fw * 2.2))
    y1 = max(0, int(fy - fh * 1.0))
    y2 = min(height, int(fy + fh * 2.0))
    ring = gray[y1:y2, x1:x2].astype(np.float32)
    face_region = gray[max(0, fy):min(height, fy + fh), max(0, fx):min(width, fx + fw)].astype(np.float32)
    if ring.size == 0 or face_region.size == 0:
        return {"score": 0.0, "hot_ratio": 0.0, "ring_minus_face": 0.0}
    hot_ratio = float((ring > 235).mean())
    ring_minus_face = float(ring.mean() - face_region.mean())
    score = min(1.0, hot_ratio * 2.6 + max(0.0, ring_minus_face) / 110.0)
    return {
        "score": round(score, 4),
        "hot_ratio": round(hot_ratio, 4),
        "ring_minus_face": round(ring_minus_face, 2),
    }


def _score_frame_quality(frame_bgr: np.ndarray, timestamp: float, face_mesh=None) -> tuple[float, dict]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    height, width = gray.shape[:2]
    face = _face_mesh_metrics(frame_bgr, face_mesh) or _haar_face_metrics(frame_bgr)

    exposure_score = max(0.0, 1.0 - abs(brightness - 118.0) / 105.0)
    sharpness_score = min(1.0, sharpness / 420.0)
    contrast_score = min(1.0, contrast / 72.0)
    rejected_reason = ""
    face_score = 0.0
    expression_score = 0.0
    backlight = {"score": 0.0, "hot_ratio": 0.0, "ring_minus_face": 0.0}

    if face:
        fx, fy, fw, fh = face["face_box"]
        face_center_y = (fy + fh / 2.0) / float(max(1, height))
        face_width_ratio = float(face["face_width_ratio"])
        face_score = min(1.0, face_width_ratio / 0.22)
        mouth_score = min(1.0, max(0.0, float(face["mouth_open"]) / 0.12))
        eye_score = min(1.0, max(0.0, float(face["eye_open"]) / 0.055))
        tilt_penalty = min(0.35, max(0.0, (float(face["head_tilt"]) - 12.0) / 70.0))
        expression_score = mouth_score * 0.58 + eye_score * 0.42 - tilt_penalty
        backlight = _backlight_metrics(gray, face["face_box"])

        if face_width_ratio < 0.10:
            rejected_reason = "face_too_small"
        elif face_center_y > 0.72:
            rejected_reason = "looking_down_or_low_face"
        elif eye_score < 0.25 and str(face.get("source", "")).startswith("mediapipe"):
            rejected_reason = "eyes_closed"
        elif sharpness_score < 0.045:
            rejected_reason = "motion_blur"
        elif backlight["score"] > 0.62:
            rejected_reason = "backlit"
    elif sharpness_score < 0.20:
        rejected_reason = "motion_blur_no_face"

    score = (
        face_score * 540.0
        + expression_score * 260.0
        + sharpness_score * 170.0
        + contrast_score * 80.0
        + exposure_score * 70.0
    )
    score *= 1.0 - min(0.40, float(backlight["score"]) * 0.40)
    if rejected_reason:
        score *= 0.55 if face and rejected_reason in ("motion_blur", "backlit") else 0.20 if face else 0.08
    if not face:
        score *= 0.18

    return score, {
        "timestamp": round(float(timestamp), 3),
        "score": round(float(score), 3),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "backlight": backlight,
        "face": face,
        "rejected_reason": rejected_reason,
    }


def extract_best_frame(
    video_path: str | np.ndarray,
    *,
    n_candidates: int = 30,
    prefer_timestamp: float | None = None,
) -> np.ndarray:
    # If already a frame, return directly
    if isinstance(video_path, np.ndarray):
        return video_path

    # Resolve path
    video_path = _normalise_path(str(video_path))
    video_path = str(Path(video_path).expanduser().resolve())

    if not Path(video_path).exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    global _LAST_FRAME_SELECTION, _LAST_TOP_FRAME_CANDIDATES
    _LAST_TOP_FRAME_CANDIDATES = []
    face_mesh = _load_face_mesh()

    # Try OpenCV first (fast for H.264/H.265)
    cap = cv2.VideoCapture(video_path)
    opencv_works = False
    if cap.isOpened():
        # Quick test: try to read one frame to see if codec is supported
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            opencv_works = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if opencv_works:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total_frames / fps
        timestamps = _candidate_timestamps(duration, n_candidates, prefer_timestamp)

        best_frame: np.ndarray | None = None
        best_score = -1.0
        candidates: list[dict] = []
        frame_candidates: list[dict] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            score, metadata = _score_frame_quality(frame, ts, face_mesh)
            candidates.append(metadata)
            frame_candidates.append({"metadata": metadata, "frame": frame.copy()})
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
        cap.release()
        if best_frame is not None:
            candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            _LAST_FRAME_SELECTION = {
                "video_path": video_path,
                "decoder": "opencv",
                "face_scorer": "mediapipe" if face_mesh is not None else "haar",
                "face_scorer_info": dict(_LAST_FACE_SCORER_INFO),
                "selected": candidates[0] if candidates else {},
                "top_candidates": candidates[:3],
            }
            frame_candidates.sort(key=lambda item: float(item["metadata"].get("score") or 0.0), reverse=True)
            _LAST_TOP_FRAME_CANDIDATES = [
                {"metadata": item["metadata"], "frame": item["frame"]}
                for item in frame_candidates[:3]
            ]
            print(
                "[ThumbnailMoveChurch] Best frame "
                f"t={_LAST_FRAME_SELECTION['selected'].get('timestamp')}s "
                f"score={best_score:.1f}"
            )
            return best_frame

    cap.release()

    # Fallback: use ffmpeg (handles AV1, VP9, etc. that OpenCV can't decode)
    print(f"[ThumbnailMoveChurch] OpenCV can't decode this codec — using ffmpeg fallback.")
    import tempfile
    duration = _get_video_duration_ffmpeg(video_path)
    timestamps = _candidate_timestamps(duration, n_candidates, prefer_timestamp)

    best_frame = None
    best_score = -1.0
    candidates = []
    frame_candidates = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            out_jpg = str(Path(tmpdir) / f"frame_{i:03d}.jpg")
            if not _ffmpeg_extract_frame(video_path, ts, out_jpg):
                continue
            frame = cv2.imread(out_jpg, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            score, metadata = _score_frame_quality(frame, ts, face_mesh)
            candidates.append(metadata)
            frame_candidates.append({"metadata": metadata, "frame": frame.copy()})
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

    if best_frame is None:
        raise RuntimeError(f"Could not extract any frame from {video_path}")
    candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    _LAST_FRAME_SELECTION = {
        "video_path": video_path,
        "decoder": "ffmpeg",
        "face_scorer": "mediapipe" if face_mesh is not None else "haar",
        "face_scorer_info": dict(_LAST_FACE_SCORER_INFO),
        "selected": candidates[0] if candidates else {},
        "top_candidates": candidates[:3],
    }
    frame_candidates.sort(key=lambda item: float(item["metadata"].get("score") or 0.0), reverse=True)
    _LAST_TOP_FRAME_CANDIDATES = [
        {"metadata": item["metadata"], "frame": item["frame"]}
        for item in frame_candidates[:3]
    ]
    print(
        "[ThumbnailMoveChurch] Best frame "
        f"t={_LAST_FRAME_SELECTION['selected'].get('timestamp')}s "
        f"score={best_score:.1f}"
    )
    return best_frame


def extract_frame_at(video_path: str, timestamp: float) -> np.ndarray:
    """Extract exactly the frame at `timestamp` (no candidate competition)."""
    global _LAST_FRAME_SELECTION, _LAST_TOP_FRAME_CANDIDATES
    video_path = str(Path(_normalise_path(str(video_path))).expanduser().resolve())
    if not Path(video_path).exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    frame = None
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
        ret, candidate = cap.read()
        if ret and candidate is not None:
            frame = candidate
    cap.release()
    if frame is None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_jpg = str(Path(tmpdir) / "frame.jpg")
            if _ffmpeg_extract_frame(video_path, float(timestamp), out_jpg):
                frame = cv2.imread(out_jpg, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not extract frame at {timestamp}s from {video_path}")

    face_mesh = _load_face_mesh()
    _score, metadata = _score_frame_quality(frame, float(timestamp), face_mesh)
    metadata["forced_timestamp"] = True
    _LAST_FRAME_SELECTION = {
        "video_path": video_path,
        "decoder": "forced_timestamp",
        "face_scorer": "mediapipe" if face_mesh is not None else "haar",
        "face_scorer_info": dict(_LAST_FACE_SCORER_INFO),
        "selected": metadata,
        "top_candidates": [metadata],
    }
    _LAST_TOP_FRAME_CANDIDATES = [{"metadata": metadata, "frame": frame.copy()}]
    print(f"[ThumbnailMoveChurch] Forced frame t={timestamp}s")
    return frame


def _normalise_path(path: str) -> str:
    # Convert Windows UNC WSL paths to native Linux paths.
    # e.g. \\wsl.localhost\Ubuntu-SSD\home\... -> /home/...
    import re
    # Replace all backslashes with forward slashes
    p = path.replace("\\", "/")
    # Strip leading slashes before wsl.localhost or wsl$
    m = re.match(r"^/+wsl[.$][^/]*/[^/]+(/.*)?$", p, re.IGNORECASE)
    if m:
        linux_part = m.group(1) or "/"
        return linux_part
    # Also handle ~ (shell doesn't expand when passed as string)
    if p.startswith("~/") or p == "~":
        import os
        return os.path.expanduser(path)
    return path


def load_source(source: str | np.ndarray | Image.Image) -> np.ndarray:
    """Normalise any source type to BGR numpy array."""
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, Image.Image):
        rgb = np.array(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path = _normalise_path(str(source))
    ext = Path(path).suffix.lower()
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return extract_best_frame(path)
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return frame


def _provider_chain(provider_name: str | None) -> list[str]:
    requested = str(provider_name or "auto").strip().lower()
    aliases = {
        "rmbg": "birefnet",
        "rmbg2": "birefnet",
        "rmbg-2.0": "birefnet",
        "birefnet_rmbg2": "birefnet",
        "grabcut": "grabcut_local",
    }
    requested = aliases.get(requested, requested)
    has_hf_auth = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if requested == "auto":
        return (["birefnet"] if has_hf_auth else []) + ["rembg", "grabcut_local"]
    if requested == "birefnet":
        return ["birefnet", "rembg", "grabcut_local"]
    if requested == "rembg":
        return ["rembg", "grabcut_local"]
    if requested == "grabcut_local":
        return ["grabcut_local"]
    return [requested, "rembg", "grabcut_local"]


def _cutout_quality_ok(coverage: float, subject_rgba: Image.Image | None) -> bool:
    if subject_rgba is None:
        return False
    if coverage < 0.035 or coverage > 0.62:
        return False
    alpha_bbox = subject_rgba.getchannel("A").getbbox()
    if not alpha_bbox:
        return False
    alpha = np.asarray(subject_rgba.getchannel("A"), dtype=np.uint8)
    strong_alpha_ratio = float(np.count_nonzero(alpha > 220)) / float(max(1, alpha.size))
    return strong_alpha_ratio > 0.02


def _clean_baked_caption_regions(frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove burned-in subtitle boxes before subject segmentation."""
    height, width = frame_bgr.shape[:2]
    y_start = int(height * 0.48)
    lower = frame_bgr[y_start:, :, :]
    if lower.size == 0:
        return frame_bgr, {"applied": False, "reason": "empty_lower_region"}

    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    # White subtitle text plus yellow karaoke text. Restrict to lower half to avoid stage lights.
    white_text = ((gray > 178) & (hsv[:, :, 1] < 95)).astype(np.uint8) * 255
    yellow_text = (
        (hsv[:, :, 0] >= 18)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 135)
    ).astype(np.uint8) * 255
    text_mask = cv2.bitwise_or(white_text, yellow_text)
    text_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)),
        iterations=1,
    )
    text_mask = cv2.dilate(
        text_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (33, 13)),
        iterations=1,
    )

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    cleanup_mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, component_count):
        x_pos, y_pos, box_w, box_h, area = [int(value) for value in stats[label]]
        if area < max(40, int(width * height * 0.00005)):
            continue
        if box_w < max(60, int(width * 0.06)) or box_h < 10:
            continue
        y_abs = y_start + y_pos
        # Keep this conservative: burned-in captions live low and are usually horizontal.
        if y_abs < int(height * 0.52):
            continue
        pad_x = max(42, int(box_w * 0.18))
        pad_y = max(24, int(box_h * 0.95))
        x1 = max(0, x_pos - pad_x)
        y1 = max(0, y_abs - pad_y)
        x2 = min(width, x_pos + box_w + pad_x)
        y2 = min(height, y_abs + box_h + pad_y)
        cleanup_mask[y1:y2, x1:x2] = 255
        boxes.append((x1, y1, x2, y2))

    coverage = float(np.count_nonzero(cleanup_mask)) / float(max(1, cleanup_mask.size))
    if not boxes or coverage > 0.22:
        return frame_bgr, {
            "applied": False,
            "boxes": boxes,
            "mask_coverage": round(coverage, 4),
            "reason": "no_caption_regions" if not boxes else "caption_mask_too_large",
        }

    cleaned = cv2.inpaint(frame_bgr, cleanup_mask, 3, cv2.INPAINT_TELEA)
    return cleaned, {
        "applied": True,
        "boxes": boxes,
        "mask_coverage": round(coverage, 4),
    }


def _subject_quality_metrics(subject_rgba: Image.Image) -> dict:
    """subject_contrast: tonal range inside the silhouette (low = washed out).
    edge_fringe: how much brighter the alpha edge band is than the core
    (high = baked-in backlight halo)."""
    rgb = np.asarray(subject_rgba.convert("RGB"), dtype=np.uint8)
    alpha = np.asarray(subject_rgba.getchannel("A"), dtype=np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    core_mask = alpha > 0.6
    if core_mask.sum() < 100:
        return {"subject_contrast": 0.0, "edge_fringe": 0.0}
    contrast = float(gray[core_mask].std()) / 64.0
    inner = np.asarray(subject_rgba.getchannel("A").filter(ImageFilter.MinFilter(7)), dtype=np.float32) / 255.0
    band = (alpha > 0.2) & (inner < 0.5)
    edge_fringe = 0.0
    if band.sum() > 50:
        edge_fringe = max(0.0, float(gray[band].mean() - gray[core_mask].mean())) / 255.0
    return {
        "subject_contrast": round(min(1.5, contrast), 4),
        "edge_fringe": round(edge_fringe, 4),
    }


def _extract_speaker_cutout(
    frame_bgr: np.ndarray,
    *,
    bg_removal_provider: str = "auto",
    relight: bool = False,
) -> dict:
    segmentation_frame_bgr, caption_cleanup = _clean_baked_caption_regions(frame_bgr)
    speaker_rgba = None
    face_box = None
    coverage = 0.0
    provider_used = None
    removal_attempts: list[dict] = []
    for provider_name in _provider_chain(bg_removal_provider):
        try:
            provider = get_background_removal_provider(provider_name)
            candidate_rgba, candidate_face_box, candidate_coverage = provider.extract_subject(segmentation_frame_bgr)
            ok = _cutout_quality_ok(candidate_coverage, candidate_rgba)
            removal_attempts.append(
                {
                    "provider": provider.name,
                    "coverage": round(float(candidate_coverage), 4),
                    "accepted": bool(ok),
                }
            )
            if not ok:
                print(
                    f"[ThumbnailMoveChurch] {provider.name} cutout rejected "
                    f"(coverage={candidate_coverage:.2f}); trying fallback."
                )
                continue
            candidate_rgba = defringe_subject(candidate_rgba)
            relit = False
            if relight:
                try:
                    from Components.SubjectRelight import relight_subject
                    candidate_rgba = relight_subject(candidate_rgba)
                    relit = True
                except Exception as exc:
                    print(f"[ThumbnailMoveChurch] Relight skipped ({exc}); procedural grade only.")
            speaker_rgba = candidate_rgba if relit else grade_subject(candidate_rgba)
            face_box = estimate_face_box(speaker_rgba) or candidate_face_box
            coverage = float(candidate_coverage)
            provider_used = provider.name
            print(f"[ThumbnailMoveChurch] Background removed with {provider.name}. Coverage: {coverage:.2f}")
            break
        except Exception as exc:
            removal_attempts.append({"provider": provider_name, "accepted": False, "error": str(exc)})
            print(f"[ThumbnailMoveChurch] {provider_name} failed: {exc}")

    return {
        "speaker_rgba": speaker_rgba,
        "face_box": face_box,
        "coverage": coverage,
        "provider_used": provider_used,
        "removal_attempts": removal_attempts,
        "caption_cleanup": caption_cleanup,
        "subject_metrics": _subject_quality_metrics(speaker_rgba) if speaker_rgba is not None else {},
    }


# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND RENDERING
# ════════════════════════════════════════════════════════════════════════════

def _render_background(width: int, height: int, template: str) -> Image.Image:
    palette = PALETTES[template]
    top    = np.array(palette["bg_top"],    dtype=np.float32)
    bottom = np.array(palette["bg_bottom"], dtype=np.float32)
    if palette.get("bg_radial"):
        # Radial gradient: bg_top = centre colour, bg_bottom = edge colour
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = width / 2.0, height / 2.0
        dist = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
        dist = np.clip(dist / dist.max(), 0.0, 1.0)[:, :, None]
        canvas = ((1.0 - dist) * top + dist * bottom).astype(np.uint8)
        return Image.fromarray(canvas, mode="RGB").convert("RGBA")
    ramp   = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    grad   = ((1.0 - ramp) * top + ramp * bottom).astype(np.uint8)
    canvas = np.repeat(grad[:, None, :], width, axis=1)
    return Image.fromarray(canvas, mode="RGB").convert("RGBA")


def _ai_background_cache_dir(output_path: str | None) -> Path | None:
    if not output_path:
        return None
    return Path(output_path).expanduser().resolve(strict=False).parent / "_ai_backgrounds"


def _render_ai_background(
    *,
    title: str,
    template: str,
    output_path: str | None,
    speaker_name: str | None = None,
    brand_label: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    size: tuple[int, int],
) -> tuple[Image.Image | None, dict]:
    width, height = size
    cache_dir = _ai_background_cache_dir(output_path)
    background, info = generate_background_image(
        title=title,
        template=template,
        speaker_name=speaker_name,
        brand_label=brand_label,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        cache_dir=cache_dir,
    )
    if background is None:
        return None, info
    background = background.convert("RGBA")
    background = ImageEnhance.Color(background).enhance(0.90)
    background = ImageEnhance.Contrast(background).enhance(1.03)
    background = ImageEnhance.Brightness(background).enhance(0.94)
    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vignette)
    for i in range(12):
        inset_x = int(width * i / 42)
        inset_y = int(height * i / 42)
        alpha = int(8 + i * 10)
        draw.rounded_rectangle(
            [inset_x, inset_y, width - inset_x, height - inset_y],
            radius=max(1, width // 24),
            outline=(0, 0, 0, alpha),
            width=max(8, width // 64),
        )
    background = Image.alpha_composite(background, vignette.filter(ImageFilter.GaussianBlur(radius=18)))
    darken = Image.new("RGBA", (width, height), (8, 10, 16, 28))
    background = Image.alpha_composite(background, darken)
    return background, info


def _render_darkened_frame_background(
    frame_bgr: np.ndarray,
    width: int,
    height: int,
    template: str,
) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    src = Image.fromarray(frame_rgb, mode="RGB").convert("RGBA")
    src_ratio = src.width / float(max(1, src.height))
    dst_ratio = width / float(max(1, height))
    if src_ratio > dst_ratio:
        new_h = height
        new_w = int(height * src_ratio)
    else:
        new_w = width
        new_h = int(width / max(1e-6, src_ratio))
    src = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    src = src.crop((left, top, left + width, top + height))
    src = ImageEnhance.Contrast(src).enhance(1.08)
    src = ImageEnhance.Color(src).enhance(0.68)
    src = ImageEnhance.Brightness(src).enhance(0.36)
    src = src.filter(ImageFilter.GaussianBlur(radius=max(4, width // 180)))

    brand_bg = _render_background(width, height, template)
    canvas = Image.blend(src, brand_bg, 0.68).convert("RGBA")
    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vignette)
    for i in range(18):
        inset_x = int(width * i / 44)
        inset_y = int(height * i / 44)
        alpha = int(10 + i * 8)
        draw.rounded_rectangle(
            [inset_x, inset_y, width - inset_x, height - inset_y],
            radius=max(1, width // 20),
            outline=(0, 0, 0, alpha),
            width=max(12, width // 55),
        )
    return Image.alpha_composite(canvas, vignette.filter(ImageFilter.GaussianBlur(radius=22)))


def _add_atmosphere(canvas: Image.Image, template: str, intensity: float = 0.7) -> Image.Image:
    """Light-based atmosphere: soft bokeh particles + diagonal light streaks
    in palette colors. Deterministic per template (seeded RNG)."""
    palette = PALETTES[template]
    w, h = canvas.size
    rng = random.Random(template)

    bokeh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bokeh)
    for color, count in ((palette["glow_a"], 9), (palette["glow_b"], 6)):
        for _ in range(count):
            r = rng.randint(max(3, int(w * 0.006)), int(w * 0.040))
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            a = int(rng.randint(20, 60) * intensity)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(*tuple(color)[:3], a))
    bokeh = bokeh.filter(ImageFilter.GaussianBlur(radius=max(5, w // 170)))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), bokeh)

    streaks = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(streaks)
    for _ in range(3):
        x0 = rng.randint(-int(w * 0.25), int(w * 0.9))
        width = rng.randint(max(8, int(w * 0.02)), int(w * 0.055))
        a = int(24 * intensity)
        sdraw.line(
            [(x0, -60), (x0 + int(w * 0.38), h + 60)],
            fill=(*tuple(palette["glow_a"])[:3], a),
            width=width,
        )
    streaks = streaks.filter(ImageFilter.GaussianBlur(radius=max(12, w // 38)))
    return Image.alpha_composite(canvas, streaks)


def _add_glow_orbs(canvas: Image.Image, template: str, intensity: float = 0.7) -> Image.Image:
    palette = PALETTES[template]
    w, h = canvas.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    orb_defs: list[tuple[float, float, float, tuple[int, int, int], float]] = []
    if template == "navy_dark":
        orb_defs = [
            (0.30, 0.0,  0.55, palette["glow_a"], intensity * 0.6),
            (0.80, 0.6,  0.45, palette["glow_b"], intensity * 0.4),
        ]
    elif template == "energy_orange":
        orb_defs = [
            (0.0,  0.40, 0.60, palette["glow_a"], intensity * 0.65),
            (1.0,  0.20, 0.55, palette["glow_b"], intensity * 0.55),
            (0.5,  1.00, 0.40, (100, 10, 100),    intensity * 0.30),
        ]
    elif template == "warm_gold":
        orb_defs = [
            (0.65, 0.05, 0.50, palette["glow_a"], intensity * 0.45),
            (0.20, 0.75, 0.50, palette["glow_b"], intensity * 0.50),
        ]
    elif template == "cinematic_dark":
        # Faint white spotlight pool top-right (main beam in _add_decorations)
        orb_defs = [
            (0.85, 0.02, 0.42, palette["glow_a"], intensity * 0.22),
        ]
    elif template == "fire_red":
        orb_defs = [
            (0.15, 0.25, 0.45, palette["glow_a"], intensity * 0.50),
            (0.85, 0.55, 0.40, palette["glow_b"], intensity * 0.45),
            (0.50, 0.95, 0.38, (255, 40, 0),      intensity * 0.30),
        ]
    elif template == "heaven_blue":
        # Soft sky-blue light from above
        orb_defs = [
            (0.50, 0.00, 0.55, palette["glow_a"], intensity * 0.55),
            (0.50, 0.30, 0.45, palette["glow_b"], intensity * 0.28),
        ]
    elif template == "sunset_warm":
        orb_defs = [
            (0.18, 0.08, 0.50, palette["glow_a"], intensity * 0.50),
            (0.80, 0.80, 0.45, palette["glow_b"], intensity * 0.40),
        ]
    # bold_minimal: deliberately no orbs

    draw = ImageDraw.Draw(overlay)
    for cx_r, cy_r, r_r, color, alpha in orb_defs:
        cx = int(cx_r * w)
        cy = int(cy_r * h)
        r  = int(r_r * min(w, h))
        a  = int(alpha * 255)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, a))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=min(w, h) // 8))
    return Image.alpha_composite(canvas, overlay)


def _add_subject_keylight(
    canvas: Image.Image,
    subject: Image.Image,
    sx: int,
    sy: int,
    template: str,
    intensity: float = 1.0,
) -> Image.Image:
    """Big multi-pass keylight bloom built from the subject silhouette.

    Painted at canvas scale right before the speaker so the halo can be huge
    without inflating the subject's bounding box. Three passes: a wide
    atmospheric wash, a mid bloom, and a hot near-white core — reads like a
    strong stage keylight from behind the speaker."""
    palette = PALETTES[template]
    w, h = canvas.size
    glow_col = tuple(palette["glow_a"])[:3]
    # Lift toward white so the bloom reads on any background (keylight = bright)
    mid_col = tuple(min(255, int(c + (255 - c) * 0.55)) for c in glow_col)
    hot_col = tuple(min(255, int(c + (255 - c) * 0.85)) for c in glow_col)

    sil = Image.new("L", (w, h), 0)
    sil.paste(subject.getchannel("A"), (sx, sy))

    # (blur_radius, color, peak_alpha) — wide → mid → hot core
    passes = (
        (max(80, int(w * 0.22)), mid_col, 170),
        (max(30, int(w * 0.075)), mid_col, 190),
        (max(12, int(w * 0.026)), hot_col, 235),
    )
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    # Screen-blend each pass so the bloom behaves like light (luminous,
    # never muddy), not like a translucent sticker.
    base_rgb = canvas.convert("RGB")
    for radius, col, peak in passes:
        a = int(peak * max(0.0, min(1.0, intensity)))
        mask = sil.filter(ImageFilter.GaussianBlur(radius=radius))
        mask = mask.point(lambda v, _a=a: v * _a // 255)
        glow_rgb = Image.new("RGB", (w, h), (0, 0, 0))
        glow_rgb.paste(Image.new("RGB", (w, h), col), mask=mask)
        base_rgb = ImageChops.screen(base_rgb, glow_rgb)
    lit = base_rgb.convert("RGBA")
    lit.putalpha(canvas.getchannel("A"))
    return lit


def _add_decorations(canvas: Image.Image, template: str, intensity: float = 0.7) -> Image.Image:
    palette = PALETTES[template]
    w, h = canvas.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Concentric rings
    if palette["rings"]:
        rings_center = palette.get(
            "rings_center",
            (-0.15, -0.13) if template == "navy_dark" else (1.15, -0.13),
        )
        cx, cy = int(w * rings_center[0]), int(h * rings_center[1])
        base_r = int(min(w, h) * 0.55)
        for i in range(5):
            r = base_r + i * int(min(w, h) * 0.085)
            a = max(4, int(18 * intensity * (1 - i * 0.18)))
            col = palette["glow_a"] if i % 2 == 0 else palette["glow_b"]
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline=(*col, a), width=2)
        # Second, mirrored ring cluster — extra background depth
        cx2, cy2 = int(w * (1.0 - rings_center[0])), int(h * 1.08)
        for i in range(3):
            r = base_r + i * int(min(w, h) * 0.085)
            a = max(3, int(13 * intensity * (1 - i * 0.22)))
            col = palette["glow_b"] if i % 2 == 0 else palette["glow_a"]
            draw.ellipse([cx2 - r, cy2 - r, cx2 + r, cy2 + r],
                         outline=(*col, a), width=2)

    # Thin swirl arcs across the frame (reference look — elegant partial
    # ellipses in palette colors, deterministic per template)
    if not palette.get("minimal"):
        arc_rng = random.Random(template + "_arcs")
        for _ in range(5):
            acx = arc_rng.randint(int(-w * 0.2), int(w * 1.2))
            acy = arc_rng.randint(int(-h * 0.1), int(h * 1.1))
            arx = arc_rng.randint(int(w * 0.30), int(w * 0.85))
            ary = int(arx * arc_rng.uniform(0.55, 1.0))
            start = arc_rng.randint(0, 360)
            extent = arc_rng.randint(70, 200)
            col = palette["glow_a"] if arc_rng.random() < 0.6 else palette["glow_b"]
            a = max(4, int(arc_rng.randint(12, 26) * intensity))
            draw.arc(
                [acx - arx, acy - ary, acx + arx, acy + ary],
                start, start + extent,
                fill=(*tuple(col)[:3], a),
                width=arc_rng.choice([2, 3, 4]),
            )

    # Energy texture for energy_orange
    if template == "energy_orange":
        for _ in range(6):
            x0 = random.randint(-w // 4, w)
            y0 = random.randint(0, h)
            x1 = x0 + random.randint(w // 2, w * 2)
            y1 = y0 + random.randint(-20, 20)
            a  = random.randint(4, 14)
            draw.line([x0, y0, x1, y1], fill=(255, 255, 255, a), width=2)

    # Fast diagonal energy lines for fire_red
    if template == "fire_red":
        for _ in range(9):
            x0 = random.randint(-w // 3, w)
            y0 = random.randint(0, h)
            x1 = x0 + random.randint(w // 2, w * 2)
            y1 = y0 + random.randint(-h // 4, h // 4)
            a  = random.randint(6, 18)
            col = (255, 120, 40, a) if random.random() < 0.6 else (255, 255, 255, a)
            draw.line([x0, y0, x1, y1], fill=col, width=random.randint(2, 3))

    canvas = Image.alpha_composite(canvas, overlay)

    # Light rays (blurred)
    ray_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ray_draw = ImageDraw.Draw(ray_overlay)
    r_col = palette["ray_color"]
    if template == "navy_dark":
        # Two diagonal beams from top
        for x_frac, rot, width_px in [(0.38, -12, 8), (0.55, 8, 12)]:
            x = int(x_frac * w)
            dx = int(h * math.tan(math.radians(rot)))
            ray_draw.line([(x, 0), (x + dx, h)], fill=r_col, width=width_px)
    elif template == "warm_gold":
        # Single spotlight from upper-right
        x = int(w * 0.60)
        ray_draw.polygon([(x - 60, 0), (x + 60, 0), (x + 200, h), (x - 200, h)],
                         fill=(*r_col[:3], int(r_col[3] * intensity * 0.5)))
    elif template == "cinematic_dark":
        # Film-noir white spotlight from top-right, falling down-left
        x = int(w * 0.88)
        ray_draw.polygon([(x - 70, 0), (x + 70, 0), (int(w * 0.45) + 260, h), (int(w * 0.45) - 260, h)],
                         fill=(*r_col[:3], int(r_col[3] * intensity * 0.6)))
    elif template == "sunset_warm":
        # Warm amber spotlight from upper-left
        x = int(w * 0.18)
        ray_draw.polygon([(x - 70, 0), (x + 70, 0), (int(w * 0.55) + 240, h), (int(w * 0.55) - 240, h)],
                         fill=(*r_col[:3], int(r_col[3] * intensity * 0.5)))
    elif template == "heaven_blue":
        # Vertical light shaft from directly above (light from heaven)
        x = int(w * 0.50)
        ray_draw.polygon([(x - 90, 0), (x + 90, 0), (x + 240, h), (x - 240, h)],
                         fill=(*r_col[:3], int(r_col[3] * intensity * 0.45)))

    ray_overlay = ray_overlay.filter(ImageFilter.GaussianBlur(radius=max(8, w // 60)))
    return Image.alpha_composite(canvas, ray_overlay)


def _add_film_grain(canvas: Image.Image, opacity: float = 0.03) -> Image.Image:
    """Subtle grey-noise grain overlay (cinematic_dark)."""
    w, h = canvas.size
    noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    alpha = np.full((h, w), int(255 * opacity), dtype=np.uint8)
    grain = Image.fromarray(np.dstack([noise, noise, noise, alpha]), mode="RGBA")
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    return Image.alpha_composite(canvas, grain)


# ════════════════════════════════════════════════════════════════════════════
# TEXT RENDERING
# ════════════════════════════════════════════════════════════════════════════

def _word_font_size(fmt: str, base_ratio: float = 0.22) -> int:
    w, h = FORMATS[fmt]
    if fmt == "9x16":
        return int(h * base_ratio)      # ~422px at default
    else:
        return int(h * base_ratio * 1.1)  # ~174px for 16:9


def _measure_spaced_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    letter_spacing: int,
    *,
    stroke_width: int = 0,
) -> tuple[int, int]:
    probe = Image.new("RGBA", (4, 4))
    draw = ImageDraw.Draw(probe)
    total_width = 0
    max_height = 1
    chars = list(text)
    for index, char in enumerate(chars):
        bbox = draw.textbbox((0, 0), char, font=font, stroke_width=stroke_width)
        total_width += max(0, bbox[2] - bbox[0])
        max_height = max(max_height, bbox[3] - bbox[1])
        if index < len(chars) - 1:
            total_width += letter_spacing
    return max(1, int(total_width)), max(1, int(max_height))


def _resolve_letter_spacing(
    word: str,
    *,
    font_size: int,
    effect_profile: str = "classic",
) -> int:
    """Return compact but still readable tracking for all-caps thumbnail text."""
    profile = _EFFECT_PROFILE_CONFIG.get(effect_profile, _EFFECT_PROFILE_CONFIG["classic"])
    tracking_scale = float(profile.get("text", {}).get("tracking_scale", 1.0))
    size = max(1, int(font_size))
    word = str(word or "").strip()

    # Base tracking is slightly tighter than the old fixed ratio so long
    # all-caps titles feel more premium and less "stretched".
    spacing = max(0, int(round(size * 0.014)))

    if len(word) <= 4:
        spacing += 1
    elif len(word) >= 10:
        spacing -= 1
    if len(word) >= 14:
        spacing -= 1

    if word.isupper() and len(word) >= 8:
        spacing -= 1

    return max(0, int(round(spacing * tracking_scale)))


def _draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int],
    letter_spacing: int,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] | None = None,
) -> None:
    x_pos, y_pos = xy
    for char in text:
        draw.text(
            (x_pos, y_pos),
            char,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        bbox = draw.textbbox((0, 0), char, font=font)
        x_pos += max(0, bbox[2] - bbox[0]) + letter_spacing


def _render_word(
    word: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int],
    glow_color: tuple[int, int, int] | None = None,
    glow_radius: int = 40,
    shadow: bool = True,
    outline_color: tuple[int, int, int, int] | None = None,
    outline_width: int = 0,
    italic_shear: float = 0.0,
    letter_spacing: int | None = None,
    extrude: int = 0,
    effect_profile: str = "classic",
) -> Image.Image:
    """Render a single word with glow, shadow, optional italic shear and
    optional 3D extrude (stacked dark offset copies)."""
    word = str(word or "").strip().upper()
    resolved_spacing = int(
        letter_spacing if letter_spacing is not None else _resolve_letter_spacing(
            word,
            font_size=getattr(font, "size", 80),
            effect_profile=effect_profile,
        )
    )
    profile = _EFFECT_PROFILE_CONFIG.get(effect_profile, _EFFECT_PROFILE_CONFIG["classic"])
    text_profile = profile.get("text", {})
    stroke_scale = float(text_profile.get("stroke_scale", 1.0))
    shadow_scale = float(text_profile.get("shadow_scale", 1.0))
    glow_scale = float(text_profile.get("glow_scale", 1.0))
    extrude_scale = float(text_profile.get("extrude_scale", 1.0))
    stroke_w = max(int(outline_width * stroke_scale), max(3, getattr(font, "size", 80) // 42))
    stroke_fill = outline_color or (3, 5, 10, 238)
    hard_stroke_w = stroke_w + 4  # hard black contour for readability on any bg
    tw, th = _measure_spaced_text(word, font, resolved_spacing, stroke_width=hard_stroke_w)
    pad = int(glow_radius * glow_scale) + hard_stroke_w + max(18, getattr(font, "size", 80) // 10)
    shear_extra = int(abs(italic_shear) * th)

    layer = Image.new("RGBA", (tw + pad * 2 + shear_extra + 6, th + pad * 2 + 6), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)
    tx, ty = pad + shear_extra, pad

    if shadow:
        shadow_col = (0, 0, 0, int(210 * shadow_scale))
        _draw_spaced_text(
            draw,
            (tx + max(4, stroke_w), ty + max(6, int((stroke_w + 2) * shadow_scale))),
            word,
            font,
            fill=shadow_col,
            letter_spacing=resolved_spacing,
            stroke_width=stroke_w,
            stroke_fill=(0, 0, 0, 180),
        )

    if glow_color:
        glow_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        glow_draw  = ImageDraw.Draw(glow_layer)
        _draw_spaced_text(
            glow_draw,
            (tx, ty),
            word,
            font,
            fill=(*glow_color, 155),
            letter_spacing=resolved_spacing,
            stroke_width=stroke_w,
            stroke_fill=(*glow_color, 145),
        )
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=max(12, int(glow_radius * glow_scale))))
        layer = Image.alpha_composite(layer, glow_layer)
        draw  = ImageDraw.Draw(layer)

    # 3D extrude: stacked dark copies receding down-right
    if extrude > 0:
        extrude = int(max(0, extrude * extrude_scale))
        ex_col = (int(fill[0] * 0.20), int(fill[1] * 0.20), int(fill[2] * 0.20), 255)
        for i in range(extrude, 0, -1):
            _draw_spaced_text(
                draw,
                (tx + i, ty + int(i * 1.15)),
                word,
                font,
                fill=ex_col,
                letter_spacing=resolved_spacing,
                stroke_width=stroke_w,
                stroke_fill=ex_col,
            )

    # Hard black 4px contour underneath the fill — keeps text readable on any bg
    _draw_spaced_text(
        draw,
        (tx, ty),
        word,
        font,
        fill=(0, 0, 0, 255),
        letter_spacing=resolved_spacing,
        stroke_width=hard_stroke_w,
        stroke_fill=(0, 0, 0, 255),
    )

    _draw_spaced_text(
        draw,
        (tx, ty),
        word,
        font,
        fill=fill,
        letter_spacing=resolved_spacing,
        stroke_width=stroke_w,
        stroke_fill=stroke_fill,
    )

    gradient = text_profile.get("gradient")
    if gradient:
        top_color, bottom_color = gradient
        gradient_layer = _build_linear_gradient(layer.size, top_color, bottom_color)
        text_mask = layer.getchannel("A")
        gradient_layer.putalpha(text_mask)
        layer = Image.alpha_composite(layer, gradient_layer)

    # Italic shear (horizontal skew for fake italic without font support)
    if abs(italic_shear) > 0.01:
        arr = np.array(layer, dtype=np.float32)
        h, w = arr.shape[:2]
        M = np.float32([[1, italic_shear, 0], [0, 1, 0]])
        arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR)
        layer = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    return crop_to_alpha(layer)


def _pack_title_lines(
    words: list[str],
    accent_word: str | None = None,
    max_chars: int = 7,
) -> list[str]:
    """Greedily merge adjacent short words onto one line (reference look:
    'ER WILL' shares a line instead of two undersized lines). Accent words
    stay alone so their colour pop survives."""
    lines: list[str] = []
    i = 0
    while i < len(words):
        cur = words[i]
        is_acc = bool(accent_word and cur.upper() == accent_word.upper())
        if i + 1 < len(words) and not is_acc:
            nxt = words[i + 1]
            nxt_acc = bool(accent_word and nxt.upper() == accent_word.upper())
            if not nxt_acc and len(cur) + 1 + len(nxt) <= max_chars:
                lines.append(f"{cur} {nxt}")
                i += 2
                continue
        lines.append(cur)
        i += 1
    return lines


def _word_fill_size(
    word: str,
    *,
    font_size: int,
    template: str,
    fill_width: int,
    effect_profile: str = "classic",
) -> int:
    """Per-word font size so the rendered word roughly fills `fill_width`.

    Clamped to 0.68×–2.60× of the base size: long words shrink to fit, short
    punch words grow toward the full line width (reference mega-stack look)."""
    base_font = _load_mc_font(font_size, template)
    probe_spacing = _resolve_letter_spacing(
        word,
        font_size=font_size,
        effect_profile=effect_profile,
    )
    probe_w, _ = _measure_spaced_text(
        word.upper(),
        base_font,
        probe_spacing,
        stroke_width=max(2, font_size // 42),
    )
    scale = fill_width / max(1, probe_w)
    scale = max(0.68, min(2.60, scale))
    return max(42, int(font_size * scale))


def _place_text_block(
    canvas: Image.Image,
    words: list[str],
    *,
    font_size: int,
    template: str,
    accent_word: str | None,
    effect_profile: str = "classic",
    x_left: int,
    y_top: int,
    line_gap_ratio: float = 0.88,
    max_width: int | None = None,
    glow: bool = True,
    shadow: bool = True,
    text_opacity: float = 1.0,
    text_color: tuple[int, int, int, int] | None = None,
    fill_width: int | None = None,
    collect_bounds: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    """Place stacked word block onto canvas, returns new canvas."""
    palette   = PALETTES[template]
    italic    = palette["italic"]
    shear     = -0.18 if italic else 0.0
    accent_col = palette["accent"]
    text_col   = text_color or palette["text"]
    glow = glow and palette.get("text_glow", True)

    resolved_opacity = max(0.0, min(1.0, text_opacity))
    minimal = bool(palette.get("minimal"))
    # Standard thumbnails should stay straight/readable; only the more poster-like
    # profiles are allowed to tilt the whole title block.
    rotate_deg = -6.0 if (not minimal and effect_profile in {"editorial", "poster"}) else 0.0
    min_line_advance_ratio = 0.72 if effect_profile in {"editorial", "poster"} else 1.02

    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    block_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    block_bounds: list[tuple[int, int, int, int]] = []

    y = y_top
    word_entries: list[tuple[Image.Image, int, int]] = []
    for word in words:
        if not word.strip():
            continue
        is_accent = accent_word and word.upper() == accent_word.upper()
        fill = (accent_col[0], accent_col[1], accent_col[2], 255) if is_accent else text_col

        word_fs = font_size
        if fill_width:
            word_fs = _word_fill_size(
                word,
                font_size=font_size,
                template=template,
                fill_width=fill_width,
                effect_profile=effect_profile,
            )
        font = _load_mc_font(word_fs, template)
        extrude = 0 if minimal else max(3, word_fs // 26)

        word_img = _render_word(
            word.upper(),
            font,
            fill=fill,
            glow_color=palette["glow_a"][:3] if glow else None,
            glow_radius=word_fs // 6,
            shadow=shadow,
            outline_width=max(4, word_fs // 45),
            italic_shear=shear,
            letter_spacing=max(1, word_fs // 46),
            extrude=extrude,
            effect_profile=effect_profile,
        )

        # Fade the whole layer (fill + stroke + glow) for depth behind the speaker
        if resolved_opacity < 1.0:
            faded_alpha = word_img.getchannel("A").point(
                lambda v: int(v * resolved_opacity)
            )
            word_img.putalpha(faded_alpha)

        # Clamp to max_width if needed
        if max_width and word_img.width > max_width:
            ratio = max_width / word_img.width
            new_h = max(1, int(word_img.height * ratio))
            word_img = word_img.resize((max_width, new_h), Image.Resampling.LANCZOS)

        # Collect entries for potential curved placement later
        word_entries.append((word_img, x_left, y))
        block_bounds.append((x_left, y, x_left + word_img.width, y + word_img.height))
        # Keep stacked title words from colliding with each other; body overlap
        # is fine, text-on-text overlap is not.
        y += max(int(word_fs * line_gap_ratio), int(word_img.height * min_line_advance_ratio))
    # If rotated style is desired, render words along a gentle arc instead
    # of a rigid block rotation so the result looks hand-set (like Illustrator).
    def _paste_words_on_arc(entries: list[tuple[Image.Image, int, int]],
                            bounds: list[tuple[int, int, int, int]],
                            out_layer: Image.Image) -> Image.Image:
        if not entries:
            return out_layer
        bx1 = min(b[0] for b in bounds)
        bx2 = max(b[2] for b in bounds)
        by2 = max(b[3] for b in bounds)
        block_w = max(1, bx2 - bx1)

        # radius proportional to block width; larger = gentler curve
        radius = max(int(block_w * 0.9), 10)
        # center below the block so text arches downward slightly
        cx = bx1 + block_w / 2.0
        cy = by2 + radius * 0.28

        for (img, x, y) in entries:
            mid_x = (x + img.width / 2.0) - bx1 - block_w / 2.0
            # angle along arc (radians)
            angle = mid_x / float(max(1.0, radius))
            # compute position on circle
            px = cx + radius * math.sin(angle) - img.width / 2.0
            py = cy - radius * math.cos(angle) - img.height / 2.0
            # rotate glyph to follow tangent (degrees)
            deg = math.degrees(angle)
            rotated = img.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
            out_layer.alpha_composite(rotated, (int(px), int(py)))
        return out_layer

    if abs(rotate_deg) > 0.05 and block_bounds:
        # Build a blank layer and paste words along an arc
        curved_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        curved_layer = _paste_words_on_arc(word_entries, block_bounds, curved_layer)
        # Slight overall rotation to keep dynamic reference look
        curved_layer = curved_layer.rotate(
            rotate_deg,
            resample=Image.Resampling.BICUBIC,
            center=( (min(b[0] for b in block_bounds) + max(b[2] for b in block_bounds)) / 2.0,
                     (min(b[1] for b in block_bounds) + max(b[3] for b in block_bounds)) / 2.0 ),
        )
        canvas = Image.alpha_composite(canvas, curved_layer)
    else:
        # Fallback: paste words exactly where calculated (no curve)
        for (img, x, y) in word_entries:
            block_layer.alpha_composite(img, (x, y))
        canvas = Image.alpha_composite(canvas, block_layer)
    if collect_bounds is not None:
        collect_bounds.extend(block_bounds)

    return canvas


def _measure_text_block(
    words: list[str],
    *,
    font_size: int,
    template: str,
    effect_profile: str = "classic",
    max_width: int | None = None,
    line_gap_ratio: float = 0.88,
    fill_width: int | None = None,
) -> tuple[int, int]:
    if not words:
        return (0, 0)
    palette = PALETTES[template]
    shear = -0.18 if palette["italic"] else 0.0
    max_word_w = 0
    rendered: list[tuple[int, int]] = []  # (word_fs, rendered_h)
    minimal = bool(palette.get("minimal"))
    for word in words:
        word_fs = font_size
        if fill_width:
            word_fs = _word_fill_size(
                word, font_size=font_size, template=template, fill_width=fill_width
            )
        font = _load_mc_font(word_fs, template)
        word_img = _render_word(
            word.upper(),
            font,
            fill=palette["text"],
            extrude=0 if minimal else max(3, word_fs // 26),
            glow_color=palette["glow_a"][:3] if palette.get("text_glow", True) else None,
            glow_radius=word_fs // 6,
            shadow=True,
            outline_width=max(4, word_fs // 45),
            italic_shear=shear,
            letter_spacing=max(1, word_fs // 46),
            effect_profile=effect_profile,
        )
        word_w, word_h = word_img.size
        if max_width and word_w > max_width:
            ratio = max_width / word_w
            word_w = max_width
            word_h = max(1, int(word_h * ratio))
        max_word_w = max(max_word_w, word_w)
        rendered.append((word_fs, int(word_h)))
    if not rendered:
        return (0, 0)
    total_h = rendered[-1][1]
    min_line_advance_ratio = 0.72 if effect_profile in {"editorial", "poster"} else 1.02
    if len(rendered) > 1:
        total_h += sum(
            max(int(fs * line_gap_ratio), int(height * min_line_advance_ratio))
            for fs, height in rendered[:-1]
        )
    return int(max_word_w), int(total_h)


def _separate_text_blocks(
    *,
    back_y: int,
    back_h: int,
    front_y: int,
    front_h: int,
    min_gap: int,
    min_back_y: int,
    max_front_y: int,
) -> tuple[int, int]:
    """Keep back/front title blocks from overlapping each other.

    Body overlap is allowed; only text-vs-text collisions are prevented here.
    """
    desired_front_y = max(front_y, back_y + back_h + min_gap)
    if desired_front_y <= max_front_y:
        return back_y, desired_front_y

    overflow = desired_front_y - max_front_y
    shifted_back_y = max(min_back_y, back_y - overflow)
    desired_front_y = max(front_y, shifted_back_y + back_h + min_gap)
    return shifted_back_y, min(max_front_y, desired_front_y)


def _fit_text_font_size(
    words: list[str],
    *,
    base_size: int,
    template: str,
    effect_profile: str = "classic",
    max_width: int,
    max_height: int,
    fill_width: int | None = None,
) -> int:
    size = int(base_size)
    while size > 42:
        block_w, block_h = _measure_text_block(
            words, font_size=size, template=template, max_width=max_width,
            fill_width=fill_width,
            effect_profile=effect_profile,
        )
        if block_w <= max_width and block_h <= max_height:
            return size
        size -= max(4, size // 18)
    return max(42, size)


# ════════════════════════════════════════════════════════════════════════════
# DECORATIVE SHAPES
# ════════════════════════════════════════════════════════════════════════════

def _add_accent_bar(
    canvas: Image.Image,
    *,
    x: int,
    y: int,
    width: int,
    template: str,
    height: int = 8,
) -> Image.Image:
    """Orange/gold brush-stroke accent bar (or thin white line, per palette)."""
    palette = PALETTES[template]
    col = palette.get("bar_color", palette["accent"])
    height = int(palette.get("bar_height", height))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Tapered bar: full colour on left, transparent on right
    for i in range(width):
        frac = 1.0 - (i / max(1, width)) ** 1.5
        a = int(frac * col[3])
        draw.line([(x + i, y), (x + i, y + height)], fill=(*col[:3], a))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.5))
    return Image.alpha_composite(canvas, overlay)


def _add_frame_border(
    canvas: Image.Image,
    *,
    x0: int, y0: int, x1: int, y1: int,
    template: str,
    width: int = 4,
    radius: int = 16,
    opacity: float = 0.25,
) -> Image.Image:
    """Decorative border (no fill, stroke only)."""
    palette = PALETTES[template]
    col = palette["accent"]
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(opacity * col[3])
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                           outline=(*col[:3], a), width=width)
    return Image.alpha_composite(canvas, overlay)


def _add_arrow(
    canvas: Image.Image,
    *,
    x: int, y: int,
    font_size: int,
    template: str,
    effect_profile: str = "classic",
    direction: str = "→",
) -> Image.Image:
    palette = PALETTES[template]
    col = palette["accent"]
    font = _load_mc_font(font_size, template)
    arrow_img = _render_word(
        direction, font,
        fill=col,
        glow_color=palette["glow_a"][:3],
        glow_radius=font_size // 5,
        shadow=True,
        effect_profile=effect_profile,
    )
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    canvas.alpha_composite(arrow_img, (x, y))
    return canvas


def _add_symbol(
    canvas: Image.Image,
    *,
    symbol: str,
    x: int, y: int,
    size: int = 120,
    opacity: float = 0.85,
) -> Image.Image:
    """Place a Unicode symbol (emoji/char) at position."""
    glyph = SYMBOLS.get(symbol, symbol)
    # Use a larger PIL font for emoji
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", size
        )
    except Exception:
        font = _load_mc_font(size)
    sym_img = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sym_img)
    draw.text((0, 0), glyph, font=font, fill=(255, 255, 255, int(255 * opacity)),
              embedded_color=True)
    sym_img = crop_to_alpha(sym_img)
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    canvas.alpha_composite(sym_img, (max(0, x), max(0, y)))
    return canvas


# ════════════════════════════════════════════════════════════════════════════
# LOGO
# ════════════════════════════════════════════════════════════════════════════

def _add_logo(
    canvas: Image.Image,
    *,
    template: str,
    logo_path: str | None = None,
    fmt: str = "9x16",
    margin: int | None = None,
) -> Image.Image:
    """Add Move Church logo (M circle + text) bottom-left."""
    w, h = canvas.size
    palette = PALETTES[template]
    col = palette["logo_color"]
    m = margin if margin is not None else int(w * 0.055)

    ring_r = int(w * 0.040)
    font_size = int(w * 0.028)
    text_font = _load_mc_font(font_size, template)
    letter_font = _load_mc_font(ring_r, template)

    logo_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo_layer)

    # Circle with "M"
    cx, cy = m + ring_r, h - m - ring_r
    draw.ellipse(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        outline=(*col[:3], 230), width=max(2, ring_r // 12),
    )
    bbox = draw.textbbox((0, 0), "M", font=letter_font)
    lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - lw // 2, cy - lh // 2), "M", font=letter_font,
              fill=(*col[:3], 230))

    # "MOVE" + "CHURCH" text
    tx = cx + ring_r + int(w * 0.018)
    ty = cy - ring_r // 2
    draw.text((tx, ty), "MOVE", font=text_font, fill=(*col[:3], 230))
    small_font = _load_mc_font(max(10, font_size - int(font_size * 0.28)), template)
    draw.text((tx, ty + font_size + 2), "CHURCH", font=small_font,
              fill=(*col[:3], 160))

    return Image.alpha_composite(canvas, logo_layer)


def _scaled_face_box(
    face_box: tuple[int, int, int, int] | None,
    *,
    scale: float,
    offset: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if face_box is None:
        return None
    x_pos, y_pos, box_w, box_h = face_box
    ox, oy = offset
    return (
        int(ox + x_pos * scale),
        int(oy + y_pos * scale),
        max(1, int(box_w * scale)),
        max(1, int(box_h * scale)),
    )


def _alpha_head_span_ratio(
    speaker_rgba: Image.Image,
    face_box: tuple[int, int, int, int] | None,
) -> float:
    if face_box is None:
        return 0.0
    alpha = np.asarray(speaker_rgba.getchannel("A"), dtype=np.uint8)
    if alpha.size == 0:
        return 0.0
    fx, fy, fw, fh = [int(value) for value in face_box]
    y1 = max(0, int(fy - fh * 0.20))
    y2 = min(alpha.shape[0], int(fy + fh * 0.82))
    if y2 <= y1:
        return 0.0
    max_span = 0
    for row in alpha[y1:y2]:
        xs = np.flatnonzero(row > 24)
        if xs.size:
            max_span = max(max_span, int(xs[-1] - xs[0] + 1))
    return float(max_span) / float(max(1, alpha.shape[1]))


def _overlaps_face_too_much(
    rect: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int] | None,
    *,
    limit: float = 0.15,
) -> bool:
    if face_box is None:
        return False
    x1, y1, x2, y2 = rect
    fx, fy, fw, fh = face_box
    ix1 = max(x1, fx)
    iy1 = max(y1, fy)
    ix2 = min(x2, fx + fw)
    iy2 = min(y2, fy + fh)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    return ((ix2 - ix1) * (iy2 - iy1)) / float(max(1, fw * fh)) > limit


def _safe_text_y(
    desired_y: int,
    *,
    block_h: int,
    canvas_w: int,
    canvas_h: int,
    face_box: tuple[int, int, int, int] | None,
    min_y: int,
    max_y: int,
    rect_x1: int = 0,
    rect_x2: int | None = None,
) -> int:
    y_pos = max(min_y, min(max_y, int(desired_y)))
    x2 = canvas_w if rect_x2 is None else rect_x2
    rect = (rect_x1, y_pos, x2, y_pos + block_h)
    if not _overlaps_face_too_much(rect, face_box):
        return y_pos
    if face_box is None:
        return y_pos
    _fx, fy, _fw, fh = face_box
    options = [
        max(min_y, min(max_y, fy - block_h - int(canvas_h * 0.035))),
        max(min_y, min(max_y, fy + fh + int(canvas_h * 0.035))),
        y_pos,
    ]
    return min(
        options,
        key=lambda candidate: _face_overlap_ratio((rect_x1, candidate, x2, candidate + block_h), face_box),
    )


def _union_bounds(bounds: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not bounds:
        return None
    return (
        min(bound[0] for bound in bounds),
        min(bound[1] for bound in bounds),
        max(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def _rect_overlap_area(
    rect_a: tuple[int, int, int, int] | list[int] | None,
    rect_b: tuple[int, int, int, int] | list[int] | None,
) -> int:
    if rect_a is None or rect_b is None:
        return 0
    ax1, ay1, ax2, ay2 = [int(value) for value in rect_a]
    bx1, by1, bx2, by2 = [int(value) for value in rect_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))


def _rect_area(rect: tuple[int, int, int, int] | list[int] | None) -> int:
    if rect is None:
        return 0
    x1, y1, x2, y2 = [int(value) for value in rect]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _face_overlap_ratio(
    rect: tuple[int, int, int, int] | None,
    face_box: tuple[int, int, int, int] | None,
) -> float:
    if rect is None or face_box is None:
        return 0.0
    x1, y1, x2, y2 = rect
    fx, fy, fw, fh = face_box
    ix1 = max(x1, fx)
    iy1 = max(y1, fy)
    ix2 = min(x2, fx + fw)
    iy2 = min(y2, fy + fh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / float(max(1, fw * fh))


def _word_face_overlap_ratio(
    word_bounds: list[tuple[int, int, int, int]],
    face_box: tuple[int, int, int, int] | None,
) -> float:
    if not word_bounds or face_box is None:
        return 0.0
    fx, fy, fw, fh = [int(value) for value in face_box]
    face_rect = (fx, fy, fx + fw, fy + fh)
    overlap = sum(_rect_overlap_area(bound, face_rect) for bound in word_bounds)
    return min(1.0, overlap / float(max(1, fw * fh)))


def _alpha_occlusion_ratio(
    alpha_layer: Image.Image | None,
    bounds: list[tuple[int, int, int, int]],
    *,
    offset: tuple[int, int],
    canvas_size: tuple[int, int],
) -> float:
    if alpha_layer is None or not bounds:
        return 0.0
    canvas_w, canvas_h = canvas_size
    alpha = np.asarray(alpha_layer.convert("L"), dtype=np.uint8)
    offset_x, offset_y = offset
    occluded = 0
    total = 0
    for x1, y1, x2, y2 in bounds:
        cx1 = max(0, min(canvas_w, int(x1)))
        cy1 = max(0, min(canvas_h, int(y1)))
        cx2 = max(cx1, min(canvas_w, int(x2)))
        cy2 = max(cy1, min(canvas_h, int(y2)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        sx1 = max(0, cx1 - offset_x)
        sy1 = max(0, cy1 - offset_y)
        sx2 = min(alpha.shape[1], cx2 - offset_x)
        sy2 = min(alpha.shape[0], cy2 - offset_y)
        total += (cx2 - cx1) * (cy2 - cy1)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        occluded += int(np.count_nonzero(alpha[sy1:sy2, sx1:sx2] > 24))
    return float(occluded) / float(max(1, total))


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT: 9:16 vs 16:9
# ════════════════════════════════════════════════════════════════════════════

def _layout_9x16(
    canvas: Image.Image,
    *,
    back_words: list[str],
    front_words: list[str],
    speaker_rgba: Image.Image | None,
    speaker_face_box: tuple[int, int, int, int] | None,
    template: str,
    font_size: int,
    accent_word: str | None,
    show_logo: bool,
    show_accent_bar: bool,
    show_frame: bool,
    show_arrow: bool,
    symbol: str | None,
    effect_profile: str = "classic",
    badge_text: str | None = None,
    badge_mode: str = "arc",
    badge_position: str = "top_right",
) -> Image.Image:
    global _LAST_LAYOUT_METADATA
    w, h = canvas.size
    palette = PALETTES[template]
    margin = int(w * 0.047)
    text_bounds: list[tuple[int, int, int, int]] = []
    back_text_bounds: list[tuple[int, int, int, int]] = []
    front_text_bounds: list[tuple[int, int, int, int]] = []

    # ── z-1+2: background + decorations already on canvas ──────────────────
    back_words = _pack_title_lines(back_words, accent_word)
    front_words = _pack_title_lines(front_words, accent_word)
    n_words = len(back_words) + len(front_words)
    effective_fs = font_size
    if n_words >= 4:
        effective_fs = int(font_size * 0.76)
    elif n_words >= 3:
        effective_fs = int(font_size * 0.86)

    max_text_w = int(w * 0.86)
    # Mega-stack: every line fills the text column (per-word sizes, clamped)
    fill_w = int(w * 0.80)
    effective_fs = _fit_text_font_size(
        [*back_words, *front_words],
        base_size=effective_fs,
        template=template,
        effect_profile=effect_profile,
        max_width=max_text_w,
        max_height=int(h * 0.56),
        fill_width=fill_w,
    )
    speaker_paste: tuple[Image.Image, int, int, float] | None = None
    speaker_alpha_for_occlusion: Image.Image | None = None
    face_canvas_box = None
    body_box = None
    closeup_subject = False
    face_height_ratio_source = 0.0
    head_span_ratio_source = 0.0
    if speaker_rgba is not None:
        sp = speaker_rgba.copy()
        resolved_face_box = speaker_face_box or estimate_face_box(speaker_rgba)
        if resolved_face_box:
            # Full standing figure → crop to waist-up so the face reads large
            # (reference look). Crop is top-anchored, so face coords stay valid.
            _fx0, _fy0, _fw0, _fh0 = resolved_face_box
            if _fh0 / float(max(1, sp.height)) < 0.22:
                waist_bottom = int(_fy0 + _fh0 * 4.4)
                if waist_bottom < sp.height:
                    sp = sp.crop((0, 0, sp.width, waist_bottom))
            face_height_ratio_source = float(resolved_face_box[3]) / float(max(1, sp.height))
            head_span_ratio_source = _alpha_head_span_ratio(sp, resolved_face_box)
        closeup_subject = face_height_ratio_source > 0.42 or head_span_ratio_source > 0.62
        target_ratio = 0.78 if n_words >= 5 else 0.82 if n_words >= 4 else 0.86
        if closeup_subject:
            target_ratio = min(target_ratio, 0.56)
        target_h = int(h * target_ratio)
        ratio = target_h / max(1, sp.height)
        sp = sp.resize((max(1, int(sp.width * ratio)), target_h), Image.Resampling.LANCZOS)
        sx = (w - sp.width) // 2
        if resolved_face_box:
            face_center_x = (resolved_face_box[0] + resolved_face_box[2] / 2.0) * ratio
            target_face_x = 0.72 if closeup_subject else 0.66 if n_words >= 4 else 0.56
            sx = int(w * target_face_x - face_center_x)
            sx = max(-int(sp.width * 0.20), min(w - int(sp.width * 0.80), sx))
        sy = int(h * 0.29) if closeup_subject else h - sp.height + int(h * 0.012)
        speaker_paste = (sp, sx, sy, ratio)
        speaker_alpha_for_occlusion = sp.getchannel("A")
        face_canvas_box = _scaled_face_box(resolved_face_box, scale=ratio, offset=(sx, sy))
        body_box = (max(0, sx), max(0, sy), min(w, sx + sp.width), min(h, sy + sp.height))

    face_center_y = 0.48
    if face_canvas_box:
        _fx, fy, _fw, fh = face_canvas_box
        face_center_y = (fy + fh / 2.0) / float(max(1, h))

    if face_center_y < 0.40:
        composition = "upper_face"
        back_y = int(h * 0.070)
        front_y = int(h * 0.610 if n_words >= 4 else h * 0.660)
        text_x = margin
    elif face_center_y < 0.58:
        composition = "center_face"
        back_y = int(h * 0.07)
        front_y = int(h * 0.64)
        text_x = margin
        if face_canvas_box:
            fx, _fy, fw, _fh = face_canvas_box
            left_space = fx
            right_space = w - (fx + fw)
            if right_space > left_space * 1.10:
                text_x = int(w * 0.42)
    else:
        composition = "lower_face"
        back_y = int(h * 0.06)
        front_y = int(h * 0.48)
        text_x = margin

    back_fs = effective_fs
    front_fs = effective_fs
    if palette.get("back_scale"):
        # bold_minimal: enormous back words (sit behind the speaker anyway) —
        # max_width clamp + _safe_text_y guard against overflow/face collision
        back_fs = int(effective_fs * palette["back_scale"])
    else:
        if composition == "upper_face":
            back_fs = min(back_fs, int(h * 0.080))
        # 5% smaller per additional back word so multi-word back zones don't clip
        if len(back_words) > 1:
            back_fs = int(back_fs * (0.95 ** (len(back_words) - 1)))
    back_line_gap_ratio = 0.96 if len(back_words) > 1 else 0.84
    front_line_gap_ratio = 0.96 if len(front_words) > 1 else 0.84
    back_line_h = int(back_fs * back_line_gap_ratio)
    front_line_h = int(front_fs * front_line_gap_ratio)

    _back_block_w, back_block_h = _measure_text_block(
        back_words,
        font_size=back_fs,
        template=template,
        effect_profile=effect_profile,
        max_width=max_text_w,
        line_gap_ratio=back_line_gap_ratio,
        fill_width=fill_w,
    )
    _front_block_w, front_block_h = _measure_text_block(
        front_words,
        font_size=front_fs,
        template=template,
        effect_profile=effect_profile,
        max_width=max_text_w,
        line_gap_ratio=front_line_gap_ratio,
        fill_width=fill_w,
    )

    text_band_min_y = int(h * 0.06)
    text_band_max_y = int(h * 0.86)
    inter_block_gap = int(h * (0.032 if back_words and front_words else 0.0))
    combined_text_h = back_block_h + front_block_h + inter_block_gap
    available_text_h = max(1, text_band_max_y - text_band_min_y)
    if combined_text_h > available_text_h:
        scale = max(0.72, available_text_h / float(combined_text_h))
        back_fs = max(42, int(back_fs * scale))
        front_fs = max(42, int(front_fs * scale))
        back_line_gap_ratio = 0.96 if len(back_words) > 1 else 0.84
        front_line_gap_ratio = 0.96 if len(front_words) > 1 else 0.84
        back_line_h = int(back_fs * back_line_gap_ratio)
        front_line_h = int(front_fs * front_line_gap_ratio)
        _back_block_w, back_block_h = _measure_text_block(
            back_words,
            font_size=back_fs,
            template=template,
            effect_profile=effect_profile,
            max_width=max_text_w,
            line_gap_ratio=back_line_gap_ratio,
            fill_width=fill_w,
        )
        _front_block_w, front_block_h = _measure_text_block(
            front_words,
            font_size=front_fs,
            template=template,
            effect_profile=effect_profile,
            max_width=max_text_w,
            line_gap_ratio=front_line_gap_ratio,
            fill_width=fill_w,
        )
    back_max_y = int(h * 0.58)
    if composition == "upper_face" and face_canvas_box:
        _fx, fy, _fw, _fh = face_canvas_box
        back_max_y = max(int(h * 0.06), fy - max(back_block_h, back_line_h * len(back_words)) - int(h * 0.018))
    front_text_x = margin
    if body_box is not None and _front_block_w > 0:
        desired_x = body_box[0] - int(_front_block_w * 0.58)
        max_x = max(margin, w - margin - min(max_text_w, max(_front_block_w, int(w * 0.34))))
        front_text_x = max(margin, min(max_x, desired_x))
    back_y = _safe_text_y(
        back_y,
        block_h=max(back_block_h, back_line_h * len(back_words)),
        canvas_w=w,
        canvas_h=h,
        face_box=face_canvas_box,
        min_y=int(h * 0.06),
        max_y=back_max_y,
        rect_x1=text_x,
        rect_x2=min(w, text_x + max(_back_block_w, int(w * 0.34))),
    )
    front_min_y = int(h * 0.54)
    if body_box is not None:
        body_anchor = 0.38 if composition == "upper_face" and n_words >= 4 else 0.48
        front_min_y = max(front_min_y, body_box[1] + int((body_box[3] - body_box[1]) * body_anchor))
    front_y = _safe_text_y(
        front_y,
        block_h=max(front_block_h, front_line_h * len(front_words)),
        canvas_w=w,
        canvas_h=h,
        face_box=face_canvas_box,
        min_y=front_min_y,
        max_y=int(h * 0.86),
        rect_x1=front_text_x,
        rect_x2=min(w, front_text_x + max(_front_block_w, int(w * 0.34))),
    )
    back_y, front_y = _separate_text_blocks(
        back_y=back_y,
        back_h=max(back_block_h, back_line_h * len(back_words)),
        front_y=front_y,
        front_h=max(front_block_h, front_line_h * len(front_words)),
        min_gap=inter_block_gap,
        min_back_y=text_band_min_y,
        max_front_y=text_band_max_y,
    )

    # ── z-2.5: dark pop + keylight bloom behind the subject ────────────────
    if speaker_paste is not None and not palette.get("minimal"):
        sp_pop, px, py, _r = speaker_paste
        pop = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pop_draw = ImageDraw.Draw(pop)
        cx = px + sp_pop.width / 2.0
        cy = py + sp_pop.height * 0.42
        rx = sp_pop.width * 0.85
        ry = sp_pop.height * 0.60
        pop_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(0, 0, 0, 80))
        pop = pop.filter(ImageFilter.GaussianBlur(radius=max(40, w // 14)))
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        canvas = Image.alpha_composite(canvas, pop)
        canvas = _add_subject_keylight(canvas, sp_pop, px, py, template)

    # ── z-3: BACK TEXT (behind speaker) ────────────────────────────────────
    canvas = _place_text_block(
        canvas, back_words,
        font_size=back_fs,
        template=template,
        accent_word=accent_word,
        effect_profile=effect_profile,
        x_left=text_x,
        y_top=back_y,
        line_gap_ratio=back_line_gap_ratio,
        max_width=max_text_w,
        glow=True,
        text_opacity=0.88,
        fill_width=fill_w,
        collect_bounds=back_text_bounds,
    )
    text_bounds.extend(back_text_bounds)

    # ── z-3.5: horizontal separator line (cinematic_dark) ──────────────────
    if palette.get("separator_color") and back_words and front_words:
        sep_y = int(((back_y + back_block_h) + front_y) / 2)
        sep_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sep_draw = ImageDraw.Draw(sep_overlay)
        sep_draw.line(
            [(text_x, sep_y), (text_x + int(w * 0.55), sep_y)],
            fill=palette["separator_color"], width=4,
        )
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        canvas = Image.alpha_composite(canvas, sep_overlay)

    # ── z-4: SPEAKER PHOTO ─────────────────────────────────────────────────
    if speaker_paste is not None:
        sp, sx, sy, _ratio = speaker_paste
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        canvas.alpha_composite(sp, (sx, sy))

    # ── z-5: FRONT TEXT (in front of speaker) ──────────────────────────────
    canvas = _place_text_block(
        canvas, front_words,
        font_size=front_fs,
        template=template,
        accent_word=accent_word,
        effect_profile=effect_profile,
        x_left=front_text_x,
        y_top=front_y,
        line_gap_ratio=front_line_gap_ratio,
        max_width=max_text_w,
        glow=True,
        text_color=palette["accent"] if palette.get("front_accent") else None,
        fill_width=fill_w,
        collect_bounds=front_text_bounds,
    )
    text_bounds.extend(front_text_bounds)

    # Accent bar below front text
    if show_accent_bar and not palette.get("minimal"):
        bar_y = front_y + int(front_fs * 0.88 * len(front_words)) + int(h * 0.012)
        canvas = _add_accent_bar(
            canvas, x=front_text_x, y=bar_y, width=int(w * 0.30), template=template
        )

    # Decorative frame border
    if show_frame:
        canvas = _add_frame_border(
            canvas,
            x0=margin // 2, y0=margin // 2,
            x1=w - margin // 2, y1=h - margin // 2,
            template=template, opacity=0.20,
        )

    # Arrow
    if show_arrow:
        arrow_y = front_y - int(font_size * 1.1)
        canvas = _add_arrow(canvas, x=front_text_x, y=max(0, arrow_y),
                            font_size=int(font_size * 0.55), template=template,
                            effect_profile=effect_profile)

    # Symbol
    if symbol:
        sym_size = int(w * 0.10)
        canvas = _add_symbol(canvas, symbol=symbol,
                             x=w - sym_size - margin, y=h - int(h * 0.18),
                             size=sym_size)

    # ── z-6: LOGO ───────────────────────────────────────────────────────────
    if show_logo:
        canvas = _add_logo(canvas, template=template, fmt="9x16")

    badge_config = _EFFECT_PROFILE_CONFIG[effect_profile].get("badge")
    resolved_badge_text = badge_text if badge_text is not None else (badge_config or {}).get("text")
    if resolved_badge_text:
        resolved_mode = badge_mode if badge_text is not None else (badge_config or {}).get("mode", "arc")
        resolved_position = badge_position if badge_text is not None else (badge_config or {}).get("position", "top_right")
        canvas = _add_path_badge(
            canvas,
            template=template,
            text=str(resolved_badge_text),
            mode=str(resolved_mode),
            position=str(resolved_position),
        )

    union = _union_bounds(text_bounds)
    back_union = _union_bounds(back_text_bounds)
    front_union = _union_bounds(front_text_bounds)
    vertical_gap_ratio = 0.0
    if back_union is not None and front_union is not None:
        vertical_gap_ratio = max(0, front_union[1] - back_union[3]) / float(max(1, h))
    speaker_offset = (speaker_paste[1], speaker_paste[2]) if speaker_paste is not None else (0, 0)
    front_area = max(1, _rect_area(front_union))
    front_body_overlap_ratio = _rect_overlap_area(front_union, body_box) / float(front_area)
    front_face_overlap_ratio = _rect_overlap_area(front_union, face_canvas_box) / float(front_area)
    back_occlusion_ratio = _alpha_occlusion_ratio(
        speaker_alpha_for_occlusion,
        back_text_bounds,
        offset=speaker_offset,
        canvas_size=(w, h),
    )
    back_word_occlusion_ratios = [
        _alpha_occlusion_ratio(
            speaker_alpha_for_occlusion,
            [bound],
            offset=speaker_offset,
            canvas_size=(w, h),
        )
        for bound in back_text_bounds
    ]
    back_max_occlusion_ratio = max(back_word_occlusion_ratios, default=0.0)
    _LAST_LAYOUT_METADATA = {
        "fmt": "9x16",
        "composition": composition,
        "font_size": effective_fs,
        "back_font_size": int(back_fs),
        "front_font_size": int(front_fs),
        "text_bounds": [int(value) for value in union] if union else None,
        "back_text_bounds": [int(value) for value in back_union] if back_union else None,
        "front_text_bounds": [int(value) for value in front_union] if front_union else None,
        "text_word_bounds": [[int(value) for value in bound] for bound in text_bounds],
        "back_text_word_bounds": [[int(value) for value in bound] for bound in back_text_bounds],
        "front_text_word_bounds": [[int(value) for value in bound] for bound in front_text_bounds],
        "face_box": [int(value) for value in face_canvas_box] if face_canvas_box else None,
        "body_box": [int(value) for value in body_box] if body_box else None,
        "closeup_subject": bool(closeup_subject),
        "source_face_height_ratio": round(float(face_height_ratio_source), 4),
        "source_head_span_ratio": round(float(head_span_ratio_source), 4),
        "front_text_x": int(front_text_x),
        "front_body_overlap_ratio": round(front_body_overlap_ratio, 4),
        "front_face_overlap_ratio": round(front_face_overlap_ratio, 4),
        # Visible text only: back words sit behind the speaker, so their rects
        # overlapping the face is the signature depth look, not a collision
        # (back-word hiding is guarded by back_text_max_occlusion_ratio).
        "face_overlap_ratio": round(_word_face_overlap_ratio(front_text_bounds, face_canvas_box), 4),
        "vertical_gap_ratio": round(vertical_gap_ratio, 4),
        "back_text_occlusion_ratio": round(back_occlusion_ratio, 4),
        "back_text_max_occlusion_ratio": round(back_max_occlusion_ratio, 4),
        "back_text_word_occlusion_ratios": [round(float(value), 4) for value in back_word_occlusion_ratios],
    }
    return canvas


def _layout_16x9(
    canvas: Image.Image,
    *,
    back_words: list[str],
    front_words: list[str],
    speaker_rgba: Image.Image | None,
    speaker_face_box: tuple[int, int, int, int] | None,
    template: str,
    font_size: int,
    accent_word: str | None,
    show_logo: bool,
    show_accent_bar: bool,
    show_frame: bool,
    show_arrow: bool,
    symbol: str | None,
    effect_profile: str = "classic",
    badge_text: str | None = None,
    badge_mode: str = "arc",
    badge_position: str = "top_right",
) -> Image.Image:
    global _LAST_LAYOUT_METADATA
    w, h = canvas.size
    margin = int(w * 0.03)
    text_zone_w = int(w * 0.48)  # left half for text

    # ── z-3: BACK TEXT (left side) ─────────────────────────────────────────
    back_y = int(h * 0.10)
    canvas = _place_text_block(
        canvas, back_words,
        font_size=font_size,
        template=template,
        accent_word=accent_word,
        effect_profile=effect_profile,
        x_left=margin,
        y_top=back_y,
        max_width=text_zone_w - margin,
        glow=True,
    )

    # ── z-4: SPEAKER PHOTO (right half) ────────────────────────────────────
    if speaker_rgba is not None:
        sp = speaker_rgba.copy()
        target_h = int(h * 1.02)
        ratio = target_h / max(1, sp.height)
        sp = sp.resize((max(1, int(sp.width * ratio)), target_h),
                       Image.Resampling.LANCZOS)
        sx = w - sp.width + int(sp.width * 0.05)  # slightly off right edge
        sy = h - sp.height
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        if not PALETTES[template].get("minimal"):
            canvas = _add_subject_keylight(canvas, sp, max(0, sx), max(0, sy), template)
        canvas.alpha_composite(sp, (max(0, sx), max(0, sy)))

    # ── z-5: FRONT TEXT (left, below back text) ────────────────────────────
    back_block_h = int(font_size * 0.88 * len(back_words))
    front_y = back_y + back_block_h + int(h * 0.03)

    canvas = _place_text_block(
        canvas, front_words,
        font_size=font_size,
        template=template,
        accent_word=accent_word,
        effect_profile=effect_profile,
        x_left=margin,
        y_top=front_y,
        max_width=text_zone_w - margin,
        glow=True,
        text_color=PALETTES[template]["accent"] if PALETTES[template].get("front_accent") else None,
    )

    if show_accent_bar and not PALETTES[template].get("minimal"):
        bar_y = front_y + int(font_size * 0.88 * len(front_words)) + int(h * 0.025)
        canvas = _add_accent_bar(
            canvas, x=margin, y=bar_y, width=int(w * 0.18), template=template
        )

    if show_frame:
        canvas = _add_frame_border(
            canvas, x0=margin // 2, y0=margin // 2,
            x1=text_zone_w, y1=h - margin // 2,
            template=template, opacity=0.15,
        )

    if show_arrow:
        arrow_y = front_y - int(font_size * 1.1)
        canvas = _add_arrow(canvas, x=margin, y=max(0, arrow_y),
                            font_size=int(font_size * 0.55), template=template,
                            effect_profile=effect_profile)

    if symbol:
        sym_size = int(h * 0.12)
        canvas = _add_symbol(canvas, symbol=symbol,
                             x=margin, y=h - int(h * 0.20),
                             size=sym_size)

    if show_logo:
        canvas = _add_logo(canvas, template=template, fmt="16x9")

    badge_config = _EFFECT_PROFILE_CONFIG[effect_profile].get("badge")
    resolved_badge_text = badge_text if badge_text is not None else (badge_config or {}).get("text")
    if resolved_badge_text:
        resolved_mode = badge_mode if badge_text is not None else (badge_config or {}).get("mode", "arc")
        resolved_position = badge_position if badge_text is not None else (badge_config or {}).get("position", "top_right")
        canvas = _add_path_badge(
            canvas,
            template=template,
            text=str(resolved_badge_text),
            mode=str(resolved_mode),
            position=str(resolved_position),
        )

    _LAST_LAYOUT_METADATA = {
        "fmt": "16x9",
        "composition": "side_speaker",
        "font_size": font_size,
        "text_bounds": [margin, back_y, max(1, text_zone_w - margin), min(h, front_y + int(font_size * 0.88 * max(1, len(front_words))))],
        "text_word_bounds": [],
        "face_box": None,
        "body_box": None,
        "face_overlap_ratio": 0.0,
    }
    return canvas


def _score_move_church_thumbnail(
    image: Image.Image,
    *,
    title: str,
    template: str,
    face_box: tuple[int, int, int, int] | None,
    coverage: float,
    provider_used: str | None,
) -> tuple[float, dict]:
    width, height = image.size
    layout = dict(_LAST_LAYOUT_METADATA or {})
    text_bounds = layout.get("text_bounds") or [int(width * 0.04), int(height * 0.04), int(width * 0.96), int(height * 0.90)]
    layout_face_box = layout.get("face_box")
    face_for_score = layout_face_box or (list(face_box) if face_box else None)
    clipping_penalty = 0.0
    if text_bounds:
        x1, y1, x2, y2 = [int(value) for value in text_bounds]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            clipping_penalty = 0.16
        clipped_area = max(0, min(width, x2) - max(0, x1)) * max(0, min(height, y2) - max(0, y1))
        original_area = max(1, (x2 - x1) * (y2 - y1))
        clipping_penalty = max(clipping_penalty, 1.0 - clipped_area / float(original_area))
    face_overlap_ratio = float(layout.get("face_overlap_ratio") or 0.0)
    face_collision_penalty = max(0.0, face_overlap_ratio - 0.15) * 1.8
    back_text_occlusion_ratio = float(layout.get("back_text_occlusion_ratio") or 0.0)
    back_text_max_occlusion_ratio = float(layout.get("back_text_max_occlusion_ratio") or 0.0)
    back_text_occlusion_penalty = (
        max(0.0, back_text_occlusion_ratio - 0.06) * 0.45
        + max(0.0, back_text_max_occlusion_ratio - 0.14) * 0.85
    )
    front_body_overlap_ratio = float(layout.get("front_body_overlap_ratio") or 0.0)
    front_face_overlap_ratio = float(layout.get("front_face_overlap_ratio") or 0.0)
    front_depth_penalty = max(0.0, 0.10 - front_body_overlap_ratio) * 0.75
    front_face_penalty = max(0.0, front_face_overlap_ratio - 0.03) * 1.20
    vertical_gap_ratio = float(layout.get("vertical_gap_ratio") or 0.0)
    composition_density_penalty = max(0.0, vertical_gap_ratio - 0.36) * 0.32
    metadata = {
        "variant": template,
        "hook_text": title,
        "subject_coverage": round(float(coverage or 0.0), 4),
        "face_box": face_for_score,
        "text_bounds": text_bounds,
        "depth_bonus": 0.08 if provider_used else 0.0,
        "keyword_visibility_bonus": 0.04,
        "edge_touch_penalty": 0.0 if provider_used else 0.08,
        "face_collision_penalty": round(face_collision_penalty, 4),
        "text_occlusion_penalty": round(clipping_penalty, 4),
        "back_text_occlusion_ratio": round(back_text_occlusion_ratio, 4),
        "back_text_max_occlusion_ratio": round(back_text_max_occlusion_ratio, 4),
        "back_text_occlusion_penalty": round(back_text_occlusion_penalty, 4),
        "front_body_overlap_ratio": round(front_body_overlap_ratio, 4),
        "front_face_overlap_ratio": round(front_face_overlap_ratio, 4),
        "front_depth_penalty": round(front_depth_penalty, 4),
        "front_face_penalty": round(front_face_penalty, 4),
        "vertical_gap_ratio": round(vertical_gap_ratio, 4),
        "composition_density_penalty": round(composition_density_penalty, 4),
    }
    raw_score = float(_score_variant(image, metadata))

    thumb = image.convert("RGB").resize((320, max(1, int(320 * height / width))), Image.Resampling.LANCZOS)
    scale = thumb.width / float(max(1, width))
    x1, y1, x2, y2 = [int(value) for value in text_bounds]
    tx1 = max(0, min(thumb.width - 1, int(x1 * scale)))
    ty1 = max(0, min(thumb.height - 1, int(y1 * scale)))
    tx2 = max(tx1 + 1, min(thumb.width, int(x2 * scale)))
    ty2 = max(ty1 + 1, min(thumb.height, int(y2 * scale)))
    gray = np.asarray(thumb.crop((tx1, ty1, tx2, ty2)).convert("L"), dtype=np.float32)
    readability = float(np.percentile(gray, 95) - np.percentile(gray, 10)) / 255.0
    score = (
        min(1.0, raw_score) * 0.88
        + min(1.0, readability) * 0.12
        - min(0.18, face_overlap_ratio * 0.28)
        - min(0.24, clipping_penalty * 0.8)
        - min(0.16, back_text_occlusion_penalty)
        - min(0.10, front_depth_penalty)
        - min(0.12, front_face_penalty)
        - min(0.07, composition_density_penalty)
    )
    score = round(max(0.0, min(0.99, float(score))), 4)
    metadata["raw_score_variant"] = round(raw_score, 4)
    metadata["readability_score"] = round(readability, 4)
    metadata["layout"] = layout
    metadata["clipping_penalty"] = round(float(clipping_penalty), 4)
    metadata["score"] = score
    return score, metadata


# ════════════════════════════════════════════════════════════════════════════
# MAIN API
# ════════════════════════════════════════════════════════════════════════════

def generate_move_church_thumbnail(
    source: str | np.ndarray | Image.Image,
    *,
    title_back: str = "",
    title_front: str = "",
    title: str = "",
    template: str = "navy_dark",
    fmt: str = "9x16",
    show_logo: bool = False,
    accent_word: str | None = None,
    symbol: str | None = None,
    show_accent_bar: bool = True,
    show_frame: bool = False,
    show_arrow: bool = False,
    show_decorations: bool = True,
    glow_intensity: float = 0.70,
    outline_preset: str = "palette_rim",
    effect_profile: str = "classic",
    badge_text: str | None = None,
    badge_mode: str = "arc",
    badge_position: str = "top_right",
    layer_specs: list[dict] | None = None,
    brief: dict | None = None,
    output_path: str | None = None,
    bg_removal_provider: str = "auto",
    relight: bool = False,
    font_size: int | None = None,
    _precomputed_subject: dict | None = None,
) -> Image.Image:
    """
    Generate a Move Church style thumbnail.

    Args:
        source:          Video path, image path, numpy BGR frame, or PIL Image.
        title_back:      Words to render BEHIND the speaker (z-3).
        title_front:     Words to render IN FRONT of the speaker (z-5).
        title:           Full title string — auto-split if title_back/front not given.
        template:        One of TEMPLATES: "navy_dark" | "energy_orange" | "warm_gold" |
                         "cinematic_dark" | "fire_red" | "heaven_blue" |
                         "bold_minimal" | "sunset_warm"
        fmt:             "9x16" (1080×1920) | "16x9" (1280×720)
        show_logo:       Show Move Church logo (bottom-left).
        accent_word:     Word to render in accent colour (orange/gold).
        symbol:          Symbol name or None (see SYMBOLS dict).
        show_accent_bar: Show orange brush-stroke accent line below front text.
        show_frame:      Show decorative border frame (no fill).
        show_arrow:      Show arrow element before front text.
        show_decorations: Render rings, glows, light rays.
        glow_intensity:  Glow/orb intensity 0.0–1.0.
        outline_preset:  "palette_rim" (template-colored rim light, default) or a
                         legacy stroke preset: "creator_white"|"creator_blue"|"sermon_gold"
        effect_profile:  "classic" | "editorial" | "premium" | "halo" | "poster"
        badge_text:      Optional path-text badge copied into the render.
        badge_mode:      "arc" or "circle" for badge rendering.
        badge_position:  Badge placement on the canvas.
        layer_specs:     Optional declarative layer stack for custom overlays.
        output_path:     Save PNG to this path (optional).
        bg_removal_provider: "auto" | "birefnet" | "rembg" | "grabcut_local"
        font_size:       Override font size in pixels.

    Returns:
        PIL Image (RGBA, size per fmt).
    """
    global _LAST_LAYOUT_METADATA
    _LAST_LAYOUT_METADATA = {}
    source_kind = "path"
    source_path = None
    if isinstance(source, np.ndarray):
        source_kind = "frame"
    elif isinstance(source, Image.Image):
        source_kind = "image"
    else:
        source_path = str(Path(source).expanduser().resolve(strict=False)) if isinstance(source, (str, Path)) else None

    if template not in PALETTES:
        raise ValueError(f"Unknown template '{template}'. Choose: {TEMPLATES}")
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Choose: {list(FORMATS)}")
    resolved_effect_profile, effect_profile_config = _resolve_effect_profile(effect_profile)

    # Auto-split title if explicit back/front not given
    if not title_back and not title_front:
        title_back, title_front = auto_split_title(title or "PREDIGT")

    back_words  = [w.strip() for w in title_back.upper().split() if w.strip()]
    front_words = [w.strip() for w in title_front.upper().split() if w.strip()]
    title_hint = " ".join([*back_words, *front_words]).strip() or title or "PREDIGT"

    W, H = FORMATS[fmt]
    fs = font_size or _word_font_size(fmt)

    # ── Load + extract speaker ──────────────────────────────────────────────
    frame_bgr = load_source(source)
    subject_info = _precomputed_subject or _extract_speaker_cutout(
        frame_bgr,
        bg_removal_provider=bg_removal_provider,
        relight=relight,
    )
    speaker_rgba = subject_info.get("speaker_rgba")
    face_box = subject_info.get("face_box")
    coverage = float(subject_info.get("coverage") or 0.0)
    provider_used = subject_info.get("provider_used")
    removal_attempts = list(subject_info.get("removal_attempts") or [])
    caption_cleanup = dict(subject_info.get("caption_cleanup") or {})

    palette = PALETTES[template]
    background_info: dict = {}

    # ── Rim light / outline (per template — rim color follows the palette) ──
    if speaker_rgba is not None:
        if outline_preset in OUTLINE_PRESETS:
            speaker_rgba = add_speaker_outline(speaker_rgba, preset_name=outline_preset)
        else:
            speaker_rgba = add_speaker_rim_light(
                speaker_rgba,
                rim_color=tuple(palette["glow_a"]),
                glow_color=tuple(palette["glow_a"]),
            )
        face_box = estimate_face_box(speaker_rgba) or face_box

    # ── Build canvas ────────────────────────────────────────────────────────
    # Non-minimal templates blend the real (blurred, darkened) frame into the
    # brand gradient — the stage light becomes atmosphere instead of an enemy.
    if palette.get("ai_background"):
        brief_data = dict(brief or {})
        canvas, background_info = _render_ai_background(
            title=title_hint,
            template=template,
            output_path=output_path,
            prompt=str(brief_data.get("background_prompt") or "").strip() or None,
            negative_prompt=str(brief_data.get("background_negative_prompt") or "").strip() or None,
            speaker_name=str(brief_data.get("speaker_name") or "").strip() or None,
            brand_label=str(brief_data.get("brand_label") or "").strip() or None,
            size=(W, H),
        )
        if canvas is None:
            canvas = _render_background(W, H, template)
    else:
        canvas = (
            _render_background(W, H, template)
            if palette.get("minimal") and speaker_rgba is not None
            else _render_darkened_frame_background(frame_bgr, W, H, template)
        )
    if show_decorations and not palette.get("minimal"):
        canvas = _add_glow_orbs(canvas, template, intensity=glow_intensity)
        canvas = _add_decorations(canvas, template, intensity=glow_intensity)
        canvas = _add_atmosphere(canvas, template, intensity=glow_intensity)

    # ── Compose layers ──────────────────────────────────────────────────────
    layout_fn = _layout_9x16 if fmt == "9x16" else _layout_16x9
    canvas = layout_fn(
        canvas,
        back_words=back_words,
        front_words=front_words,
        speaker_rgba=speaker_rgba,
        speaker_face_box=face_box,
        template=template,
        font_size=fs,
        accent_word=accent_word,
        show_logo=show_logo,
        show_accent_bar=show_accent_bar,
        show_frame=show_frame,
        show_arrow=show_arrow,
        symbol=symbol,
        effect_profile=resolved_effect_profile,
        badge_text=badge_text,
        badge_mode=badge_mode,
        badge_position=badge_position,
    )

    default_layer_text = " ".join([*back_words, *front_words]).strip() or title or ""
    canvas, profile_overlay_layers = _apply_profile_overlay_stack(
        canvas,
        template=template,
        effect_profile=resolved_effect_profile,
        face_box=face_box,
    )
    custom_layer_meta: list[dict] = []
    if layer_specs:
        canvas, custom_layer_meta = apply_custom_layer_stack(
            canvas,
            layer_specs,
            template=template,
            default_text=default_layer_text,
        )

    # Film grain finish (cinematic_dark)
    if palette.get("grain"):
        canvas = _add_film_grain(canvas)

    canvas = _apply_canvas_finish(
        canvas,
        effect_profile=resolved_effect_profile,
        template=template,
    )

    # ── Save ────────────────────────────────────────────────────────────────
    full_title = " ".join([*back_words, *front_words]).strip()
    render_layers = [
        {
            "name": "background",
            "kind": (
                "comfyui_background"
                if background_info.get("backend") == "comfyui"
                else "brand_plate"
                if palette.get("minimal") and speaker_rgba is not None
                else "frame_blend"
            ),
            "template": template,
            "decorations": bool(show_decorations and not palette.get("minimal")),
            "generation": background_info or None,
        },
        {
            "name": "speaker_cutout",
            "kind": "smart_layer",
            "provider": provider_used,
            "relight": bool(relight),
            "coverage": round(float(coverage or 0.0), 4),
        },
        {
            "name": "headline_back",
            "kind": "text",
            "profile": resolved_effect_profile,
        },
        {
            "name": "headline_front",
            "kind": "text",
            "profile": resolved_effect_profile,
        },
    ]
    render_layers.extend(profile_overlay_layers)
    render_layers.extend(custom_layer_meta)
    badge_cfg = effect_profile_config.get("badge")
    resolved_badge_text = badge_text if badge_text is not None else (badge_cfg or {}).get("text")
    if resolved_badge_text:
        render_layers.append(
            {
                "name": "badge",
                "kind": "text_path",
                "text": str(resolved_badge_text),
                "mode": badge_mode if badge_text is not None else (badge_cfg or {}).get("mode", "arc"),
                "position": badge_position if badge_text is not None else (badge_cfg or {}).get("position", "top_right"),
            }
        )
    render_layers.append(
        {
            "name": "finish",
            "kind": "adjustment_layer",
            "profile": resolved_effect_profile,
            "contrast": effect_profile_config["finish"].get("contrast"),
            "color": effect_profile_config["finish"].get("color"),
            "brightness": effect_profile_config["finish"].get("brightness"),
            "vignette": effect_profile_config["finish"].get("vignette"),
            "grain": effect_profile_config["finish"].get("grain"),
        }
    )
    score, score_metadata = _score_move_church_thumbnail(
        canvas,
        title=full_title,
        template=template,
        face_box=face_box,
        coverage=coverage,
        provider_used=provider_used,
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.convert("RGB").save(output_path, "PNG", optimize=True)
        print(f"[ThumbnailMoveChurch] Saved → {output_path}")
        report_path = Path(output_path).with_suffix(".thumbnail_report.json")
        score_metadata.update(dict(subject_info.get("subject_metrics") or {}))
        report = {
            "output": str(output_path),
            "title": full_title,
            "template": template,
            "format": fmt,
            "effect_profile": resolved_effect_profile,
            "score": score,
            "provider_used": provider_used,
            "source_kind": source_kind,
            "source_path": source_path,
            "brief": brief or {},
            "background_generation": background_info,
            "background_removal_attempts": removal_attempts,
            "caption_cleanup": caption_cleanup,
            "render_layers": render_layers,
            "frame_selection": _LAST_FRAME_SELECTION,
            "metrics": score_metadata,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return canvas


def generate_all_variants(
    source: str | np.ndarray | Image.Image,
    *,
    title: str,
    fmt: str = "9x16",
    show_logo: bool = False,
    symbol: str | None = None,
    output_dir: str = ".",
    brief: dict | None = None,
    **kwargs,
) -> dict[str, str]:
    """
    Generate all templates (see TEMPLATES) for a given source + title.
    Returns dict mapping template_name → file_path.
    """
    out: dict[str, str] = {}
    stem = Path(str(source)).stem if isinstance(source, str) else "thumbnail"
    output_root = Path(output_dir)
    variant_root = output_root / "thumbnail_variants"
    variant_root.mkdir(parents=True, exist_ok=True)
    variants_report: list[dict] = []
    frame_bgr = load_source(source)
    frame_candidates = list(_LAST_TOP_FRAME_CANDIDATES) or [{"metadata": dict(_LAST_FRAME_SELECTION.get("selected") or {}), "frame": frame_bgr}]
    for frame_index, frame_item in enumerate(frame_candidates[:3], start=1):
        candidate_frame = frame_item["frame"]
        subject_info = _extract_speaker_cutout(
            candidate_frame,
            bg_removal_provider=str(kwargs.get("bg_removal_provider") or "auto"),
            relight=bool(kwargs.get("relight")),
        )
        for tmpl in TEMPLATES:
            variant_name = f"{stem}_frame{frame_index:02d}_{tmpl}_{fmt}.png"
            path = str(variant_root / variant_name)
            generate_move_church_thumbnail(
                candidate_frame,
                title=title,
                template=tmpl,
                fmt=fmt,
                show_logo=show_logo,
                symbol=symbol,
                output_path=path,
                brief=brief,
                _precomputed_subject=subject_info,
                **kwargs,
            )
            report_path = Path(path).with_suffix(".thumbnail_report.json")
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report["frame_variant_index"] = frame_index
                    report["frame_variant_metadata"] = frame_item.get("metadata") or {}
                    frame_score = float((frame_item.get("metadata") or {}).get("score") or 0.0)
                    frame_quality = min(1.0, max(0.0, frame_score / 520.0))
                    rejected_reason = str((frame_item.get("metadata") or {}).get("rejected_reason") or "")
                    selection_score = float(report.get("score") or 0.0) + frame_quality * 0.035
                    if rejected_reason:
                        selection_score -= 0.025
                    report["selection_score"] = round(selection_score, 4)
                    variants_report.append(report)
                except Exception:
                    pass
            out[f"frame{frame_index}_{tmpl}"] = path
    if variants_report:
        for tmpl in TEMPLATES:
            template_reports = [item for item in variants_report if item.get("template") == tmpl]
            if not template_reports:
                continue
            template_best = max(template_reports, key=lambda item: float(item.get("selection_score", item.get("score") or 0.0)))
            source_path = Path(str(template_best.get("output") or ""))
            compat_path = output_root / f"{stem}_{tmpl}_{fmt}.png"
            if source_path.exists():
                shutil.copy2(source_path, compat_path)
                out[tmpl] = str(compat_path)

        best = max(variants_report, key=lambda item: float(item.get("selection_score", item.get("score") or 0.0)))
        best_source = Path(str(best.get("output") or ""))
        best_path = output_root / "thumbnail_best.png"
        if best_source.exists():
            shutil.copy2(best_source, best_path)
            out["best"] = str(best_path)
        best_metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
        best_layout = best_metrics.get("layout") if isinstance(best_metrics.get("layout"), dict) else {}
        aggregate = {
            "selected": str(best_path),
            "selected_template": best.get("template"),
            "selected_score": best.get("score"),
            "selected_selection_score": best.get("selection_score"),
            "selected_provider": best.get("provider_used"),
            "selected_quality_score": best_metrics.get("score"),
            "selected_readability_320": best_metrics.get("readability_score"),
            "selected_face_overlap_ratio": best_layout.get("face_overlap_ratio"),
            "selected_back_occlusion_ratio": best_metrics.get("back_text_occlusion_ratio"),
            "selected_front_body_overlap_ratio": best_metrics.get("front_body_overlap_ratio"),
            "selected_caption_cleanup": best.get("caption_cleanup"),
            "selected_variant": best,
            "title": title,
            "format": fmt,
            "frame_selection": _LAST_FRAME_SELECTION,
            "variants_dir": str(variant_root),
            "variants": variants_report,
        }
        (output_root / "thumbnail_report.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Move Church Thumbnail Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Video (.mp4/.mov) or image (.jpg/.png) path")
    parser.add_argument("--title",
                        help='Full title (auto-split). E.g. "EINE WIE KEINE"')
    parser.add_argument("--back",
                        help='Words behind speaker. E.g. "EINE WIE"')
    parser.add_argument("--front",
                        help='Words in front of speaker. E.g. "KEINE"')
    parser.add_argument("--template", default="navy_dark",
                        choices=TEMPLATES, help="Visual template")
    parser.add_argument("--fmt", default="9x16",
                        choices=list(FORMATS), help="Output format")
    parser.add_argument("--output", default="thumbnail.png",
                        help="Output PNG path")
    parser.add_argument("--logo", action="store_true",
                        help="Show Move Church logo")
    parser.add_argument("--accent", default=None,
                        help="Accent word (rendered in orange/gold)")
    parser.add_argument("--symbol", default=None,
                        choices=list(SYMBOLS) + [None],
                        help="Optional symbol")
    parser.add_argument("--frame", action="store_true",
                        help="Add decorative border frame")
    parser.add_argument("--arrow", action="store_true",
                        help="Add arrow element")
    parser.add_argument("--all-templates", action="store_true",
                        help="Generate all 3 template variants")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "birefnet", "rmbg", "rembg", "grabcut_local"],
                        help="Background removal provider")
    parser.add_argument("--outline", default="palette_rim",
                        choices=["palette_rim", "creator_white", "creator_blue", "sermon_gold"],
                        help="Speaker outline preset")
    parser.add_argument("--effect-profile", default="classic",
                        choices=EFFECT_PROFILES,
                        help="High-end render preset")
    parser.add_argument("--badge-text", default=None,
                        help="Optionaler Badge-Text für Path-/Arc-Text")
    parser.add_argument("--badge-mode", default="arc",
                        choices=["arc", "circle"],
                        help="Badge-Text als Bogen oder Kreis")
    parser.add_argument("--badge-position", default="top_right",
                        choices=["top_left", "top_right", "upper_left", "bottom_left"],
                        help="Badge-Position auf dem Thumbnail")
    parser.add_argument("--layers-json", default=None,
                        help="JSON string or @path to a list of custom layer specs")
    parser.add_argument("--timestamp", type=float, default=None,
                        help="Preferred video timestamp in seconds for frame extraction")

    args = parser.parse_args()
    layer_specs = None
    if args.layers_json:
        raw = args.layers_json.strip()
        payload = Path(raw[1:]).read_text(encoding="utf-8") if raw.startswith("@") else Path(raw).read_text(encoding="utf-8") if Path(raw).exists() else raw
        layer_specs = json.loads(payload)
        if not isinstance(layer_specs, list):
            raise ValueError("--layers-json must decode to a list of layer dicts")

    # Handle video timestamp
    source = args.source
    if args.timestamp is not None and Path(source).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        print(f"[CLI] Extracting frame at {args.timestamp}s …")
        source = extract_best_frame(source, prefer_timestamp=args.timestamp)

    if args.all_templates:
        results = generate_all_variants(
            source,
            title=args.title or f"{args.back or ''} {args.front or ''}".strip(),
            fmt=args.fmt,
            show_logo=args.logo,
            symbol=args.symbol,
            output_dir=str(Path(args.output).parent),
            bg_removal_provider=args.provider,
            outline_preset=args.outline,
            effect_profile=args.effect_profile,
            badge_text=args.badge_text,
            badge_mode=args.badge_mode,
            badge_position=args.badge_position,
            layer_specs=layer_specs,
        )
        for tmpl, path in results.items():
            print(f"  {tmpl:20s} → {path}")
    else:
        generate_move_church_thumbnail(
            source,
            title=args.title,
            title_back=args.back or "",
            title_front=args.front or "",
            template=args.template,
            fmt=args.fmt,
            show_logo=args.logo,
            accent_word=args.accent,
            symbol=args.symbol,
            show_frame=args.frame,
            show_arrow=args.arrow,
            output_path=args.output,
            bg_removal_provider=args.provider,
            outline_preset=args.outline,
            effect_profile=args.effect_profile,
            badge_text=args.badge_text,
            badge_mode=args.badge_mode,
            badge_position=args.badge_position,
            layer_specs=layer_specs,
        )


if __name__ == "__main__":
    _cli()
