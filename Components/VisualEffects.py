"""Visual effect presets for short-form video clips.

Each preset is a bundle of FFmpeg filter expressions that are chained
*after* the subtitle burn step.  The filters are applied in a single
encoding pass (GPU-accelerated via NVENC).

Usage:
    from Components.VisualEffects import random_style, get_ffmpeg_filters
    style = random_style()
    extra_vf = get_ffmpeg_filters(style)
    # chain with subtitles:  f"subtitles={f},{extra_vf}"
"""

import random

# ── Style presets ────────────────────────────────────────────────
# Each preset is a dict with:
#   "vf"   — FFmpeg video-filter expression to append (comma-joined)
#   "desc" — Human-readable description
#
# All colour adjustments are subtle to keep sermon content readable.
# Vignette uses FFmpeg's built-in filter; film grain uses geq noise.

STYLE_PRESETS = {
    "clean": {
        "desc": "No colour grading — clean, neutral look",
        "vf": "",
    },
    "warm": {
        "desc": "Warm, inviting tones — slightly boosted reds/yellows",
        "vf": (
            "eq=brightness=0.03:saturation=1.12,"
            "colorbalance=rs=0.06:gs=0.02:bs=-0.04:rm=0.04:gm=0.01:bm=-0.03"
        ),
    },
    "cool": {
        "desc": "Cool, modern tones — lifted shadows, slight blue cast",
        "vf": (
            "eq=brightness=0.02:saturation=1.05,"
            "colorbalance=rs=-0.03:gs=0.0:bs=0.06:rm=-0.02:gm=0.0:bm=0.04"
        ),
    },
    "cinematic": {
        "desc": "Cinematic — contrast boost, desaturated shadows, warm highlights",
        "vf": (
            "eq=contrast=1.10:brightness=-0.01:saturation=1.08,"
            "colorbalance=rs=0.04:gs=0.01:bs=-0.03:rh=0.05:gh=0.02:bh=-0.02"
        ),
    },
    "vintage": {
        "desc": "Vintage — faded blacks, slight sepia, reduced contrast",
        "vf": (
            "eq=contrast=0.92:brightness=0.04:saturation=0.85,"
            "colorbalance=rs=0.07:gs=0.03:bs=-0.05:rm=0.04:gm=0.02:bm=-0.02"
        ),
    },
    "high_contrast": {
        "desc": "Punchy — elevated contrast, vivid colours",
        "vf": (
            "eq=contrast=1.18:brightness=-0.02:saturation=1.15"
        ),
    },
}

# Which presets are eligible for random selection.
# "clean" is intentionally included — not every short needs grading.
_RANDOM_POOL = ["clean", "warm", "cool", "cinematic", "vintage", "high_contrast"]

# Overlay fragments that can be combined with any colour grade.
VIGNETTE_FILTER = "vignette=PI/4"
LIGHT_GRAIN_FILTER = "noise=alls=6:allf=t"  # subtle temporal noise


def random_style():
    """Return a random style preset name from the pool."""
    return random.choice(_RANDOM_POOL)


def get_ffmpeg_filters(style_name, vignette=True, grain=False):
    """Build a complete FFmpeg ``-vf`` fragment for the given style.

    Parameters
    ----------
    style_name : str
        Key in STYLE_PRESETS (e.g. ``"cinematic"``).
    vignette : bool
        Whether to add a subtle vignette darkening around the edges.
    grain : bool
        Whether to add light film grain.

    Returns
    -------
    str
        A comma-separated FFmpeg filter expression ready to append after
        ``subtitles=...``.  Returns ``""`` if nothing to add.
    """
    preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["clean"])
    parts = []

    if preset["vf"]:
        parts.append(preset["vf"])
    if vignette:
        parts.append(VIGNETTE_FILTER)
    if grain:
        parts.append(LIGHT_GRAIN_FILTER)

    return ",".join(parts)
