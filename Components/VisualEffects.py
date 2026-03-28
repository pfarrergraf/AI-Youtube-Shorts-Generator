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
# TUNING GUIDE (important):
#
# This file is the safest central place to tweak "look" in production.
# The values below directly feed FFmpeg filters and are applied in a single
# NVENC pass after subtitles.
#
# 1) eq=contrast=...,brightness=...,saturation=...
#    - contrast:
#      * normal usable range: 0.90 .. 1.20
#      * <1.0 = flatter image, >1.0 = punchier image
#      * above ~1.22 often clips highlights / loses shadow detail
#    - brightness:
#      * normal usable range: -0.04 .. +0.05
#      * positive lifts entire image, negative darkens globally
#      * keep small; large values wash out skin or crush blacks
#    - saturation:
#      * normal usable range: 0.85 .. 1.18
#      * 1.0 = unchanged
#      * >1.15 can look synthetic on skin tones
#
# 2) colorbalance=...
#    FFmpeg channels:
#      rs/gs/bs = shadow tint (red/green/blue in dark regions)
#      rm/gm/bm = midtone tint
#      rh/gh/bh = highlight tint
#    Typical per-channel micro-adjustment range: -0.08 .. +0.08
#    Practical target for sermon footage: mostly -0.05 .. +0.06
#
# 3) VIGNETTE + GRAIN
#    - vignette helps focus attention to center, but too strong can make
#      subtitles harder to read near lower thirds.
#    - grain adds "texture" but can hurt codec efficiency and subtitle clarity.
#
# Each preset is a dict with:
#   "vf"   — FFmpeg video-filter expression to append (comma-joined)
#   "desc" — Human-readable description

STYLE_PRESETS = {
    "clean": {
        "desc": "No colour grading — clean, neutral look",
        # Keep empty for passthrough. This is your baseline sanity check style.
        "vf": "",
    },
    "warm": {
        "desc": "Warm, inviting tones — slightly boosted reds/yellows",
        # Warm recipe:
        # - brightness +0.03: gentle lift for faces
        # - saturation 1.12: modest color boost
        # - colorbalance: small warm push in shadows/mids
        "vf": (
            "eq=brightness=0.03:saturation=1.12,"
            "colorbalance=rs=0.06:gs=0.02:bs=-0.04:rm=0.04:gm=0.01:bm=-0.03"
        ),
    },
    "cool": {
        "desc": "Cool, modern tones — lifted shadows, slight blue cast",
        # Cool recipe:
        # - small brightness/saturation lift to keep image alive
        # - colorbalance shifts toward blue in shadows/mids
        "vf": (
            "eq=brightness=0.02:saturation=1.05,"
            "colorbalance=rs=-0.03:gs=0.0:bs=0.06:rm=-0.02:gm=0.0:bm=0.04"
        ),
    },
    "cinematic": {
        "desc": "Cinematic — contrast boost, desaturated shadows, warm highlights",
        # Cinematic recipe:
        # - contrast 1.10 for depth
        # - tiny negative brightness to preserve highlight headroom
        # - warm highlight tint so skin doesn't look sterile
        "vf": (
            "eq=contrast=1.10:brightness=-0.01:saturation=1.08,"
            "colorbalance=rs=0.04:gs=0.01:bs=-0.03:rh=0.05:gh=0.02:bh=-0.02"
        ),
    },
    "vintage": {
        "desc": "Vintage — faded blacks, slight sepia, reduced contrast",
        # Vintage recipe:
        # - contrast below 1.0 for softer tonal separation
        # - lower saturation for aged look
        # - warm sepia-ish channel bias
        "vf": (
            "eq=contrast=0.92:brightness=0.04:saturation=0.85,"
            "colorbalance=rs=0.07:gs=0.03:bs=-0.05:rm=0.04:gm=0.02:bm=-0.02"
        ),
    },
    "high_contrast": {
        "desc": "Punchy — elevated contrast, vivid colours",
        # Use carefully for fast clips; this can be too aggressive on faces
        # if source already has high contrast.
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
    # Unknown style names fail-safe to "clean" so batch runs never crash.
    preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["clean"])
    parts = []

    if preset["vf"]:
        parts.append(preset["vf"])
    if vignette:
        # Keep vignette subtle in shorts: stronger values can darken subtitle
        # lanes and clash with lower-third readability.
        parts.append(VIGNETTE_FILTER)
    if grain:
        # Grain is intentionally light; heavy noise hurts compression quality.
        parts.append(LIGHT_GRAIN_FILTER)

    return ",".join(parts)
