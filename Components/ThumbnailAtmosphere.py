"""
Light and atmosphere for the "epic" thumbnail look. Pure numpy/Pillow.
=====================================================================

The measured gap between our renders and the references is not colour, it is
*dynamic range*. References sit at roughly 45% near-black and 11% blown-out
white — real chiaroscuro. Our legacy renders are ~72% mid-dark with no
highlight at all, because ``_apply_canvas_finish`` applies Brightness 0.94,
Color 0.90, a 28-step vignette and a gradient blend, which between them make a
true white impossible by construction.

So this module does the opposite of that finish pass: it *adds* one light
source, blooms it, and only then sets the black point. Nothing here needs a GPU
or a running server.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Light colours matching the reference palette.
LIGHT_COLORS: dict[str, tuple[int, int, int]] = {
    "warm": (255, 214, 150),
    "gold": (255, 190, 90),
    "cool": (170, 220, 255),
    "cyan": (120, 240, 240),
    "white": (255, 255, 255),
    "red": (255, 90, 70),
    "magenta": (255, 120, 200),
}


def _as_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr.clip(0.0, 1.0) * 255.0).astype(np.uint8), "RGB")


def _screen(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - layer)


# ────────────────────────────────────────────────────────────────────────────
# Light sources
# ────────────────────────────────────────────────────────────────────────────

def back_glow(
    size: tuple[int, int],
    *,
    center: tuple[float, float] = (0.5, 0.42),
    radius: float = 0.42,
    color: tuple[int, int, int] | str = "warm",
    power: float = 2.2,
    intensity: float = 1.0,
) -> np.ndarray:
    """Radial hot halo, as an additive RGB layer in [0, 1].

    This is the single light source the composition is built around — the thing
    behind the speaker's head in almost every reference.
    """
    w, h = size
    rgb = np.asarray(LIGHT_COLORS.get(color, color) if isinstance(color, str) else color, dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx / w - center[0]) / max(1e-6, radius)
    dy = (yy / h - center[1]) / max(1e-6, radius * (w / h))
    dist = np.sqrt(dx * dx + dy * dy)
    falloff = np.clip(1.0 - dist, 0.0, 1.0) ** power
    return (falloff[..., None] * rgb[None, None, :]) * intensity


def god_rays(
    size: tuple[int, int],
    *,
    origin: tuple[float, float] = (0.5, 0.10),
    color: tuple[int, int, int] | str = "warm",
    n_rays: int = 13,
    spread: float = 0.62,
    length: float = 1.05,
    jitter: float = 0.45,
    intensity: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Bundle of volumetric shafts radiating from `origin`.

    Seed-stable: the same seed always yields the same ray pattern, so a render
    is reproducible.
    """
    w, h = size
    rgb = np.asarray(LIGHT_COLORS.get(color, color) if isinstance(color, str) else color, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ox, oy = origin[0] * w, origin[1] * h
    dx, dy = xx - ox, yy - oy
    dist = np.sqrt(dx * dx + dy * dy) / float(max(w, h))
    angle = np.arctan2(dy, dx)

    acc = np.zeros((h, w), dtype=np.float32)
    base = np.pi / 2.0
    for i in range(n_rays):
        t = (i / max(1, n_rays - 1)) * 2.0 - 1.0
        theta = base + t * spread + float(rng.normal(0.0, 0.05 * jitter))
        width = 0.020 + 0.055 * float(rng.random()) * (0.4 + jitter)
        weight = 0.45 + 0.55 * float(rng.random())
        delta = np.abs(np.angle(np.exp(1j * (angle - theta))))
        acc += weight * np.exp(-(delta ** 2) / (2.0 * width * width))

    acc *= np.clip(1.0 - dist / max(1e-6, length), 0.0, 1.0) ** 1.6
    acc /= max(1e-6, acc.max())
    acc = np.asarray(
        Image.fromarray((acc * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(2, w // 220))),
        dtype=np.float32,
    ) / 255.0
    return (acc[..., None] * rgb[None, None, :]) * intensity


def light_shaft(
    size: tuple[int, int],
    *,
    apex: tuple[float, float] = (0.5, 0.0),
    base_width: float = 0.55,
    color: tuple[int, int, int] | str = "cool",
    softness: float = 0.16,
    intensity: float = 0.9,
) -> np.ndarray:
    """Single wedge of light widening downward — the doorway in "KEINE ANGST"."""
    w, h = size
    rgb = np.asarray(LIGHT_COLORS.get(color, color) if isinstance(color, str) else color, dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ax, ay = apex[0] * w, apex[1] * h
    depth = np.clip((yy - ay) / max(1.0, h - ay), 0.0, 1.0)
    half = (depth * base_width * w) / 2.0 + 1.0
    offset = np.abs(xx - ax) / half
    core = np.clip(1.0 - offset, 0.0, 1.0)
    shaft = core ** (1.0 / max(1e-6, softness)) if softness < 1 else core
    shaft = np.clip(shaft + core * 0.35, 0.0, 1.0) * (1.0 - depth * 0.35)
    return (shaft[..., None] * rgb[None, None, :]) * intensity


def atmospheric_haze(
    size: tuple[int, int],
    *,
    color: tuple[int, int, int] | str = "warm",
    density: float = 0.10,
    seed: int = 0,
) -> np.ndarray:
    """Low-frequency volumetric haze so the light has something to travel through."""
    w, h = size
    rgb = np.asarray(LIGHT_COLORS.get(color, color) if isinstance(color, str) else color, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, h // 90), max(2, w // 90))).astype(np.float32)
    field = np.asarray(
        Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    return (field[..., None] * rgb[None, None, :]) * density


# ────────────────────────────────────────────────────────────────────────────
# Subject treatment
# ────────────────────────────────────────────────────────────────────────────

def rim_light_from_alpha(
    subject_rgba: Image.Image,
    *,
    color: tuple[int, int, int] | str = "warm",
    direction: tuple[float, float] = (0.0, -1.0),
    width: int = 5,
    intensity: float = 0.85,
    feather: int = 3,
) -> Image.Image:
    """Directional edge light derived from the subject's own alpha.

    Narrower and harder than the legacy ``add_speaker_rim_light``, which paints
    a wide soft glow that reads as a matting halo rather than as a light.
    """
    rgba = subject_rgba.convert("RGBA")
    alpha = rgba.split()[-1]
    a = np.asarray(alpha, dtype=np.float32) / 255.0

    shifted = np.roll(a, int(round(direction[1] * width)), axis=0)
    shifted = np.roll(shifted, int(round(direction[0] * width)), axis=1)
    edge = np.clip(a - shifted, 0.0, 1.0)

    edge_img = Image.fromarray((edge * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(1, feather))
    )
    edge = np.asarray(edge_img, dtype=np.float32) / 255.0
    edge *= (a > 0.5)  # keep the light on the subject, never spilling outside

    rgb = np.asarray(LIGHT_COLORS.get(color, color) if isinstance(color, str) else color, dtype=np.float32) / 255.0
    base = np.asarray(rgba, dtype=np.float32) / 255.0
    lit = _screen(base[..., :3], edge[..., None] * rgb[None, None, :] * intensity)

    out = np.dstack([lit, base[..., 3:4]])
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8), "RGBA")


# ────────────────────────────────────────────────────────────────────────────
# Finishing
# ────────────────────────────────────────────────────────────────────────────

def bloom(image: Image.Image, *, threshold: float = 0.78, radius: int = 60, strength: float = 0.55) -> Image.Image:
    """Bleed the highlights. This is what produces the missing hot pixels.

    Extracts everything above `threshold`, blurs it wide, and screens it back.
    """
    arr = _as_float(image)
    luma = arr.mean(axis=2)
    mask = np.clip((luma - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    highlights = arr * mask[..., None]
    blurred = np.asarray(
        _to_image(highlights).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    ) / 255.0
    return _to_image(_screen(arr, blurred * strength))


def vignette(image: Image.Image, *, strength: float = 0.35, radius: float = 0.95, power: float = 1.8) -> Image.Image:
    """Gentle corner falloff. Deliberately far milder than the legacy 28-step ramp."""
    arr = _as_float(image)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx / w - 0.5) * 2.0
    dy = (yy / h - 0.5) * 2.0
    dist = np.sqrt(dx * dx + dy * dy) / max(1e-6, radius)
    falloff = 1.0 - strength * np.clip(dist, 0.0, 1.0) ** power
    return _to_image(arr * falloff[..., None])


def film_grain(image: Image.Image, *, opacity: float = 0.035, seed: int = 0) -> Image.Image:
    arr = _as_float(image)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    return _to_image(arr + noise[..., None] * opacity)


def cinematic_finish(
    image: Image.Image,
    *,
    black_point: float = 0.045,
    white_point: float = 0.985,
    contrast: float = 1.10,
    saturation: float = 1.20,
) -> Image.Image:
    """Set the black point and stretch to a true white.

    The inverse of the legacy finish: it lifts the top end instead of pulling it
    down, which is the only way `peak_luma` can reach the reference band.
    """
    arr = _as_float(image)
    lo = np.percentile(arr, black_point * 100.0)
    hi = np.percentile(arr, white_point * 100.0)
    arr = (arr - lo) / max(1e-6, hi - lo)
    arr = np.clip(arr, 0.0, 1.0)

    arr = np.clip((arr - 0.5) * contrast + 0.5, 0.0, 1.0)

    grey = arr.mean(axis=2, keepdims=True)
    arr = np.clip(grey + (arr - grey) * saturation, 0.0, 1.0)
    return _to_image(arr)


def apply_light_stack(
    canvas: Image.Image,
    layers: list[np.ndarray],
) -> Image.Image:
    """Screen a list of additive light layers onto the canvas in order."""
    arr = _as_float(canvas)
    for layer in layers:
        arr = _screen(arr, np.clip(layer, 0.0, 1.0))
    return _to_image(arr)
