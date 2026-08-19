"""
Reference-look typography for 9:16 thumbnails.
==============================================

What separates the reference thumbnails from the legacy renderer is not the
typeface, it is three measurable things:

1. **Every line is scaled independently to fill the measure.** "DAS" (3 glyphs)
   is set much larger than "PAKET" (5 glyphs) so both span the same width. The
   legacy renderer clamps per-word growth to 2.60x, so a short word can never
   reach the full column.
2. **Leading is driven by ink, not by point size.** Lines almost touch: measured
   on reference 5, the gap between lines is 4-5% of a line's ink height. Point-
   size leading always leaves the font's internal ascender/descender air in the
   gap, which makes a stack that tight impossible.
3. **No contour, no extrude, no offset shadow.** Just a soft dark bloom behind
   the glyph. The legacy hard black contour eats the bright glyph edge, which is
   why our renders measure zero near-white pixels in the upper half.

The title is always rendered here, deterministically, with Pillow — never by an
image model. Umlauts and ``ß`` must survive verbatim; see
:func:`Components.ThumbnailReferenceGate.validate_exact_spelling`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from Components.ThumbnailReferenceGate import safe_upper

_FONT_DIR = Path.home() / ".local/share/fonts/mc_thumbnails"

# Upright heavy condensed faces only. The legacy renderer points all eight
# templates at BarlowCondensed-ExtraBoldItalic; the references are never italic.
FONTS: dict[str, str] = {
    "anton": "anton/Anton-Regular.ttf",
    "barlow_black": "barlow-condensed/BarlowCondensed-Black.ttf",
    "bebas": "bebas-neue/BebasNeue-Regular.ttf",
    "oswald_bold": "oswald/Oswald-Bold.ttf",
}
DEFAULT_FONT = "anton"

# Accent colours seen in the references: red, yellow, cyan.
ACCENT_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 34, 24),
    "yellow": (255, 205, 20),
    "cyan": (34, 226, 226),
    "white": (255, 255, 255),
}

MAX_LINES = 4
# Largest allowed ratio between the biggest and smallest line on one thumbnail.
# Measured on reference 5 (DEINE/GABE/WARTET) on a clean dark background, where
# per-line ink can be read reliably: caps 283/299/196 px, a 1.53x spread.
MAX_SIZE_SPREAD = 1.9

# Upper bound on mean cap height as a fraction of frame height. The references
# top out at 0.163; without this a 2- or 3-line hook fills the height budget and
# overshoots the calibrated band.
MAX_CAP_RATIO = 0.152

# Gap between two lines as a fraction of cap height. Measured on reference 5:
# ink heights 283/299/196 px separated by gaps of 11 and 13 px, i.e. 4-5%.
# Note this is a *gap*, added to each line's ink height — treating it as a
# multiplier on cap height instead makes every line collide with the next,
# because an all-caps line's ink height already is one cap.
LINE_GAP_RATIO = 0.045
_MIN_FONT_PX = 24
_MAX_FONT_PX = 900


@dataclass
class RenderedLine:
    text: str
    font_size: int
    cap_height: float
    ink_width: int
    fill_ratio: float
    is_accent: bool
    box: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class TypeLayout:
    image: Image.Image
    alpha: Image.Image
    lines: list[RenderedLine] = field(default_factory=list)
    block_box: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def texts(self) -> list[str]:
        return [ln.text for ln in self.lines]

    @property
    def mean_cap_height(self) -> float:
        return float(np.mean([ln.cap_height for ln in self.lines])) if self.lines else 0.0

    @property
    def min_fill_ratio(self) -> float:
        return float(min((ln.fill_ratio for ln in self.lines), default=0.0))

    def metrics(self) -> dict:
        return {
            "n_lines": len(self.lines),
            "mean_cap_height_px": round(self.mean_cap_height, 2),
            "min_fill_ratio": round(self.min_fill_ratio, 4),
            "fill_ratios": [round(ln.fill_ratio, 4) for ln in self.lines],
            "font_sizes": [ln.font_size for ln in self.lines],
            "block_box": list(self.block_box),
        }


# ────────────────────────────────────────────────────────────────────────────
# Font handling
# ────────────────────────────────────────────────────────────────────────────

def resolve_font_path(name: str = DEFAULT_FONT) -> Path:
    rel = FONTS.get(name)
    if rel is None:
        raise ValueError(f"unknown font {name!r} (known: {sorted(FONTS)})")
    path = _FONT_DIR / rel
    if not path.exists():
        for fallback in FONTS.values():
            candidate = _FONT_DIR / fallback
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no headline font found under {_FONT_DIR}")
    return path


_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    key = (str(path), int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(path), int(size))
    return _font_cache[key]


def cap_height_of(font: ImageFont.FreeTypeFont) -> float:
    """Cap height in px, measured from the actual glyph rather than the size."""
    x0, y0, x1, y1 = font.getbbox("H")
    return float(y1 - y0)


def _tracking_px(font_size: int, text: str) -> float:
    """Optical letter spacing. Large sizes read looser, so tighten them.

    Negative tracking at display sizes is what keeps a huge line from falling
    apart into separate letters.
    """
    base = font_size * -0.012
    if len(text) <= 4:
        base += font_size * 0.004
    return base


def _measure_tracked(text: str, font: ImageFont.FreeTypeFont, tracking: float) -> tuple[int, int, int, int]:
    """Ink bounding box of `text` drawn with manual tracking, origin at (0, 0)."""
    x = 0.0
    x0_min, y0_min, x1_max, y1_max = None, None, None, None
    for ch in text:
        bbox = font.getbbox(ch)
        if bbox[2] > bbox[0]:  # non-empty glyph
            cx0, cy0, cx1, cy1 = bbox[0] + x, bbox[1], bbox[2] + x, bbox[3]
            x0_min = cx0 if x0_min is None else min(x0_min, cx0)
            y0_min = cy0 if y0_min is None else min(y0_min, cy0)
            x1_max = cx1 if x1_max is None else max(x1_max, cx1)
            y1_max = cy1 if y1_max is None else max(y1_max, cy1)
        x += font.getlength(ch) + tracking
    if x0_min is None:
        return (0, 0, 0, 0)
    return (int(x0_min), int(y0_min), int(np.ceil(x1_max)), int(np.ceil(y1_max)))


def _draw_tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    tracking: float,
) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking


def _size_to_fill(text: str, font_path: Path, measure_px: int) -> int:
    """Largest point size whose *ink* width fits the measure.

    Binary search on ink extent, not on advance width — trailing side bearings
    would otherwise leave the line visibly short of the measure.
    """
    lo, hi = _MIN_FONT_PX, _MAX_FONT_PX
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _font(font_path, mid)
        x0, _, x1, _ = _measure_tracked(text, font, _tracking_px(mid, text))
        if (x1 - x0) <= measure_px:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ────────────────────────────────────────────────────────────────────────────
# Line breaking
# ────────────────────────────────────────────────────────────────────────────

def _partitions(words: list[str], n_lines: int) -> list[list[str]]:
    """All ways to split `words` into `n_lines` contiguous, non-empty groups."""
    n = len(words)
    if n_lines < 1 or n_lines > n:
        return []
    out = []
    for cuts in itertools.combinations(range(1, n), n_lines - 1):
        bounds = (0, *cuts, n)
        out.append([" ".join(words[bounds[i]:bounds[i + 1]]) for i in range(n_lines)])
    return out


def choose_lines(
    hook: str,
    *,
    font_path: Path,
    measure_px: int,
    max_block_h: int,
    line_gap: float,
    max_cap_px: float,
    max_size_spread: float = MAX_SIZE_SPREAD,
) -> tuple[list[str], list[int]]:
    """Pick the line break that makes the type as large as possible.

    Every candidate partition is costed by the mean cap height it achieves once
    the block has been shrunk to fit `max_block_h`. More lines means each line
    is narrower and therefore set larger, but the block grows taller.

    The spread constraint is what keeps the result looking like the references.
    Since every line is scaled to fill the measure, a partition that puts a
    3-glyph line next to an 11-glyph one sets them at a 3x size difference —
    "DAS" / "VOLLE PAKET" was the first thing this engine produced. Measured on
    the references, per-line width lands at 0.90-1.00 of the measure with a size
    spread never worse than about 1.6x, because their partitions are balanced.
    Constraining spread and then maximising cap height reproduces that.
    """
    words = [w for w in safe_upper(hook).split() if w]
    if not words:
        return [], []

    scored: list[tuple[float, float, list[str], list[int]]] = []
    for n_lines in range(1, min(MAX_LINES, len(words)) + 1):
        for candidate in _partitions(words, n_lines):
            sizes = [_size_to_fill(line, font_path, measure_px) for line in candidate]
            caps = [cap_height_of(_font(font_path, s)) for s in sizes]
            natural_cap = float(np.mean(caps))
            block_h = sum(caps) + line_gap * natural_cap * (len(caps) - 1)
            scale = 1.0
            if block_h > 0:
                scale = min(scale, max_block_h / block_h)
            if natural_cap > 0:
                scale = min(scale, max_cap_px / natural_cap)
            mean_cap = natural_cap * scale
            spread = max(sizes) / max(1, min(sizes))
            scaled = [max(_MIN_FONT_PX, int(s * scale)) for s in sizes]
            scored.append((spread, mean_cap, candidate, scaled))

    balanced = [s for s in scored if s[0] <= max_size_spread]
    pool = balanced or [min(scored, key=lambda s: s[0])]
    # Once cap height is clamped, several partitions tie on it exactly. Break the
    # tie toward more lines: "DAS VOLLE / PAKET" and "DAS / VOLLE / PAKET" score
    # identically, and the stacked three-line version is the reference look.
    best = max(pool, key=lambda s: (round(s[1], 1), len(s[2]), -s[0]))
    return best[2], best[3]


# ────────────────────────────────────────────────────────────────────────────
# Glyph rendering
# ────────────────────────────────────────────────────────────────────────────

def _grain_texture(size: tuple[int, int], seed: int, strength: float) -> np.ndarray:
    """Seed-stable print/concrete grain, returned as a multiplier in [1-s, 1]."""
    rng = np.random.default_rng(seed)
    w, h = size
    coarse = rng.random((max(1, h // 6), max(1, w // 6))).astype(np.float32)
    noise = np.asarray(
        Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    fine = rng.random((h, w)).astype(np.float32)
    mixed = 0.65 * noise + 0.35 * fine
    return 1.0 - strength * (1.0 - mixed)


def render_line_image(
    text: str,
    font_path: Path,
    font_size: int,
    *,
    color: tuple[int, int, int],
    grain: float = 0.0,
    seed: int = 0,
) -> Image.Image:
    """One line as a tight RGBA image, cropped to its ink.

    Deliberately a single fill pass. No stroke, no extrude, no offset shadow —
    separation from the background is the bloom layer's job, applied once to the
    whole block rather than per glyph.
    """
    font = _font(font_path, font_size)
    tracking = _tracking_px(font_size, text)
    x0, y0, x1, y1 = _measure_tracked(text, font, tracking)
    pad = max(4, font_size // 20)
    img = Image.new("RGBA", (x1 - x0 + pad * 2, y1 - y0 + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_tracked(draw, (pad - x0, pad - y0), text, font, (*color, 255), tracking)

    if grain > 0:
        arr = np.asarray(img, dtype=np.float32)
        tex = _grain_texture(img.size, seed, grain)
        arr[..., 3] *= tex
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")

    return img.crop(img.getbbox() or (0, 0, img.width, img.height))


def layout_and_render(
    hook: str,
    *,
    canvas_size: tuple[int, int],
    measure_ratio: float = 0.90,
    block_top_ratio: float = 0.05,
    max_block_ratio: float = 0.50,
    max_cap_ratio: float = MAX_CAP_RATIO,
    align: str = "center",
    font: str = DEFAULT_FONT,
    color: tuple[int, int, int] = (255, 255, 255),
    accent_line: int | None = None,
    accent_color: tuple[int, int, int] | str = "red",
    line_gap: float = LINE_GAP_RATIO,
    grain: float = 0.11,
    bloom_px: int | None = None,
    bloom_opacity: float = 0.55,
    seed: int = 0,
) -> TypeLayout:
    """Set `hook` in the reference style and return the rendered layer + metrics.

    `line_gap` is the gap between lines as a fraction of cap height, added to
    each line's ink height — not a multiplier on it.
    """
    w, h = canvas_size
    measure_px = int(w * measure_ratio)
    max_block_h = int(h * max_block_ratio)
    font_path = resolve_font_path(font)
    if isinstance(accent_color, str):
        accent_color = ACCENT_COLORS.get(accent_color, ACCENT_COLORS["red"])

    texts, sizes = choose_lines(
        hook,
        font_path=font_path,
        measure_px=measure_px,
        max_block_h=max_block_h,
        line_gap=line_gap,
        max_cap_px=h * max_cap_ratio,
    )
    if not texts:
        empty = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        return TypeLayout(image=empty, alpha=empty.split()[-1])

    # The LLM picks an accent line before the layout exists, so it can name a
    # line that this hook does not have. Clamp rather than fail.
    if accent_line is not None:
        accent_line = max(0, min(int(accent_line), len(texts) - 1))

    rendered: list[Image.Image] = []
    lines: list[RenderedLine] = []
    for idx, (text, size) in enumerate(zip(texts, sizes)):
        is_accent = accent_line is not None and idx == accent_line
        img = render_line_image(
            text,
            font_path,
            size,
            color=accent_color if is_accent else color,
            grain=grain,
            seed=seed + idx,
        )
        rendered.append(img)
        lines.append(
            RenderedLine(
                text=text,
                font_size=size,
                cap_height=cap_height_of(_font(font_path, size)),
                ink_width=img.width,
                fill_ratio=img.width / max(1, measure_px),
                is_accent=is_accent,
            )
        )

    # Advance by each line's own ink height plus a small gap, so lines nearly
    # touch without colliding. Umlauts make a line's ink taller than its cap,
    # which is exactly why the advance cannot be derived from cap height alone.
    gap_px = float(np.mean([ln.cap_height for ln in lines])) * line_gap
    advances = [img.height + gap_px for img in rendered]
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    y = float(h * block_top_ratio)
    boxes: list[tuple[int, int, int, int]] = []
    for idx, (img, ln) in enumerate(zip(rendered, lines)):
        if align == "left":
            x = int((w - measure_px) / 2) if measure_ratio >= 1.0 else int(w * (1 - measure_ratio) / 2)
        elif align == "right":
            x = w - int(w * (1 - measure_ratio) / 2) - img.width
        else:
            x = (w - img.width) // 2
        canvas.alpha_composite(img, (int(x), int(y)))
        box = (int(x), int(y), int(x) + img.width, int(y) + img.height)
        ln.box = box
        boxes.append(box)
        y += advances[idx]

    block_box = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )

    if bloom_px is None:
        bloom_px = max(8, int(np.mean([ln.cap_height for ln in lines]) * 0.22))
    layer = _apply_bloom(canvas, radius=bloom_px, opacity=bloom_opacity)

    return TypeLayout(image=layer, alpha=canvas.split()[-1], lines=lines, block_box=block_box)


def _apply_bloom(text_layer: Image.Image, *, radius: int, opacity: float) -> Image.Image:
    """Soft dark halo behind the glyphs, then the glyphs on top.

    Replaces the legacy stack of drop shadow + glow + 3D extrude + hard black
    contour. Centred and blurred rather than offset, so it darkens the
    background without ever cutting into the bright glyph edge.
    """
    alpha = text_layer.split()[-1]
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius))
    shadow_alpha = shadow_alpha.point(lambda v: int(min(255, v * opacity * 1.8)))
    shadow = Image.new("RGBA", text_layer.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    out = Image.alpha_composite(shadow, text_layer)
    return out
