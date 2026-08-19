"""
Reference-calibrated quality gate for the "epic" thumbnail look.
================================================================

Two kinds of checks:

* **Hard gates** — boolean, failing one means the render is rejected outright.
* **Calibrated metrics** — continuous, compared against bands *derived* from the
  reference images in ``thumbnail_ideal_examples/`` (see
  ``tools/calibrate_reference_gate.py``). Bands are never hand-written: the
  reference set has to pass its own gate by construction.

The band file is ``reference_gate_bands.json`` next to this module. When it is
missing the gate runs hard gates only and reports metrics without verdicts, so a
fresh checkout degrades to "no opinion" rather than to "everything fails".
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

_BANDS_PATH = Path(__file__).with_name("reference_gate_bands.json")

# Metrics that are calibrated from the reference distribution.
#
# `cap_height_ratio` — cap height of one line over frame height — is *the*
# discriminating type metric, and finding that took three attempts:
#
#   * block height / frame height: references 0.08–0.49, our current render
#     0.31. Sits mid-range, discriminates nothing.
#   * glyph ink / frame area: references 0.035–0.163, ours 0.030. Inside the
#     padded band, because a single-line reference is legitimately sparse.
#   * cap height / frame height: references 0.080–0.163, ours 0.062. Below the
#     band — it fails, correctly, because our type is simply too small.
#
# `type_ink_ratio` is kept as a weaker secondary signal.
CALIBRATED_METRICS = (
    "cap_height_ratio",
    "type_ink_ratio",
    "peak_luma",
    "hot_fraction",
    "dark_fraction",
    "mean_saturation",
)

# Metrics that express a design decision rather than an observation, so they are
# constants and deliberately not fitted to the references.
DESIGN_CONSTANTS = {
    # The widest line should span most of the measure. Not 0.85: once the cap
    # height is clamped to the reference band the whole block scales down, and a
    # 3-line hook legitimately lands near 0.67 — reference 1 ("DAS VOLLE PAKET")
    # spans 0.64 of its frame width. cap_height_ratio already gates type size;
    # this only catches a line that failed to fill at all.
    "line_fill_ratio": (0.55, None),
    "halo_score": (None, 0.15),
}

GATE_TIERS = {
    "off": ("spelling_exact",),
    "normal": ("spelling_exact", "dims_9x16", "text_within_canvas", "not_blank"),
    "strict": (
        "spelling_exact",
        "dims_9x16",
        "text_within_canvas",
        "not_blank",
        "face_not_covered",
        "accent_present",
    ),
}


@dataclass
class GateResult:
    passed: bool
    tier: str
    hard_failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    out_of_band: dict[str, tuple[float, float | None, float | None]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "tier": self.tier,
            "hard_failures": list(self.hard_failures),
            "metrics": {k: round(float(v), 5) for k, v in self.metrics.items()},
            "out_of_band": {
                k: {"value": round(float(v), 5), "low": lo, "high": hi}
                for k, (v, lo, hi) in self.out_of_band.items()
            },
            "notes": list(self.notes),
        }


# ────────────────────────────────────────────────────────────────────────────
# Spelling — the golden rule. Ported from the layered pipeline.
# ────────────────────────────────────────────────────────────────────────────

def validate_exact_spelling(source: str, rendered_lines: list[str]) -> bool:
    """True when `rendered_lines` reconstruct `source` character for character.

    Umlauts and ``ß`` must survive. ``ß`` is never allowed to become ``SS`` —
    Python's ``str.upper()`` does exactly that, so uppercasing must go through
    :func:`safe_upper`.
    """
    joined = " ".join(rendered_lines)
    return _normalise_for_compare(joined) == _normalise_for_compare(source)


def safe_upper(text: str) -> str:
    """Uppercase that keeps ``ß`` intact instead of expanding it to ``SS``."""
    return "".join("ß" if ch == "ß" else ch.upper() for ch in text)


def _normalise_for_compare(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = safe_upper(text)
    return " ".join(text.split())


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def image_metrics(image: Image.Image) -> dict[str, float]:
    """Colour/luminance metrics that work on any RGB image, ours or a reference."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = rgb.mean(axis=2)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    return {
        "peak_luma": float(np.percentile(luma, 95)),
        "hot_fraction": float((luma > 230).mean()),
        "dark_fraction": float((luma < 40).mean()),
        "mean_saturation": float(sat.mean()),
    }


def type_ink_ratio_from_alpha(text_alpha: Image.Image, canvas_size: tuple[int, int]) -> float:
    """Exact glyph coverage from the alpha the type engine produced.

    Used for our own renders, where the glyph mask is known exactly rather than
    estimated.
    """
    a = np.asarray(text_alpha.convert("L"), dtype=np.float32) / 255.0
    w, h = canvas_size
    return float(a.sum() / max(1.0, float(w * h)))


def type_ink_ratio_from_box(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Estimate glyph coverage inside an annotated title box, frame-relative.

    Splits the box into two luminance clusters and treats the brighter one as
    ink. Restricting to the annotated box is what makes this workable —
    globally, a light shaft reads as "bright" just like white type does.

    Luminance only, deliberately: folding saturation into the feature makes a
    saturated background register as ink. Our own orange-gradient render has
    mean saturation 0.74, and a combined ``max(luma, sat)`` feature scored it
    0.119 — four times its true ink coverage.
    """
    w, h = image.size
    x1, y1, x2, y2 = box
    rgb = np.asarray(image.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.float32)
    if rgb.size == 0:
        return 0.0

    ink = _two_cluster_ink_mask((rgb.mean(axis=2) / 255.0).ravel())
    return float(ink.sum() / max(1.0, float(w * h)))


def _two_cluster_ink_mask(feat: np.ndarray, iterations: int = 25) -> np.ndarray:
    """1-D k-means (k=2); returns a boolean mask of the higher-valued cluster."""
    lo, hi = float(feat.min()), float(feat.max())
    if hi - lo < 1e-6:
        return np.zeros_like(feat, dtype=bool)
    c_lo, c_hi = lo, hi
    for _ in range(iterations):
        mid = (c_lo + c_hi) / 2.0
        upper = feat >= mid
        if not upper.any() or upper.all():
            break
        new_lo = float(feat[~upper].mean())
        new_hi = float(feat[upper].mean())
        if abs(new_lo - c_lo) < 1e-6 and abs(new_hi - c_hi) < 1e-6:
            break
        c_lo, c_hi = new_lo, new_hi
    return feat >= (c_lo + c_hi) / 2.0


def type_block_ratio(block_box: tuple[int, int, int, int], canvas_size: tuple[int, int]) -> float:
    """Title-block height over frame height. Reported for context, not gated."""
    _, y1, _, y2 = block_box
    _, h = canvas_size
    return float(max(0, y2 - y1) / max(1, h))


def cap_height_ratio(
    block_box: tuple[int, int, int, int],
    n_lines: int,
    canvas_size: tuple[int, int],
) -> float:
    """Cap height of a single line over frame height — how big the type reads.

    Approximated as block height / line count, which holds because the look
    uses near-zero leading. The type engine reports this exactly.
    """
    _, y1, _, y2 = block_box
    _, h = canvas_size
    return float(max(0, y2 - y1) / max(1, h) / max(1, n_lines))


def halo_score(subject_alpha: Image.Image, composed: Image.Image, band_px: int = 6) -> float:
    """Brightness ring just *inside* the subject edge — the rembg fringe artefact.

    Compares mean luminance in a thin band inside the alpha edge against the
    subject's interior. A clean matte scores ~0; a bright fringe scores high.

    `subject_alpha` must be the alpha at its final size and position on the
    canvas. Passing the raw cutout instead compares two images that differ in
    both scale and position, which silently measures nothing.
    """
    alpha = np.asarray(subject_alpha.convert("L").resize(composed.size), dtype=np.float32) / 255.0
    luma = np.asarray(composed.convert("L"), dtype=np.float32)

    inside = alpha > 0.5
    if not inside.any():
        return 0.0

    from scipy import ndimage  # noqa: PLC0415 - optional heavy import, only needed here

    # Erode inward: the fringe lives on the subject's own edge pixels, which is
    # where rembg leaves background colour behind.
    core = ndimage.binary_erosion(inside, iterations=band_px * 3)
    edge = inside & ~ndimage.binary_erosion(inside, iterations=band_px)
    if not edge.any() or not core.any():
        return 0.0
    return float(max(0.0, (luma[edge].mean() - luma[core].mean()) / 255.0))


def _halo_score_fallback(subject_alpha: Image.Image, composed: Image.Image, band_px: int = 6) -> float:
    """scipy-free variant using Pillow's MinFilter for erosion."""
    from PIL import ImageFilter  # noqa: PLC0415

    alpha = subject_alpha.convert("L").resize(composed.size)

    def erode(img: Image.Image, iterations: int) -> np.ndarray:
        cur = img
        for _ in range(iterations):
            cur = cur.filter(ImageFilter.MinFilter(3))
        return np.asarray(cur, dtype=np.float32) > 127

    inside = np.asarray(alpha, dtype=np.float32) > 127
    if not inside.any():
        return 0.0
    core = erode(alpha, band_px * 3)
    edge = inside & ~erode(alpha, band_px)
    if not edge.any() or not core.any():
        return 0.0
    luma = np.asarray(composed.convert("L"), dtype=np.float32)
    return float(max(0.0, (luma[edge].mean() - luma[core].mean()) / 255.0))


def compute_halo_score(subject_alpha: Image.Image, composed: Image.Image, band_px: int = 6) -> float:
    try:
        return halo_score(subject_alpha, composed, band_px)
    except ImportError:
        return _halo_score_fallback(subject_alpha, composed, band_px)


# ────────────────────────────────────────────────────────────────────────────
# Bands
# ────────────────────────────────────────────────────────────────────────────

def load_bands(path: Path | None = None) -> dict[str, dict]:
    p = path or _BANDS_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("bands", {})
    except (OSError, json.JSONDecodeError):
        return {}


def derive_bands(samples: dict[str, list[float]], *, margin: float = 0.10) -> dict[str, dict]:
    """Band = [min - margin*range, max + margin*range] over the sample distribution.

    Deriving rather than hand-writing is what makes the reference set pass its
    own gate. A hand-written band of e.g. peak_luma >= 240 rejects reference 9
    (202.7), which would make the calibration test meaningless.
    """
    bands: dict[str, dict] = {}
    for name, values in samples.items():
        if not values:
            continue
        lo, hi = float(min(values)), float(max(values))
        span = hi - lo
        pad = span * margin if span > 0 else max(abs(hi) * 0.05, 1e-6)
        bands[name] = {
            "low": lo - pad,
            "high": hi + pad,
            "observed_min": lo,
            "observed_max": hi,
            "n": len(values),
        }
    return bands


# ────────────────────────────────────────────────────────────────────────────
# Gate
# ────────────────────────────────────────────────────────────────────────────

def run_gate(
    image: Image.Image,
    *,
    tier: str = "normal",
    title: str | None = None,
    rendered_lines: list[str] | None = None,
    text_alpha: Image.Image | None = None,
    text_block_box: tuple[int, int, int, int] | None = None,
    n_lines: int | None = None,
    cap_height_px: float | None = None,
    face_box: tuple[int, int, int, int] | None = None,
    subject_alpha: Image.Image | None = None,
    line_fill_ratio: float | None = None,
    accent_present: bool | None = None,
    bands: dict[str, dict] | None = None,
) -> GateResult:
    if tier not in GATE_TIERS:
        raise ValueError(f"unknown gate tier: {tier!r} (expected one of {sorted(GATE_TIERS)})")

    checks = GATE_TIERS[tier]
    bands = bands if bands is not None else load_bands()
    result = GateResult(passed=True, tier=tier)

    w, h = image.size
    metrics = image_metrics(image)

    if text_alpha is not None:
        metrics["type_ink_ratio"] = type_ink_ratio_from_alpha(text_alpha, (w, h))
    elif text_block_box is not None:
        metrics["type_ink_ratio"] = type_ink_ratio_from_box(image, text_block_box)
    if text_block_box is not None:
        metrics["type_block_ratio"] = type_block_ratio(text_block_box, (w, h))

    # Exact when the type engine reports the cap height; approximated from the
    # block otherwise (which is what the reference annotations rely on).
    if cap_height_px is not None:
        metrics["cap_height_ratio"] = float(cap_height_px) / max(1, h)
    elif text_block_box is not None and n_lines:
        metrics["cap_height_ratio"] = cap_height_ratio(text_block_box, n_lines, (w, h))

    if line_fill_ratio is not None:
        metrics["line_fill_ratio"] = float(line_fill_ratio)
    if subject_alpha is not None:
        metrics["halo_score"] = compute_halo_score(subject_alpha, image)

    result.metrics = metrics

    # ── hard gates ────────────────────────────────────────────────────────
    if "spelling_exact" in checks and title is not None and rendered_lines is not None:
        if not validate_exact_spelling(title, rendered_lines):
            result.hard_failures.append("spelling_exact")

    if "dims_9x16" in checks:
        if abs((w / max(1, h)) - (9 / 16)) > 0.01:
            result.hard_failures.append("dims_9x16")

    if "text_within_canvas" in checks and text_block_box is not None:
        x1, y1, x2, y2 = text_block_box
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            result.hard_failures.append("text_within_canvas")

    if "not_blank" in checks:
        if float(np.asarray(image.convert("L"), dtype=np.float32).std()) < 6.0:
            result.hard_failures.append("not_blank")

    if "face_not_covered" in checks and face_box is not None and text_block_box is not None:
        if _overlap_ratio(text_block_box, face_box) > 0.12:
            result.hard_failures.append("face_not_covered")

    if "accent_present" in checks and accent_present is not None and not accent_present:
        result.hard_failures.append("accent_present")

    # ── calibrated bands + design constants ───────────────────────────────
    if not bands:
        result.notes.append("no calibration file — hard gates only, metrics reported without verdict")
    for name in CALIBRATED_METRICS:
        if name not in metrics or name not in bands:
            continue
        lo, hi = bands[name].get("low"), bands[name].get("high")
        val = metrics[name]
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            result.out_of_band[name] = (val, lo, hi)

    for name, (lo, hi) in DESIGN_CONSTANTS.items():
        if name not in metrics:
            continue
        val = metrics[name]
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            result.out_of_band[name] = (val, lo, hi)

    result.passed = not result.hard_failures
    return result


def _overlap_ratio(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box
    bx, by, bw, bh = other if len(other) == 4 and other[2] < ax2 else other
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_b)


def image_fingerprint(path: Path) -> str:
    """md5 of the file bytes — used to drop the duplicate reference image."""
    return hashlib.md5(path.read_bytes()).hexdigest()
