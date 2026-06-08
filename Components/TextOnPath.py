"""Minimal Text-on-Path utilities.

This module gives the thumbnail pipeline a practical text-on-path layer system:

- ``curved_text_layer``: simple bezier headline helper
- ``path_text_layer`` / ``text_on_path_layer``: arbitrary sampled path
- ``text_on_svg_path_layer``: lightweight SVG ``M/L/C/Q`` path support
- ``arc_text_layer`` / ``circle_text_layer``: arc and badge text helpers

The default renderer uses Pillow and per-glyph placement. That is sufficient for
Latin uppercase thumbnail typography and path text experiments.

An optional ``harfbuzz_advance`` backend is included for better advance/spacing
calculation when ``uharfbuzz`` + ``freetype`` are installed, but rasterization is
still Pillow-based. For true outline-quality shaping, use HarfBuzz/Pango + a
vector renderer in a later phase.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Tuple
from PIL import Image, ImageFont, ImageDraw


Point = Tuple[float, float]


TEXT_PATH_PRESETS = {
    "energetic_arc": {
        "start_offset": "50%",
        "align": "center",
        "tracking": 2.0,
        "baseline_offset": 12.0,
        "side": "above",
    },
    "badge_top": {
        "start_offset": "50%",
        "align": "center",
        "tracking": 1.0,
        "baseline_offset": 10.0,
        "side": "above",
    },
    "badge_bottom": {
        "start_offset": "50%",
        "align": "center",
        "tracking": 1.0,
        "baseline_offset": -10.0,
        "side": "below",
        "reverse_path": True,
    },
    "swoosh_headline": {
        "start_offset": "50%",
        "align": "center",
        "tracking": 2.0,
        "baseline_offset": 8.0,
        "side": "above",
    },
}


def bezier_points(p0: Point, p1: Point, p2: Point, p3: Point, n: int = 300) -> List[Point]:
    pts: List[Point] = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t ** 2
        d = t ** 3
        x = a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0]
        y = a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]
        pts.append((x, y))
    return pts


def quadratic_bezier_points(p0: Point, p1: Point, p2: Point, n: int = 220) -> List[Point]:
    pts: List[Point] = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 2
        b = 2 * (1 - t) * t
        c = t ** 2
        x = a * p0[0] + b * p1[0] + c * p2[0]
        y = a * p0[1] + b * p1[1] + c * p2[1]
        pts.append((x, y))
    return pts


def path_length(points: Iterable[Point]) -> float:
    pts = list(points)
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        total += math.hypot(dx, dy)
    return total


def sample_point_at(points: List[Point], distance: float) -> Tuple[Point, float]:
    """Return (point, angle_rad) at given distance along the polyline points.

    Angle is tangent direction in radians.
    """
    if distance <= 0:
        a, b = points[0], points[1]
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        return a, ang
    acc = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if acc + seg >= distance:
            rem = distance - acc
            t = rem / seg if seg > 0 else 0.0
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            return (x, y), ang
        acc += seg
    # fallback: last segment direction
    a, b = points[-2], points[-1]
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    return points[-1], ang


def reverse_path_points(points: List[Point]) -> List[Point]:
    return list(reversed(points))


def _parse_length(value, total: float, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("%"):
        try:
            return total * float(text[:-1]) / 100.0
        except ValueError:
            return fallback
    try:
        return float(text)
    except ValueError:
        return fallback


def _glyph_metrics(font, text: str, renderer_backend: str):
    if renderer_backend == "harfbuzz_advance":
        shaped = _shape_text_harfbuzz(font, text)
        if shaped:
            return shaped
    glyphs = []
    for ch in text:
        try:
            bbox = font.getbbox(ch)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = font.getsize(ch)
        glyphs.append({"text": ch, "advance": float(w), "width": float(w), "height": float(h)})
    return glyphs


def _shape_text_harfbuzz(font, text: str):
    """Optional HarfBuzz shaping.

    This returns per-cluster advances where available. Raster output still uses
    Pillow text drawing, so this mainly improves spacing for supported scripts.
    If dependencies are missing, return ``None``.
    """
    try:
        import uharfbuzz as hb
        from freetype import Face
    except Exception:
        return None

    font_path = getattr(font, "path", None)
    if not font_path:
        return None

    try:
        with open(font_path, "rb") as handle:
            blob = hb.Blob(handle.read())
        face = hb.Face(blob, 0)
        hb_font = hb.Font(face)
        scale = max(1, int(getattr(font, "size", 64)))
        hb_font.scale = (scale * 64, scale * 64)

        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        hb.shape(hb_font, buffer, {"kern": True, "liga": True})

        infos = buffer.glyph_infos
        positions = buffer.glyph_positions
        if not infos or not positions:
            return None

        shaped = []
        for info, pos in zip(infos, positions):
            cluster = info.cluster
            char = text[cluster:cluster + 1] if cluster < len(text) else ""
            try:
                bbox = font.getbbox(char or " ")
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            except Exception:
                width, height = font.getsize(char or " ")
            shaped.append(
                {
                    "text": char or " ",
                    "advance": float(pos.x_advance) / 64.0,
                    "width": float(width),
                    "height": float(height),
                }
            )
        return shaped
    except Exception:
        return None


def parse_svg_path(svg_path: str, samples_per_curve: int = 120) -> List[Point]:
    """Parse a lightweight SVG path string into sampled points.

    Supported commands: ``M``, ``L``, ``Q``, ``C``, ``Z`` and lowercase variants.
    This is intentionally narrow for thumbnail specs, not a full SVG engine.
    """
    tokens = re.findall(r"[MLCQZmlcqz]|-?\d*\.?\d+(?:e[-+]?\d+)?", svg_path)
    points: List[Point] = []
    idx = 0
    cmd = None
    cur = (0.0, 0.0)
    start = None

    def read_point(relative: bool) -> Point:
        nonlocal idx, cur
        x = float(tokens[idx])
        y = float(tokens[idx + 1])
        idx += 2
        if relative:
            return cur[0] + x, cur[1] + y
        return x, y

    while idx < len(tokens):
        token = tokens[idx]
        if re.fullmatch(r"[MLCQZmlcqz]", token):
            cmd = token
            idx += 1
        if cmd is None:
            break

        relative = cmd.islower()
        op = cmd.upper()

        if op == "M":
            cur = read_point(relative)
            start = cur
            points.append(cur)
            cmd = "l" if relative else "L"
        elif op == "L":
            nxt = read_point(relative)
            if not points or points[-1] != cur:
                points.append(cur)
            points.append(nxt)
            cur = nxt
        elif op == "Q":
            p1 = read_point(relative)
            p2 = read_point(relative)
            curve = quadratic_bezier_points(cur, p1, p2, samples_per_curve)
            points.extend(curve[1:] if points else curve)
            cur = p2
        elif op == "C":
            p1 = read_point(relative)
            p2 = read_point(relative)
            p3 = read_point(relative)
            curve = bezier_points(cur, p1, p2, p3, samples_per_curve)
            points.extend(curve[1:] if points else curve)
            cur = p3
        elif op == "Z":
            if start is not None:
                points.append(start)
                cur = start
            start = None
            cmd = None
        else:
            break

    return points


def text_on_path_layer(
    text: str,
    font_path: str,
    font_size: int,
    path_points: List[Point],
    image_size: Tuple[int, int],
    start_offset: float | str = 0.0,
    tracking: float = 0.0,
    side: str = "center",
    baseline_offset: float = 0.0,
    align: str = "start",
    reverse_path: bool = False,
    flip_tangent: bool = False,
    renderer_backend: str = "pillow",
    fill=(255, 255, 255, 255),
    stroke=None,
) -> Image.Image:
    """Render `text` along `path_points` onto a transparent layer.

    - `start_offset`: pixels or a string like ``"50%"``
    - `tracking`: extra px between glyphs (positive = looser)
    - `align`: ``start`` / ``center`` / ``end`` positioning along path
    - `side`: ``center`` / ``above`` / ``below`` controls normal direction
    - `baseline_offset`: px offset normal to path (positive moves outward)
    """
    img = Image.new("RGBA", image_size, (0, 0, 0, 0))
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    if reverse_path:
        path_points = reverse_path_points(path_points)

    # Precompute glyph advances
    glyphs = _glyph_metrics(font, text, renderer_backend)
    total_advance = 0.0
    for glyph in glyphs:
        total_advance += glyph["advance"] + tracking

    path_len = path_length(path_points)
    if path_len <= 1:
        return img

    start_px = _parse_length(start_offset, path_len, fallback=0.0)
    if align == "center":
        start_px -= total_advance / 2.0
    elif align == "end":
        start_px -= total_advance
    start_px = max(0.0, min(path_len, start_px))

    if side == "above":
        baseline_offset = abs(baseline_offset)
    elif side == "below":
        baseline_offset = -abs(baseline_offset)

    cur = start_px
    for glyph in glyphs:
        ch = glyph["text"]
        w = glyph["width"]
        h = glyph["height"]
        advance = glyph["advance"]
        # advance half glyph to place at center
        cur += advance / 2.0
        (px, py), ang = sample_point_at(path_points, cur)
        if flip_tangent:
            ang += math.pi

        # render glyph to its own image
        glyph_img = Image.new("RGBA", (int(w * 3), int(h * 3)), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glyph_img)
        gx = (glyph_img.width - w) // 2
        gy = (glyph_img.height - h) // 2
        if stroke:
            gdraw.text((gx, gy), ch, font=font, fill=stroke)
        gdraw.text((gx, gy), ch, font=font, fill=fill)

        # rotate glyph according to tangent; PIL rotates counter-clockwise
        deg = math.degrees(ang)
        rotated = glyph_img.rotate(deg, resample=Image.BICUBIC, expand=True)

        # baseline offset normal to tangent
        nx = -math.sin(ang)
        ny = math.cos(ang)
        bx = px + nx * baseline_offset - rotated.width / 2.0
        by = py + ny * baseline_offset - rotated.height / 2.0

        img.alpha_composite(rotated, (int(bx), int(by)))

        cur += advance / 2.0 + tracking

    return img


def path_text_layer(
    text: str,
    font_path: str,
    font_size: int,
    path_points: List[Point],
    image_size: Tuple[int, int],
    preset: str | None = None,
    **kwargs,
) -> Image.Image:
    options = dict(TEXT_PATH_PRESETS.get(preset or "", {}))
    options.update(kwargs)
    return text_on_path_layer(text, font_path, font_size, path_points, image_size, **options)


def text_on_svg_path_layer(
    text: str,
    font_path: str,
    font_size: int,
    svg_path: str,
    image_size: Tuple[int, int],
    preset: str | None = None,
    **kwargs,
) -> Image.Image:
    points = parse_svg_path(svg_path)
    return path_text_layer(text, font_path, font_size, points, image_size, preset=preset, **kwargs)


def curved_text_layer(
    text: str,
    font_path: str,
    font_size: int,
    image_size: Tuple[int, int],
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    preset: str | None = "swoosh_headline",
    **kwargs,
) -> Image.Image:
    points = bezier_points(p0, p1, p2, p3)
    return path_text_layer(text, font_path, font_size, points, image_size, preset=preset, **kwargs)


def arc_text_layer(
    text: str,
    font_path: str,
    font_size: int,
    center: Point,
    radius: float,
    start_angle_deg: float,
    end_angle_deg: float,
    image_size: Tuple[int, int],
    **kwargs,
) -> Image.Image:
    # sample arc as polyline
    start = math.radians(start_angle_deg)
    end = math.radians(end_angle_deg)
    length_angle = end - start
    segments = max(64, int(abs(length_angle) / (math.pi * 2) * 300))
    pts: List[Point] = []
    for i in range(segments + 1):
        t = i / segments
        ang = start + t * length_angle
        x = center[0] + radius * math.cos(ang)
        y = center[1] + radius * math.sin(ang)
        pts.append((x, y))
    return path_text_layer(text, font_path, font_size, pts, image_size, **kwargs)


def circle_text_layer(
    text: str,
    font_path: str,
    font_size: int,
    center: Point,
    radius: float,
    image_size: Tuple[int, int],
    outside: bool = True,
    preset: str | None = None,
    **kwargs,
):
    start_angle = -90.0
    end_angle = 270.0
    options = dict(kwargs)
    if outside:
        options.setdefault("side", "above")
    else:
        options.setdefault("side", "below")
        options.setdefault("flip_tangent", True)
        options.setdefault("reverse_path", True)
    return arc_text_layer(
        text,
        font_path,
        font_size,
        center,
        radius,
        start_angle,
        end_angle,
        image_size,
        preset=preset,
        **options,
    )
