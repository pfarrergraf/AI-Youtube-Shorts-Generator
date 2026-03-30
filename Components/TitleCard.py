"""Title-card / thumbnail-hook generator for short-form sermon clips.

Two modes:
  **thumbnail** (default) — TikTok/YouTube-style scroll-stopping cover:
    • Speaker face clearly visible (sharp frame, not heavy-blurred)
    • Bottom gradient for text readability
    • Bold 2-5 word hook text with one accent-color keyword
    • Face-aware text placement (avoids covering the speaker)
    • Short duration (1.0 s) or overlay on first frames

  **cinematic** — After-Effects-style animated intro card:
    • Blurred first frame with slow Ken Burns zoom + vignette
    • Text fades in with upward drift, accent line, fade-out
    • 2.5 s prepended intro
"""

import math
import os
import random
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ── Shared defaults ──────────────────────────────────────────────
THUMBNAIL_DURATION = 1.5  # seconds — short cover
CINEMATIC_DURATION = 2.5  # seconds — animated intro

# Accent colour for the keyword (warm red, like TikTok thumbnails)
ACCENT_COLOUR = (220, 40, 40, 255)  # RGBA red
ACCENT_COLOUR_HEX = "#860BF8"

# Font discovery order (first match wins)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
]

# Thumbnail text sizing
_THUMB_FONT_RATIO = 0.10  # bigger than cinematic
_THUMB_MAX_TEXT_W_RATIO = 0.84  # 8% safe margin each side (was 0.88)

# Cinematic defaults (kept from previous version)
_CINE_BLUR_RADIUS = 25
_CINE_OVERLAY_OPACITY = 110
_CINE_FONT_RATIO = 0.065
_CINE_ACCENT_LINE_W_RATIO = 0.25
_CINE_ACCENT_LINE_THICKNESS = 3
_CINE_FADE_IN_START = 0.25
_CINE_FADE_IN_DUR = 0.60
_CINE_FADE_OUT_OFFSET = 0.30
_CINE_TEXT_DRIFT_PX = 20
_CINE_ZOOM_FACTOR = 0.018

NVENC_FLAGS = [
    "-c:v", "h264_nvenc",
    "-preset", "p7",
    "-rc", "constqp",
    "-qp", "18",
    "-b:v", "0",
    "-gpu", "0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
]

# Thumbnail style catalogue
THUMBNAIL_STYLES = ("classic", "text_reveal", "dramatic")
# Extended fancy styles from ai_after-effects (rendered frame-by-frame)
FANCY_STYLES = ("kinetic", "glitch", "neon", "distortion")
ALL_STYLES = THUMBNAIL_STYLES + FANCY_STYLES

# Active style pool for random selection:
# - text_reveal: track-matte (speaker visible through letters) — most preferred
# - classic: visible speaker + bottom gradient + white/red text
# dramatic + FANCY_STYLES removed: dramatic looks too dark/bloody, fancy styles
# don't word-wrap and render as white-on-black which looks cheap.
_ACTIVE_STYLES = ("text_reveal", "classic")
_ACTIVE_STYLE_WEIGHTS = (0.70, 0.30)

_TEXT_REVEAL_FONT_RATIO = 0.12  # larger text for mask/cutout effect
_FANCY_FONT_RATIO = 0.10        # font ratio for fancy title styles
_TEXT_REVEAL_END_REVEAL_SEC = 0.50  # final reveal window for track-matte zoom-out

# Background-music defaults
# Make background music 4 dB quieter by default
_BG_MUSIC_VOLUME_DB = -20  # background music level (was -16)
_BG_FADE_OUT_SEC = 2.0     # fade-out duration at end
_TARGET_PEAK_DB = -2.0     # target peak for speech audio


# ── Helpers ──────────────────────────────────────────────────────

def _find_font():
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _wrap_text(text, font, max_width, draw):
    """Word-wrap *text* so each line fits within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return fps
    finally:
        cap.release()


def _extract_first_frame(video_path):
    """Read the first frame from *video_path* using OpenCV.  Returns BGR ndarray."""
    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read first frame from {video_path}")
        return frame
    finally:
        cap.release()


def probe_peak_db(audio_path):
    """Return the peak volume in dB of *audio_path* using FFmpeg volumedetect.

    Returns a float, e.g. ``-8.2`` means peak is at -8.2 dB.
    Falls back to ``-10.0`` on error so the pipeline never crashes.
    """
    import re as _re
    cmd = [
        "ffmpeg", "-hide_banner", "-i", os.path.abspath(audio_path),
        "-af", "volumedetect", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # volumedetect prints to stderr:  max_volume: -8.2 dB
    match = _re.search(r"max_volume:\s*([-\d.]+)\s*dB", result.stderr)
    if match:
        return float(match.group(1))
    return -10.0  # safe fallback


def probe_lufs_integrated(audio_path) -> float:
    """Return integrated LUFS of *audio_path* using FFmpeg loudnorm.

    Uses a short analysis pass (no re-encode).  Falls back to ``-23.0``
    (EBU R128 target) if the measurement cannot be obtained.
    """
    import re as _re, json as _json
    cmd = [
        "ffmpeg", "-hide_banner", "-i", os.path.abspath(audio_path),
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints JSON summary to stderr after the run
    m = _re.search(r"\{[^}]+\}", result.stderr, _re.DOTALL)
    if m:
        try:
            data = _json.loads(m.group())
            return float(data["input_i"])  # integrated loudness in LUFS
        except (KeyError, ValueError, _json.JSONDecodeError):
            pass
    return -23.0  # EBU R128 target as conservative fallback


def _get_media_duration(media_path):
    """Return duration in seconds of *media_path* via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(media_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 60.0  # safe fallback


def _bg_music_filter(input_idx, total_duration, volume_db=_BG_MUSIC_VOLUME_DB,
                     fade_out_sec=_BG_FADE_OUT_SEC):
    """Return FFmpeg filter fragment that loops, trims, levels and fades music.

    Example output::

        [3:a]aloop=loop=-1:size=2e+09,atrim=0:45.0,
             volume=-15dB,afade=t=out:st=40.0:d=5.0[music]
    """
    fade_start = max(0, total_duration - fade_out_sec)
    return (
        f"[{input_idx}:a]aloop=loop=-1:size=2e+09,"
        f"atrim=0:{total_duration:.2f},"
        f"volume={volume_db}dB,"
        f"afade=t=out:st={fade_start:.2f}:d={fade_out_sec:.1f}[music]"
    )


def _thumb_font_setup(w, h, hook_text, font_ratio=_THUMB_FONT_RATIO,
                      max_w_ratio=_THUMB_MAX_TEXT_W_RATIO):
    """Shared font setup: find font, auto-shrink for overflow, wrap text.

    Returns ``(font_path, font, font_size, lines, max_text_w, hook_upper)``.
    """
    font_path = _find_font()
    font_size = max(36, int(h * font_ratio))
    font = (ImageFont.truetype(font_path, font_size)
            if font_path else ImageFont.load_default())

    tmp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(tmp)
    max_text_w = int(w * max_w_ratio)
    hook_upper = hook_text.upper()

    words = hook_upper.split()
    while font_size > 36 and words:
        longest = max(words, key=len)
        bbox = draw.textbbox((0, 0), longest, font=font)
        if bbox[2] - bbox[0] <= max_text_w:
            break
        font_size -= 4
        font = (ImageFont.truetype(font_path, font_size)
                if font_path else ImageFont.load_default())

    lines = _wrap_text(hook_upper, font, max_text_w, draw)
    return font_path, font, font_size, lines, max_text_w, hook_upper


# ══════════════════════════════════════════════════════════════════
#  THUMBNAIL MODE  — scroll-stopping cover frame
# ══════════════════════════════════════════════════════════════════

def _select_best_frame(video_path, window_sec=1.5, sample_count=15):
    """Pick the best frame from the first *window_sec* seconds.

    Scores by: face presence, face size, sharpness (Laplacian variance),
    brightness (not too dark/bright).  Falls back to first frame.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = min(int(fps * window_sec), total_frames)

    if window_frames < 2:
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
        raise RuntimeError(f"Could not read any frame from {video_path}")

    # Sample evenly across the window
    step = max(1, window_frames // sample_count)
    candidates = []

    # Load face detector (same as FaceCrop.py uses)
    face_cascade = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        pass

    for i in range(0, window_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpness: Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness: aim for 80-170 range
        brightness = gray.mean()
        bright_score = 1.0 - abs(brightness - 125) / 125.0

        # Face score
        face_score = 0.0
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(w // 8, h // 8))
            if len(faces) > 0:
                # Prefer larger face
                biggest = max(faces, key=lambda f: f[2] * f[3])
                face_area_ratio = (biggest[2] * biggest[3]) / (w * h)
                face_score = min(1.0, face_area_ratio * 10)  # 10% of frame = 1.0

        score = sharpness * 0.3 + bright_score * 100 * 0.2 + face_score * 100 * 0.5
        candidates.append((score, frame.copy()))

    cap.release()

    if not candidates:
        return _extract_first_frame(video_path)

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _detect_face_box(frame_bgr):
    """Return (x, y, w, h) of the largest face, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_bgr.shape[:2]
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(w // 10, h // 10))
        if len(faces) > 0:
            return tuple(max(faces, key=lambda f: f[2] * f[3]))
    except Exception:
        pass
    return None


def _pick_text_lane(face_box, width, height):
    """Choose upper-third or mid-lower lane based on face position.

    Returns (lane_top, lane_bottom) in pixels.
    """
    upper = (int(height * 0.06), int(height * 0.40))
    lower = (int(height * 0.45), int(height * 0.72))

    if face_box is None:
        return upper  # default to upper third

    fx, fy, fw, fh = face_box
    face_center_y = fy + fh / 2

    # If face is in the upper half, use lower lane
    if face_center_y < height * 0.45:
        return lower
    # If face is in the lower half, use upper lane
    return upper


def _face_aware_darken(img, face_box, width, height):
    """Darken the frame but preserve brightness around the detected face.

    If *face_box* is None, applies a uniform light darken.
    Otherwise, creates a radial "spotlight" that keeps the face area bright
    while darkening the surrounding frame more aggressively for text contrast.
    """
    if face_box is None:
        # No face — uniform light darken
        darken = Image.new("RGBA", (width, height), (0, 0, 0, 45))
        return Image.alpha_composite(img, darken)

    fx, fy, fw, fh = face_box
    # Face center and radius (expand slightly for natural falloff)
    cx = fx + fw // 2
    cy = fy + fh // 2
    face_radius = max(fw, fh) * 0.75  # generous radius around face

    # Build per-pixel alpha mask: high alpha far from face, low alpha near face
    mask = Image.new("L", (width, height), 0)
    mask_arr = np.zeros((height, width), dtype=np.float32)

    # Create coordinate grids
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Normalise distance: 0 at face center, 1 at face_radius, >1 beyond
    norm_dist = dist / max(face_radius, 1)

    # Smooth ramp: face area (dist<0.8) = very light, beyond = darker
    # Uses a clamped quadratic: alpha = clamp(norm_dist - 0.8, 0, 1)^1.5 * max_alpha
    ramp = np.clip(norm_dist - 0.8, 0, 2.0) / 2.0  # 0..1
    max_alpha = 70  # darkest areas around the periphery
    min_alpha = 10  # even the face gets a very slight darken for consistency
    alpha_arr = (min_alpha + ramp * (max_alpha - min_alpha)).astype(np.uint8)

    mask = Image.fromarray(alpha_arr, mode="L")
    darken = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    darken.putalpha(mask)
    return Image.alpha_composite(img, darken)


# ── Style dispatcher ─────────────────────────────────────────────

def _render_thumbnail(frame_bgr, hook_text, accent_keyword="", style="classic"):
    """Dispatch to the requested thumbnail style renderer.

    Parameters
    ----------
    style : str
        ``"classic"`` | ``"text_reveal"`` | ``"dramatic"`` | ``"random"`` |
        ``"auto"``.  ``"random"`` picks uniformly; ``"auto"`` picks based on
        face detection (text_reveal when a large face is present).
    """
    if style == "random":
        style = random.choice(THUMBNAIL_STYLES)
    elif style == "auto":
        face_box = _detect_face_box(frame_bgr)
        h, w = frame_bgr.shape[:2]
        if face_box is not None:
            _, _, fw, fh = face_box
            if (fw * fh) / (w * h) > 0.04:
                style = random.choice(THUMBNAIL_STYLES)
            else:
                style = random.choice(("classic", "dramatic"))
        else:
            style = random.choice(("classic", "dramatic"))

    if style == "text_reveal":
        return _render_thumbnail_text_reveal(frame_bgr, hook_text, accent_keyword)
    elif style == "dramatic":
        return _render_thumbnail_dramatic(frame_bgr, hook_text, accent_keyword)
    return _render_thumbnail_classic(frame_bgr, hook_text, accent_keyword)


# ── Style: classic ───────────────────────────────────────────────

def _render_thumbnail_classic(frame_bgr, hook_text, accent_keyword=""):
    """TikTok/YouTube-style thumbnail: visible speaker + gradient + bold text."""
    h, w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).convert("RGBA")

    face_box = _detect_face_box(frame_bgr)
    img = _face_aware_darken(img, face_box, w, h)

    # Bottom gradient
    gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    grad_draw = ImageDraw.Draw(gradient)
    gradient_start = int(h * 0.50)
    for y_pos in range(gradient_start, h):
        frac = (y_pos - gradient_start) / (h - gradient_start)
        alpha = int(180 * frac * frac)
        grad_draw.line([(0, y_pos), (w, y_pos)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, gradient)

    lane_top, lane_bottom = _pick_text_lane(face_box, w, h)
    font_path, title_font, font_size, lines, max_text_w, hook_upper = (
        _thumb_font_setup(w, h, hook_text)
    )

    draw = ImageDraw.Draw(img)
    line_height = int(font_size * 1.30)
    total_text_h = line_height * len(lines)
    lane_h = lane_bottom - lane_top
    text_y = lane_top + max(0, (lane_h - total_text_h) // 2)

    accent_upper = accent_keyword.upper() if accent_keyword else ""

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) // 2

        if accent_upper and accent_upper in line:
            _draw_line_with_accent(
                draw, x, text_y, line, accent_upper, title_font, ACCENT_COLOUR
            )
        else:
            for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 3), (0, -3)]:
                draw.text((x + dx, text_y + dy), line, font=title_font,
                          fill=(0, 0, 0, 200))
            draw.text((x, text_y), line, font=title_font,
                      fill=(255, 255, 255, 255))

        text_y += line_height

    return img


# ── Style: text_reveal (alpha-mask) ─────────────────────────────

def _build_text_reveal_assets(frame_bgr, hook_text, accent_keyword=""):
    """Build reusable assets for track-matte style and its end-transition.

    Returns ``(base, dark, mask, outline)`` as RGBA/L images.
    """
    h, w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(frame_rgb).convert("RGB")
    base = ImageEnhance.Color(base).enhance(1.35)
    base = ImageEnhance.Contrast(base).enhance(1.20)
    base = base.convert("RGBA")

    # Pure black plate; the video is revealed only by the text mask.
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 255))

    font_path, title_font, font_size, lines, _max_text_w, _hook_upper = (
        _thumb_font_setup(w, h, hook_text, font_ratio=_TEXT_REVEAL_FONT_RATIO)
    )

    line_height = int(font_size * 1.20)
    total_text_h = line_height * len(lines)
    # Vertical safe zone: at least 6% from top and bottom edges
    safe_top = int(h * 0.06)
    safe_bottom = int(h * 0.94)
    safe_h = safe_bottom - safe_top
    text_y_start = safe_top + max(0, (safe_h - total_text_h) // 2)

    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)

    text_y = text_y_start
    for line in lines:
        bbox = mask_draw.textbbox((0, 0), line, font=title_font)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) // 2
        mask_draw.text(
            (x, text_y),
            line,
            font=title_font,
            fill=255,
            stroke_width=5,
            stroke_fill=255,
        )
        text_y += line_height

    outline = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline)
    accent_upper = accent_keyword.upper() if accent_keyword else ""

    text_y = text_y_start
    for line in lines:
        bbox = outline_draw.textbbox((0, 0), line, font=title_font)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) // 2

        stroke_colour = (255, 255, 255, 190)
        if accent_upper and accent_upper in line:
            idx = line.find(accent_upper)
            if idx >= 0:
                before = line[:idx]
                accent = line[idx:idx + len(accent_upper)]
                after = line[idx + len(accent_upper):]
                cur_x = x
                if before:
                    outline_draw.text(
                        (cur_x, text_y),
                        before,
                        font=title_font,
                        fill=(0, 0, 0, 0),
                        stroke_width=2,
                        stroke_fill=stroke_colour,
                    )
                    cur_x += outline_draw.textbbox((0, 0), before, font=title_font)[2]
                if accent:
                    outline_draw.text(
                        (cur_x, text_y),
                        accent,
                        font=title_font,
                        fill=(0, 0, 0, 0),
                        stroke_width=2,
                        stroke_fill=(*ACCENT_COLOUR[:3], 220),
                    )
                    cur_x += outline_draw.textbbox((0, 0), accent, font=title_font)[2]
                if after:
                    outline_draw.text(
                        (cur_x, text_y),
                        after,
                        font=title_font,
                        fill=(0, 0, 0, 0),
                        stroke_width=2,
                        stroke_fill=stroke_colour,
                    )
            else:
                outline_draw.text(
                    (x, text_y),
                    line,
                    font=title_font,
                    fill=(0, 0, 0, 0),
                    stroke_width=2,
                    stroke_fill=stroke_colour,
                )
        else:
            outline_draw.text(
                (x, text_y),
                line,
                font=title_font,
                fill=(0, 0, 0, 0),
                stroke_width=2,
                stroke_fill=stroke_colour,
            )
        text_y += line_height

    return base, dark, mask, outline


def _render_thumbnail_text_reveal(frame_bgr, hook_text, accent_keyword=""):
    """Static track-matte frame used as the title-card thumbnail image."""
    base, dark, mask, outline = _build_text_reveal_assets(
        frame_bgr,
        hook_text,
        accent_keyword=accent_keyword,
    )
    result = Image.composite(base, dark, mask)
    return Image.alpha_composite(result, outline)


def _render_text_reveal_transition_frames(
    frame_bgr,
    hook_text,
    accent_keyword,
    fps,
    duration,
):
    """Track-matte animation with end zoom/reveal transition.

    Behavior requested:
    1) Start as black plate with speaker visible only through text.
    2) Last ~0.5 s: text mask scales up rapidly ("zoom to infinity").
    3) During that zoom, blend to full-frame frozen speaker image until 100% visible.
    """
    h, w = frame_bgr.shape[:2]
    total_frames = max(1, int(fps * duration))
    reveal_frames = max(1, int(fps * _TEXT_REVEAL_END_REVEAL_SEC))
    reveal_start = max(0, total_frames - reveal_frames)

    base, dark, mask, outline = _build_text_reveal_assets(
        frame_bgr,
        hook_text,
        accent_keyword=accent_keyword,
    )

    frames: list[Image.Image] = []
    for idx in range(total_frames):
        if idx < reveal_start:
            comp = Image.composite(base, dark, mask)
            comp = Image.alpha_composite(comp, outline)
            frames.append(comp)
            continue

        p = (idx - reveal_start) / max(1, reveal_frames - 1)  # 0..1

        # Exponential scale gives the "unlimited" growth feel near the end.
        scale = 1.0 + (6.0 * (p ** 2.4))
        sw = max(2, int(w * scale))
        sh = max(2, int(h * scale))

        scaled_mask = mask.resize((sw, sh), Image.Resampling.BICUBIC)
        scaled_outline = outline.resize((sw, sh), Image.Resampling.BICUBIC)

        # Re-center scaled assets onto canvas.
        canvas_mask = Image.new("L", (w, h), 0)
        canvas_outline = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ox = (w - sw) // 2
        oy = (h - sh) // 2
        canvas_mask.paste(scaled_mask, (ox, oy))
        canvas_outline.paste(scaled_outline, (ox, oy), scaled_outline)

        # Start with matte composite.
        comp = Image.composite(base, dark, canvas_mask)
        comp = Image.alpha_composite(comp, canvas_outline)

        # Blend toward full frame so video becomes 100% visible by the last frame.
        full_mix_alpha = int(255 * p)
        full_mix = Image.blend(comp, base, p)
        if full_mix_alpha >= 255:
            frames.append(base.copy())
        else:
            frames.append(full_mix)

    return frames


# ── Style: dramatic ──────────────────────────────────────────────

def _render_thumbnail_dramatic(frame_bgr, hook_text, accent_keyword=""):
    """Desaturated high-contrast style with heavy vignette + bold text."""
    h, w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(frame_rgb).convert("RGB")

    # Partial desaturation (70 % toward grayscale) + contrast boost
    img = ImageEnhance.Color(base).enhance(0.30)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = img.convert("RGBA")

    # Face-aware darkening
    face_box = _detect_face_box(frame_bgr)
    img = _face_aware_darken(img, face_box, w, h)

    # Heavy elliptical vignette
    vig = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vig_draw = ImageDraw.Draw(vig)
    cx, cy = w // 2, h // 2
    max_r = math.hypot(cx, cy)
    for i in range(40, 0, -1):
        frac = i / 40
        alpha = int(120 * frac * frac)
        r = int(max_r * frac)
        vig_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, vig)

    # Strong bottom gradient
    gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    grad_draw = ImageDraw.Draw(gradient)
    grad_start = int(h * 0.40)
    for y_pos in range(grad_start, h):
        frac = (y_pos - grad_start) / (h - grad_start)
        alpha = int(200 * frac * frac)
        grad_draw.line([(0, y_pos), (w, y_pos)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, gradient)

    # Text (same layout logic as classic)
    lane_top, lane_bottom = _pick_text_lane(face_box, w, h)
    font_path, title_font, font_size, lines, max_text_w, hook_upper = (
        _thumb_font_setup(w, h, hook_text)
    )

    draw = ImageDraw.Draw(img)
    line_height = int(font_size * 1.30)
    total_text_h = line_height * len(lines)
    lane_h = lane_bottom - lane_top
    text_y = lane_top + max(0, (lane_h - total_text_h) // 2)

    accent_upper = accent_keyword.upper() if accent_keyword else ""

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) // 2

        if accent_upper and accent_upper in line:
            _draw_line_with_accent(
                draw, x, text_y, line, accent_upper, title_font, ACCENT_COLOUR
            )
        else:
            for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 3), (0, -3)]:
                draw.text((x + dx, text_y + dy), line, font=title_font,
                          fill=(0, 0, 0, 200))
            draw.text((x, text_y), line, font=title_font,
                      fill=(255, 255, 255, 255))

        text_y += line_height

    return img


def _draw_line_with_accent(draw, x, y, line, accent_word, font, accent_colour):
    """Draw a line of text with one word in accent colour, rest in white."""
    # Find accent word position in line
    idx = line.upper().find(accent_word.upper())
    if idx < 0:
        # Fallback: all white
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        return

    before = line[:idx]
    accent_text = line[idx:idx + len(accent_word)]
    after = line[idx + len(accent_word):]

    cur_x = x

    # Draw before-accent text (white)
    if before:
        for dx, dy in [(-2, 2), (2, 2), (0, 3)]:
            draw.text((cur_x + dx, y + dy), before, font=font, fill=(0, 0, 0, 200))
        draw.text((cur_x, y), before, font=font, fill=(255, 255, 255, 255))
        bbox = draw.textbbox((0, 0), before, font=font)
        cur_x += bbox[2] - bbox[0]

    # Draw accent keyword (red with glow)
    if accent_text:
        # Red glow
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
            draw.text((cur_x + dx, y + dy), accent_text, font=font,
                      fill=(accent_colour[0], accent_colour[1], accent_colour[2], 80))
        # Drop shadow
        for dx, dy in [(-2, 2), (2, 2), (0, 3)]:
            draw.text((cur_x + dx, y + dy), accent_text, font=font,
                      fill=(0, 0, 0, 200))
        draw.text((cur_x, y), accent_text, font=font, fill=accent_colour)
        bbox = draw.textbbox((0, 0), accent_text, font=font)
        cur_x += bbox[2] - bbox[0]

    # Draw after-accent text (white)
    if after:
        for dx, dy in [(-2, 2), (2, 2), (0, 3)]:
            draw.text((cur_x + dx, y + dy), after, font=font, fill=(0, 0, 0, 200))
        draw.text((cur_x, y), after, font=font, fill=(255, 255, 255, 255))


# ══════════════════════════════════════════════════════════════════
#  FANCY TITLE STYLES — frame-by-frame animated titles
# ══════════════════════════════════════════════════════════════════

def _center_text_layer(width, height, text, font, color=(255, 255, 255)):
    """Create a transparent RGBA layer with text centred horizontally."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=3)
    x = (width - (bbox[2] - bbox[0])) / 2
    y = height * 0.38
    draw.text((x, y), text, font=font, fill=color + (255,),
              stroke_width=3, stroke_fill=(0, 0, 0, 190))
    return img


def _word_sprite(text, font, color=(255, 255, 255)):
    """Return a tightly-cropped RGBA sprite for one word.

    This avoids full-frame text layers, which caused multi-word kinetic titles
    to stack at nearly the same location after resizing/rotation.
    """
    temp = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=3)
    tw = max(2, bbox[2] - bbox[0] + 8)
    th = max(2, bbox[3] - bbox[1] + 8)

    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(img)
    d2.text(
        (4 - bbox[0], 4 - bbox[1]),
        text,
        font=font,
        fill=color + (255,),
        stroke_width=3,
        stroke_fill=(0, 0, 0, 190),
    )
    return img


def _add_vignette(frame_rgb, strength=0.12):
    """Add a subtle vignette to an RGB ndarray."""
    h, w = frame_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    max_r = math.hypot(cx, cy)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    factor = 1.0 - strength * (dist / max_r) ** 2
    factor = np.clip(factor, 0, 1)
    return np.clip(frame_rgb * factor[:, :, np.newaxis], 0, 255).astype(np.uint8)


def _rgb_split(frame, offset):
    """Simple RGB channel-shift glitch."""
    h, w = frame.shape[:2]
    result = frame.copy()
    result[:, max(0, offset):, 0] = frame[:, :w - max(0, offset), 0]  # R right
    result[:, :w - max(0, offset), 2] = frame[:, max(0, offset):, 2]  # B left
    return result


def _render_fancy_frames(frame_bgr, hook_text, style, fps, duration):
    """Render a frame-by-frame animated title returning a list of PIL RGBA images.

    *frame_bgr* is the background video frame (BGR ndarray).
    """
    h, w = frame_bgr.shape[:2]
    total_frames = max(1, int(fps * duration))

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base_pil = Image.fromarray(frame_rgb).convert("RGBA")
    # Keep the speaker clearly visible in the title background.
    # Normal practical range here is alpha 100-170:
    # - lower = cleaner/brighter frame
    # - higher = stronger title contrast
    dark_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 135))
    dark_base = Image.alpha_composite(base_pil, dark_overlay)

    font_path = _find_font()
    hook_upper = hook_text.upper()

    frames = []
    for idx in range(total_frames):
        p = idx / max(1, total_frames - 1)  # progress 0..1
        bg = dark_base.copy()

        if style == "kinetic":
            bg = _render_kinetic_frame(bg, hook_upper, p, w, h, font_path)
        elif style == "glitch":
            bg = _render_glitch_frame(bg, hook_upper, p, w, h, font_path)
        elif style == "neon":
            bg = _render_neon_frame(bg, hook_upper, p, w, h, font_path)
        elif style == "distortion":
            bg = _render_distortion_frame(bg, hook_upper, p, w, h, font_path)

        arr = np.asarray(bg.convert("RGB"), dtype=np.uint8)
        arr = _add_vignette(arr, 0.12)
        frames.append(Image.fromarray(arr).convert("RGBA"))

    return frames


def _render_kinetic_frame(base, text, p, w, h, font_path):
    """Kinetic word-by-word pop-in with scale + rotation."""
    words = text.split()
    word_font_size = max(40, int(w * _FANCY_FONT_RATIO))

    for idx, word in enumerate(words):
        local = max(0.0, min(1.0, p * len(words) - idx))
        if local <= 0.0:
            continue

        font = (ImageFont.truetype(font_path, word_font_size)
            if font_path else ImageFont.load_default())
        layer = _word_sprite(word, font)

        # Scale up with ease-out
        scale = 0.5 + 0.5 * (1.0 - (1.0 - local) ** 3)
        new_w = max(2, int(layer.width * scale))
        new_h = max(2, int(layer.height * scale))
        layer = layer.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Rotation
        rot = (idx - len(words) / 2) * 6.0 * (1.0 - local)
        layer = layer.rotate(rot, expand=True, resample=Image.Resampling.BICUBIC)

        # Spread words across vertical lanes to avoid same-position stacking.
        # For two-word hooks this lands naturally on line-1 and line-2.
        if len(words) == 2:
            lane_ratios = [0.38, 0.62]
            y_center = h * lane_ratios[idx]
        else:
            y_center = h * (0.26 + idx * min(0.11, 0.48 / max(len(words), 1)))
        x = int(w * 0.5 - layer.width / 2)
        y = int(y_center - layer.height / 2)
        base.alpha_composite(layer, (max(0, x), max(0, y)))

    return base


def _render_glitch_frame(base, text, p, w, h, font_path):
    """RGB split + noise glitch effect."""
    font_size = max(48, int(w * _FANCY_FONT_RATIO))
    font = (ImageFont.truetype(font_path, font_size)
            if font_path else ImageFont.load_default())
    layer = _center_text_layer(w, h, text, font)

    arr = np.asarray(layer.convert("RGB"), dtype=np.uint8)
    offset = max(2, int(2 + 8 * (0.5 + 0.5 * math.sin(p * math.pi * 6.0))))
    arr = _rgb_split(arr, offset)

    # Add noise
    rng = np.random.RandomState(int(p * 999) + 7)
    noise = rng.randint(0, 30, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

    layer = Image.fromarray(arr, mode="RGB").convert("RGBA")
    base.alpha_composite(layer)
    return base


def _render_neon_frame(base, text, p, w, h, font_path):
    """Neon glow bloom effect with cycling color."""
    font_size = max(48, int(w * _FANCY_FONT_RATIO))
    font = (ImageFont.truetype(font_path, font_size)
            if font_path else ImageFont.load_default())
    color = (70, 255, 220)
    layer = _center_text_layer(w, h, text, font, color=color)

    # Double gaussian glow
    glow = layer.filter(ImageFilter.GaussianBlur(radius=18))
    glow2 = layer.filter(ImageFilter.GaussianBlur(radius=8))
    base.alpha_composite(glow)
    base.alpha_composite(glow2)
    base.alpha_composite(layer)
    return base


def _render_distortion_frame(base, text, p, w, h, font_path):
    """Sine-wave distortion of text."""
    font_size = max(46, int(w * _FANCY_FONT_RATIO))
    font = (ImageFont.truetype(font_path, font_size)
            if font_path else ImageFont.load_default())
    layer = _center_text_layer(w, h, text, font)

    arr = np.asarray(layer.convert("RGB"), dtype=np.uint8)
    lh, lw = arr.shape[:2]
    yy, xx = np.indices((lh, lw), dtype=np.float32)
    map_x = xx + 12.0 * np.sin(yy / 18.0 + p * 6.0)
    map_y = yy
    arr = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    base.alpha_composite(Image.fromarray(arr, mode="RGB").convert("RGBA"))
    return base


def generate_thumbnail_card(
    video_path,
    hook_text,
    accent_keyword="",
    duration=THUMBNAIL_DURATION,
    output_path=None,
    thumbnail_image_path=None,
    style="random",
):
    """Create a thumbnail-style hook video (speaker visible, bold text overlay).

    Also exports a standalone thumbnail image (JPG) for platform use.

    Parameters
    ----------
    video_path : str
        Path to the cropped 9:16 video.
    hook_text : str
        Short 2–5 word hook (will be uppercased).
    accent_keyword : str
        Single word from the hook to render in accent colour.
    duration : float
        Length of the thumbnail card (default 1.0 s).
    output_path : str, optional
        Where to write the thumbnail card video.
    thumbnail_image_path : str, optional
        Where to write the standalone thumbnail image.
        Defaults to ``<output_dir>/<video_basename>_thumb.jpg``.
    style : str
        ``"random"`` (default), ``"auto"``, ``"classic"``,
        ``"text_reveal"``, or ``"dramatic"``.

    Returns
    -------
    tuple[str, str, str]
        ``(video_path, thumbnail_image_path, style_used)`` — paths to the
        generated thumbnail video, the standalone thumbnail image, and the
        name of the style that was rendered.
    """
    frame = _select_best_frame(video_path)
    fps = _get_video_fps(video_path)
    h_frame, w_frame = frame.shape[:2]
    total_frames = int(fps * duration)

    # Resolve the actual style name before rendering.
    # "random" / "auto" both draw from _ACTIVE_STYLES (text_reveal + classic only).
    resolved_style = style
    if style in ("random", "auto"):
        resolved_style = random.choices(_ACTIVE_STYLES, weights=_ACTIVE_STYLE_WEIGHTS, k=1)[0]

    # ── Export standalone thumbnail image ──
    if thumbnail_image_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        thumbnail_image_path = os.path.join(
            os.path.dirname(video_path), f"{base}_thumb.jpg"
        )

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(video_path), "_thumbnail.mp4"
        )

    tmp_dir = tempfile.mkdtemp(prefix="thumbnail_")

    # ── Fancy styles: frame-by-frame rendering ──
    if resolved_style in FANCY_STYLES:
        fancy_frames = _render_fancy_frames(
            frame, hook_text, resolved_style, fps, duration
        )
        # Save first frame as thumbnail image
        fancy_frames[0].convert("RGB").save(thumbnail_image_path, "JPEG", quality=92)

        # Write PNG sequence
        for i, fr in enumerate(fancy_frames):
            fr.convert("RGB").save(os.path.join(tmp_dir, f"frame_{i:04d}.png"))

        print(f"  Title card style: {resolved_style} ({len(fancy_frames)} frames)")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%04d.png"),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-vf", f"fade=t=out:st={duration - 0.20}:d=0.20",
            "-t", str(duration),
            *NVENC_FLAGS,
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg fancy title generation failed:\n{result.stderr}"
            )

        # Cleanup
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

        return output_path, thumbnail_image_path, resolved_style

    # ── Track-matte style: frame-by-frame end reveal zoom ──
    if resolved_style == "text_reveal":
        reveal_frames = _render_text_reveal_transition_frames(
            frame,
            hook_text,
            accent_keyword,
            fps,
            duration,
        )
        reveal_frames[0].convert("RGB").save(thumbnail_image_path, "JPEG", quality=92)

        for i, fr in enumerate(reveal_frames):
            fr.convert("RGB").save(os.path.join(tmp_dir, f"frame_{i:04d}.png"))

        print(
            "  Title card style: text_reveal "
            f"({len(reveal_frames)} frames, end reveal {_TEXT_REVEAL_END_REVEAL_SEC:.2f}s)"
        )

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%04d.png"),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", str(duration),
            *NVENC_FLAGS,
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg text-reveal transition generation failed:\n{result.stderr}"
            )

        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

        return output_path, thumbnail_image_path, resolved_style

    # ── Classic / text_reveal / dramatic: single-frame + zoompan ──
    img = _render_thumbnail(frame, hook_text, accent_keyword,
                            style=resolved_style)

    img.convert("RGB").save(thumbnail_image_path, "JPEG", quality=92)

    png_path = os.path.join(tmp_dir, "thumb.png")
    img.convert("RGB").save(png_path)

    # Random motion effect for visual energy (NO fade-in so frame 0
    # = clean thumbnail for platform cover images).
    motion_style = random.choice(["shake", "zoom_pulse", "drift", "push_in"])
    if motion_style == "shake":
        # Quick horizontal/vertical shake — 8 Hz, ±12 px
        zoom_expr = (
            f"zoompan="
            f"z='1.03':"
            f"x='iw/2-(iw/zoom/2)+12*sin(on*8*2*PI/{total_frames})':"
            f"y='ih/2-(ih/zoom/2)+8*cos(on*11*2*PI/{total_frames})':"
            f"d={total_frames}:"
            f"s={w_frame}x{h_frame}:"
            f"fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )
    elif motion_style == "zoom_pulse":
        # Zoom from 1.0 → 1.08 with a fast ease-out
        zoom_expr = (
            f"zoompan="
            f"z='1+0.08*(1-pow(1-on/{total_frames},3))':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:"
            f"s={w_frame}x{h_frame}:"
            f"fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )
    elif motion_style == "drift":
        # Slow diagonal drift — moves ~20 px total
        zoom_expr = (
            f"zoompan="
            f"z='1.04':"
            f"x='iw/2-(iw/zoom/2)+20*on/{total_frames}':"
            f"y='ih/2-(ih/zoom/2)-15*on/{total_frames}':"
            f"d={total_frames}:"
            f"s={w_frame}x{h_frame}:"
            f"fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )
    else:
        # push_in: original subtle 1.2% push-in zoom
        zoom_expr = (
            f"zoompan="
            f"z='1+0.012*on/{total_frames}':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:"
            f"s={w_frame}x{h_frame}:"
            f"fps={fps},"
            f"fade=t=out:st={duration - 0.20}:d=0.20"
        )
    print(f"  Title card motion: {motion_style}")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", png_path,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", f"[0:v]{zoom_expr}[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-t", str(duration),
        *NVENC_FLAGS,
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg thumbnail generation failed:\n{result.stderr}"
        )

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except OSError:
        pass

    return output_path, thumbnail_image_path, resolved_style


# ══════════════════════════════════════════════════════════════════
#  CINEMATIC MODE  — animated intro card (previous default)
# ══════════════════════════════════════════════════════════════════

def _render_background(frame_bgr):
    """Blurred frame + dark overlay + radial vignette → RGBA Pillow image."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.filter(ImageFilter.GaussianBlur(radius=_CINE_BLUR_RADIUS))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, _CINE_OVERLAY_OPACITY))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)

    w, h = img.size
    vig = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vig_draw = ImageDraw.Draw(vig)
    cx, cy = w // 2, h // 2
    max_r = math.hypot(cx, cy)
    steps = 30
    for i in range(steps, 0, -1):
        frac = i / steps
        alpha = int(100 * frac * frac)
        r = int(max_r * frac)
        vig_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, vig)
    return img


def _render_text_overlay(width, height, title, subtitle=None):
    """Cinematic mode: title text + accent line on transparent RGBA canvas."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title_size = max(28, int(height * _CINE_FONT_RATIO))
    sub_size = max(18, int(height * 0.032))

    font_path = _find_font()
    if font_path:
        title_font = ImageFont.truetype(font_path, title_size)
        sub_font = ImageFont.truetype(font_path, sub_size)
    else:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    max_text_w = int(width * 0.80)

    title_lines = _wrap_text(title, title_font, max_text_w, draw)
    line_height = int(title_size * 1.40)
    total_title_h = line_height * len(title_lines)

    accent_gap = int(height * 0.018)
    accent_h = _CINE_ACCENT_LINE_THICKNESS

    sub_lines: list[str] = []
    sub_line_height = int(sub_size * 1.35)
    total_sub_h = 0
    if subtitle:
        sub_lines = _wrap_text(subtitle, sub_font, max_text_w, draw)
        total_sub_h = sub_line_height * len(sub_lines)

    sub_gap = int(height * 0.025) if sub_lines else 0
    total_h = total_title_h + accent_gap + accent_h + sub_gap + total_sub_h
    y = (height - total_h) // 2

    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        x = (width - lw) // 2
        for dx, dy, a in [(-2, -2, 40), (2, -2, 40), (-2, 2, 40), (2, 2, 40),
                          (0, 3, 60), (0, -3, 60), (3, 0, 60), (-3, 0, 60)]:
            draw.text((x + dx, y + dy), line, font=title_font, fill=(255, 255, 255, a))
        draw.text((x + 2, y + 3), line, font=title_font, fill=(0, 0, 0, 180))
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255))
        y += line_height

    y += accent_gap
    accent_w = int(width * _CINE_ACCENT_LINE_W_RATIO)
    ax = (width - accent_w) // 2
    draw.rectangle([ax, y, ax + accent_w, y + accent_h], fill=(255, 255, 255, 180))
    y += accent_h + sub_gap

    for line in sub_lines:
        bbox = draw.textbbox((0, 0), line, font=sub_font)
        lw = bbox[2] - bbox[0]
        x = (width - lw) // 2
        draw.text((x + 1, y + 1), line, font=sub_font, fill=(0, 0, 0, 140))
        draw.text((x, y), line, font=sub_font, fill=(220, 220, 220, 255))
        y += sub_line_height

    return img


def _render_title_image(frame_bgr, title, subtitle=None):
    """Return a Pillow RGBA image compositing cinematic background + text.

    Kept for backward compatibility and tests.
    """
    bg = _render_background(frame_bgr)
    w, h = bg.size
    txt = _render_text_overlay(w, h, title, subtitle)
    return Image.alpha_composite(bg, txt)


def generate_title_card(
    video_path,
    title,
    subtitle=None,
    duration=CINEMATIC_DURATION,
    output_path=None,
):
    """Create an animated cinematic title-card video.

    Layers:
      1. Background (blurred frame) with slow Ken Burns zoom + vignette
      2. Text overlay fades in with upward drift, then fades to black.
    """
    frame = _extract_first_frame(video_path)
    fps = _get_video_fps(video_path)
    h_frame, w_frame = frame.shape[:2]
    total_frames = int(fps * duration)

    bg = _render_background(frame)
    txt = _render_text_overlay(w_frame, h_frame, title, subtitle)

    tmp_dir = tempfile.mkdtemp(prefix="titlecard_")
    bg_path = os.path.join(tmp_dir, "bg.png")
    txt_path = os.path.join(tmp_dir, "txt.png")
    bg.convert("RGB").save(bg_path)
    txt.save(txt_path)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(video_path), "_titlecard.mp4")

    fade_out_start = duration - _CINE_FADE_OUT_OFFSET
    fade_in_end = _CINE_FADE_IN_START + _CINE_FADE_IN_DUR

    zoompan_expr = (
        f"zoompan="
        f"z='1+{_CINE_ZOOM_FACTOR}*on/{total_frames}':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={w_frame}x{h_frame}:fps={fps}"
    )

    txt_fade = (
        f"format=rgba,"
        f"fade=t=in:st={_CINE_FADE_IN_START}:d={_CINE_FADE_IN_DUR}:alpha=1"
    )
    txt_y = (
        f"if(lt(t,{_CINE_FADE_IN_START}),{_CINE_TEXT_DRIFT_PX},"
        f"if(lt(t,{fade_in_end}),"
        f"floor({_CINE_TEXT_DRIFT_PX}*(1-(t-{_CINE_FADE_IN_START})/{_CINE_FADE_IN_DUR})"
        f"*(1-(t-{_CINE_FADE_IN_START})/{_CINE_FADE_IN_DUR})),0))"
    )

    filter_complex = (
        f"[0:v]{zoompan_expr}[bg];"
        f"[1:v]{txt_fade}[txt];"
        f"[bg][txt]overlay=x=0:y='{txt_y}':format=auto:shortest=1[comp];"
        f"[comp]fade=t=out:st={fade_out_start}:d={_CINE_FADE_OUT_OFFSET}[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", bg_path,
        "-loop", "1", "-i", txt_path,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "2:a",
        "-t", str(duration),
        *NVENC_FLAGS,
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg title-card generation failed:\n{result.stderr}")

    for p in (bg_path, txt_path):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return output_path


# ══════════════════════════════════════════════════════════════════
#  CONCAT / ASSEMBLY helpers
# ══════════════════════════════════════════════════════════════════

def prepend_title_card(title_card_path, main_video_path, output_path):
    """Concatenate *title_card_path* + *main_video_path* into *output_path*.

    Uses the FFmpeg concat demuxer for frame-accurate lossless join.
    Both inputs must share the same codec / resolution / fps.
    """
    tmp_dir = tempfile.mkdtemp(prefix="concat_")
    list_path = os.path.join(tmp_dir, "concat.txt")
    with open(list_path, "w") as f:
        f.write(f"file '{os.path.abspath(title_card_path)}'\n")
        f.write(f"file '{os.path.abspath(main_video_path)}'\n")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg concat failed:\n{result.stderr}"
        )

    # Cleanup
    try:
        os.remove(list_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return output_path


def assemble_with_title_card(
    title_card_path,
    subtitled_video_path,
    audio_source_path,
    output_path,
    title_duration=CINEMATIC_DURATION,
    speech_gain_db=0.0,
    bg_music_path=None,
    music_gain_db=None,
):
    """Assemble final short: title card video + subtitled video + delayed audio.

    1. Concatenates the title-card and subtitled video streams.
    2. Takes the audio from *audio_source_path*, boosts by *speech_gain_db*,
       and delays it by *title_duration* so speech starts after the title card.
    3. Optionally mixes in background music at -15 dB with a 5 s fade-out.
    """
    delay_ms = int(title_duration * 1000)

    # Total duration = title card + subtitled video
    title_dur = _get_media_duration(title_card_path)
    sub_dur = _get_media_duration(subtitled_video_path)
    total_dur = title_dur + sub_dur

    # Build speech filter chain
    speech_filter = f"[2:a]adelay={delay_ms}|{delay_ms}"
    if speech_gain_db:
        speech_filter += f",volume={speech_gain_db:.1f}dB"
    speech_filter += "[speech]"

    inputs = [
        "-i", os.path.abspath(title_card_path),
        "-i", os.path.abspath(subtitled_video_path),
        "-i", os.path.abspath(audio_source_path),
    ]

    if bg_music_path and os.path.isfile(bg_music_path):
        inputs += ["-i", os.path.abspath(bg_music_path)]
        _music_db = music_gain_db if music_gain_db is not None else _BG_MUSIC_VOLUME_DB
        music_filter = _bg_music_filter(3, total_dur, volume_db=_music_db)
        filter_complex = (
            f"[0:v][1:v]concat=n=2:v=1:a=0[v];"
            f"{speech_filter};"
            f"{music_filter};"
            f"[speech][music]amix=inputs=2:duration=first:normalize=0[a]"
        )
    else:
        filter_complex = (
            f"[0:v][1:v]concat=n=2:v=1:a=0[v];"
            f"{speech_filter.replace('[speech]', '[a]')}"
        )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        *NVENC_FLAGS,
        "-c:a", "aac", "-b:a", "192k",
        os.path.abspath(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg title-card assembly failed:\n{result.stderr}"
        )

    return output_path
