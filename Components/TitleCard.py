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
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Shared defaults ──────────────────────────────────────────────
THUMBNAIL_DURATION = 1.0  # seconds — short cover
CINEMATIC_DURATION = 2.5  # seconds — animated intro

# Accent colour for the keyword (warm red, like TikTok thumbnails)
ACCENT_COLOUR = (220, 40, 40, 255)  # RGBA red
ACCENT_COLOUR_HEX = "#DC2828"

# Font discovery order (first match wins)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
]

# Thumbnail text sizing
_THUMB_FONT_RATIO = 0.080  # bigger than cinematic
_THUMB_MAX_TEXT_W_RATIO = 0.88

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


def _render_thumbnail(frame_bgr, hook_text, accent_keyword=""):
    """Render a TikTok/YouTube-style thumbnail cover frame.

    Returns a Pillow RGBA image: visible speaker + bottom gradient + bold text.
    Face-aware darkening preserves the speaker's face while adding contrast
    to surrounding areas for text readability.
    """
    h, w = frame_bgr.shape[:2]

    # Convert to Pillow (NO heavy blur — speaker must be visible)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).convert("RGBA")

    # Face-aware selective darkening (Ergänzung 3)
    face_box = _detect_face_box(frame_bgr)
    img = _face_aware_darken(img, face_box, w, h)

    # Bottom gradient: dark at bottom, transparent at top
    gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    grad_draw = ImageDraw.Draw(gradient)
    gradient_start = int(h * 0.50)  # gradient starts at 50% height
    for y_pos in range(gradient_start, h):
        frac = (y_pos - gradient_start) / (h - gradient_start)
        alpha = int(180 * frac * frac)  # quadratic ramp
        grad_draw.line([(0, y_pos), (w, y_pos)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, gradient)

    # Face-aware text placement (reuse already-detected face_box)
    lane_top, lane_bottom = _pick_text_lane(face_box, w, h)

    # ── Render hook text ──
    font_path = _find_font()
    font_size = max(36, int(h * _THUMB_FONT_RATIO))
    if font_path:
        title_font = ImageFont.truetype(font_path, font_size)
    else:
        title_font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    max_text_w = int(w * _THUMB_MAX_TEXT_W_RATIO)
    hook_upper = hook_text.upper()
    lines = _wrap_text(hook_upper, title_font, max_text_w, draw)
    line_height = int(font_size * 1.30)
    total_text_h = line_height * len(lines)

    # Center text vertically in the chosen lane
    lane_h = lane_bottom - lane_top
    text_y = lane_top + max(0, (lane_h - total_text_h) // 2)

    accent_upper = accent_keyword.upper() if accent_keyword else ""

    for line in lines:
        # Measure full line width for centering
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) // 2

        if accent_upper and accent_upper in line:
            # Draw word by word, accent keyword in color
            _draw_line_with_accent(
                draw, x, text_y, line, accent_upper, title_font, ACCENT_COLOUR
            )
        else:
            # Strong drop shadow
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


def generate_thumbnail_card(
    video_path,
    hook_text,
    accent_keyword="",
    duration=THUMBNAIL_DURATION,
    output_path=None,
    thumbnail_image_path=None,
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

    Returns
    -------
    tuple[str, str]
        ``(video_path, thumbnail_image_path)`` — paths to the generated
        thumbnail video and the standalone thumbnail image.
    """
    frame = _select_best_frame(video_path)
    fps = _get_video_fps(video_path)
    h_frame, w_frame = frame.shape[:2]
    total_frames = int(fps * duration)

    img = _render_thumbnail(frame, hook_text, accent_keyword)

    # ── Export standalone thumbnail image (Ergänzung 1) ──
    if thumbnail_image_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        thumbnail_image_path = os.path.join(
            os.path.dirname(video_path), f"{base}_thumb.jpg"
        )
    img.convert("RGB").save(thumbnail_image_path, "JPEG", quality=92)

    # Write as PNG for FFmpeg input (lossless intermediate)
    tmp_dir = tempfile.mkdtemp(prefix="thumbnail_")
    png_path = os.path.join(tmp_dir, "thumb.png")
    img.convert("RGB").save(png_path)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(video_path), "_thumbnail.mp4"
        )

    # Very subtle 1.2% push-in zoom + text fade-in over 0.3s + fade-out last 0.2s
    zoom_expr = (
        f"zoompan="
        f"z='1+0.012*on/{total_frames}':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:"
        f"s={w_frame}x{h_frame}:"
        f"fps={fps},"
        f"fade=t=in:st=0:d=0.15,"
        f"fade=t=out:st={duration - 0.20}:d=0.20"
    )

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
        os.remove(png_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return output_path, thumbnail_image_path


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
):
    """Assemble final short: title card video + subtitled video + delayed audio.

    This replaces the separate ``prepend_title_card`` + ``combine_videos`` steps
    when a title card is used.  It:

    1. Concatenates the title-card and subtitled video streams.
    2. Takes the audio from *audio_source_path* and delays it by *title_duration*
       so the speech starts after the title card finishes.
    3. Outputs a single .mp4 with synced video + audio.
    """
    delay_ms = int(title_duration * 1000)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", os.path.abspath(title_card_path),
        "-i", os.path.abspath(subtitled_video_path),
        "-i", os.path.abspath(audio_source_path),
        "-filter_complex",
        (
            f"[0:v][1:v]concat=n=2:v=1:a=0[v];"
            f"[2:a]adelay={delay_ms}|{delay_ms}[a]"
        ),
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
