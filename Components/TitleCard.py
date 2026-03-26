"""Title-card generator for short-form sermon clips.

Creates a 2.5-second animated intro card with After-Effects-style motion:
  • Background: blurred first frame with slow Ken Burns zoom + vignette
  • Text: fades in with upward drift (0.3–0.9 s), accent line under title
  • Overall fade-out in the last 0.3 s

The result is concatenated with the main clip via FFmpeg.
"""

import math
import os
import subprocess
import tempfile

import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Visual defaults ──────────────────────────────────────────────
TITLE_DURATION = 2.5  # seconds
BLUR_RADIUS = 25  # moderate – the scene is still recognisable
OVERLAY_OPACITY = 110  # lighter than before so the background shows through
TITLE_FONT_RATIO = 0.065  # fraction of image height
SUBTITLE_FONT_RATIO = 0.032
ACCENT_LINE_WIDTH_RATIO = 0.25  # fraction of image width
ACCENT_LINE_THICKNESS = 3

# Animation timing (seconds)
_FADE_IN_START = 0.25
_FADE_IN_DUR = 0.60
_FADE_OUT_START_OFFSET = 0.30  # before end
_TEXT_DRIFT_PX = 20  # pixels the text drifts upward during fade-in
_ZOOM_FACTOR = 0.018  # total zoom increase over the duration (1.0 → 1.018)

# Font discovery order (first match wins)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
]

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


# ── Layer renderers ──────────────────────────────────────────────

def _render_background(frame_bgr):
    """Blurred frame + dark overlay + radial vignette → RGB Pillow image."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Moderate Gaussian blur (scene still recognisable)
    img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # Semi-transparent dark overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, OVERLAY_OPACITY))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)

    # Radial vignette: darken edges, keep centre brighter
    w, h = img.size
    vig = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vig_draw = ImageDraw.Draw(vig)
    cx, cy = w // 2, h // 2
    max_r = math.hypot(cx, cy)
    steps = 30
    for i in range(steps, 0, -1):
        frac = i / steps  # 1.0 = outermost
        alpha = int(100 * frac * frac)  # quadratic falloff
        r = int(max_r * frac)
        vig_draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(0, 0, 0, alpha),
        )
    img = Image.alpha_composite(img, vig)

    return img


def _render_text_overlay(width, height, title, subtitle=None):
    """Title text + accent line on a transparent RGBA canvas."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Font sizes
    title_size = max(28, int(height * TITLE_FONT_RATIO))
    sub_size = max(18, int(height * SUBTITLE_FONT_RATIO))

    font_path = _find_font()
    if font_path:
        title_font = ImageFont.truetype(font_path, title_size)
        sub_font = ImageFont.truetype(font_path, sub_size)
    else:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    max_text_w = int(width * 0.80)

    # ── Title lines ──
    title_lines = _wrap_text(title, title_font, max_text_w, draw)
    line_height = int(title_size * 1.40)
    total_title_h = line_height * len(title_lines)

    # ── Accent line ──
    accent_gap = int(height * 0.018)
    accent_h = ACCENT_LINE_THICKNESS

    # ── Subtitle lines ──
    sub_lines: list[str] = []
    sub_line_height = int(sub_size * 1.35)
    total_sub_h = 0
    if subtitle:
        sub_lines = _wrap_text(subtitle, sub_font, max_text_w, draw)
        total_sub_h = sub_line_height * len(sub_lines)

    sub_gap = int(height * 0.025) if sub_lines else 0
    total_h = total_title_h + accent_gap + accent_h + sub_gap + total_sub_h

    y = (height - total_h) // 2

    # Draw title lines — white with soft glow
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        x = (width - lw) // 2
        # Soft glow (draw at offsets with low alpha)
        for dx, dy, a in [(-2, -2, 40), (2, -2, 40), (-2, 2, 40), (2, 2, 40),
                          (0, 3, 60), (0, -3, 60), (3, 0, 60), (-3, 0, 60)]:
            draw.text((x + dx, y + dy), line, font=title_font, fill=(255, 255, 255, a))
        # Drop shadow
        draw.text((x + 2, y + 3), line, font=title_font, fill=(0, 0, 0, 180))
        # Main text
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255))
        y += line_height

    y += accent_gap

    # ── Accent line (thin white rule, centred) ──
    accent_w = int(width * ACCENT_LINE_WIDTH_RATIO)
    ax = (width - accent_w) // 2
    draw.rectangle([ax, y, ax + accent_w, y + accent_h], fill=(255, 255, 255, 180))
    y += accent_h + sub_gap

    # ── Subtitle lines (light grey) ──
    for line in sub_lines:
        bbox = draw.textbbox((0, 0), line, font=sub_font)
        lw = bbox[2] - bbox[0]
        x = (width - lw) // 2
        draw.text((x + 1, y + 1), line, font=sub_font, fill=(0, 0, 0, 140))
        draw.text((x, y), line, font=sub_font, fill=(220, 220, 220, 255))
        y += sub_line_height

    return img


# ── Legacy compatibility shim ────────────────────────────────────

def _render_title_image(frame_bgr, title, subtitle=None):
    """Return a Pillow RGBA image: blurred frame + overlay + text.

    Kept for backward compatibility and tests.  Internally delegates to
    the new two-layer renderers.
    """
    bg = _render_background(frame_bgr)
    w, h = bg.size
    txt = _render_text_overlay(w, h, title, subtitle)
    return Image.alpha_composite(bg, txt)


# ── Frame / FPS helpers ──────────────────────────────────────────

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


def _get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return fps
    finally:
        cap.release()


# ── Animated title-card generation ───────────────────────────────

def generate_title_card(
    video_path,
    title,
    subtitle=None,
    duration=TITLE_DURATION,
    output_path=None,
):
    """Create an animated title-card video with After-Effects-style motion.

    Layers:
      1. Background (blurred frame) with slow Ken Burns zoom + vignette
      2. Text overlay fades in with upward drift, then the whole card
         fades to black at the end.
    """
    frame = _extract_first_frame(video_path)
    fps = _get_video_fps(video_path)
    h_frame, w_frame = frame.shape[:2]
    total_frames = int(fps * duration)

    bg = _render_background(frame)
    txt = _render_text_overlay(w_frame, h_frame, title, subtitle)

    # Write layers as PNGs
    tmp_dir = tempfile.mkdtemp(prefix="titlecard_")
    bg_path = os.path.join(tmp_dir, "bg.png")
    txt_path = os.path.join(tmp_dir, "txt.png")
    bg.convert("RGB").save(bg_path)
    txt.save(txt_path)  # keep RGBA for alpha overlay

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(video_path), "_titlecard.mp4"
        )

    fade_out_start = duration - _FADE_OUT_START_OFFSET
    fade_in_end = _FADE_IN_START + _FADE_IN_DUR

    # Expressions for FFmpeg:
    # Ken Burns slow zoom centred on the image
    zoompan_expr = (
        f"zoompan="
        f"z='1+{_ZOOM_FACTOR}*on/{total_frames}':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:"
        f"s={w_frame}x{h_frame}:"
        f"fps={fps}"
    )

    # Text overlay: fade alpha in + drift upward
    # y offset: starts at _TEXT_DRIFT_PX, eases to 0 during fade-in
    # Drift uses smoothstep for ease-out feel
    txt_fade = (
        f"format=rgba,"
        f"fade=t=in:st={_FADE_IN_START}:d={_FADE_IN_DUR}:alpha=1"
    )
    txt_y = (
        f"if(lt(t,{_FADE_IN_START}),{_TEXT_DRIFT_PX},"
        f"if(lt(t,{fade_in_end}),"
        f"floor({_TEXT_DRIFT_PX}*(1-(t-{_FADE_IN_START})/{_FADE_IN_DUR})*(1-(t-{_FADE_IN_START})/{_FADE_IN_DUR})),"
        f"0))"
    )

    filter_complex = (
        f"[0:v]{zoompan_expr}[bg];"
        f"[1:v]{txt_fade}[txt];"
        f"[bg][txt]overlay=x=0:y='{txt_y}':format=auto:shortest=1[comp];"
        f"[comp]fade=t=out:st={fade_out_start}:d={_FADE_OUT_START_OFFSET}[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", bg_path,
        "-loop", "1", "-i", txt_path,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "2:a",
        "-t", str(duration),
        *NVENC_FLAGS,
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg title-card generation failed:\n{result.stderr}"
        )

    # Cleanup temp files
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


# ── Concat / assembly helpers ────────────────────────────────────

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
    title_duration=TITLE_DURATION,
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
