"""Title-card generator for short-form sermon clips.

Creates a 2–3 second intro card by blurring the first frame of the
cropped video, adding a dark overlay, and rendering title text on top.
The result is concatenated with the main clip via FFmpeg.
"""

import os
import subprocess
import tempfile

import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Visual defaults ──────────────────────────────────────────────
TITLE_DURATION = 2.5  # seconds
BLUR_RADIUS = 40
OVERLAY_OPACITY = 160  # 0-255, higher = darker
TITLE_FONT_RATIO = 0.065  # fraction of image height
SUBTITLE_FONT_RATIO = 0.032

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
    lines = []
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


def _render_title_image(frame_bgr, title, subtitle=None):
    """Return a Pillow RGBA image: blurred frame + overlay + text."""
    # Convert BGR (OpenCV) → RGB (Pillow)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Heavy Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # Dark semi-transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, OVERLAY_OPACITY))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Font sizes
    title_size = max(28, int(h * TITLE_FONT_RATIO))
    sub_size = max(18, int(h * SUBTITLE_FONT_RATIO))

    font_path = _find_font()
    if font_path:
        title_font = ImageFont.truetype(font_path, title_size)
        sub_font = ImageFont.truetype(font_path, sub_size)
    else:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # Text area: centered, max 80% width
    max_text_w = int(w * 0.80)
    margin_x = int(w * 0.10)

    # ── Title ──
    title_lines = _wrap_text(title, title_font, max_text_w, draw)
    line_height = int(title_size * 1.35)
    total_title_h = line_height * len(title_lines)

    # ── Subtitle ──
    sub_lines = []
    sub_line_height = int(sub_size * 1.35)
    total_sub_h = 0
    if subtitle:
        sub_lines = _wrap_text(subtitle, sub_font, max_text_w, draw)
        total_sub_h = sub_line_height * len(sub_lines)

    gap = int(h * 0.03) if sub_lines else 0
    total_h = total_title_h + gap + total_sub_h

    y = (h - total_h) // 2

    # Draw title lines (white, centered)
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        x = (w - lw) // 2
        # Drop shadow
        draw.text((x + 2, y + 2), line, font=title_font, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255))
        y += line_height

    y += gap

    # Draw subtitle lines (light grey, centered)
    for line in sub_lines:
        bbox = draw.textbbox((0, 0), line, font=sub_font)
        lw = bbox[2] - bbox[0]
        x = (w - lw) // 2
        draw.text((x + 1, y + 1), line, font=sub_font, fill=(0, 0, 0, 160))
        draw.text((x, y), line, font=sub_font, fill=(220, 220, 220, 255))
        y += sub_line_height

    return img


def _extract_first_frame(video_path):
    """Read the first frame from *video_path* using OpenCV. Returns BGR ndarray."""
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


def generate_title_card(
    video_path,
    title,
    subtitle=None,
    duration=TITLE_DURATION,
    output_path=None,
):
    """Create a title-card video clip (still image + silent audio).

    Parameters
    ----------
    video_path : str
        Path to the cropped 9:16 video whose first frame is used as background.
    title : str
        Main title text (large, centered).
    subtitle : str, optional
        Secondary text below the title (smaller, grey).
    duration : float
        Length of the title card in seconds (default 2.5).
    output_path : str, optional
        Where to write the title card video.  If *None*, a temp file is used.

    Returns
    -------
    str
        Path to the generated title-card .mp4 file.
    """
    frame = _extract_first_frame(video_path)
    fps = _get_video_fps(video_path)

    img = _render_title_image(frame, title, subtitle)

    # Write the frame as a temporary PNG
    tmp_dir = tempfile.mkdtemp(prefix="titlecard_")
    png_path = os.path.join(tmp_dir, "title.png")
    img.convert("RGB").save(png_path)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(video_path), "_titlecard.mp4"
        )

    # FFmpeg: still image → video with silent audio
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1",
        "-i", png_path,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-t", str(duration),
        "-r", str(fps),
        *NVENC_FLAGS,
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg title-card generation failed:\n{result.stderr}"
        )

    # Cleanup temp PNG
    try:
        os.remove(png_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return output_path


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
