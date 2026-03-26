import os
import shutil
import subprocess
import tempfile

import cv2


MAX_WORDS_PER_CAPTION = 4
MAX_LINES_PER_CAPTION = 2

# TikTok-style colour palette (ASS uses &HAABBGGRR)
# Primary (active word):  bright yellow
ACTIVE_COLOUR = "&H0000FFFF"     # yellow  (#FFFF00)
# Inactive words:         pure white
INACTIVE_COLOUR = "&H00FFFFFF"   # white   (#FFFFFF)
# Outline:                dark shadow
OUTLINE_COLOUR = "&H00000000"    # black
# Box background:         semi-transparent dark
BOX_COLOUR = "&H96000000"        # ~60% opaque black

NVENC_FLAGS = [
    "-c:v",
    "h264_nvenc",
    "-preset",
    "p7",
    "-rc",
    "constqp",
    "-qp",
    "18",
    "-b:v",
    "0",
    "-gpu",
    "0",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
]


def _run_ffmpeg(command, description, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"{description} failed: {stderr}")


def _seconds_to_ass_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _escape_ass_text(text):
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _format_caption_lines(words):
    if len(words) <= 3 or MAX_LINES_PER_CAPTION == 1:
        return " ".join(words)

    split_index = min(3, max(2, (len(words) + 1) // 2))
    first_line = " ".join(words[:split_index])
    second_line = " ".join(words[split_index:])

    if not second_line:
        return first_line
    return f"{first_line}\n{second_line}"


def _chunk_transcriptions(transcriptions):
    chunked = []

    for text, start, end in transcriptions:
        words = text.split()
        if not words:
            continue

        total_words = len(words)
        duration = max(0.01, end - start)

        for index in range(0, total_words, MAX_WORDS_PER_CAPTION):
            chunk_words = words[index:index + MAX_WORDS_PER_CAPTION]
            chunk_start = start + (index / total_words) * duration
            chunk_end = start + (
                min(index + MAX_WORDS_PER_CAPTION, total_words) / total_words
            ) * duration
            chunked.append(
                [_format_caption_lines(chunk_words), chunk_start, chunk_end]
            )

    return chunked


def _read_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for subtitle burn: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    return width, height, duration


def _write_ass_file(subtitle_path, video_width, video_height, chunks,
                    word_events=None):
    """Write an ASS subtitle file.

    If *word_events* is provided (list of caption groups, each group being a
    list of ``{"text", "start", "end"}`` dicts), generate TikTok-style
    word-by-word karaoke.  Otherwise fall back to the legacy chunk approach.
    """
    font_size = max(36, int(video_height * 0.055))
    margin_v = max(100, int(video_height * 0.14))
    margin_h = max(40, int(video_width * 0.08))
    outline = max(3, int(video_height * 0.004))
    shadow = max(2, int(video_height * 0.002))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        # Default style: white bold text, dark box background (BorderStyle=3)
        (
            f"Style: Default,Arial Black,"
            f"{font_size},{INACTIVE_COLOUR},&H000000FF,"
            f"{OUTLINE_COLOUR},{BOX_COLOUR},"
            f"1,0,0,0,100,100,1,0,3,{outline},{shadow},2,"
            f"{margin_h},{margin_h},{margin_v},1"
        ),
        # Karaoke style: same but used for highlighted word override
        (
            f"Style: Active,Arial Black,"
            f"{font_size},{ACTIVE_COLOUR},&H000000FF,"
            f"{OUTLINE_COLOUR},{BOX_COLOUR},"
            f"1,0,0,0,100,100,1,0,3,{outline},{shadow},2,"
            f"{margin_h},{margin_h},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if word_events:
        # TikTok-style: each caption group shown as a block, active word
        # highlighted.  Each word's Dialogue spans from its start until the
        # NEXT word begins so that there are never gaps (no flickering).
        # The last word of a group extends to the next group's start; the
        # very last word holds for an extra 1.5 s.
        for g_idx, group in enumerate(word_events):
            if not group:
                continue
            group_words = [_escape_ass_text(w["text"]) for w in group]

            # Time at which this caption block should disappear: extend to
            # the start of the next group so captions stay on-screen during
            # speech pauses with the last word still highlighted.
            if g_idx + 1 < len(word_events) and word_events[g_idx + 1]:
                group_visible_end = word_events[g_idx + 1][0]["start"]
            else:
                group_visible_end = group[-1]["end"] + 1.5

            for word_idx, w in enumerate(group):
                # Display from this word's start until the next word begins
                # (or until the group disappears for the last word).
                display_start = w["start"]
                if word_idx + 1 < len(group):
                    display_end = group[word_idx + 1]["start"]
                else:
                    display_end = group_visible_end

                if display_end <= display_start:
                    display_end = display_start + 0.05

                # Build text line: all words shown, current one in yellow
                parts = []
                for j, gw in enumerate(group_words):
                    if j == word_idx:
                        parts.append(
                            r"{\c" + ACTIVE_COLOUR + r"\fscx105\fscy105}" + gw
                            + r"{\c" + INACTIVE_COLOUR + r"\fscx100\fscy100}"
                        )
                    else:
                        parts.append(gw)
                caption_text = _format_caption_lines(parts)
                caption_text = caption_text.replace("\n", r"\N")
                # Fade-in only on first word of group; no fade-out
                prefix = r"{\fad(150,0)}" if word_idx == 0 else ""
                safe = prefix + caption_text

                lines.append(
                    "Dialogue: 0,"
                    f"{_seconds_to_ass_time(display_start)},"
                    f"{_seconds_to_ass_time(display_end)},"
                    f"Default,,0,0,0,,{safe}"
                )
    else:
        # Legacy fallback: plain chunked subtitles with fade
        for text, start, end in chunks:
            safe_text = _escape_ass_text(text.strip())
            if not safe_text:
                continue
            fade = r"{\fad(150,80)}"
            lines.append(
                "Dialogue: 0,"
                f"{_seconds_to_ass_time(start)},"
                f"{_seconds_to_ass_time(end)},"
                f"Default,,0,0,0,,{fade}{safe_text}"
            )

    with open(subtitle_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _build_word_events(word_timestamps, video_start_time, video_duration):
    """Group word timestamps into caption blocks of MAX_WORDS_PER_CAPTION words.

    Each group is a list of ``{"text", "start", "end"}`` dicts with times
    already adjusted relative to the clip.
    """
    # Adjust timestamps for clip offset
    adjusted = []
    for w in word_timestamps:
        start = w["start"] - video_start_time
        end = w["end"] - video_start_time
        if end <= 0:
            continue
        if video_duration > 0 and start >= video_duration:
            continue
        start = max(0, start)
        if video_duration > 0:
            end = min(video_duration, end)
        text = w["text"].strip()
        if text and not text.startswith("["):
            adjusted.append({"text": text, "start": start, "end": end})

    if not adjusted:
        return []

    # Group into caption blocks
    groups = []
    for i in range(0, len(adjusted), MAX_WORDS_PER_CAPTION):
        group = adjusted[i:i + MAX_WORDS_PER_CAPTION]
        if group:
            groups.append(group)
    return groups


def add_subtitles_to_video(input_video, output_video, transcriptions,
                           video_start_time=0, word_timestamps=None):
    """
    Add subtitles to video based on transcription segments.

    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        transcriptions: List of [text, start, end] from transcribeAudio
        video_start_time: Start time offset if video was cropped
        word_timestamps: Optional list of {"text", "start", "end"} dicts for
            TikTok-style word-by-word highlighting
    """
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)

    video_width, video_height, video_duration = _read_video_metadata(input_video)

    # Build word-level karaoke events if real word timestamps are available
    word_events = None
    if word_timestamps:
        word_events = _build_word_events(word_timestamps, video_start_time, video_duration)

    relevant_transcriptions = []
    for text, start, end in transcriptions:
        adjusted_start = start - video_start_time
        adjusted_end = end - video_start_time

        if adjusted_end <= 0:
            continue
        if video_duration > 0 and adjusted_start >= video_duration:
            continue

        adjusted_start = max(0, adjusted_start)
        if video_duration > 0:
            adjusted_end = min(video_duration, adjusted_end)

        stripped_text = text.strip()
        if stripped_text and not stripped_text.startswith("["):
            relevant_transcriptions.append([stripped_text, adjusted_start, adjusted_end])

    if not relevant_transcriptions and not word_events:
        print("No transcriptions found for this video segment")
        shutil.copyfile(input_video, output_video)
        return

    chunked = _chunk_transcriptions(relevant_transcriptions) if relevant_transcriptions else []
    if not chunked and not word_events:
        print("No subtitle chunks generated for this video segment")
        shutil.copyfile(input_video, output_video)
        return

    subtitle_dir = os.path.dirname(output_video) or os.getcwd()
    subtitle_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ass",
            prefix="captions_",
            dir=subtitle_dir,
            delete=False,
        ) as handle:
            subtitle_path = handle.name

        _write_ass_file(subtitle_path, video_width, video_height, chunked,
                        word_events=word_events)

        n_events = sum(len(g) for g in word_events) if word_events else len(chunked)
        mode = "TikTok karaoke" if word_events else "chunked"
        print(
            f"Adding {n_events} subtitle events ({mode}) to video with FFmpeg NVENC..."
        )
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-vf",
            f"subtitles={os.path.basename(subtitle_path)}",
            "-an",
            *NVENC_FLAGS,
            output_video,
        ]
        _run_ffmpeg(command, "subtitle burn", cwd=subtitle_dir)
        print(f"Subtitles added successfully -> {output_video}")
    finally:
        if subtitle_path and os.path.exists(subtitle_path):
            os.remove(subtitle_path)
