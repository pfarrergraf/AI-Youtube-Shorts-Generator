import os
import re
import shutil
import subprocess
import tempfile

import cv2


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

# Phrase segmentation defaults
HARD_PUNCT = {".", "?", "!"}
SOFT_PUNCT = {",", ";", ":"}
MIN_WORDS_BEFORE_SOFT_BREAK = 4
MIN_PHRASE_DURATION = 1.0
MAX_PHRASE_DURATION = 4.5
MAX_PHRASE_WORDS = 10
HOLD_AFTER_PHRASE_SEC = 0.4

# Stable wrapping defaults
TARGET_CHARS_PER_LINE = 18
MAX_CHARS_PER_LINE = 26

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


# ---------------------------------------------------------------------------
#  Phrase segmentation — group words into meaningful speech units
# ---------------------------------------------------------------------------

def _ends_with_hard_punct(word):
    """Check if the last character of *word* is a hard break punctuation."""
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in HARD_PUNCT


def _ends_with_soft_punct(word):
    """Check if the last character of *word* is a soft break punctuation."""
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in SOFT_PUNCT


def segment_words_into_phrases(words):
    """Group timestamped word dicts into semantic phrase blocks.

    Each input word is ``{"text": str, "start": float, "end": float}``.
    Returns a list of phrases, where each phrase is a list of word dicts.

    Break rules:
    - Hard break after ``.``, ``?``, ``!`` — always starts a new phrase.
    - Soft break after ``,``, ``;``, ``:`` — only if the phrase already has
      at least *MIN_WORDS_BEFORE_SOFT_BREAK* words **and** at least
      *MIN_PHRASE_DURATION* seconds.
    - Force-break if phrase exceeds *MAX_PHRASE_WORDS* or *MAX_PHRASE_DURATION*.
    """
    if not words:
        return []

    phrases = []
    current = []

    for w in words:
        current.append(w)
        phrase_dur = current[-1]["end"] - current[0]["start"]
        n = len(current)

        # Force-break: phrase too long
        if n >= MAX_PHRASE_WORDS or phrase_dur >= MAX_PHRASE_DURATION:
            phrases.append(current)
            current = []
            continue

        # Hard break: sentence-ending punctuation
        if _ends_with_hard_punct(w["text"]):
            phrases.append(current)
            current = []
            continue

        # Soft break: clause punctuation, only if phrase is substantial enough
        if _ends_with_soft_punct(w["text"]):
            if n >= MIN_WORDS_BEFORE_SOFT_BREAK and phrase_dur >= MIN_PHRASE_DURATION:
                phrases.append(current)
                current = []
                continue

    if current:
        phrases.append(current)

    return phrases


# ---------------------------------------------------------------------------
#  Stable text wrapping — balanced 2-line layout
# ---------------------------------------------------------------------------

def wrap_phrase_text(phrase_words):
    """Wrap a list of plain-text words into at most 2 balanced lines.

    Returns a string with ``\\n`` separating lines.  Tries to keep line
    widths roughly equal and within *TARGET_CHARS_PER_LINE* /
    *MAX_CHARS_PER_LINE* limits.
    """
    if not phrase_words:
        return ""

    text = " ".join(phrase_words)

    # Single line fits
    if len(text) <= MAX_CHARS_PER_LINE or MAX_LINES_PER_CAPTION == 1:
        return text

    # Try to split into 2 roughly balanced lines
    best_split = None
    best_diff = float("inf")
    running = 0

    for i, word in enumerate(phrase_words):
        running += len(word) + (1 if i > 0 else 0)
        remaining = len(text) - running - 1  # account for the space we remove
        if remaining < 0:
            break
        diff = abs(running - remaining)
        if diff < best_diff:
            best_diff = diff
            best_split = i + 1

    if best_split is None or best_split >= len(phrase_words):
        return text

    line1 = " ".join(phrase_words[:best_split])
    line2 = " ".join(phrase_words[best_split:])

    if not line2:
        return line1
    return f"{line1}\n{line2}"


def _format_caption_lines(words):
    """Legacy helper kept for backward compat with chunked subtitles."""
    if len(words) <= 3 or MAX_LINES_PER_CAPTION == 1:
        return " ".join(words)

    split_index = min(3, max(2, (len(words) + 1) // 2))
    first_line = " ".join(words[:split_index])
    second_line = " ".join(words[split_index:])

    if not second_line:
        return first_line
    return f"{first_line}\n{second_line}"


def _chunk_transcriptions(transcriptions):
    """Legacy fallback: split transcriptions into small caption chunks."""
    _MAX_CHUNK = 4
    chunked = []

    for text, start, end in transcriptions:
        words = text.split()
        if not words:
            continue

        total_words = len(words)
        duration = max(0.01, end - start)

        for index in range(0, total_words, _MAX_CHUNK):
            chunk_words = words[index:index + _MAX_CHUNK]
            chunk_start = start + (index / total_words) * duration
            chunk_end = start + (
                min(index + _MAX_CHUNK, total_words) / total_words
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
        # Default style: yellow karaoke fill (PrimaryColour = spoken word),
        # white unspoken text (SecondaryColour = not yet spoken),
        # dark box background (BorderStyle=3)
        (
            f"Style: Default,Arial Black,"
            f"{font_size},{ACTIVE_COLOUR},{INACTIVE_COLOUR},"
            f"{OUTLINE_COLOUR},{BOX_COLOUR},"
            f"1,0,0,0,100,100,1,0,3,{outline},{shadow},2,"
            f"{margin_h},{margin_h},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if word_events:
        # Phrase-based karaoke: ONE Dialogue per phrase with \kf tags.
        # All words are rendered simultaneously; the fill colour sweeps
        # through them progressively — no text redraws, no layout jitter.
        for phrase in word_events:
            if not phrase:
                continue

            phrase_start = phrase[0]["start"]
            # Hold phrase on-screen for HOLD_AFTER_PHRASE_SEC past last word
            phrase_end = phrase[-1]["end"] + HOLD_AFTER_PHRASE_SEC

            if phrase_end <= phrase_start:
                phrase_end = phrase_start + 0.1

            # Build \kf karaoke tag sequence
            # Format: {\kfDUR}word  where DUR is in centiseconds
            kf_parts = []
            for w_idx, w in enumerate(phrase):
                word_dur_cs = max(1, round((w["end"] - w["start"]) * 100))
                escaped = _escape_ass_text(w["text"])
                # Add space before word (except first)
                space = " " if w_idx > 0 else ""
                kf_parts.append(
                    f"{space}{{\\kf{word_dur_cs}}}{escaped}"
                )

            # Wrap into balanced 2-line layout (using \N for ASS line break)
            plain_words = [w["text"] for w in phrase]
            wrapped = wrap_phrase_text(plain_words)
            # If wrapping produced a line break, insert it into the kf sequence
            if "\n" in wrapped:
                split_at = wrapped.index("\n")
                chars_before_break = len(wrapped[:split_at])
                # Find the word index where the break goes
                running_chars = 0
                break_after_word = 0
                for wi, pw in enumerate(plain_words):
                    running_chars += len(pw) + (1 if wi > 0 else 0)
                    if running_chars >= chars_before_break:
                        break_after_word = wi
                        break
                # Rebuild kf_parts with \N inserted after break_after_word
                kf_parts_wrapped = []
                for wi, part in enumerate(kf_parts):
                    kf_parts_wrapped.append(part)
                    if wi == break_after_word and wi < len(kf_parts) - 1:
                        kf_parts_wrapped.append(r"\N")
                kf_parts = kf_parts_wrapped

            # Fade-in at phrase start
            karaoke_text = r"{\fad(150,0)}" + "".join(kf_parts)

            lines.append(
                "Dialogue: 0,"
                f"{_seconds_to_ass_time(phrase_start)},"
                f"{_seconds_to_ass_time(phrase_end)},"
                f"Default,,0,0,0,,{karaoke_text}"
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
    """Build phrase-segmented caption groups from word timestamps.

    Returns a list of phrases, where each phrase is a list of
    ``{"text", "start", "end"}`` dicts with times adjusted to the clip.
    Phrases are split at sentence/clause boundaries using
    ``segment_words_into_phrases()``.
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

    return segment_words_into_phrases(adjusted)


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

        n_events = len(word_events) if word_events else len(chunked)
        mode = "phrase karaoke" if word_events else "chunked"
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
