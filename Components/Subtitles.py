import os
import shutil
import subprocess
import tempfile

import cv2


MAX_LINES_PER_CAPTION = 2
MAX_WORDS_PER_PHRASE = 6

# ASS colours use &HAABBGGRR
ACTIVE_COLOUR = "&H0000FFFF"      # yellow
INACTIVE_COLOUR = "&H00FFFFFF"    # white
OUTLINE_COLOUR = "&H00101010"     # dark gray outline
BOX_COLOUR = "&HCC000000"         # 80% transparent black box

# Phrase segmentation defaults
HARD_PUNCT = {".", "?", "!"}
SOFT_PUNCT = {",", ";", ":"}
MIN_WORDS_BEFORE_SOFT_BREAK = 4
MIN_PHRASE_DURATION = 0.4
MAX_PHRASE_DURATION = 2
MAX_PHRASE_WORDS = MAX_WORDS_PER_PHRASE
MIN_PHRASE_WORDS = 1
HOLD_AFTER_PHRASE_SEC = 0.1

# Stable wrapping defaults
TARGET_CHARS_PER_LINE = 16
MAX_CHARS_PER_LINE = 22

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


def _ends_with_hard_punct(word):
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in HARD_PUNCT


def _ends_with_soft_punct(word):
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in SOFT_PUNCT


def _phrase_duration(words):
    if not words:
        return 0.0
    return max(0.0, words[-1]["end"] - words[0]["start"])


def _should_force_break(current_phrase):
    if not current_phrase:
        return False
    return (
        len(current_phrase) >= MAX_PHRASE_WORDS
        or _phrase_duration(current_phrase) >= MAX_PHRASE_DURATION
    )


def _should_soft_break(current_phrase, current_word):
    if not _ends_with_soft_punct(current_word["text"]):
        return False
    return (
        len(current_phrase) >= MIN_WORDS_BEFORE_SOFT_BREAK
        and _phrase_duration(current_phrase) >= MIN_PHRASE_DURATION
    )


def _merge_short_phrases(phrases):
    if not phrases:
        return []

    merged = []
    for phrase in phrases:
        if not phrase:
            continue

        if not merged:
            merged.append(phrase)
            continue

        is_too_short = (
            len(phrase) < MIN_PHRASE_WORDS
            or _phrase_duration(phrase) < MIN_PHRASE_DURATION
        )

        prev = merged[-1]
        prev_ends_hard = _ends_with_hard_punct(prev[-1]["text"])

        if is_too_short or not prev_ends_hard:
            candidate = prev + phrase
            if (
                len(candidate) <= MAX_PHRASE_WORDS
                and _phrase_duration(candidate) <= MAX_PHRASE_DURATION
            ):
                merged[-1] = candidate
            else:
                merged.append(phrase)
        else:
            merged.append(phrase)

    if len(merged) >= 2:
        last = merged[-1]
        if len(last) < MIN_PHRASE_WORDS or _phrase_duration(last) < MIN_PHRASE_DURATION:
            candidate = merged[-2] + last
            if (
                len(candidate) <= MAX_PHRASE_WORDS
                and _phrase_duration(candidate) <= MAX_PHRASE_DURATION
            ):
                merged[-2] = candidate
                merged.pop()

    return merged


def segment_words_into_phrases(words):
    if not words:
        return []

    phrases = []
    current = []

    for w in words:
        text = (w.get("text") or "").strip()
        if not text:
            continue

        current.append(w)

        if _ends_with_hard_punct(text):
            phrases.append(current)
            current = []
            continue

        if _should_force_break(current):
            phrases.append(current)
            current = []
            continue

        if _should_soft_break(current, w):
            phrases.append(current)
            current = []
            continue

    if current:
        phrases.append(current)

    return _merge_short_phrases(phrases)


def _join_words(words):
    return " ".join(words).strip()


def _split_index_candidates(words):
    candidates = []
    for i in range(1, len(words)):
        left = words[i - 1]
        left_last = left[-1] if left else ""
        score_bonus = 0

        if left_last in {",", ";", ":"}:
            score_bonus -= 4
        elif left_last in {".", "?", "!"}:
            score_bonus -= 6

        if i < 2:
            score_bonus += 5
        if len(words) - i < 2:
            score_bonus += 5

        candidates.append((i, score_bonus))
    return candidates


def wrap_phrase_words(phrase_words, max_lines=MAX_LINES_PER_CAPTION):
    words = [w for w in phrase_words if w]
    if not words:
        return [""]
    if max_lines <= 1 or len(words) == 1:
        return [_join_words(words)]

    if len(words) > MAX_WORDS_PER_PHRASE:
        words = words[:MAX_WORDS_PER_PHRASE]

    full_text = _join_words(words)
    if len(full_text) <= MAX_CHARS_PER_LINE:
        return [full_text]

    best = None
    best_score = float("inf")

    for split_idx, bonus in _split_index_candidates(words):
        line1 = _join_words(words[:split_idx])
        line2 = _join_words(words[split_idx:])

        if not line1 or not line2:
            continue

        len1 = len(line1)
        len2 = len(line2)
        longest = max(len1, len2)
        shortest = min(len1, len2)

        score = 0
        score += abs(len1 - len2) * 1.8
        score += abs(len1 - TARGET_CHARS_PER_LINE) * 1.0
        score += abs(len2 - TARGET_CHARS_PER_LINE) * 1.0

        if longest > MAX_CHARS_PER_LINE:
            score += (longest - MAX_CHARS_PER_LINE) * 10

        if shortest < 7:
            score += (7 - shortest) * 5

        score += bonus

        if score < best_score:
            best_score = score
            best = [line1, line2]

    if best is not None:
        return best

    midpoint = max(1, len(words) // 2)
    return [_join_words(words[:midpoint]), _join_words(words[midpoint:])]


def wrap_phrase_text(phrase_words):
    return "\n".join(wrap_phrase_words(phrase_words))


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


def _build_phrase_layout_metadata(phrase):
    plain_words = [w["text"] for w in phrase][:MAX_WORDS_PER_PHRASE]
    wrapped_lines = wrap_phrase_words(plain_words)

    line_ranges = []
    cursor = 0
    for line in wrapped_lines:
        count = len(line.split())
        line_ranges.append((cursor, cursor + count))
        cursor += count

    return wrapped_lines, line_ranges


def _colour_word(text, active=False):
    escaped = _escape_ass_text(text)
    colour = ACTIVE_COLOUR if active else INACTIVE_COLOUR
    return f"{{\\c{colour}}}{escaped}"


def _build_highlight_text_for_word(phrase, active_word_idx):
    wrapped_lines, line_ranges = _build_phrase_layout_metadata(phrase)
    lines = []

    for start_idx, end_idx in line_ranges:
        rendered_words = []
        for word_idx in range(start_idx, end_idx):
            rendered_words.append(
                _colour_word(
                    phrase[word_idx]["text"],
                    active=(word_idx == active_word_idx),
                )
            )
        lines.append(" ".join(rendered_words))

    return r"\N".join(lines)


def _normalise_phrase_timings(word_events):
    """Make all word Dialogue events strictly sequential — no overlaps ever.

    A global cursor tracks the end of the last emitted event.  Each new
    word starts at ``max(original_start, cursor)`` so two events can never
    occupy the same time range.  Within a phrase, each word extends to the
    next word's start (no gap).  Between phrases, the last word extends to
    ``min(natural_end, next_phrase_start)`` which may leave a deliberate
    silent gap.
    """
    if not word_events:
        return word_events

    normalised = []
    cursor = 0.0          # end-time of the last emitted Dialogue event
    total = len(word_events)

    for i, phrase in enumerate(word_events):
        if not phrase:
            continue
        phrase = phrase[:MAX_WORDS_PER_PHRASE]
        n = len(phrase)
        copied = []

        for j, w in enumerate(phrase):
            ws = max(w["start"], cursor)

            if j + 1 < n:
                # Mid-phrase: extend seamlessly to next word's start
                we = max(ws + 0.04, phrase[j + 1]["start"])
            elif i + 1 < total and word_events[i + 1]:
                # Last word of phrase: hold, but never overlap next phrase
                natural = w["end"] + HOLD_AFTER_PHRASE_SEC
                next_start = word_events[i + 1][0]["start"]
                we = min(natural, next_start)
                we = max(we, ws + 0.04)
            else:
                # Very last word overall
                we = max(ws + 0.04, w["end"] + HOLD_AFTER_PHRASE_SEC)

            copied.append({"text": w["text"], "start": ws, "end": we})
            cursor = we

        if copied:
            normalised.append(copied)

    return normalised


def _write_ass_file(subtitle_path, video_width, video_height, chunks, word_events=None):
    # slightly smaller than before
    font_size = max(33, int(video_height * 0.053) - 2)

    # keep captions clearly in lower third and away from center
    margin_v = max(190, int(video_height * 0.21))
    margin_h = max(70, int(video_width * 0.10))

    outline = max(3, int(video_height * 0.0035))
    shadow = max(1, int(video_height * 0.0012))

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
        (
            f"Style: Default,Arial Black,"
            f"{font_size},{INACTIVE_COLOUR},{INACTIVE_COLOUR},"
            f"{OUTLINE_COLOUR},{BOX_COLOUR},"
            f"1,0,0,0,100,100,0,0,3,{outline},{shadow},2,"
            f"{margin_h},{margin_h},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if word_events:
        word_events = _normalise_phrase_timings(word_events)

        for phrase in word_events:
            if not phrase:
                continue

            for word_idx, word in enumerate(phrase):
                event_start = word["start"]
                event_end = word["end"]

                if event_end <= event_start:
                    event_end = event_start + 0.03

                highlight_text = _build_highlight_text_for_word(phrase, word_idx)
                # Fade-in only when the phrase first appears (first word)
                prefix = r"{\fad(100,0)}" if word_idx == 0 else ""

                lines.append(
                    "Dialogue: 0,"
                    f"{_seconds_to_ass_time(event_start)},"
                    f"{_seconds_to_ass_time(event_end)},"
                    f"Default,,0,0,0,,{prefix}{highlight_text}"
                )
    else:
        for text, start, end in chunks:
            safe_text = _escape_ass_text(text.strip())
            if not safe_text:
                continue
            fade = r"{\fad(80,40)}"
            lines.append(
                "Dialogue: 0,"
                f"{_seconds_to_ass_time(start)},"
                f"{_seconds_to_ass_time(end)},"
                f"Default,,0,0,0,,{fade}{safe_text}"
            )

    with open(subtitle_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _build_word_events(word_timestamps, video_start_time, video_duration):
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

        text = (w["text"] or "").strip()
        if text and not text.startswith("["):
            adjusted.append({"text": text, "start": start, "end": end})

    if not adjusted:
        return []

    return segment_words_into_phrases(adjusted)


def add_subtitles_to_video(input_video, output_video, transcriptions,
                           video_start_time=0, word_timestamps=None):
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)

    video_width, video_height, video_duration = _read_video_metadata(input_video)

    word_events = None
    if word_timestamps:
        word_events = _build_word_events(
            word_timestamps,
            video_start_time,
            video_duration,
        )

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

        _write_ass_file(
            subtitle_path,
            video_width,
            video_height,
            chunked,
            word_events=word_events,
        )

        n_events = len(word_events) if word_events else len(chunked)
        mode = "phrase highlight" if word_events else "chunked"
        print(f"Adding {n_events} subtitle events ({mode}) to video with FFmpeg NVENC...")

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