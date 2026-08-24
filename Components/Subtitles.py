import os
import shutil
import subprocess
import tempfile

import cv2


MAX_LINES_PER_CAPTION = 2
MAX_WORDS_PER_PHRASE = 4  # was 6 — shorter/punchier lines for social video

# ASS colours use &HAABBGGRR
ACTIVE_COLOUR = "&H0000FFFF"      # yellow
INACTIVE_COLOUR = "&H00FFFFFF"    # white
OUTLINE_COLOUR = "&H00101010"     # dark gray outline
BOX_COLOUR = "&HCC000000"         # 80% transparent black box

# Phrase segmentation defaults
HARD_PUNCT = {".", "?", "!"}
SOFT_PUNCT = {",", ";", ":"}
MIN_WORDS_BEFORE_SOFT_BREAK = 3   # was 4 — break earlier on commas
MIN_PHRASE_DURATION = 0.4
MAX_PHRASE_DURATION = 2.0         # was 1.6 — German compound words need more time
MAX_PHRASE_WORDS = MAX_WORDS_PER_PHRASE
MIN_PHRASE_WORDS = 1
MIN_WORD_DISPLAY_SEC = 0.08   # minimum time each word stays highlighted (was 0.02)
HOLD_AFTER_PHRASE_SEC = 1.5   # last word of phrase holds until next phrase starts, or this long (was 0.0)
SILENCE_GAP_BREAK_SEC = 0.8  # force phrase break on silence gaps >= this

# Stable wrapping defaults
TARGET_CHARS_PER_LINE = 14   # was 16 — narrower lines, more readable on portrait
MAX_CHARS_PER_LINE = 18      # was 22

# ---------------------------------------------------------------------------
# Caption styles
# ---------------------------------------------------------------------------
# "Arial Black" is not installed here — fontconfig silently substitutes
# NotoSans-Regular and libass fakes the bold, so the classic style has never
# rendered in the face it names.  The new styles ship an explicit font
# directory instead.  Use the full family name with Bold=0: synthetically
# emboldening a Black cut distorts it.
BLACK_FONT_NAME = "Barlow Semi Condensed Black"
BLACK_FONT_DIR = os.path.expanduser(
    "~/.local/share/fonts/mc_thumbnails/barlow-semi-condensed"
)

CAPTION_STYLE_PRESETS = {
    # The old classic style used BorderStyle 3 (opaque black box). Audience
    # feedback found that block visually heavy, so classic now uses the same
    # clean contour treatment as the newer styles.
    "classic": {
        "fontname": "Arial Black",
        "fontsdir": None,
        "bold": 1,
        "border_style": 1,
        "outline_colour": "&H00000000",
        "back_colour": "&H80000000",
        "outline_ratio": 0.005,
        "shadow_ratio": 0.002,
        "uppercase": False,
        "font_ratio": 0.053,
        "emphasis_scale": 1.0,
        "max_chars_per_line": MAX_CHARS_PER_LINE,
        "active_ramp": False,
        "solo_slow": False,
    },
    # Mixed typography, contour instead of a box: a box drawn per line steps
    # visibly when the lines have different heights, which is exactly what
    # mixed sizing produces.
    "emphasis": {
        "fontname": BLACK_FONT_NAME,
        "fontsdir": BLACK_FONT_DIR,
        "bold": 0,
        "border_style": 1,
        "outline_colour": "&H00000000",
        "back_colour": "&H80000000",
        "outline_ratio": 0.006,
        "shadow_ratio": 0.003,
        "uppercase": False,
        # Measured against the reference clips: their caption cap height is
        # ~5% of frame height, the old ratio yields 3.6% in this face.
        "font_ratio": 0.070,
        "emphasis_scale": 1.5,
        # Fallback only — the real budget is measured from the font (see
        # _measured_char_budget); a guessed character count either wraps
        # captions that fit or lets them run past the frame edge.
        "max_chars_per_line": 14,
        "active_ramp": True,
        "solo_slow": False,
    },
    # Full social look: hard outline instead of a box, all caps, and single
    # words popping in one at a time whenever the speaker slows down.
    "impact": {
        "fontname": BLACK_FONT_NAME,
        "fontsdir": BLACK_FONT_DIR,
        "bold": 0,
        "border_style": 1,          # outline + shadow, no box
        "outline_colour": "&H00000000",   # pure black contour
        "back_colour": "&H80000000",      # soft drop shadow
        "outline_ratio": 0.006,
        "shadow_ratio": 0.003,
        "uppercase": True,
        "font_ratio": 0.070,
        "emphasis_scale": 1.5,
        "max_chars_per_line": 14,
        "active_ramp": True,
        "solo_slow": True,
    },
}
DEFAULT_CAPTION_STYLE = "emphasis"
# Pop modifiers combine with any style: "none" leaves the style untouched,
# "balloon" makes appearing words inflate and rise from half to full opacity.
CAPTION_POPS = ("none", "balloon")
DEFAULT_CAPTION_POP = "balloon"

# Word-by-word mode: a phrase qualifies when the speaker is genuinely slow.
SLOW_MEDIAN_ONSET_SEC = 0.48   # median word-to-word onset distance (<= ~2 words/s)
SLOW_SOLO_WORD_SEC = 0.9       # a single word held this long carries an event alone
SOLO_HOLD_SEC = 0.35           # solo words must not inherit HOLD_AFTER_PHRASE_SEC
SOLO_POP_MS = 110              # scale-up duration of the pop-in

# Balloon pop: an independent modifier that combines with every caption style.
# The word inflates fast, overshoots slightly and settles — the overshoot is
# what makes it read as a balloon rather than a plain zoom — while its opacity
# rises from half to full.
BALLOON_START_SCALE = 58        # % of final size at the start of the inflate
BALLOON_OVERSHOOT_SCALE = 107   # % at the peak, before settling back to 100
BALLOON_INFLATE_MS = 105        # start → overshoot
BALLOON_SETTLE_MS = 85          # overshoot → final size
BALLOON_START_ALPHA = "&H80&"   # 50% transparent
BALLOON_FULL_ALPHA = "&H00&"    # fully opaque
BALLOON_FADE_MS = 130           # duration of the opacity rise
# A word popping in before it is spoken needs the inflate+settle to be done
# by the time the voice hits — that natural lead-in is exactly the pop's own
# duration.
BALLOON_LEAD_IN_SEC = (BALLOON_INFLATE_MS + BALLOON_SETTLE_MS) / 1000.0
# The balloon deliberately dominates far more than the shared 1.5x emphasis
# scale — the punch word is meant to shove its neighbours aside.
BALLOON_EMPHASIS_SCALE = 2.2
# Balloon motion is readable only in genuinely slow delivery. WPM is measured
# from the phrase's real word timestamps; synthetic/retimed timings never
# qualify because they describe the retimer rather than the speaker.
BALLOON_MAX_WPM = 90.0

# Bottom-aligned ASS captions move downward when MarginV becomes smaller.
# 38% keeps them below the previous 45%-from-bottom placement without entering
# the platform UI / lower-third danger zone.
CAPTION_MARGIN_BOTTOM_RATIO = 0.38

# Emphasis fallback for every path without an LLM signal (raw ASR, captions
# stage, --skip-llm-cleanup).  German function words never carry the accent.
DE_STOPWORDS = {
    "aber", "alle", "als", "also", "am", "an", "auch", "auf", "aus", "bei",
    "bin", "bis", "bist", "da", "damit", "dann", "das", "dass", "dein", "dem",
    "den", "denn", "der", "des", "dich", "die", "dir", "doch", "dort", "du",
    "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er", "es",
    "euch", "euer", "für", "ganz", "gar", "gegen", "gewesen", "habe", "haben",
    "hat", "hatte", "hier", "hin", "ich", "ihm", "ihn", "ihr", "im", "immer",
    "in", "ist", "ja", "jetzt", "kann", "kein", "keine", "können", "mal",
    "man", "mehr", "mein", "mich", "mir", "mit", "muss", "nach", "nein",
    "nicht", "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "schon",
    "sein", "seine", "sich", "sie", "sind", "so", "soll", "sondern", "über",
    "um", "und", "uns", "unser", "unter", "vom", "von", "vor", "war", "waren",
    "was", "weil", "weiter", "wenn", "wer", "werden", "wie", "wieder", "wir",
    "wird", "wo", "wurde", "zu", "zum", "zur",
}

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

        # Force phrase break when there is a long silence gap before this word
        if current and w.get("start") is not None and current[-1].get("end") is not None:
            gap = w["start"] - current[-1]["end"]
            if gap >= SILENCE_GAP_BREAK_SEC:
                phrases.append(current)
                current = []

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


_CHAR_BUDGET_CACHE: dict[tuple, int] = {}


def _measured_char_budget(preset, font_size, usable_width, fallback):
    """How many average characters actually fit across *usable_width*.

    A hand-picked character count silently goes wrong whenever the face or
    the size changes: too low and captions wrap that would have fit, too high
    and they run past the frame edge with no error anywhere.
    """
    font_dir = preset.get("fontsdir")
    if not font_dir or not os.path.isdir(font_dir):
        return fallback

    key = (font_dir, preset["fontname"], font_size, usable_width)
    if key in _CHAR_BUDGET_CACHE:
        return _CHAR_BUDGET_CACHE[key]

    try:
        from PIL import ImageFont

        candidates = sorted(
            name for name in os.listdir(font_dir) if name.lower().endswith((".ttf", ".otf"))
        )
        black = [name for name in candidates if "black" in name.lower()] or candidates
        font = ImageFont.truetype(os.path.join(font_dir, black[0]), font_size)
        sample = "der Samen einer Erweckung"
        average = font.getlength(sample) / len(sample)
        budget = max(8, int(usable_width / average)) if average > 0 else fallback
    except Exception:
        budget = fallback

    _CHAR_BUDGET_CACHE[key] = budget
    return budget


def resolve_caption_style(style):
    name = str(style or DEFAULT_CAPTION_STYLE).strip().lower()
    return CAPTION_STYLE_PRESETS.get(name, CAPTION_STYLE_PRESETS[DEFAULT_CAPTION_STYLE])


def safe_upper(text):
    """Uppercase that keeps ``ß`` intact instead of expanding it to ``SS``."""
    return "".join("ß" if ch == "ß" else ch.upper() for ch in text)


def _emphasis_token(text):
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _pick_emphasis_indices(phrase):
    """Indices of the words rendered oversized in a phrase (at most two).

    An explicit signal on the words wins — that is the LLM's pick, carried
    through the cleanup cache.  Otherwise fall back to a heuristic so every
    path (raw ASR, ``--skip-llm-cleanup``, the captions-only render stage)
    still gets mixed typography rather than a flat line.
    """
    explicit = [i for i, w in enumerate(phrase) if w.get("emphasis")]
    if explicit:
        return set(explicit[:2])

    candidates = []
    for index, word in enumerate(phrase):
        token = _emphasis_token(word.get("text"))
        if len(token) < 4 or token in DE_STOPWORDS:
            continue
        # German capitalises nouns, which is exactly what should be shouted.
        is_noun_like = str(word.get("text") or "")[:1].isupper()
        # Ties go to the earlier word: the punch word tends to open the
        # phrase, and a late accent reads as an afterthought.
        candidates.append((len(token) + (3 if is_noun_like else 0), -index, index))

    if not candidates:
        return set()
    return {max(candidates)[2]}


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _phrase_is_slow(phrase):
    # Retimed words carry a distributed, not a measured, duration — reading a
    # speaking rate off them would measure the retimer, not the speaker.
    if any(w.get("synthetic") for w in phrase):
        return False
    if any((w["end"] - w["start"]) >= SLOW_SOLO_WORD_SEC for w in phrase):
        return True
    if len(phrase) < 2:
        return False
    onsets = [phrase[i + 1]["start"] - phrase[i]["start"] for i in range(len(phrase) - 1)]
    return _median(onsets) >= SLOW_MEDIAN_ONSET_SEC


def _phrase_wpm(phrase):
    """Measured words per minute for one phrase, or infinity if unreliable."""
    if not phrase or any(w.get("synthetic") for w in phrase):
        return float("inf")
    try:
        duration = float(phrase[-1]["end"]) - float(phrase[0]["start"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    if duration <= 0:
        return float("inf")
    return len(phrase) * 60.0 / duration


def _mark_balloon_eligibility(phrases):
    for phrase in phrases:
        eligible = _phrase_wpm(phrase) < BALLOON_MAX_WPM
        for word in phrase:
            word["balloon_eligible"] = eligible
    return phrases


def _solo_units(phrase):
    """Group a phrase into the smallest units that still mean something.

    Splitting strictly per word puts a lone "einer" full-screen, which reads
    as a glitch rather than emphasis. Function words therefore ride along with
    the content word they belong to, giving the one-or-two-word units the
    reference clips use.
    """
    units: list[list[dict]] = []
    pending: list[dict] = []
    for word in phrase:
        pending.append(word)
        if _emphasis_token(word.get("text")) not in DE_STOPWORDS:
            units.append(pending)
            pending = []
    if pending:
        if units:
            units[-1].extend(pending)
        else:
            units.append(pending)
    return units


def split_slow_phrases(phrases):
    """Break slowly spoken phrases into single-unit events.

    A solo event has no neighbours, so scaling it cannot reflow anything —
    that is what makes the pop-in animation safe where a per-word ``\\fs``
    inside a multi-word line would shift the whole centred phrase.
    """
    result = []
    for phrase in phrases:
        if not phrase or not _phrase_is_slow(phrase):
            result.append(phrase)
            continue
        for unit in _solo_units(phrase):
            solo = {
                "text": " ".join(str(w["text"]).strip() for w in unit),
                "start": unit[0]["start"],
                "end": unit[-1]["end"],
                "solo": True,
            }
            if any(w.get("emphasis") for w in unit):
                solo["emphasis"] = True
            if all(w.get("balloon_eligible") for w in unit):
                solo["balloon_eligible"] = True
            result.append([solo])
    return result


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


def _weighted_len(words, weights, start, stop):
    """Line length in *rendered* characters, not literal ones.

    An oversized word occupies proportionally more of the line, so the
    wrapper has to count it that way or a phrase with one 1.5× word
    overflows the frame while scoring as comfortably short.
    """
    span = range(start, stop)
    if not span:
        return 0
    total = sum(len(words[i]) * weights[i] for i in span)
    return total + (len(span) - 1)


def wrap_phrase_words(phrase_words, max_lines=MAX_LINES_PER_CAPTION, weights=None,
                      max_chars=MAX_CHARS_PER_LINE):
    words = [w for w in phrase_words if w]
    if not words:
        return [""]
    if max_lines <= 1 or len(words) == 1:
        return [_join_words(words)]

    if len(words) > MAX_WORDS_PER_PHRASE:
        words = words[:MAX_WORDS_PER_PHRASE]

    if weights is None:
        weights = [1.0] * len(words)
    else:
        weights = list(weights[:len(words)]) + [1.0] * max(0, len(words) - len(weights))

    if _weighted_len(words, weights, 0, len(words)) <= max_chars:
        return [_join_words(words)]

    best = None
    best_score = float("inf")

    for split_idx, bonus in _split_index_candidates(words):
        line1 = _join_words(words[:split_idx])
        line2 = _join_words(words[split_idx:])

        if not line1 or not line2:
            continue

        len1 = _weighted_len(words, weights, 0, split_idx)
        len2 = _weighted_len(words, weights, split_idx, len(words))
        longest = max(len1, len2)
        shortest = min(len1, len2)

        score = 0
        score += abs(len1 - len2) * 1.8
        score += abs(len1 - TARGET_CHARS_PER_LINE) * 1.0
        score += abs(len2 - TARGET_CHARS_PER_LINE) * 1.0

        if longest > max_chars:
            score += (longest - max_chars) * 10

        if shortest < 7:
            score += (7 - shortest) * 5

        # An oversized word reads hardest when it shares a line with small
        # ones, so prefer splits that give it a line of its own — but only
        # when the other line still holds more than a single stranded word.
        if len(words) >= 3 and any(weights[i] > 1.0 for i in range(split_idx)) != any(
            weights[i] > 1.0 for i in range(split_idx, len(words))
        ):
            score -= 6

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


def _starts_mid_segment(transcriptions, subtitle_start_time):
    for text, start, end in transcriptions or []:
        stripped = str(text or "").strip()
        if not stripped or stripped.startswith("["):
            continue
        if float(start) < float(subtitle_start_time) < float(end):
            return True
    return False


def _leading_alpha(text):
    for ch in str(text or "").lstrip("\"'([{„‚«»"):
        if ch.isalpha():
            return ch
    return ""


def _drop_leading_partial_phrase(phrases, *, subtitle_start_time, transcriptions):
    if len(phrases) <= 1:
        return phrases
    if not _starts_mid_segment(transcriptions, subtitle_start_time):
        return phrases

    first_phrase = phrases[0]
    if not first_phrase:
        return phrases[1:]

    first_word = first_phrase[0]
    first_alpha = _leading_alpha(first_word.get("text", ""))
    starts_lowercase = (
        bool(first_alpha)
        and first_alpha == first_alpha.lower()
        and first_alpha != first_alpha.upper()
    )
    starts_immediately = float(first_word.get("start", 0.0)) <= 0.35
    short_phrase = len(first_phrase) <= 2 or _phrase_duration(first_phrase) <= 0.8

    if starts_immediately and short_phrase and starts_lowercase:
        dropped_text = " ".join(w.get("text", "").strip() for w in first_phrase).strip()
        if dropped_text:
            print(f"[Subtitles] Dropping orphan opening phrase: {dropped_text}")
        return phrases[1:]

    return phrases


def _word_scales(phrase, preset, max_chars=None, scale_override=None):
    """Per-word size multipliers, constant for every event of the phrase.

    Constant is the whole point: each word's Dialogue event re-renders the
    entire phrase, so an override set that changed between events would
    reflow the centred line and make the caption jitter word by word.
    """
    scale = float(scale_override if scale_override is not None else preset.get("emphasis_scale", 1.0))
    words = phrase[:MAX_WORDS_PER_PHRASE]
    if scale <= 1.0:
        return [1.0] * len(words)

    scales = [1.0] * len(words)
    for index in _pick_emphasis_indices(words):
        word_scale = scale
        # A long German compound at 1.5× overflows the frame on its own;
        # step it down rather than shrink the whole caption.
        budget = max_chars or preset["max_chars_per_line"]
        while word_scale > 1.0 and len(words[index]["text"]) * word_scale > budget:
            word_scale -= 0.25
        scales[index] = max(1.0, word_scale)
    return scales


def _build_phrase_layout_metadata(phrase, scales=None, max_chars=MAX_CHARS_PER_LINE):
    plain_words = [w["text"] for w in phrase][:MAX_WORDS_PER_PHRASE]
    wrapped_lines = wrap_phrase_words(plain_words, weights=scales, max_chars=max_chars)

    line_ranges = []
    cursor = 0
    for line in wrapped_lines:
        count = len(line.split())
        line_ranges.append((cursor, cursor + count))
        cursor += count

    return wrapped_lines, line_ranges


def _render_word(text, *, active, scale, base_font_size, preset, balloon=False):
    escaped = _escape_ass_text(safe_upper(text) if preset["uppercase"] else text)
    colour = ACTIVE_COLOUR if active else INACTIVE_COLOUR

    if active and preset["active_ramp"]:
        # Start on the inactive colour and animate into the highlight, so the
        # word visibly ignites instead of cutting.  Colour-only animation
        # cannot change glyph advances, so the line never moves while the
        # highlight travels across it.
        colour_tags = f"\\c{INACTIVE_COLOUR}\\t(0,90,\\c{ACTIVE_COLOUR})"
    else:
        colour_tags = f"\\c{colour}"

    if active and balloon:
        # Only the opacity half of the balloon can run here: the word shares
        # its line with the rest of the phrase, so scaling it would push the
        # others sideways. The inflate rides on the phrase entrance instead.
        colour_tags += _balloon_alpha_tags()

    if scale > 1.0:
        # Inline \fs, never \r: a style reset would also wipe the karaoke
        # colour of the active word.
        opening = f"\\fs{int(round(base_font_size * scale))}{colour_tags}"
        closing = f"{{\\fs{base_font_size}}}"
    else:
        opening = colour_tags
        closing = ""

    return f"{{{opening}}}{escaped}{closing}"


def _balloon_scale_tags():
    """Inflate-with-overshoot, safe only where nothing shares the event.

    Scaling changes glyph advances, so this may only be applied to an event
    that carries a single word or the whole caption at once — never to one
    word inside a shared line, which would reflow the others.
    """
    return (
        f"\\fscx{BALLOON_START_SCALE}\\fscy{BALLOON_START_SCALE}"
        f"\\t(0,{BALLOON_INFLATE_MS},\\fscx{BALLOON_OVERSHOOT_SCALE}\\fscy{BALLOON_OVERSHOOT_SCALE})"
        f"\\t({BALLOON_INFLATE_MS},{BALLOON_INFLATE_MS + BALLOON_SETTLE_MS},\\fscx100\\fscy100)"
    )


def _balloon_alpha_tags():
    return (
        f"\\alpha{BALLOON_START_ALPHA}"
        f"\\t(0,{BALLOON_FADE_MS},\\alpha{BALLOON_FULL_ALPHA})"
    )


def _build_highlight_text_for_word(phrase, active_word_idx, preset=None, base_font_size=100,
                                   max_chars=None, balloon=False):
    if preset is None:
        preset = CAPTION_STYLE_PRESETS[DEFAULT_CAPTION_STYLE]
    if max_chars is None:
        max_chars = preset["max_chars_per_line"]

    # The balloon deliberately reverses the no-reflow invariant every other
    # style relies on: words build up one at a time instead of all standing
    # there at once, so the wrapping still has to be computed from the FULL
    # phrase (otherwise the line break would jump as words are revealed) but
    # only words up to the active one are actually emitted.
    scale_override = BALLOON_EMPHASIS_SCALE if balloon else None
    scales = _word_scales(phrase, preset, max_chars=max_chars, scale_override=scale_override)
    wrapped_lines, line_ranges = _build_phrase_layout_metadata(
        phrase, scales=scales, max_chars=max_chars,
    )
    lines = []

    for start_idx, end_idx in line_ranges:
        if balloon and start_idx > active_word_idx:
            continue  # this line has not been reached by the spoken word yet
        reveal_end = min(end_idx, active_word_idx + 1) if balloon else end_idx
        rendered_words = []
        for word_idx in range(start_idx, reveal_end):
            rendered_words.append(
                _render_word(
                    phrase[word_idx]["text"],
                    active=(word_idx == active_word_idx),
                    scale=scales[word_idx],
                    base_font_size=base_font_size,
                    preset=preset,
                    balloon=balloon,
                )
            )
        if rendered_words:
            lines.append(" ".join(rendered_words))

    return r"\N".join(lines)


def _build_solo_word_text(word, *, base_font_size, preset, video_width, video_height, margin_v,
                          max_chars=None, balloon=False):
    """One big word, centred, popping in — used when the speaker slows down."""
    text = _escape_ass_text(safe_upper(word["text"]) if preset["uppercase"] else word["text"])
    size = int(round(base_font_size * max(1.25, preset["emphasis_scale"])))
    # A two-word unit at 1.5x can outrun the frame; scale it to the same
    # character budget the wrapped captions use.
    budget = max_chars or preset["max_chars_per_line"]
    length = len(word["text"])
    if length > budget:
        size = max(base_font_size, int(size * budget / length))
    # \pos measures from the top, MarginV from the bottom — land the solo word
    # on the same optical line the wrapped captions occupy.
    pos_y = max(size, int(video_height - margin_v - size * 0.5))
    colour = ACTIVE_COLOUR if word.get("emphasis") else INACTIVE_COLOUR
    if balloon:
        # A solo event stands alone, so the full balloon is safe here.
        motion = _balloon_scale_tags() + _balloon_alpha_tags()
    else:
        motion = f"\\fscx70\\fscy70\\t(0,{SOLO_POP_MS},\\fscx100\\fscy100)\\fad(40,0)"
    return (
        f"{{\\an5\\pos({video_width // 2},{pos_y})\\fs{size}\\c{colour}{motion}}}{text}"
    )


def _normalise_phrase_timings(word_events, caption_cutoff=None, lead_in=0.0):
    """Make all word Dialogue events strictly sequential — no overlaps ever.

    A global cursor tracks the end of the last emitted event.  Each new
    word starts at ``max(original_start, cursor)`` so two events can never
    occupy the same time range.  Within a phrase, each word extends to the
    next word's start (no gap).  Between phrases, the last word extends to
    ``min(natural_end, next_phrase_start)`` which may leave a deliberate
    silent gap.

    ``caption_cutoff``, when given, is the clip-relative end of the last
    audible word in the render window. ``HOLD_AFTER_PHRASE_SEC`` otherwise
    holds the final caption 1.5s past its word regardless of how much clip
    is actually left — measured on real renders to overrun the clip end by
    up to +1.44s of caption sitting on top of silence (HANDOVER_CAPTIONS.md,
    Aufgabe 2). Clamping every event's end to the cutoff removes exactly
    that overrun without touching anything that is still within earshot.

    ``lead_in``, when given (balloon mode only), pulls every event's start
    forward so a word's inflate+settle animation finishes by the time the
    voice actually reaches it, instead of popping in on top of the syllable.
    The strict-sequencing cursor still applies, so it never overlaps the
    still-running previous event.
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

        phrase_lead_in = lead_in if phrase[0].get("balloon_eligible") else 0.0
        for j, w in enumerate(phrase):
            ws = max(w["start"] - phrase_lead_in, cursor)
            # A solo word is its own phrase, so it would otherwise inherit the
            # full inter-phrase hold and linger on screen long after it was
            # spoken — which defeats the point of word-by-word pacing.
            hold = SOLO_HOLD_SEC if w.get("solo") else HOLD_AFTER_PHRASE_SEC

            if j + 1 < n:
                # Mid-phrase: use next word's start (no gap within phrase).
                # The next event pulls its own start forward by lead_in too,
                # so this one must end at that same earlier point.
                we = max(ws + 0.001, phrase[j + 1]["start"] - phrase_lead_in)
            elif i + 1 < total and word_events[i + 1]:
                # Last word of phrase: hold briefly, but never overlap next phrase
                natural = w["end"] + hold
                next_phrase = word_events[i + 1]
                next_lead_in = lead_in if next_phrase[0].get("balloon_eligible") else 0.0
                next_start = next_phrase[0]["start"] - next_lead_in
                we = min(natural, next_start)
                we = max(we, ws + 0.001)
            else:
                # Very last word overall
                we = max(ws + 0.001, w["end"] + hold)

            if caption_cutoff is not None:
                we = min(we, max(ws + 0.001, caption_cutoff))

            carried = {"text": w["text"], "start": ws, "end": we}
            for key in ("emphasis", "solo", "synthetic", "balloon_eligible"):
                if w.get(key):
                    carried[key] = w[key]
            copied.append(carried)
            cursor = we

        if copied:
            normalised.append(copied)

    return normalised


def _write_ass_file(subtitle_path, video_width, video_height, chunks, word_events=None,
                    style=DEFAULT_CAPTION_STYLE, pop=DEFAULT_CAPTION_POP, caption_cutoff=None):
    preset = resolve_caption_style(style)
    balloon = str(pop or "").lower() == "balloon"

    # slightly smaller than before
    font_size = max(33, int(video_height * preset["font_ratio"]) - 2)

    # Audience feedback preferred the captions lower than the previous 45%
    # placement, while still leaving room for platform controls.
    margin_v = max(420, int(video_height * CAPTION_MARGIN_BOTTOM_RATIO))
    margin_h = max(70, int(video_width * 0.10))

    max_chars = _measured_char_budget(
        preset, font_size, video_width - 2 * max(70, int(video_width * 0.10)),
        preset["max_chars_per_line"],
    )

    outline = max(3, int(video_height * preset["outline_ratio"]))
    shadow = max(1, int(video_height * preset["shadow_ratio"]))

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
            f"Style: Default,{preset['fontname']},"
            f"{font_size},{INACTIVE_COLOUR},{INACTIVE_COLOUR},"
            f"{preset['outline_colour']},{preset['back_colour']},"
            f"{preset['bold']},0,0,0,100,100,0,0,{preset['border_style']},{outline},{shadow},2,"
            f"{margin_h},{margin_h},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if word_events:
        if balloon:
            word_events = _mark_balloon_eligibility(word_events)
        if preset["solo_slow"]:
            before = len(word_events)
            word_events = split_slow_phrases(word_events)
            solo_count = sum(1 for phrase in word_events if phrase and phrase[0].get("solo"))
            # Worth printing on every run: solo mode silently does nothing when
            # the timings are synthetic, and the result then looks identical to
            # the emphasis style with no indication why.
            print(f"[Subtitles] Word-by-word: {solo_count} solo word(s) from {before} phrase(s)")
        # A word popping in cold on the exact frame it is spoken reads as a
        # cut, not a pop — bring every event forward so inflate+settle lands
        # just before the voice does.
        lead_in = BALLOON_LEAD_IN_SEC if balloon else 0.0
        word_events = _normalise_phrase_timings(
            word_events, caption_cutoff=caption_cutoff, lead_in=lead_in,
        )

        for phrase in word_events:
            if not phrase:
                continue

            phrase_balloon = balloon and bool(phrase[0].get("balloon_eligible"))

            for word_idx, word in enumerate(phrase):
                event_start = word["start"]
                event_end = word["end"]

                if event_end <= event_start:
                    event_end = event_start + 0.03

                if word.get("solo"):
                    event_text = _build_solo_word_text(
                        word,
                        base_font_size=font_size,
                        preset=preset,
                        video_width=video_width,
                        video_height=video_height,
                        margin_v=margin_v,
                        max_chars=max_chars,
                        balloon=phrase_balloon,
                    )
                else:
                    highlight_text = _build_highlight_text_for_word(
                        phrase, word_idx, preset=preset, base_font_size=font_size,
                        max_chars=max_chars, balloon=phrase_balloon,
                    )
                    if phrase_balloon:
                        # Every new word is its own arrival, not just the
                        # phrase's first: scaling the whole event is
                        # reflow-free by definition — every currently visible
                        # word grows together, so nothing shifts relative to
                        # anything else, and libass keeps re-centring it while
                        # it grows. That growth is also what makes a new word
                        # visibly shove the existing ones aside — the effect
                        # is intentional, not a bug.
                        prefix = "{" + _balloon_scale_tags() + "}"
                    elif word_idx == 0:
                        # Fade-in only when the phrase first appears (first word)
                        prefix = r"{\fad(100,0)}"
                    else:
                        prefix = ""
                    event_text = f"{prefix}{highlight_text}"

                lines.append(
                    "Dialogue: 0,"
                    f"{_seconds_to_ass_time(event_start)},"
                    f"{_seconds_to_ass_time(event_end)},"
                    f"Default,,0,0,0,,{event_text}"
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


def _build_word_events(
    word_timestamps,
    video_start_time,
    video_duration,
    *,
    transcriptions=None,
    trim_leading_partial_phrase=False,
    highlight_start_time=None,
    highlight_end_time=None,
):
    adjusted = []
    dropped = 0
    # If a highlight start is provided, compute offset (relative to clip start)
    highlight_offset = None
    if highlight_start_time is not None:
        try:
            highlight_offset = float(highlight_start_time) - float(video_start_time)
        except Exception:
            highlight_offset = None

    # Symmetric end boundary: a word starting after the highlight's own end
    # belongs to whatever comes next in the sermon, not this clip — it would
    # otherwise render as a caption for speech that was cut out of the video.
    highlight_end_offset = None
    if highlight_end_time is not None:
        try:
            highlight_end_offset = float(highlight_end_time) - float(video_start_time)
        except Exception:
            highlight_end_offset = None

    for w in word_timestamps:
        start = w["start"] - video_start_time
        end = w["end"] - video_start_time
        # If highlight_offset is set, strictly exclude any word that starts
        # before the highlight (no tolerance, no shifting).
        if highlight_offset is not None:
            if start < highlight_offset:
                # word starts before highlight — drop it
                continue
        if highlight_end_offset is not None:
            if start > highlight_end_offset:
                # word starts after the highlight — drop it
                continue
        if end <= 0:
            continue
        if video_duration > 0 and start > video_duration:
            continue

        start = max(0, start)
        if video_duration > 0:
            end = min(video_duration, end)

        text = (w["text"] or "").strip()
        if text and not text.startswith("["):
            entry = {"text": text, "start": start, "end": end}
            for key in ("emphasis", "synthetic"):
                if w.get(key):
                    entry[key] = w[key]
            adjusted.append(entry)
        elif text:
            dropped += 1

    if dropped:
        print(f"[Subtitles] Dropped {dropped} marker words (e.g. [AUDIENCE REACTION])")

    total_in_range = len(adjusted) + dropped
    total_input = len(word_timestamps)
    if total_in_range < total_input:
        print(f"[Subtitles] {total_input} input words → {len(adjusted)} in clip range "
              f"(filtered {total_input - total_in_range} outside clip boundaries)")
    else:
        print(f"[Subtitles] All {len(adjusted)} words in clip range")

    if not adjusted:
        return []

    phrases = segment_words_into_phrases(adjusted)
    if trim_leading_partial_phrase:
        phrases = _drop_leading_partial_phrase(
            phrases,
            subtitle_start_time=video_start_time,
            transcriptions=transcriptions,
        )
    return phrases


def add_subtitles_to_video(input_video, output_video, transcriptions,
                           video_start_time=0, word_timestamps=None,
                           trim_leading_partial_phrase=False,
                           highlight_start_time=None,
                           highlight_end_time=None,
                           caption_cutoff=None,
                           extra_vf="",
                           caption_style=DEFAULT_CAPTION_STYLE,
                           caption_pop=DEFAULT_CAPTION_POP):
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)

    video_width, video_height, video_duration = _read_video_metadata(input_video)

    word_events = None
    if word_timestamps:
        word_events = _build_word_events(
            word_timestamps,
            video_start_time,
            video_duration,
            transcriptions=transcriptions,
            trim_leading_partial_phrase=trim_leading_partial_phrase,
            highlight_start_time=highlight_start_time,
            highlight_end_time=highlight_end_time,
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
            style=caption_style,
            pop=caption_pop,
            caption_cutoff=caption_cutoff,
        )

        preset = resolve_caption_style(caption_style)
        n_events = len(word_events) if word_events else len(chunked)
        mode = "phrase highlight" if word_events else "chunked"
        pop_note = f", pop: {caption_pop}" if str(caption_pop or "none").lower() != "none" else ""
        print(f"Adding {n_events} subtitle events ({mode}, style: {caption_style}{pop_note}) "
              "to video with FFmpeg NVENC...")

        vf_chain = f"subtitles={os.path.basename(subtitle_path)}"
        # Without an explicit directory libass resolves the family through
        # fontconfig and silently substitutes whatever is closest.
        if preset["fontsdir"] and os.path.isdir(preset["fontsdir"]):
            vf_chain += f":fontsdir={preset['fontsdir']}"
        if extra_vf:
            vf_chain = f"{vf_chain},{extra_vf}"

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-vf",
            vf_chain,
            "-an",
            *NVENC_FLAGS,
            output_video,
        ]
        _run_ffmpeg(command, "subtitle burn", cwd=subtitle_dir)
        print(f"Subtitles added successfully -> {output_video}")
    finally:
        if subtitle_path and os.path.exists(subtitle_path):
            os.remove(subtitle_path)
