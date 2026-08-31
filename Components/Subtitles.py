import json
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
HOLD_AFTER_PHRASE_SEC = 0.6   # last word of phrase holds until next phrase starts, or this long (Benjamin, 2026-08-25: was 1.5 — lingered too long; this value also bounds how long any silence gap can show a caption, since _normalise_phrase_timings clamps the hold to min(this, gap to next phrase's own start) — no separate pause-hiding logic needed)
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

# Word-level effect layer ("--caption-fx"), combinable with every style and
# every pop. Only the reflow-safe half of the reference vocabulary lives here:
# alpha and blur leave glyph advances untouched, so they may animate a single
# word inside a shared, centred line without shifting its neighbours. Scale
# and font swaps do move the line and therefore stay on solo events.
CAPTION_FX_CHOICES = ("none", "fade_up", "blur_words", "typewriter", "flicker", "font_mix")
DEFAULT_CAPTION_FX = "none"
FADE_UP_START_ALPHA = "&HC8&"    # ~78% transparent when the word arrives
FADE_UP_MS = 170
FADE_UP_RISE_RATIO = 0.5         # of the font size; solo events only
BLUR_WORDS_START = 6
BLUR_WORDS_MS = 150

# Letter animation without reflow. A character that has not been typed yet is
# rendered fully transparent instead of being left out: a transparent glyph
# still occupies its advance, so the centred line never moves while the word
# types itself. That also means no glyph measuring is needed anywhere.
TYPEWRITER_STEP_MS = 38          # per character
TYPEWRITER_FADE_MS = 20          # how hard each character lands
TYPEWRITER_MAX_MS = 420          # a long compound must not type past its word
FLICKER_STEP_MS = 45             # per character offset of the flicker pattern
FLICKER_DIP_ALPHA = "&HA0&"      # how far a character dips
FLICKER_DIPS = 2

# Font mix instead of one more highlight colour. Applied to the word the style
# already renders oversized, so the choice is constant across every event of a
# phrase — a font swap changes glyph advances, and a *changing* swap would
# reflow the centred line exactly the way an animated \fs once did.
# Benjamin, 2026-08-25: wanted the punch word in "super kursiver Schrift"
# instead of the previous upright Anton swap. Mrs Saint Delafield (Google
# Fonts, OFL — freely usable here; his first pick, Adobe Fonts' "Absolute
# Beauty Script", is licensed only for use inside Adobe apps/hosted CSS, not
# for extraction into this ffmpeg/libass pipeline). A connected script reads
# as a blob of loops in full caps (verified render), so this word keeps its
# natural case instead of the style's usual safe_upper() — see the two
# ``mixed and "font_mix" in fx`` case checks below.
MIX_FONT_NAME = "Mrs Saint Delafield"
MIX_FONT_DIR = os.path.expanduser("~/.local/share/fonts/mc_thumbnails/mrs-saint-delafield")
MIX_FONT_FILE = "MrsSaintDelafield-Regular.ttf"
# libass resolves a single fontsdir; the mix needs both families in one place.
CAPTION_FX_FONT_DIR = os.path.expanduser("~/.local/share/fonts/kanzelclips_caption_mix")

# Caption-keyed sound design ("--caption-sfx"). The reference edits put a click
# on the accent and a typing tick under a word-by-word passage; both live at
# -15..-30 dB, and both stop being an accent the moment they fire on every
# word. The rate limits below, not the sample choice, are what keeps them
# tasteful.
# Caption glow. "Deep Glow like salt": a bright-core bloom lifted off the
# caption itself, never a filter over the whole frame. The caption is rendered
# a second time onto a transparent layer because a bloom needs its own alpha —
# the production burn composites the captions in a single pass.
CAPTION_GLOW_MODES = ("off", "soft", "strong")
DEFAULT_CAPTION_GLOW = "off"
CAPTION_GLOW_PROFILES = {
    # threshold: only the bright core blooms, so the black contour cannot smear
    # a grey halo across the footage.
    "soft": {"threshold": 190, "sigma": 14, "opacity": 0.45},
    "strong": {"threshold": 170, "sigma": 20, "opacity": 0.60},
}

CAPTION_SFX_MODES = ("off", "click", "typing", "both")
DEFAULT_CAPTION_SFX = "off"
CAPTION_SFX_ACCENT_EVENT = "caption_accent"
CAPTION_SFX_LETTER_EVENT = "caption_letter"
CAPTION_SFX_LEAD_IN_SEC = 0.35   # never fire into the cut from the hero intro
CAPTION_SFX_ACCENT_MIN_GAP_SEC = 1.2
CAPTION_SFX_ACCENT_WINDOW_SEC = 10.0
CAPTION_SFX_ACCENT_PER_WINDOW = 2
CAPTION_SFX_LETTER_MIN_GAP_SEC = 0.22
CAPTION_SFX_LETTER_MAX = 10

# Word-by-word mode: a phrase qualifies when the speaker is genuinely slow.
SLOW_MEDIAN_ONSET_SEC = 0.48   # median word-to-word onset distance (<= ~2 words/s)
SLOW_SOLO_WORD_SEC = 0.9       # a single word held this long carries an event alone
SOLO_HOLD_SEC = 0.35           # solo words must not inherit HOLD_AFTER_PHRASE_SEC
SOLO_POP_MS = 110              # scale-up duration of the pop-in

# Balloon pop: an independent modifier that combines with every caption style.
# This is intentionally a visible motion preset, not a barely perceptible
# easing. The whole event inflates from a small, blurred silhouette, overshoots
# clearly, then settles; the spoken word itself fades in separately.
BALLOON_START_SCALE = 38        # % of final size at the start of the inflate
BALLOON_OVERSHOOT_SCALE = 124   # % at the peak, before settling back to 100
BALLOON_INFLATE_MS = 145        # start → overshoot
BALLOON_SETTLE_MS = 115         # overshoot → final size
BALLOON_START_ALPHA = "&HFF&"   # fully transparent at the start
BALLOON_FULL_ALPHA = "&H00&"    # fully opaque
BALLOON_FADE_MS = 180           # duration of the opacity rise
BALLOON_START_BLUR = 7          # resolves with the spoken-word reveal
# Benjamin, 2026-08-25: captions were arriving audibly before the voice —
# the old lead-in was the pop's full inflate+settle duration (260ms), so the
# animation always finished before the word was heard. "Gleichzeitig" beats
# "pop fully settled": a small fixed lead-in instead, independent of how long
# the pop itself takes. The animation now keeps running into the spoken word.
BALLOON_LEAD_IN_SEC = 0.0
# Benjamin, 2026-08-30: still reading as "ahead of the voice". Measured on the
# Oberlahnstein 30.08. audio over 91 words that follow a real pause (so the
# onset detector cannot latch onto the previous word): the audible onset sits a
# median of +75 ms AFTER the ASR timestamp, and later in 82% of cases. The word
# times are systematically early, so every caption was too - independently of
# the balloon lead-in, which is now zero. Shifting the display times by this
# amount is what makes "exactly with the audio" true rather than nominal.
_DEFAULT_ASR_ONSET_CORRECTION_SEC = 0.075


def _configured_asr_onset_correction_sec():
    """Read the per-recording-lane offset, failing closed on bad values."""
    raw = os.getenv("CAPTION_ASR_ONSET_CORRECTION_SEC")
    if raw is None or not raw.strip():
        return _DEFAULT_ASR_ONSET_CORRECTION_SEC
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("CAPTION_ASR_ONSET_CORRECTION_SEC must be a number") from exc
    if not -0.25 <= value <= 0.35:
        raise ValueError("CAPTION_ASR_ONSET_CORRECTION_SEC must be between -0.25 and 0.35")
    return value


ASR_ONSET_CORRECTION_SEC = _configured_asr_onset_correction_sec()
# The balloon deliberately dominates far more than the shared 1.5x emphasis
# scale — the punch word is meant to shove its neighbours aside.
BALLOON_EMPHASIS_SCALE = 2.2
# The medium tier makes ``emphasis + balloon`` visibly distinct at ordinary
# sermon pacing without turning every phrase into motion graphics. WPM is
# always measured from real ASR anchors; synthetic/retimed timings never
# qualify because they describe the retimer rather than the speaker.
BALLOON_MAX_WPM = 90.0
BALLOON_MEDIUM_MAX_WPM = 150.0
BALLOON_MOTION_PROFILES = {
    "strong": {
        "start_scale": BALLOON_START_SCALE,
        "overshoot_scale": BALLOON_OVERSHOOT_SCALE,
        "inflate_ms": BALLOON_INFLATE_MS,
        "settle_ms": BALLOON_SETTLE_MS,
        "start_alpha": BALLOON_START_ALPHA,
        "fade_ms": BALLOON_FADE_MS,
        "start_blur": BALLOON_START_BLUR,
        "emphasis_scale": BALLOON_EMPHASIS_SCALE,
    },
    "medium": {
        "start_scale": 64,
        "overshoot_scale": 112,
        "inflate_ms": 120,
        "settle_ms": 95,
        "start_alpha": "&HD0&",
        "fade_ms": 150,
        "start_blur": 3,
        "emphasis_scale": 1.8,
    },
    # Fast delivery. Measured on the Oberlahnstein 30.08. sermon: median 194 wpm,
    # i.e. a new word roughly every 300 ms. A strong pop at that rate is visual
    # noise, so this tier keeps the word-by-word build-up but almost drops the
    # motion - the reveal itself carries the rhythm.
    "subtle": {
        "start_scale": 84,
        "overshoot_scale": 104,
        "inflate_ms": 70,
        "settle_ms": 55,
        "start_alpha": "&HB0&",
        "fade_ms": 90,
        "start_blur": 1,
        "emphasis_scale": 1.5,
    },
}

# Bottom-aligned ASS captions move downward when MarginV becomes smaller.
# 38% keeps them below the previous 45%-from-bottom placement without entering
# the platform UI / lower-third danger zone.
CAPTION_MARGIN_BOTTOM_RATIO = 0.38

# The Instagram guide is deliberately represented as a small, versioned
# profile instead of scattering magic coordinates throughout the renderer.
# ``social_vertical`` is the production default; ``off`` preserves the legacy
# centred placement for forensic comparisons only.
DEFAULT_CAPTION_SAFE_AREA = "social_vertical"
CAPTION_SAFE_AREA_OFF = "off"
_CAPTION_SAFE_AREAS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "caption_safe_areas.json"
)


def _load_caption_safe_area_profiles():
    with open(_CAPTION_SAFE_AREAS_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    profiles = payload.get("profiles") or {}
    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeError("caption_safe_areas.json has no profiles")
    return profiles


def resolve_caption_safe_area(name=DEFAULT_CAPTION_SAFE_AREA):
    """Resolve a versioned caption-safe-area profile, or the legacy ``off`` mode."""
    selected = str(name or DEFAULT_CAPTION_SAFE_AREA).strip().lower()
    if selected == CAPTION_SAFE_AREA_OFF:
        return None
    profiles = _load_caption_safe_area_profiles()
    if selected not in profiles:
        choices = ", ".join((CAPTION_SAFE_AREA_OFF, *sorted(profiles)))
        raise ValueError(f"Unsupported caption safe area: {selected} (choose {choices})")
    return profiles[selected]


def _caption_layout(video_width, video_height, safe_area=DEFAULT_CAPTION_SAFE_AREA):
    """Return a stable ASS anchor and conservative wrapping width.

    The profile's base width deliberately leaves enough room for the 124 %
    Balloon peak, outline, and blur. A fixed ``\\pos`` anchor also means that
    karaoke events cannot drift as their highlighted word changes.
    """
    profile = resolve_caption_safe_area(safe_area)
    margin_v = max(420, int(video_height * CAPTION_MARGIN_BOTTOM_RATIO))
    margin_h = max(70, int(video_width * 0.10))
    if profile is None:
        usable = video_width - 2 * margin_h
        return {
            "anchor_x": video_width // 2,
            "anchor_y": video_height - margin_v,
            "usable_width": usable,
            "isolated_word_usable_width": usable,
            "margin_v": margin_v,
            "margin_h": margin_h,
            "enabled": False,
        }

    canvas = profile["reference_canvas"]
    scale_x = video_width / float(canvas["width"])
    scale_y = video_height / float(canvas["height"])
    return {
        "anchor_x": int(round(profile["caption_anchor"]["x"] * scale_x)),
        "anchor_y": int(round(profile["caption_anchor"]["y"] * scale_y)),
        "usable_width": int(round(profile["max_base_caption_width"] * scale_x)),
        # Wider than usable_width on purpose: usable_width is sized for a
        # full multi-word phrase at rest, conservative so it still peaks
        # safely under Balloon's 124%. A single unbreakable token with no
        # line-mates (an oversized compound, or a solo word) doesn't need
        # that multi-word margin and can use the real measured corridor
        # instead — see isolated_word_safe_width in caption_safe_areas.json.
        "isolated_word_usable_width": int(round(
            profile.get("isolated_word_safe_width", profile["max_base_caption_width"]) * scale_x
        )),
        "margin_v": margin_v,
        "margin_h": margin_h,
        "enabled": True,
    }

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
        crosses_silence_break = (
            float(phrase[0]["start"]) - float(prev[-1]["end"])
            >= SILENCE_GAP_BREAK_SEC
        )

        if (is_too_short or not prev_ends_hard) and not crosses_silence_break:
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
        crosses_silence_break = (
            float(last[0]["start"]) - float(merged[-2][-1]["end"])
            >= SILENCE_GAP_BREAK_SEC
        )
        if (
            not crosses_silence_break
            and (len(last) < MIN_PHRASE_WORDS or _phrase_duration(last) < MIN_PHRASE_DURATION)
        ):
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
_MEASUREMENT_FONT_CACHE: dict[tuple, object] = {}
_WORD_PIXEL_CACHE: dict[tuple, float] = {}
_AVERAGE_GLYPH_SAMPLE = "der Samen einer Erweckung"
# SAFE-25 measured a real libass render undershooting this same PIL proxy by
# up to ~4.3% on long German compounds even when using a real font file —
# so every width check below adds this margin on top of the direct
# measurement rather than trusting it at face value.
_WIDTH_SAFETY_MARGIN = 1.05
# A defensive floor only — German compounds have no upper bound on length,
# so there is no scale that is always "big enough"; the safe-area gate (0
# unsafe pixels) outranks a legibility minimum. This just stops a
# pathological case from collapsing to an unreadable sliver.
_MIN_WORD_SCALE = 0.2


def _measurement_font(preset, font_size):
    """The concrete PIL font used to size captions, or ``None`` if unavailable.

    Cached per (font dir, family, size) — loading a TrueType face repeatedly
    for every word of every phrase would be needlessly slow.
    """
    font_dir = preset.get("fontsdir")
    if not font_dir or not os.path.isdir(font_dir):
        return None

    key = (font_dir, preset["fontname"], font_size)
    if key in _MEASUREMENT_FONT_CACHE:
        return _MEASUREMENT_FONT_CACHE[key]

    try:
        from PIL import ImageFont

        candidates = sorted(
            name for name in os.listdir(font_dir) if name.lower().endswith((".ttf", ".otf"))
        )
        black = [name for name in candidates if "black" in name.lower()] or candidates
        font = ImageFont.truetype(os.path.join(font_dir, black[0]), font_size)
    except Exception:
        font = None

    _MEASUREMENT_FONT_CACHE[key] = font
    return font


def _measured_char_budget(preset, font_size, usable_width, fallback):
    """How many average characters actually fit across *usable_width*.

    A hand-picked character count silently goes wrong whenever the face or
    the size changes: too low and captions wrap that would have fit, too high
    and they run past the frame edge with no error anywhere.
    """
    font = _measurement_font(preset, font_size)
    if font is None:
        return fallback

    key = (id(font), usable_width)
    if key in _CHAR_BUDGET_CACHE:
        return _CHAR_BUDGET_CACHE[key]

    average = font.getlength(_AVERAGE_GLYPH_SAMPLE) / len(_AVERAGE_GLYPH_SAMPLE)
    budget = max(8, int(usable_width / average)) if average > 0 else fallback

    _CHAR_BUDGET_CACHE[key] = budget
    return budget


def _measured_pixel_width(preset, font_size, text):
    """Real rendered width of *text* at *font_size*, in pixels, plus margin.

    Returns ``None`` when no real font file is available to measure — callers
    must treat that as "cannot verify", not "fits".
    """
    font = _measurement_font(preset, font_size)
    if font is None:
        return None

    key = (id(font), text)
    if key in _WORD_PIXEL_CACHE:
        return _WORD_PIXEL_CACHE[key]

    width = font.getlength(text) * _WIDTH_SAFETY_MARGIN
    _WORD_PIXEL_CACHE[key] = width
    return width


def _measured_char_length(preset, font_size, text, fallback):
    """*text*'s real rendered width, in the same average-character units
    ``_measured_char_budget`` counts in.

    A raw ``len(text)`` badly misjudges a German compound, whose glyphs run
    narrower or wider than the sample average — this is DEFEKT 1 from
    SAFE-25: the character-count guard in ``_word_scales`` only ever
    stepped an *emphasised* word back down to 1.0, and never caught a word
    that was already too wide at scale 1.0.
    """
    font = _measurement_font(preset, font_size)
    if font is None:
        return fallback

    average = font.getlength(_AVERAGE_GLYPH_SAMPLE) / len(_AVERAGE_GLYPH_SAMPLE)
    if average <= 0:
        return fallback

    pixel = _measured_pixel_width(preset, font_size, text)
    return pixel / average if pixel is not None else fallback


def resolve_caption_style(style):
    name = str(style or DEFAULT_CAPTION_STYLE).strip().lower()
    return CAPTION_STYLE_PRESETS.get(name, CAPTION_STYLE_PRESETS[DEFAULT_CAPTION_STYLE])


def resolve_caption_fx(spec=DEFAULT_CAPTION_FX):
    """Resolve a comma-separated fx spec into the set of active effects.

    Unlike ``--caption-style``/``--caption-pop``, a comma here means *combine*,
    not "render one variant each": these effects stack on the same caption.
    """
    parts = [part.strip().lower() for part in str(spec or "none").split(",")]
    active = set()
    for part in parts:
        if not part or part == "none":
            continue
        if part not in CAPTION_FX_CHOICES:
            raise ValueError(f"Unknown caption fx: {part}")
        active.add(part)
    # fade_up, typewriter and flicker all drive the same \alpha channel on the
    # same word; two of them at once do not layer, they overwrite each other.
    alpha_effects = active & {"fade_up", "typewriter", "flicker"}
    if len(alpha_effects) > 1:
        raise ValueError(
            "These caption fx all animate opacity and cannot be combined: "
            + ", ".join(sorted(alpha_effects))
        )
    return frozenset(active)


def resolve_caption_glow(mode=DEFAULT_CAPTION_GLOW):
    selected = str(mode or DEFAULT_CAPTION_GLOW).strip().lower()
    if selected not in CAPTION_GLOW_MODES:
        raise ValueError(f"Unknown caption glow: {selected}")
    return selected


def _caption_glow_filter_complex(mode, *, subtitle_name, fonts_dir, width, height, fps,
                                 extra_vf=""):
    """Composite: base video, bloom of the caption core, then the sharp caption."""
    profile = CAPTION_GLOW_PROFILES[mode]
    ass_opts = f"filename={subtitle_name}:alpha=1"
    if fonts_dir and os.path.isdir(fonts_dir):
        ass_opts += f":fontsdir={fonts_dir}"
    tail = f",{extra_vf}" if extra_vf else ""
    return (
        f"color=c=black@0.0:s={width}x{height}:r={fps:.6f},format=rgba,ass={ass_opts}[cap];"
        "[cap]split=2[capsharp][capglow];"
        "[capglow]format=rgba,split=2[glowrgb][glowluma];"
        f"[glowluma]format=gray,lutyuv=y='if(gt(val,{profile['threshold']}),val,0)'[glowmask];"
        f"[glowrgb][glowmask]alphamerge,gblur=sigma={profile['sigma']}:steps=2,"
        f"colorchannelmixer=aa={profile['opacity']}[glow];"
        "[0:v][glow]overlay=shortest=1[withglow];"
        f"[withglow][capsharp]overlay=shortest=1{tail}[vout]"
    )


def caption_fx_fonts_dir(fx):
    """Font directory for the burn, or ``None`` to keep the style's own.

    The mix directory is materialised on demand as symlinks, so neither family
    is duplicated in the repo and both stay resolvable by libass.
    """
    if "font_mix" not in fx:
        return None
    try:
        os.makedirs(CAPTION_FX_FONT_DIR, exist_ok=True)
        for source_dir in (BLACK_FONT_DIR, MIX_FONT_DIR):
            for name in sorted(os.listdir(source_dir)):
                if not name.lower().endswith((".ttf", ".otf")):
                    continue
                link = os.path.join(CAPTION_FX_FONT_DIR, name)
                if not os.path.exists(link):
                    os.symlink(os.path.join(source_dir, name), link)
    except OSError as error:
        print(f"[Subtitles] Font mix unavailable ({error}); keeping the style font")
        return None
    return CAPTION_FX_FONT_DIR


_MIX_FONT_RATIO_CACHE = {}


def _mix_font_size_ratio(budget_text, render_text, size):
    """Size factor that keeps the mixed word at the width it was measured for.

    The line budget was measured on ``budget_text`` (the word's normal,
    style-cased form — usually uppercase) in the style face; the second
    family has to hold that same measure while actually rendering
    ``render_text`` (its own case — a connected script keeps natural case,
    see ``MIX_FONT_NAME``), so both are measured, never guessed. Mrs Saint
    Delafield sets *much narrower* than Barlow Semi Condensed Black at equal
    point size (measured ~1.5x-1.7x on real German words: "Gott" 1.64x,
    "Vergebung" 1.70x, a 29-char compound 1.71x) — a script face's letters
    are simply slimmer than a condensed black sans — so the clamp has to
    reach further than the old Anton-vs-Barlow pairing (0.7-1.2) needed.
    """
    key = (budget_text, render_text, size)
    if key in _MIX_FONT_RATIO_CACHE:
        return _MIX_FONT_RATIO_CACHE[key]

    ratio = 1.0
    try:
        from PIL import ImageFont

        base = ImageFont.truetype(
            os.path.join(BLACK_FONT_DIR, "BarlowSemiCondensed-Black.ttf"), size,
        )
        mixed = ImageFont.truetype(os.path.join(MIX_FONT_DIR, MIX_FONT_FILE), size)
        base_width = base.getlength(budget_text)
        mixed_width = mixed.getlength(render_text)
        if mixed_width > 0 and base_width > 0:
            ratio = max(0.7, min(2.0, base_width / mixed_width))
    except Exception:
        ratio = 1.0

    _MIX_FONT_RATIO_CACHE[key] = ratio
    return ratio


def _letter_reveal(text, pattern):
    """Per-character alpha animation inside one Dialogue event.

    ``pattern`` returns the override tags for character *index*; the character
    itself is always emitted, so the line keeps its full width from the first
    frame and cannot reflow while the animation runs.
    """
    pieces = []
    for index, char in enumerate(text):
        tags = pattern(index)
        escaped = _escape_ass_text(char)
        pieces.append(f"{{{tags}}}{escaped}" if tags else escaped)
    return "".join(pieces)


def _typewriter_pattern(length):
    step = min(TYPEWRITER_STEP_MS, max(8, TYPEWRITER_MAX_MS // max(1, length)))

    def pattern(index):
        start = index * step
        return f"\\alpha&HFF&\\t({start},{start + TYPEWRITER_FADE_MS},\\alpha&H00&)"

    return pattern


def _flicker_pattern(length):
    def pattern(index):
        # Deterministic, not random: the same word has to render identically on
        # every rerun, or a caption stops being reproducible.
        cursor = ((index * 7) % 5) * 12
        tags = ""
        for dip in range(FLICKER_DIPS):
            tags += (
                f"\\t({cursor},{cursor + FLICKER_STEP_MS // 2},\\alpha{FLICKER_DIP_ALPHA})"
                f"\\t({cursor + FLICKER_STEP_MS // 2},{cursor + FLICKER_STEP_MS},\\alpha&H00&)"
            )
            cursor += FLICKER_STEP_MS + 25 * (dip + 1)
        return "\\alpha&H00&" + tags

    return pattern


def resolve_caption_sfx(mode=DEFAULT_CAPTION_SFX):
    selected = str(mode or DEFAULT_CAPTION_SFX).strip().lower()
    if selected not in CAPTION_SFX_MODES:
        raise ValueError(f"Unknown caption sfx mode: {selected}")
    return selected


def plan_caption_sfx_cues(word_events, mode=DEFAULT_CAPTION_SFX, preset=None):
    """Body-relative SFX cues derived from already-normalised caption timings.

    Candidates are the punch words (click) and the words of a word-by-word
    passage (typing tick). Everything else is rate limiting, and the limits are
    the whole point: an accent that fires on every punch word is a metronome.

    The punch word is resolved exactly the way the renderer resolves it, via
    ``_pick_emphasis_indices`` — reading ``word["emphasis"]`` alone would make
    the click depend on the LLM emphasis map being present, and it would land
    on a different word than the one the viewer sees grow. A style that renders
    no oversized word at all (``classic``) gets no click: there is nothing for
    it to accent.
    """
    selected = resolve_caption_sfx(mode)
    if selected == "off" or not word_events:
        return []

    want_typing = selected in ("typing", "both")
    want_click = selected in ("click", "both") and (
        preset is None or float(preset.get("emphasis_scale", 1.0)) > 1.0
    )

    cues = []
    accent_times = []
    letter_count = 0
    last_letter = None

    for phrase in word_events:
        accented = _pick_emphasis_indices((phrase or [])[:MAX_WORDS_PER_PHRASE])
        for index, word in enumerate(phrase or []):
            start = word.get("start")
            if start is None or start < CAPTION_SFX_LEAD_IN_SEC:
                continue
            if word.get("solo"):
                if not want_typing or letter_count >= CAPTION_SFX_LETTER_MAX:
                    continue
                if last_letter is not None and start - last_letter < CAPTION_SFX_LETTER_MIN_GAP_SEC:
                    continue
                cues.append({"event": CAPTION_SFX_LETTER_EVENT, "time_sec": float(start)})
                last_letter = start
                letter_count += 1
                continue
            if not want_click or index not in accented:
                continue
            if accent_times and start - accent_times[-1] < CAPTION_SFX_ACCENT_MIN_GAP_SEC:
                continue
            recent = [t for t in accent_times if start - t < CAPTION_SFX_ACCENT_WINDOW_SEC]
            if len(recent) >= CAPTION_SFX_ACCENT_PER_WINDOW:
                continue
            cues.append({"event": CAPTION_SFX_ACCENT_EVENT, "time_sec": float(start)})
            accent_times.append(start)

    cues.sort(key=lambda cue: cue["time_sec"])
    return cues


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


def _ms_int(seconds):
    """Integer milliseconds, rounded exactly the way the contract producer's
    own ``_ms`` rounds -- the two must never disagree."""
    return int(round(float(seconds) * 1000.0))


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
        wpm = _phrase_wpm(phrase)
        # The speaking rate decides how big the pop is, NOT whether the words
        # build up one at a time. Gating the reveal on it meant that at this
        # speaker's 194 wpm median, 69% of phrases showed all four words at
        # once - which reads as the caption running ahead of the voice, because
        # words b, c and d stand there before they are spoken.
        if wpm == float("inf"):
            # Retimed or otherwise unreliable timings. The build-up would be
            # keyed to invented word starts, so it stays off entirely - the
            # rate here measures the retimer, not the speaker.
            level = None
        elif wpm < BALLOON_MAX_WPM:
            level = "strong"
        elif wpm < BALLOON_MEDIUM_MAX_WPM:
            level = "medium"
        else:
            level = "subtle"
        for word in phrase:
            word["balloon_eligible"] = level is not None
            # The rate this level was decided from, kept on the words so a
            # later solo split (which re-groups tokens into ~350ms fragments
            # whose own rate no longer means anything) can carry the parent
            # passage's pacing forward instead of re-measuring the fragment.
            if wpm != float("inf"):
                word["pacing_wpm"] = wpm
            if level is not None:
                word["balloon_level"] = level
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
        parent_wpm = _phrase_wpm(phrase)
        for unit in _solo_units(phrase):
            solo = {
                "text": " ".join(str(w["text"]).strip() for w in unit),
                "start": unit[0]["start"],
                "end": unit[-1]["end"],
                "solo": True,
            }
            # A solo unit's own span is a fragment of the phrase that decided
            # the pacing; carry the parent's measured rate, never re-measure.
            if parent_wpm != float("inf"):
                solo["pacing_wpm"] = parent_wpm
            if any(w.get("emphasis") for w in unit):
                solo["emphasis"] = True
            if all(w.get("balloon_eligible") for w in unit):
                solo["balloon_eligible"] = True
                solo["balloon_level"] = unit[0].get("balloon_level", "strong")
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
    return width, height, duration, fps


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


def _word_scales(phrase, preset, max_chars=None, scale_override=None, font_size=None,
                 peak_multiplier=1.0, isolated_usable_width_px=None):
    """Per-word size multipliers, constant for every event of the phrase.

    Constant is the whole point: each word's Dialogue event re-renders the
    entire phrase, so an override set that changed between events would
    reflow the centred line and make the caption jitter word by word.

    ``peak_multiplier`` is the Balloon's whole-event ``\\fscx`` overshoot
    (e.g. 1.24) that gets applied on top of this word's own ``\\fs`` at
    render time — a shared-line word must fit *after* that multiplication,
    not just at rest. ``isolated_usable_width_px`` is the wider, real
    measured safe corridor (see ``isolated_word_usable_width`` in
    ``_caption_layout``) that a word with no line-mates can use instead of
    ``max_chars``' multi-word budget.
    """
    scale = float(scale_override if scale_override is not None else preset.get("emphasis_scale", 1.0))
    words = phrase[:MAX_WORDS_PER_PHRASE]
    budget = max_chars or preset["max_chars_per_line"]

    scales = [1.0] * len(words)
    if scale > 1.0:
        for index in _pick_emphasis_indices(words):
            word_scale = scale
            # A long German compound at 1.5× overflows the frame on its own;
            # step it down rather than shrink the whole caption.
            while word_scale > 1.0 and len(words[index]["text"]) * word_scale > budget:
                word_scale -= 0.25
            scales[index] = max(1.0, word_scale)

    # SAFE-25 DEFEKT 1: the emphasis loop above only ever steps a word back
    # down *to* 1.0 — it never shrinks a word below that, so a long German
    # compound that is already too wide at the style's base size rendered at
    # full size regardless, and wrap_phrase_words cannot split it any
    # further. This pass covers every word, at whatever scale it already
    # has, against the budget already discounted by the Balloon peak so the
    # *animated* width — not just the resting width — stays in bounds.
    #
    # A word that trips this check has no line-mates by construction: it was
    # already too wide for the multi-word budget on its own, so
    # wrap_phrase_words gives it its own line. It therefore doesn't need that
    # budget's multi-word margin either — measured against the real safe
    # corridor (isolated_usable_width_px) it needs far less shrinking than
    # the conservative multi-word budget alone would suggest (e.g. a 12-char
    # word landed at ~0.57x against the multi-word budget but only needed
    # ~0.75x against the real corridor — reported by Benjamin as captions
    # rendering needlessly small on real material).
    if font_size:
        if isolated_usable_width_px:
            effective_budget_px = isolated_usable_width_px / max(peak_multiplier, 1.0)
            for index, word in enumerate(words):
                pixel_width = _measured_pixel_width(preset, font_size, word["text"])
                if pixel_width is None or pixel_width <= 0:
                    continue
                if pixel_width * scales[index] > effective_budget_px:
                    fitted = effective_budget_px / pixel_width
                    scales[index] = max(_MIN_WORD_SCALE, min(scales[index], fitted))
        else:
            effective_budget = budget / max(peak_multiplier, 1.0)
            for index, word in enumerate(words):
                length = _measured_char_length(preset, font_size, word["text"], None)
                if length is None or length <= 0:
                    continue
                if length * scales[index] > effective_budget:
                    fitted = effective_budget / length
                    scales[index] = max(_MIN_WORD_SCALE, min(scales[index], fitted))

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


def _fx_word_tags(fx, *, skip_alpha=False):
    """Animation tags for the arriving word, plus the reset the line needs.

    Inline overrides persist for the rest of the line, so a word that fades or
    resolves from blur has to hand the following words their sharp, opaque
    state back explicitly.
    """
    tags = ""
    resets = ""
    if "fade_up" in fx and not skip_alpha:
        tags += (
            f"\\alpha{FADE_UP_START_ALPHA}"
            f"\\t(0,{FADE_UP_MS},\\alpha&H00&)"
        )
        resets += "\\alpha&H00&"
    if "blur_words" in fx:
        tags += f"\\blur{BLUR_WORDS_START}\\t(0,{BLUR_WORDS_MS},\\blur0)"
        resets += "\\blur0"
    return tags, resets


def _render_word(text, *, active, scale, base_font_size, preset, balloon_level=None,
                 fx=frozenset(), mixed=False):
    budget_plain = safe_upper(text) if preset["uppercase"] else text
    # The mix font is a connected script — full caps render as an unreadable
    # blob of loops (verified render, 2026-08-25), so the swapped word keeps
    # its natural case instead of the style's usual uppercasing. The line's
    # width budget was still measured on ``budget_plain``; see
    # ``_mix_font_size_ratio``.
    plain = text if (mixed and "font_mix" in fx) else budget_plain
    escaped = _escape_ass_text(plain)
    colour = ACTIVE_COLOUR if active else INACTIVE_COLOUR
    letters = (fx & {"typewriter", "flicker"}) if active else frozenset()

    if active and preset["active_ramp"] and not letters:
        # Start on the inactive colour and animate into the highlight, so the
        # word visibly ignites instead of cutting.  Colour-only animation
        # cannot change glyph advances, so the line never moves while the
        # highlight travels across it.
        #
        # Skipped when a letter effect is also revealing this word: the
        # per-character alpha fade already *is* the arrival animation, and
        # stacking a 90 ms white-to-yellow colour ramp on top of a handful of
        # fast, staggered per-letter alpha fades produced a muddy,
        # low-contrast blend for the first letters typed — real ICF render,
        # 2026-08-25: the "F" of "Frage," briefly rendered almost invisible
        # against the background. Reported by Benjamin as a caption looking
        # "cut off/invisible".
        colour_tags = f"\\c{INACTIVE_COLOUR}\\t(0,90,\\c{ACTIVE_COLOUR})"
    else:
        colour_tags = f"\\c{colour}"

    if active and balloon_level:
        # Only the opacity half of the balloon can run here: the word shares
        # its line with the rest of the phrase, so scaling it would push the
        # others sideways. The inflate rides on the phrase entrance instead.
        colour_tags += _balloon_alpha_tags(balloon_level)

    fx_reset = ""
    if active and fx:
        # The balloon already carries the opacity rise; a second \alpha ramp on
        # the same word would only fight it. The same is true for the letter
        # effects, which drive \alpha per character. (``letters`` computed
        # above, ahead of the colour_tags decision.)
        fx_tags, fx_reset = _fx_word_tags(
            fx, skip_alpha=bool(balloon_level) or bool(letters),
        )
        colour_tags += fx_tags
        if letters:
            pattern = (_typewriter_pattern if "typewriter" in letters else _flicker_pattern)(
                len(plain)
            )
            escaped = _letter_reveal(plain, pattern)
            fx_reset += "\\alpha&H00&"

    size = base_font_size * scale
    if mixed and "font_mix" in fx:
        # The punch word carries the second family. Constant across every event
        # of the phrase by construction — the picked index does not change —
        # and width-compensated, so the line it sits in still fits.
        size *= _mix_font_size_ratio(budget_plain, plain, int(round(size)))
        colour_tags = f"\\fn{MIX_FONT_NAME}" + colour_tags
        fx_reset += f"\\fn{preset['fontname']}"
        scale = max(scale, 1.0001)  # force the explicit \fs below

    if scale != 1.0:
        # Inline \fs, never \r: a style reset would also wipe the karaoke
        # colour of the active word. scale < 1.0 happens too now (SAFE-25
        # DEFEKT 1) -- a word too wide even at the style's base size has to
        # shrink below it, not just step back down to it.
        opening = f"\\fs{int(round(size))}{colour_tags}"
        closing = f"\\fs{base_font_size}{fx_reset}"
    else:
        opening = colour_tags
        closing = fx_reset
    closing = f"{{{closing}}}" if closing else ""

    return f"{{{opening}}}{escaped}{closing}"


def _balloon_scale_tags(level="strong"):
    """Inflate-with-overshoot, safe only where nothing shares the event.

    Scaling changes glyph advances, so this may only be applied to an event
    that carries a single word or the whole caption at once — never to one
    word inside a shared line, which would reflow the others.
    """
    profile = BALLOON_MOTION_PROFILES[level]
    return (
        f"\\fscx{profile['start_scale']}\\fscy{profile['start_scale']}"
        f"\\blur{profile['start_blur']}"
        f"\\t(0,{profile['inflate_ms']},\\fscx{profile['overshoot_scale']}\\fscy{profile['overshoot_scale']})"
        f"\\t(0,{profile['fade_ms']},\\blur0)"
        f"\\t({profile['inflate_ms']},{profile['inflate_ms'] + profile['settle_ms']},\\fscx100\\fscy100)"
    )


def _balloon_alpha_tags(level="strong"):
    profile = BALLOON_MOTION_PROFILES[level]
    return (
        f"\\alpha{profile['start_alpha']}"
        f"\\t(0,{profile['fade_ms']},\\alpha{BALLOON_FULL_ALPHA})"
    )


def _build_highlight_text_for_word(phrase, active_word_idx, preset=None, base_font_size=100,
                                   max_chars=None, balloon_level=None, fx=frozenset(),
                                   isolated_usable_width_px=None):
    if preset is None:
        preset = CAPTION_STYLE_PRESETS[DEFAULT_CAPTION_STYLE]
    if max_chars is None:
        max_chars = preset["max_chars_per_line"]

    # The balloon deliberately reverses the no-reflow invariant every other
    # style relies on: words build up one at a time instead of all standing
    # there at once, so the wrapping still has to be computed from the FULL
    # phrase (otherwise the line break would jump as words are revealed) but
    # only words up to the active one are actually emitted.
    scale_override = (
        BALLOON_MOTION_PROFILES[balloon_level]["emphasis_scale"]
        if balloon_level else None
    )
    # The Balloon's overshoot scales the *whole shared-line event*, not just
    # the active word -- every word visible in that event peaks at this
    # factor too, so the width check has to budget for it up front.
    peak_multiplier = (
        BALLOON_MOTION_PROFILES[balloon_level]["overshoot_scale"] / 100.0
        if balloon_level else 1.0
    )
    scales = _word_scales(
        phrase, preset, max_chars=max_chars, scale_override=scale_override,
        font_size=base_font_size, peak_multiplier=peak_multiplier,
        isolated_usable_width_px=isolated_usable_width_px,
    )
    # The compound guard in _word_scales can step an oversized word back to 1.0;
    # the font mix still belongs on that word, so read the pick directly.
    mix_indices = (
        _pick_emphasis_indices(phrase[:MAX_WORDS_PER_PHRASE]) if "font_mix" in fx else set()
    )
    wrapped_lines, line_ranges = _build_phrase_layout_metadata(
        phrase, scales=scales, max_chars=max_chars,
    )
    lines = []

    for start_idx, end_idx in line_ranges:
        if balloon_level and start_idx > active_word_idx:
            continue  # this line has not been reached by the spoken word yet
        reveal_end = min(end_idx, active_word_idx + 1) if balloon_level else end_idx
        rendered_words = []
        for word_idx in range(start_idx, reveal_end):
            rendered_words.append(
                _render_word(
                    phrase[word_idx]["text"],
                    active=(word_idx == active_word_idx),
                    scale=scales[word_idx],
                    base_font_size=base_font_size,
                    preset=preset,
                    balloon_level=balloon_level,
                    fx=fx,
                    mixed=(word_idx in mix_indices),
                )
            )
        if rendered_words:
            lines.append(" ".join(rendered_words))

    return r"\N".join(lines)


def _solo_word_font_size_px(word, *, base_font_size, preset, max_chars=None,
                             balloon_level=None, usable_width=None):
    """Resolved static size (px) for a solo word — factored out of
    ``_build_solo_word_text`` so a contract producer (MAT-47) can obtain the
    exact same number without re-deriving it by hand and risking drift.
    """
    size = int(round(base_font_size * max(1.25, preset["emphasis_scale"])))
    # SAFE-25 DEFEKT 2: a solo word has no line-mates, so — like the
    # shared-line path — it should be checked against its real measured
    # width and the real safe corridor (``usable_width``, the wider,
    # isolated-word bound: see ``isolated_word_usable_width`` in
    # ``_caption_layout``), not a raw character count against the
    # conservative multi-word budget. Checked at the size the Balloon
    # overshoot actually scales up to, since that peak — not the resting
    # size — is the true footprint that has to stay in bounds.
    peak_scale = (
        BALLOON_MOTION_PROFILES[balloon_level]["overshoot_scale"] / 100.0
        if balloon_level else 1.0
    )
    if usable_width:
        peak_width = _measured_pixel_width(preset, int(round(size * peak_scale)), word["text"])
        if peak_width is not None and peak_width > usable_width:
            size = max(int(base_font_size * _MIN_WORD_SCALE), int(size * usable_width / peak_width))
    else:
        # No measured safe corridor available — fall back to the coarser
        # character-count clamp against the multi-word budget.
        budget = max_chars or preset["max_chars_per_line"]
        length = len(word["text"])
        if length > budget:
            size = max(base_font_size, int(size * budget / length))
    return size


def _build_solo_word_text(word, *, base_font_size, preset, video_width, video_height, margin_v,
                          max_chars=None, balloon_level=None, anchor_x=None, anchor_y=None,
                          fx=frozenset(), usable_width=None):
    """One big word, centred, popping in — used when the speaker slows down."""
    budget_plain = safe_upper(word["text"]) if preset["uppercase"] else word["text"]
    # Same case exception as _render_word: the mix font is a connected
    # script, unreadable in full caps, so it keeps the word's natural case.
    plain = word["text"] if "font_mix" in fx else budget_plain
    text = _escape_ass_text(plain)
    letters = fx & {"typewriter", "flicker"}
    if letters:
        pattern = (_typewriter_pattern if "typewriter" in letters else _flicker_pattern)(len(plain))
        text = _letter_reveal(plain, pattern)
    size = _solo_word_font_size_px(
        word, base_font_size=base_font_size, preset=preset, max_chars=max_chars,
        balloon_level=balloon_level, usable_width=usable_width,
    )
    # \pos measures from the top, MarginV from the bottom — land the solo word
    # on the same optical line the wrapped captions occupy.
    pos_x = int(anchor_x if anchor_x is not None else video_width // 2)
    baseline_y = int(anchor_y if anchor_y is not None else video_height - margin_v)
    pos_y = max(size, int(baseline_y - size * 0.5))
    colour = ACTIVE_COLOUR if word.get("emphasis") else INACTIVE_COLOUR
    if balloon_level:
        # A solo event stands alone, so the full balloon is safe here.
        motion = _balloon_scale_tags(balloon_level) + _balloon_alpha_tags(balloon_level)
    else:
        motion = f"\\fscx70\\fscy70\\t(0,{SOLO_POP_MS},\\fscx100\\fscy100)"
        if "fade_up" not in fx:
            motion += "\\fad(40,0)"
    if "fade_up" in fx:
        # Nothing shares this event, so "fade up" can be the literal reference
        # effect here: the word rises into place as it becomes opaque. \move is
        # the only way to translate a caption and it is event-wide — which is
        # exactly why this half of the effect cannot run inside a shared line.
        rise = int(round(size * FADE_UP_RISE_RATIO))
        entrance = f"\\move({pos_x},{pos_y + rise},{pos_x},{pos_y},0,{FADE_UP_MS})"
        if not balloon_level:
            entrance += f"\\alpha{FADE_UP_START_ALPHA}\\t(0,{FADE_UP_MS},\\alpha&H00&)"
        motion = entrance + motion
        position = "\\an5"
    else:
        position = f"\\an5\\pos({pos_x},{pos_y})"
    if "blur_words" in fx:
        motion += f"\\blur{BLUR_WORDS_START}\\t(0,{BLUR_WORDS_MS},\\blur0)"
    if letters:
        # The per-character \alpha animation owns opacity here; an event-wide
        # fade would only overwrite it.
        motion = motion.replace("\\fad(40,0)", "")
    font = ""
    if "font_mix" in fx:
        font = f"\\fn{MIX_FONT_NAME}"
        size = int(round(size * _mix_font_size_ratio(budget_plain, plain, size)))
    return (
        f"{{{position}{font}\\fs{size}\\c{colour}{motion}}}{text}"
    )


def _normalise_phrase_timings(word_events, caption_cutoff=None, lead_in=0.0,
                              body_duration_ms=None, cutoff_sink=None):
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

    ``body_duration_ms``/``cutoff_sink`` (INT-51, contract mode): the caller's
    own ``caption_cutoff`` and the contract's ``caption_cutoff_ms`` were
    computed from *different* word sets, with *different* end clamps, and
    quantised to milliseconds in a *different order* -- three independent
    divergences that made the contract's display intervals overrun its own
    cutoff by exactly 1 ms on 6 of 15 real ICF highlights. When
    ``body_duration_ms`` is given, the cutoff is instead derived here, in
    integer milliseconds, from the same post-truncation word set the contract
    producer sees, and reported back through ``cutoff_sink`` so producer and
    renderer cannot disagree by construction. The ASS emitter only has
    centisecond resolution (``_seconds_to_ass_time``), so this is provably
    invisible to rendered output.
    """
    if not word_events:
        return word_events

    if body_duration_ms is not None:
        truncated_ends = [
            _ms_int(w["end"])
            for phrase in word_events if phrase
            for w in phrase[:MAX_WORDS_PER_PHRASE]
        ]
        if truncated_ends:
            cutoff_ms = min(int(body_duration_ms), max(truncated_ends))
            caption_cutoff = cutoff_ms / 1000.0
            if cutoff_sink is not None:
                cutoff_sink["caption_cutoff_ms"] = cutoff_ms

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
            ws = max(w["start"] + ASR_ONSET_CORRECTION_SEC - phrase_lead_in, cursor)
            # A solo word is its own phrase, so it would otherwise inherit the
            # full inter-phrase hold and linger on screen long after it was
            # spoken — which defeats the point of word-by-word pacing.
            hold = SOLO_HOLD_SEC if w.get("solo") else HOLD_AFTER_PHRASE_SEC

            if j + 1 < n:
                # Mid-phrase: use next word's start (no gap within phrase).
                # The next event pulls its own start forward by lead_in too,
                # so this one must end at that same earlier point.
                we = max(ws + 0.001,
                         phrase[j + 1]["start"] + ASR_ONSET_CORRECTION_SEC - phrase_lead_in)
            elif i + 1 < total and word_events[i + 1]:
                # Last word of phrase: hold briefly, but never overlap next phrase
                natural = w["end"] + hold
                next_phrase = word_events[i + 1]
                next_lead_in = lead_in if next_phrase[0].get("balloon_eligible") else 0.0
                next_start = next_phrase[0]["start"] + ASR_ONSET_CORRECTION_SEC - next_lead_in
                we = min(natural, next_start)
                we = max(we, ws + 0.001)
            else:
                # Very last word overall
                we = max(ws + 0.001, w["end"] + hold)

            if caption_cutoff is not None:
                we = min(we, max(ws + 0.001, caption_cutoff))

            # MAT-47: keep the pre-normalisation (spoken) interval alongside
            # the just-computed display interval. Nothing in the ASS
            # emission path reads these two extra keys, but a contract
            # producer materialised from this same list otherwise has no way
            # to recover "when was this word actually spoken" — that
            # information is gone the moment ``start``/``end`` are
            # overwritten with the lead-in/hold-adjusted display values.
            carried = {
                "text": w["text"], "start": ws, "end": we,
                "spoken_start": w["start"], "spoken_end": w["end"],
            }
            for key in ("emphasis", "solo", "synthetic", "balloon_eligible",
                        "balloon_level", "pacing_wpm", "probability", "timing_source"):
                if key in w:
                    carried[key] = w[key]
            copied.append(carried)
            cursor = we

        if copied:
            normalised.append(copied)

    return normalised


def _write_ass_file(subtitle_path, video_width, video_height, chunks, word_events=None,
                    style=DEFAULT_CAPTION_STYLE, pop=DEFAULT_CAPTION_POP, caption_cutoff=None,
                    caption_safe_area=DEFAULT_CAPTION_SAFE_AREA,
                    caption_fx=DEFAULT_CAPTION_FX, caption_sfx=DEFAULT_CAPTION_SFX,
                    contract_sink=None, body_duration_ms=None):
    """Write the ASS file and return the body-relative caption SFX cues.

    ``contract_sink`` (MAT-47/INT-50): an optional, write-only, empty dict a
    caller can pass to receive the exact post-normalisation render state
    (``word_events``, ``preset``, ``font_size``, ``max_chars``, ``layout``,
    ``balloon``) needed for ``Components.CaptionContract.build_caption_contract``.
    Populated exactly once, right after ``_normalise_phrase_timings``, before
    any ASS ``Dialogue`` line is emitted -- never touched when ``None`` (the
    default), so the existing render path is provably unaffected. Kept as a
    side-channel rather than a return value so this function's return type
    (the SFX cue list) never has to change for callers that don't ask for a
    contract. Must be empty on entry: this function does not clear or merge
    into it, and a caller rendering multiple clips in a loop must pass a
    fresh dict every time.
    """
    if contract_sink is not None and contract_sink:
        raise ValueError("contract_sink must be an empty dict on entry")
    preset = resolve_caption_style(style)
    balloon = str(pop or "").lower() == "balloon"
    fx = resolve_caption_fx(caption_fx)

    # slightly smaller than before
    font_size = max(33, int(video_height * preset["font_ratio"]) - 2)

    # Audience feedback preferred the captions lower than the previous 45%
    # placement, while still leaving room for platform controls.
    layout = _caption_layout(video_width, video_height, caption_safe_area)
    margin_v = layout["margin_v"]
    margin_h = layout["margin_h"]

    max_chars = _measured_char_budget(
        preset, font_size, layout["usable_width"],
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

    sfx_cues = []
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
        cutoff_sink = {} if contract_sink is not None else None
        word_events = _normalise_phrase_timings(
            word_events, caption_cutoff=caption_cutoff, lead_in=lead_in,
            body_duration_ms=body_duration_ms, cutoff_sink=cutoff_sink,
        )
        if contract_sink is not None:
            # Exactly the state build_caption_contract() needs, captured at
            # exactly this point -- after normalisation, before any Dialogue
            # line is emitted. See the docstring above.
            contract_sink.update(
                word_events=word_events, preset=preset, font_size=font_size,
                max_chars=max_chars, layout=layout, balloon=balloon,
            )
            # The resolved cutoff, so build_caption_contract can use exactly
            # this number instead of recomputing it a fourth way.
            contract_sink.update(cutoff_sink or {})
        # Planned from the normalised timings, so a cue can never sit on a
        # word whose Dialogue event was moved, clamped, or dropped.
        sfx_cues = plan_caption_sfx_cues(word_events, caption_sfx, preset=preset)

        for phrase in word_events:
            if not phrase:
                continue

            phrase_balloon_level = (
                phrase[0].get("balloon_level") if balloon else None
            )

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
                        anchor_x=layout["anchor_x"],
                        anchor_y=layout["anchor_y"],
                        max_chars=max_chars,
                        balloon_level=phrase_balloon_level,
                        fx=fx,
                        usable_width=layout["isolated_word_usable_width"],
                    )
                else:
                    highlight_text = _build_highlight_text_for_word(
                        phrase, word_idx, preset=preset, base_font_size=font_size,
                        max_chars=max_chars, balloon_level=phrase_balloon_level,
                        fx=fx, isolated_usable_width_px=layout["isolated_word_usable_width"],
                    )
                    position = f"\\an2\\pos({layout['anchor_x']},{layout['anchor_y']})"
                    if phrase_balloon_level:
                        # Every new word is its own arrival, not just the
                        # phrase's first: scaling the whole event is
                        # reflow-free by definition — every currently visible
                        # word grows together, so nothing shifts relative to
                        # anything else, and libass keeps re-centring it while
                        # it grows. That growth is also what makes a new word
                        # visibly shove the existing ones aside — the effect
                        # is intentional, not a bug.
                        prefix = "{" + position + _balloon_scale_tags(phrase_balloon_level) + "}"
                    elif word_idx == 0:
                        # Fade-in only when the phrase first appears (first word)
                        prefix = "{" + position + r"\fad(100,0)}"
                    else:
                        prefix = "{" + position + "}"
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
            fade = "{" + f"\\an2\\pos({layout['anchor_x']},{layout['anchor_y']})" + r"\fad(80,40)}"
            lines.append(
                "Dialogue: 0,"
                f"{_seconds_to_ass_time(start)},"
                f"{_seconds_to_ass_time(end)},"
                f"Default,,0,0,0,,{fade}{safe_text}"
            )

    with open(subtitle_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return sfx_cues


_OVERLAP_EPSILON = 1e-9
# Two milliseconds: comfortably above the millisecond quantisation the caption
# contract's timeline uses, so a repaired span can never round to zero length.
_MIN_REPAIRED_SPAN_SEC = 0.002


def _repair_degenerate_word_runs(words, video_duration=0.0):
    """Give overlapping real ASR words their own non-overlapping spans.

    Real German sermon ASR (measured on ICF material, verified against the
    *raw* transcript -- these are not the retimer's interpolated words) emits
    runs of 2-22 consecutive words sharing near-identical start timestamps,
    with up to 940 ms of mutual overlap. A naive ``a.end = b.start`` nudge
    would produce a zero-duration word in 290 of 315 measured cases, so the
    run is redistributed instead:

    a) an immediately-repeating token cycle inside the run is collapsed to its
       first occurrence (an ASR loop such as "krank. krank." at one timestamp
       is one spoken word, not two);
    b) the survivors are spread evenly across the run's own frame, extended
       forward to the next distinct word's start when that frame is too narrow
       to give each survivor a visible slot;
    c) every survivor is marked ``synthetic``. This is the honest part: these
       timings were never measured, and ``_phrase_wpm``/``_phrase_is_slow``
       already correctly refuse to read a speaking rate off synthetic words --
       so a repaired run loses motion/SFX eligibility instead of driving it
       off a number the retimer invented.

    Applied to *both* the classic and the contract render path on purpose:
    gating it would reintroduce exactly the contract-vs-render drift the
    contract exists to rule out.
    """
    if len(words) < 2:
        return _enforce_word_span_invariant(words, video_duration)

    repaired = []
    index = 0
    total = len(words)
    while index < total:
        end = index
        while end + 1 < total and words[end]["end"] > words[end + 1]["start"] + _OVERLAP_EPSILON:
            end += 1
        if end == index:
            repaired.append(words[index])
            index += 1
            continue

        run = words[index:end + 1]

        survivors = [run[0]]
        for word in run[1:]:
            if _emphasis_token(word.get("text")) != _emphasis_token(survivors[-1].get("text")):
                survivors.append(word)

        # The run's own words can be out of order (that is what makes it
        # degenerate), so the frame comes from the extremes, never from run[0],
        # and never earlier than the last word already emitted.
        frame_start = min(w["start"] for w in run)
        if repaired:
            frame_start = max(frame_start, repaired[-1]["start"] + _MIN_REPAIRED_SPAN_SEC)
        frame_end = max(w["end"] for w in run)
        needed = len(survivors) * MIN_WORD_DISPLAY_SEC
        if frame_end - frame_start < needed:
            # Grow into the gap before the next real word; the display pass
            # re-sequences everything afterwards, so a small spoken-time
            # overlap with that word is harmless -- a zero-length word is not.
            frame_end = max(frame_end, frame_start + needed)
            if end + 1 < total:
                frame_end = min(frame_end, max(words[end + 1]["end"], frame_start + needed))
        if video_duration > 0:
            frame_end = max(min(frame_end, video_duration), frame_start)

        # Whatever room is left has to give every survivor a strictly positive
        # millisecond span. If it cannot, the run is an ASR loop at a single
        # timestamp against the clip end: keep as many words as fit rather
        # than emit spans the contract (rightly) rejects as invalid.
        while len(survivors) > 1 and (frame_end - frame_start) / len(survivors) < _MIN_REPAIRED_SPAN_SEC:
            survivors.pop()
        width = max((frame_end - frame_start) / len(survivors), _MIN_REPAIRED_SPAN_SEC)

        for position, word in enumerate(survivors):
            repaired.append({
                **word,
                "start": frame_start + position * width,
                "end": frame_start + (position + 1) * width,
                "synthetic": True,
            })

        index = end + 1

    return _enforce_word_span_invariant(repaired, video_duration)


def _enforce_word_span_invariant(words, video_duration=0.0):
    """Guarantee strictly increasing millisecond starts and positive spans.

    The redistribution above fixes *overlapping* runs, but real material also
    carries words that are degenerate on their own: the cleanup retimer can
    distribute one millisecond across three tokens (0.33 ms each, which rounds
    to a zero-length span), and ASR occasionally emits an end before its own
    start. Both are invisible in the classic centisecond ASS output and both
    make a caption contract structurally invalid, so this pass states the
    postcondition once, in one place, instead of leaving it implicit.

    Anything it has to move is marked ``synthetic`` -- the same honesty rule as
    the redistribution: a timing this pass invented must never be read as a
    measured speaking rate.
    """
    out = []
    cursor_ms = None
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        start_ms = _ms_int(start)
        touched = False

        if cursor_ms is not None and start_ms <= cursor_ms:
            start = (cursor_ms + 1) / 1000.0
            start_ms = cursor_ms + 1
            touched = True
        if _ms_int(end) <= start_ms:
            end = start + _MIN_REPAIRED_SPAN_SEC
            touched = True
        if video_duration > 0 and end > video_duration:
            end = video_duration
            if _ms_int(end) <= start_ms:
                # No room left in the clip for this word at all.
                continue
            touched = True

        if touched:
            word = {**word, "start": start, "end": end, "synthetic": True}
        out.append(word)
        cursor_ms = start_ms

    return out


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
        # _smart_pad_end can collapse the first word beyond the clip boundary
        # to zero length; that boundary-only word is correctly dropped below.
        # A zero-span word *inside* the clip is different: Whisper really did
        # emit its text, so preserve it for the synthetic repair pass.
        if end <= start:
            # Whisper can assign several real tokens the same boundary.  They
            # must reach the repair pass below; dropping them silently removes
            # spoken words from karaoke captions.  The provisional span is
            # explicitly synthetic and may be redistributed with neighbours.
            if video_duration > 0 and start >= video_duration:
                continue
            end = start + _MIN_REPAIRED_SPAN_SEC
            if video_duration > 0:
                end = min(video_duration, end)
            if end <= start:
                continue

        text = (w["text"] or "").strip()
        if text and not text.startswith("["):
            entry = {"text": text, "start": start, "end": end}
            for key in ("emphasis", "synthetic", "probability", "timing_source"):
                if key in w:
                    entry[key] = w[key]
            if float(w["end"]) <= float(w["start"]):
                entry["synthetic"] = True
                entry["timing_source"] = "repaired_zero_span"
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

    adjusted = _repair_degenerate_word_runs(adjusted, video_duration)

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
                           caption_pop=DEFAULT_CAPTION_POP,
                           caption_safe_area=DEFAULT_CAPTION_SAFE_AREA,
                           caption_fx=DEFAULT_CAPTION_FX,
                           caption_sfx=DEFAULT_CAPTION_SFX,
                           caption_glow=DEFAULT_CAPTION_GLOW,
                           contract_sink=None, body_duration_ms=None):
    """Burn the captions in and return the body-relative caption SFX cues.

    ``contract_sink``: see ``_write_ass_file`` -- passed straight through.
    """
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)

    video_width, video_height, video_duration, video_fps = _read_video_metadata(input_video)

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
        return []

    chunked = _chunk_transcriptions(relevant_transcriptions) if relevant_transcriptions else []
    if not chunked and not word_events:
        print("No subtitle chunks generated for this video segment")
        shutil.copyfile(input_video, output_video)
        return []

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

        sfx_cues = _write_ass_file(
            subtitle_path,
            video_width,
            video_height,
            chunked,
            word_events=word_events,
            style=caption_style,
            pop=caption_pop,
            caption_cutoff=caption_cutoff,
            caption_safe_area=caption_safe_area,
            caption_fx=caption_fx,
            caption_sfx=caption_sfx,
            contract_sink=contract_sink,
            body_duration_ms=body_duration_ms,
        )

        preset = resolve_caption_style(caption_style)
        fonts_dir = caption_fx_fonts_dir(resolve_caption_fx(caption_fx)) or preset["fontsdir"]
        n_events = len(word_events) if word_events else len(chunked)
        mode = "phrase highlight" if word_events else "chunked"
        pop_note = f", pop: {caption_pop}" if str(caption_pop or "none").lower() != "none" else ""
        fx_active = sorted(resolve_caption_fx(caption_fx))
        fx_note = f", fx: {'+'.join(fx_active)}" if fx_active else ""
        if resolve_caption_glow(caption_glow) != "off":
            fx_note += f", glow: {resolve_caption_glow(caption_glow)}"
        print(f"Adding {n_events} subtitle events ({mode}, style: {caption_style}{pop_note}{fx_note}) "
              "to video with FFmpeg NVENC...")
        sfx_mode = resolve_caption_sfx(caption_sfx)
        if sfx_mode != "off":
            # Printed even at zero: a requested effect that silently plans
            # nothing is the failure mode worth seeing in the log.
            print(f"[Subtitles] Caption SFX: {len(sfx_cues)} cue(s) ({sfx_mode})")

        glow = resolve_caption_glow(caption_glow)
        if glow != "off":
            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", input_video,
                "-filter_complex", _caption_glow_filter_complex(
                    glow,
                    subtitle_name=os.path.basename(subtitle_path),
                    fonts_dir=fonts_dir,
                    width=video_width,
                    height=video_height,
                    fps=video_fps,
                    extra_vf=extra_vf or "",
                ),
                "-map", "[vout]",
                "-an",
                *NVENC_FLAGS,
                output_video,
            ]
        else:
            vf_chain = f"subtitles={os.path.basename(subtitle_path)}"
            # Without an explicit directory libass resolves the family through
            # fontconfig and silently substitutes whatever is closest.
            if fonts_dir and os.path.isdir(fonts_dir):
                vf_chain += f":fontsdir={fonts_dir}"
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
        return sfx_cues
    finally:
        if subtitle_path and os.path.exists(subtitle_path):
            os.remove(subtitle_path)
