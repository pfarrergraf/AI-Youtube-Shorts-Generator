from dotenv import load_dotenv
import json as _json
from difflib import SequenceMatcher
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.error

load_dotenv()

def _first_env(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


# Prefer local vLLM naming, keep OpenAI-compatible names as compatibility fallbacks.
api_key = _first_env(
    "VLLM_API_KEY",
    "LOCAL_LLM_API_KEY",
    "OPENAI_API",
    "OPENAI_API_KEY",
    default="local-vllm",
)
api_base = _first_env(
    "VLLM_BASE_URL",
    "LOCAL_LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    default="http://127.0.0.1:1234/v1",
)
model_name = _first_env(
    "VLLM_MODEL",
    "LOCAL_LLM_MODEL",
    "LLM_MODEL",
    default="qwen2.5-72b-instruct",
)


def _llm_backend_mode() -> str:
    explicit = (_first_env("LLM_BACKEND", "LOCAL_LLM_BACKEND", default="") or "").strip().lower()
    if explicit in {"vllm", "local-vllm", "local_vllm"}:
        return "vllm"
    if explicit in {"lmstudio", "lm-studio", "lms"}:
        return "lmstudio"
    if any(os.getenv(name) for name in ("VLLM_API_KEY", "VLLM_BASE_URL", "VLLM_MODEL")):
        return "vllm"
    return "lmstudio"


# ---------- Local LLM server detection / optional LM Studio auto-start ----------

_LMS_CLI = (
    shutil.which("lms")
    or os.path.expandvars(r"%USERPROFILE%\.lmstudio\bin\lms.exe")
)

_llm_server_ready = False  # cached flag so we only check once per process


def _llm_server_reachable() -> bool:
    """Return True if the LLM server responds on the configured port."""
    try:
        url = (api_base or "http://localhost:1234/v1").rstrip("/")
        # strip /v1 to get base
        base = url.rsplit("/v1", 1)[0]
        req = urllib.request.Request(f"{base}/v1/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        urllib.request.urlopen(req, timeout=5)
        return True
    except urllib.error.HTTPError as e:
        # 401/403 means server IS running but needs auth — treat as reachable
        return e.code in (401, 403)
    except Exception:
        return False


def _llm_has_model_loaded() -> bool:
    """Return True if at least one model is loaded and ready."""
    try:
        url = (api_base or "http://localhost:1234/v1").rstrip("/")
        base = url.rsplit("/v1", 1)[0]
        req = urllib.request.Request(f"{base}/v1/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        r = urllib.request.urlopen(req, timeout=5)
        data = _json.loads(r.read())
        models = data.get("data", [])
        if not models:
            return False
        return True
    except Exception:
        return False


def _lmstudio_run(args: list, timeout: int = 120) -> bool:
    """Run an LM Studio CLI command. Returns True on success."""
    exe = _LMS_CLI
    if not exe or not os.path.isfile(exe):
        print(f"  [LM Studio] lms CLI not found at {exe}")
        return False
    try:
        cmd = [exe] + args
        print(f"  [LM Studio] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  [LM Studio] lms exited {result.returncode}: {(result.stderr or result.stdout)[:300]}")
            return False
        return True
    except FileNotFoundError:
        print(f"  [LM Studio] lms CLI not found")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [LM Studio] lms command timed out after {timeout}s")
        return False


def _ensure_llm_server():
    """Ensure the configured local LLM endpoint is reachable.

    For vLLM, this only checks the configured server.
    For LM Studio, this also tries to start the server and load the model.
    """
    global _llm_server_ready
    if _llm_server_ready:
        return
    backend_mode = _llm_backend_mode()

    if _llm_server_reachable():
        if _llm_has_model_loaded():
            print(f"  [LLM] {backend_mode.upper()} server has model(s) loaded — ready.")
            _llm_server_ready = True
            return
        if backend_mode != "lmstudio":
            print(f"  [LLM] {backend_mode.upper()} server reachable at {api_base}, but no models were reported.")
            return

    if backend_mode != "lmstudio":
        print(f"  [LLM] vLLM server not reachable at {api_base}. Start your local vLLM server first.")
        return

    # --- Step 1: server ---
    print("  [LM Studio] Server not reachable — starting it...")
    _lmstudio_run(["server", "start"])
    # Wait up to 30s for the server to come up
    for _ in range(15):
        time.sleep(2)
        if _llm_server_reachable():
            print("  [LM Studio] Server is up.")
            break
    else:
        print("  [LM Studio] WARNING: server did not start within 30s")
        return

    # --- Step 2: model ---
    # Check via API first (works with vLLM and LM Studio)
    if _llm_has_model_loaded():
        print(f"  [LLM] Server has model(s) loaded — ready.")
        _llm_server_ready = True
        return

    # Derive the model key from the configured model_name.
    # LM Studio identifiers can look like "qwen/qwen3-32b" or just "qwen3-32b".
    model_key = model_name.split("/")[-1] if "/" in model_name else model_name
    print(f"  [LM Studio] Ensuring model '{model_key}' is loaded...")
    # Use `lms ps` to check if the specific model is already loaded
    try:
        exe = _LMS_CLI
        if exe and os.path.isfile(exe):
            ps_result = subprocess.run(
                [exe, "ps"], capture_output=True, text=True, timeout=10,
            )
            if model_key.lower() in (ps_result.stdout or "").lower():
                print(f"  [LM Studio] Model '{model_key}' already loaded.")
                _llm_server_ready = True
                return
    except Exception:
        pass

    # Not loaded — load it
    ok = _lmstudio_run(["load", model_key, "-y", "--gpu", "max"], timeout=180)
    if ok:
        print(f"  [LM Studio] Model '{model_key}' loaded successfully.")
    else:
        print(f"  [LM Studio] WARNING: could not auto-load model '{model_key}'")

    _llm_server_ready = True


def _ensure_lm_studio():
    """Backward-compatible alias for older callers."""
    _ensure_llm_server()

SYSTEM_PROMPT = """\
You are a viral-content editor. The input is a timestamped transcription of a talk or sermon (often German).

The transcription has TWO entry types:
- **Speech segments**: timestamped text from the speaker.
- **Audience reaction markers**: `[AUDIENCE REACTION — loud, 2.3s]` with timestamps.

Your job: find the single best self-contained episode for a short-form video clip.

HOW TO FIND A COMPLETE SEQUENCE:
1. **Read the ENTIRE transcription first.** Identify where each distinct topic/story/joke BEGINS and ENDS.
2. **A sequence starts** where the speaker introduces a new topic, premise, or story. Look for topic shifts: "Also...", "Und dann...", "Ich war mal...", "Stellt euch vor...", or simply a new subject after a pause/reaction.
3. **A sequence ends** when:
   - The speaker moves to a DIFFERENT topic/premise (not just the next sentence in the same story)
   - The final audience reaction for this topic has completely finished
   - There is a clear pause before new content begins
4. **Your start time** = the EXACT timestamp where the speaker begins the setup for this topic. Going even 3 seconds too late means the viewer misses context and won't understand the joke.
5. **Your end time** = the END timestamp of the LAST audience reaction belonging to this topic. If the speaker continues the SAME joke after a reaction (escalation), include that too until the topic truly changes.

CRITICAL RULES:
- **Complete story arc**: setup → build-up → punchline → reaction → (optional escalation → bigger reaction). Missing the setup ruins the clip.
- **Never start mid-story**: If the joke is about "my father at the airport", start where the speaker first mentions the airport, NOT halfway through.
- **Never cut during reactions**: Your end MUST be >= the END timestamp of the last [AUDIENCE REACTION].
- **Stop before new content**: Do NOT include the beginning of the NEXT topic.
- **Duration**: 30-180 seconds. Too long is better than too short.
- **Complete sentences only**: Never start or end mid-sentence.

Return ONLY a JSON object (no markdown fences):
{
    "start": <start seconds — EXACT timestamp where this topic's setup begins>,
    "content": "<brief summary of what happens>",
    "end": <end seconds — after the last reaction, before the next topic starts>
}"""

MULTI_HIGHLIGHT_PROMPT = """\
You are a viral-content editor who understands rhetoric, theology, and audience psychology. The input is a timestamped transcription of a sermon, talk, or comedy set (often German).

The transcription has TWO entry types:
- **Speech segments**: timestamped text from the speaker.
- **Audience reaction markers**: `[AUDIENCE REACTION — loud, 2.3s]` with timestamps.

Your job: find ALL self-contained episodes that would each make a compelling short-form video clip.

WHAT MAKES A CLIP GREAT (in order of importance):
1. **Complete story arc with a PAYOFF** — Every clip MUST end with a clear payoff: a punchline -- better multiple punchlines each adding another layer --, a surprising twist or multiple twists, an emotional revelation, a moment of laughter, or a profound insight. A story without its ending is WORTHLESS, unless it is a strong part conveying a message. The payoff is what makes people rewatch.
2. **Audience impact** — The clip should make the viewer feel something: laugh, think, get goosebumps, feel convicted, or be genuinely surprised. Rate this honestly.
3. **Rhetorical power** — Vivid imagery, compelling analogies, well-timed pauses, rhetorical questions with answers, escalating tension, voice modulation implied by the text (exclamations, short punchy sentences, repetition).
4. **Self-contained meaning** — A first-time viewer who has NEVER seen the full video must fully understand the clip. No dangling references, no "as I said earlier".
5. **Psychological hooks** — Stories with conflict, unexpected turns, relatable situations, or statements that challenge assumptions.
6. **Theological/intellectual precision** — Interesting biblical insights, counter-intuitive interpretations, connections the audience hasn't heard before.
7. Humor — jokes, witty remarks, playful language, or funny situations. Humor is a strong engagement driver but must be complete with the punchline and reaction.

HOW TO IDENTIFY COMPLETE SEQUENCES:
1. **Read the ENTIRE transcription first.** Map out where each distinct topic, story, joke or argument begins and ends.
2. **A sequence starts** where the speaker introduces a new premise or topic. Signals: topic shift, setup phrase ("Also...", "Und dann...", "Stellt euch vor...", "Ich weiß noch..."), ("Wenn du...") or a new subject after a pause/reaction.
3. **A sequence MUST end AFTER the payoff.** This is the most critical rule:
   - If a story leads to a funny moment → include the laughter/reaction COMPLETELY
   - If an argument builds to a conclusion → include the conclusion sentence
   - If there's an audience reaction → your end time MUST be AFTER the END timestamp of the last reaction
   - If someone says something witty and the audience laughs → that laugh IS the ending, don't cut before it
   - If we have a story with multiple punchlines → include them all until the story truly ends
4. **Never end a clip during setup.** If the story is "X happened, and then Y said Z" — you MUST include what Z said and how the audience reacted.

CRITICAL RULES:
- **The payoff is NON-NEGOTIABLE.** A 90-second clip that includes the punchline beats a 45-second clip that cuts before it. If the punchline is at second 88 of a story that starts at second 0, the clip is 88+ seconds. So be it.
- **Find ALL highlights.** A 5-minute video may have 1-2. A 60-minute sermon has 10-25.
- **No overlaps.** Clips must not overlap in time.
- **Duration**: 30-180 seconds. Shorter is fine if the payoff is strong and the story is complete.
- **Complete sentences only**: Never start or end mid-sentence.
- **Skip boring passages.** Flat exposition, repetitive explanations, or administrative remarks are not clips.
- **German idioms/context**: Understand that speakers may use idioms ("mit allem Drum und Dran", "Tacheles reden", "auf Herz und Nieren prüfen"). These are part of the rhetorical texture — include them in context.

IMPACT SCORING (1-10):
- 10: Audience erupts, unforgettable moment, instant rewatch
- 8-9: Strong emotional reaction, great story with clear payoff, powerful rhetoric
- 6-7: Solid content, interesting insight, decent audience engagement
- 4-5: Decent but not remarkable, might hold attention
- 1-3: Do not include — not strong enough for short-form

Return ONLY a JSON array (no markdown fences). Each element:
{
    "start": <start seconds — EXACT timestamp where this topic's setup begins>,
    "end": <end seconds — AFTER the payoff and any audience reaction>,
    "content": "<1-2 sentence summary: what happens AND what the payoff/punchline is>",
    "impact": <1-10 integer — honest audience impact score>,
    "why": "<brief explanation of what makes this clip compelling>"
}

Return [] if no good clips exist."""


def GetHighlight(Transcription):
    from openai import OpenAI

    try:
        _ensure_llm_server()
        client = OpenAI(api_key=api_key, base_url=api_base)

        print(f"Calling LLM ({model_name}) for highlight selection...")
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": Transcription + "\n\n/no_think"},
            ],
        )

        text = (completion.choices[0].message.content or "").strip()
        # Strip <think>...</think> blocks from reasoning models
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0].strip()

        parsed = _json.loads(text)
        if isinstance(parsed, list):
            parsed = parsed[0]

        response_start = parsed.get("start")
        response_end = parsed.get("end")
        response_content = parsed.get("content", "")
        
        if response_start is None or response_end is None:
            print("ERROR: LLM response missing start/end fields")
            print(f"  Raw response: {text}")
            return None, None
        
        try:
            Start = float(response_start)
            End = float(response_end)
        except (ValueError, TypeError) as e:
            print(f"ERROR: Could not parse start/end times from response")
            print(f"  start: {response_start}")
            print(f"  end: {response_end}")
            print(f"  Error: {e}")
            return None, None
        
        # Validate times
        if Start < 0 or End < 0:
            print(f"ERROR: Negative time values - Start: {Start}s, End: {End}s")
            return None, None
        
        if End <= Start:
            print(f"ERROR: Invalid time range - Start: {Start}s, End: {End}s (end must be > start)")
            return None, None
        
        # Log the selected segment
        print(f"\n{'='*60}")
        print(f"SELECTED SEGMENT DETAILS:")
        print(f"Time: {Start}s - {End}s ({End-Start}s duration)")
        print(f"Content: {response_content}")
        print(f"{'='*60}\n")
        
        if Start==End:
            Ask = input("Error - Get Highlights again (y/n) -> ").lower()
            if Ask == "y":
                Start, End = GetHighlight(Transcription)
            return Start, End
        return Start,End
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR IN GetHighlight FUNCTION:")
        print(f"{'='*60}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print(f"\nTranscription length: {len(Transcription)} characters")
        print(f"First 200 chars: {Transcription[:200]}...")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return None, None


def _call_llm(system_prompt, user_content, temperature=0.7, _retries=3):
    """Shared helper: call LLM, strip think blocks / fences, return raw text.

    Automatically checks the configured local LLM server.
    Retries on transient "No models loaded" or connection errors.
    """
    from openai import OpenAI
    import re as _re

    _ensure_llm_server()

    last_err = None
    for attempt in range(1, _retries + 1):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content + "\n\n/no_think"},
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0].strip()
            return text
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            is_model_error = "no models loaded" in err_msg
            is_conn_error = "connection" in err_msg or "refused" in err_msg

            if is_model_error or is_conn_error:
                global _llm_server_ready
                _llm_server_ready = False  # force re-check
                # For vLLM: no auto-start possible, fail fast
                if _llm_backend_mode() != "lmstudio":
                    print(f"  [LLM] Attempt {attempt}/{_retries} failed: {e}")
                    if not _llm_server_reachable():
                        print(f"  [LLM] vLLM server not reachable — skipping remaining retries.")
                        break
                    # Server reachable but request failed — short retry
                    time.sleep(2)
                else:
                    wait = 10 * attempt
                    print(f"  [LLM] Attempt {attempt}/{_retries} failed: {e}")
                    print(f"  [LLM] Re-checking local LLM server in {wait}s...")
                    time.sleep(wait)
                    _ensure_llm_server()
            else:
                raise  # non-recoverable error, don't retry

    raise last_err  # all retries exhausted


def _chunk_transcription(trans_text, max_chars=12000, overlap_chars=1500):
    """Split a long transcription into overlapping chunks that fit within
    LLM context limits.  Each chunk is a string of transcription lines."""
    lines = trans_text.strip().splitlines()
    chunks = []
    current_lines = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            # Keep last ~overlap_chars worth of lines for context continuity
            overlap_lines = []
            overlap_len = 0
            for ol in reversed(current_lines):
                if overlap_len + len(ol) > overlap_chars:
                    break
                overlap_lines.insert(0, ol)
                overlap_len += len(ol)
            current_lines = overlap_lines
            current_len = overlap_len
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


TRANSCRIPT_CLEANUP_PROMPT = """\
You are an expert German subtitle editor with deep knowledge of spoken German, theology, and rhetoric.

You receive numbered transcript segments from an ASR-transcribed sermon or talk. Your job is to make the text readable as subtitles while preserving the speaker's voice and meaning.

PRIMARY OBJECTIVE (HIGHEST PRIORITY):
- Keep the ACTUAL spoken words faithful. When unsure, keep the original ASR words.
- Prefer minimal edits over stylistic rewrites.

MUST FIX:
- **Filler words**: Remove "äh", "ähm", "eh", "hm", "also" (when used as filler, not as "therefore"), "ja" (when used as filler, not as "yes"), "ne", "gell", "halt", and other verbal fillers that add no meaning.
- **ASR errors**: Fix obvious misrecognitions. Common patterns:
  - Misheard German idioms: "Drum und Ranner" → "Drum und Dran", "im Endeffekte" → "im Endeffekt"
  - Theological vocabulary: proper names of biblical figures, places, books of the Bible
  - Compound words split or mangled by ASR
  - Numbers and dates misheard
- **Sentence boundaries**: Add proper punctuation. Split run-on sentences. Capitalize sentence starts.
- **German idioms**: Recognize and correct common idioms the ASR may have mangled:
  - "mit allem Drum und Dran", "auf Herz und Nieren", "Tacheles reden", "ins Schwarze treffen"
  - Theological idioms: "Buße tun", "den Glauben bekennen", "im Geist wandeln"
- **Proper nouns**: Correct biblical names (Elisa/Elischa, Joasch, Aram, Paulus, Petrus, etc.)

MUST PRESERVE:
- The speaker's natural voice and style — do NOT make it sound like written text
- Colloquial expressions that are intentional (e.g., "Verdammt noch einmal" when the speaker actually says it)
- The emotional tone: emphatic repetition, rhetorical questions, exclamations
- Segment indexes, timestamps, and order — do NOT merge, split, reorder, or drop segments
- Bracketed markers like audience reactions — leave unchanged

DO NOT:
- Summarize or condense — every segment keeps its full meaning
- Invent content that wasn't said
- Replace core lexical content words with plausible alternatives just because they fit context
    (example of FORBIDDEN behavior: "Messias" -> "Christkind" unless the segment clearly contains that exact word)
- Change the register (don't make casual speech formal)
- Over-correct spoken German into written German — "Da hab ich gesagt" stays, don't change to "Da habe ich gesagt"

IF UNCERTAIN:
- Return the segment text unchanged.
- Only perform changes when the correction is highly obvious from the segment itself.

Return ONLY JSON:
[
  {"index": 0, "text": "Corrected text"},
  {"index": 1, "text": "Corrected text"}
]
"""


# Conservative safety rails for cleanup application.
# Goal: reject semantic drift / hallucinated rewrites while still allowing
# punctuation, filler removal, and minor obvious ASR typo fixes.
_CLEANUP_MIN_SIMILARITY = 0.72
_CLEANUP_MAX_NEW_TOKEN_RATIO = 0.28
_CLEANUP_STRIP_CHARS = " \t\n\r.,;:!?\"'`´()[]{}<>/\\|+-=_*~"


def _cleanup_tokens(text: str) -> list[str]:
    tokens = []
    for part in str(text).split():
        token = part.strip(_CLEANUP_STRIP_CHARS).lower()
        if token:
            tokens.append(token)
    return tokens


def _cleanup_change_is_conservative(original_text: str, candidate_text: str) -> bool:
    """Return True if candidate is a conservative edit of original.

    We allow punctuation and small lexical fixes, but reject broad rewrites
    that introduce too many new content tokens.
    """
    original = str(original_text or "").strip()
    candidate = str(candidate_text or "").strip()

    if not candidate:
        return False
    if candidate == original:
        return True

    similarity = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()

    old_tokens = _cleanup_tokens(original)
    new_tokens = _cleanup_tokens(candidate)
    if not old_tokens or not new_tokens:
        return similarity >= _CLEANUP_MIN_SIMILARITY

    old_set = set(old_tokens)
    new_only = [tok for tok in new_tokens if tok not in old_set]
    new_ratio = len(new_only) / max(len(new_tokens), 1)

    # Always reject highly dissimilar rewrites.
    if similarity < _CLEANUP_MIN_SIMILARITY:
        return False

    # Reject candidates that introduce many new words unless the segment is tiny.
    if len(old_tokens) >= 4 and new_ratio > _CLEANUP_MAX_NEW_TOKEN_RATIO:
        return False

    return True


def _chunk_segments_for_cleanup(transcriptions, max_chars=9000, max_segments=60):
    """Split speech segments into LLM-sized chunks while preserving indexes."""
    chunks = []
    current = []
    current_len = 0

    for idx, (text, start, end) in enumerate(transcriptions):
        raw_text = str(text).strip()
        if not raw_text or raw_text.startswith("["):
            continue

        line = f"[{idx}] {float(start):.2f} - {float(end):.2f} | {raw_text}"
        line_len = len(line) + 1

        if current and (len(current) >= max_segments or current_len + line_len > max_chars):
            chunks.append(current)
            current = []
            current_len = 0

        current.append(
            {
                "index": idx,
                "start": float(start),
                "end": float(end),
                "text": raw_text,
            }
        )
        current_len += line_len

    if current:
        chunks.append(current)

    return chunks


def _parse_cleanup_response(text):
    parsed = _json.loads(text)
    if isinstance(parsed, dict):
        for key in ("segments", "items", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def CleanTranscriptSegments(transcriptions, language="de"):
    """Conservatively clean segment text for subtitle rendering.

    Keeps segment timing untouched and returns the same ``[[text, start, end], ...]``
    shape as the input.  This is intentionally segment-level cleanup so it can be
    used before subtitle generation without remapping word timestamps.
    """
    if not transcriptions:
        return []

    cleaned = [[str(text), float(start), float(end)] for text, start, end in transcriptions]
    chunks = _chunk_segments_for_cleanup(cleaned)
    if not chunks:
        return cleaned

    print(f"Cleaning transcript text in {len(chunks)} chunk(s)...")
    updated = 0
    rejected = 0

    for chunk_idx, chunk in enumerate(chunks, start=1):
        lines = []
        for item in chunk:
            lines.append(
                f"[{item['index']}] {item['start']:.2f} - {item['end']:.2f} | {item['text']}"
            )

        user_content = (
            f"Language hint: {language}\n"
            "Correct these subtitle transcript segments conservatively.\n"
            "Return only JSON.\n\n"
            + "\n".join(lines)
        )

        try:
            response = _call_llm(TRANSCRIPT_CLEANUP_PROMPT, user_content, temperature=0.2)
            items = _parse_cleanup_response(response)
        except Exception as exc:
            print(f"  [Cleanup] Chunk {chunk_idx}/{len(chunks)} failed ({exc}); keeping original text.")
            continue

        valid_indexes = {item["index"] for item in chunk}
        applied = 0
        for item in items:
            try:
                idx = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if idx not in valid_indexes:
                continue

            new_text = str(item.get("text", "")).strip()
            if not new_text:
                continue

            old_text = cleaned[idx][0]
            if old_text != new_text:
                if not _cleanup_change_is_conservative(old_text, new_text):
                    rejected += 1
                    continue
                cleaned[idx][0] = new_text
                updated += 1
                applied += 1

        print(f"  [Cleanup] Chunk {chunk_idx}/{len(chunks)} updated {applied} segment(s).")

    print(f"[Cleanup] Updated {updated} segment(s) total.")
    if rejected:
        print(f"[Cleanup] Rejected {rejected} risky rewrite(s); kept original ASR wording.")
    return cleaned


def GetAllHighlights(Transcription):
    """Analyze the full transcription and return ALL highlight-worthy segments.

    Returns a list of dicts sorted by impact score (highest first):
    [{"start": float, "end": float, "content": str, "impact": int, "why": str}, ...]
    """
    try:
        chunks = _chunk_transcription(Transcription)
        print(f"Analyzing transcription in {len(chunks)} chunk(s) for ALL highlights...")

        all_highlights = []

        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")

            text = _call_llm(MULTI_HIGHLIGHT_PROMPT, chunk, temperature=0.5)
            try:
                parsed = _json.loads(text)
            except _json.JSONDecodeError:
                print(f"  Warning: Could not parse LLM response for chunk {i+1}")
                continue

            if not isinstance(parsed, list):
                parsed = [parsed]

            for item in parsed:
                try:
                    s = float(item["start"])
                    e = float(item["end"])
                except (KeyError, ValueError, TypeError):
                    continue
                if e > s and (e - s) >= 20:
                    # Accept both new "impact" field and legacy "quality" field
                    impact = 5  # default
                    if "impact" in item:
                        try:
                            impact = max(1, min(10, int(item["impact"])))
                        except (ValueError, TypeError):
                            pass
                    elif item.get("quality") == "high":
                        impact = 8
                    elif item.get("quality") == "medium":
                        impact = 5

                    all_highlights.append({
                        "start": s,
                        "end": e,
                        "content": item.get("content", ""),
                        "impact": impact,
                        "why": item.get("why", ""),
                    })

        # Sort by start time first for de-overlap pass
        all_highlights.sort(key=lambda h: h["start"])

        # Remove overlaps: keep the higher-impact clip
        cleaned = []
        for h in all_highlights:
            if cleaned and h["start"] < cleaned[-1]["end"]:
                prev = cleaned[-1]
                if h["impact"] > prev["impact"]:
                    cleaned[-1] = h
                # else keep previous (higher or equal impact)
            else:
                cleaned.append(h)

        # Final sort by impact score (best first)
        cleaned.sort(key=lambda h: h["impact"], reverse=True)

        print(f"\n{'='*60}")
        print(f"FOUND {len(cleaned)} HIGHLIGHT(S) (ranked by impact):")
        for i, h in enumerate(cleaned):
            dur = h['end'] - h['start']
            print(f"  {i+1}. [{h['start']:.1f}s - {h['end']:.1f}s] ({dur:.0f}s) "
                  f"[impact={h['impact']}] {h['content'][:70]}")
            if h.get('why'):
                print(f"     → {h['why'][:80]}")
        print(f"{'='*60}\n")

        return cleaned

    except Exception as e:
        print(f"GetAllHighlights failed: {e}")
        import traceback
        traceback.print_exc()
        return []


JUMPCUT_PROMPT = """\

You are a conservative video editor making minimal jump cuts in short-form content.
Below is the transcription of a selected video clip (often in German).

Your job: identify 0-2 SHORT sections (each 1-4 seconds MAX) of pure dead air
that can be removed to tighten the pacing.

You may ONLY cut:
- Silent gaps with NO speech and NO audience reaction (true dead air)
- Isolated filler words ("äh", "ähm") where removing them doesn't break flow

You MUST NEVER cut:
- [AUDIENCE REACTION] markers or ANY time near them (before or after)
- ANY sentence or part of a sentence — even if it seems boring, it may be setup
- Story setup, context, or character introductions — these are ESSENTIAL
- The punchline, climax, or any emotional moment
- Transitions like "und dann", "dann fragt er" — these build narrative tension
- ANY speech content at all — if someone is talking, don't cut it
- If in doubt, return [] — it is much better to leave a boring second in than to
  accidentally cut a story beat

Return ONLY a JSON array (or empty []):
[
  {"cut_start": <seconds>, "cut_end": <seconds>, "reason": "<brief reason>"}
]"""


def GetJumpCuts(transcription_text, start, end):
    """Ask the LLM to identify boring filler sections within the selected
    highlight range that should be cut.  Returns a sorted list of
    (cut_start, cut_end) tuples in absolute seconds, or [] if nothing
    to cut."""
    from openai import OpenAI

    try:
        _ensure_llm_server()
        client = OpenAI(api_key=api_key, base_url=api_base)

        # Extract only the transcription lines within the selected range
        segment_lines = []
        for line in transcription_text.strip().splitlines():
            parts = line.split(" - ", 1)
            if len(parts) < 2:
                continue
            try:
                line_start = float(parts[0].strip())
            except ValueError:
                continue
            if line_start >= start and line_start < end:
                segment_lines.append(line)

        segment_text = "\n".join(segment_lines)
        if not segment_text.strip():
            return []

        user_msg = (
            f"Here is the clip transcription ({start}s - {end}s):\n\n"
            f"{segment_text}\n\n/no_think"
        )

        print(f"Calling LLM ({model_name}) for jump-cut analysis...")
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            messages=[
                {"role": "system", "content": JUMPCUT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

        text = (completion.choices[0].message.content or "").strip()
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0].strip()

        cuts_raw = _json.loads(text)
        if not isinstance(cuts_raw, list):
            print("Jump-cut LLM returned non-list; skipping jump cuts.")
            return []

        cuts = []
        for item in cuts_raw:
            cs = float(item["cut_start"])
            ce = float(item["cut_end"])
            reason = item.get("reason", "")
            if ce > cs and cs >= start and ce <= end:
                cuts.append((cs, ce))
                print(f"  Jump cut: {cs:.1f}s - {ce:.1f}s  ({reason})")

        cuts.sort()
        return cuts

    except Exception as e:
        print(f"Jump-cut analysis failed ({e}); proceeding without jump cuts.")
        return []


# ── Title-card hook generation ───────────────────────────────────

TITLE_HOOK_PROMPT = """\
You write short, bold title text for social media sermon/talk clips.  The text appears over the speaker's face as a scroll-stopping thumbnail overlay.

You receive:
- The actual TRANSCRIPT of the clip (the literal spoken words).
- A content summary.
- The video title.

Your job: Return TWO lines, nothing else.
Line 1: HOOK — 2-5 words, max 30 characters.  This is the large bold text.
Line 2: KEYWORD — the single most powerful word from the hook that should be color-accented.

The hook MUST be extremely short and punchy — think YouTube thumbnail / TikTok cover text.
It must include at least one word DIRECTLY from the transcript.

Rules:
- 2-5 words MAXIMUM. Shorter is better.  Think "BROKEN HEART", "WHY DO WE PRAY?", "FASTING", "GOD'S DNA".
- ALL CAPS is fine and encouraged for punch.
- Communicate the core message or provoke curiosity in the fewest words possible.
- German if the transcript is German, English if English.
- NO hashtags, emojis, or clickbait filler ("UNGLAUBLICH", "DAS MUSST DU SEHEN").
- The keyword on line 2 must be ONE word from the hook that carries the most weight.

GOOD examples:
DREI MAL SCHLAGEN
SCHLAGEN

GOTTES PLAN
GOTTES

GNADE REICHT
GNADE

NICHT AUFHÖREN
AUFHÖREN

Return EXACTLY two lines. No quotes, no explanation, no JSON."""


def GenerateTitleHook(clip_content, clip_transcript="", video_title="", language="de"):
    """Ask the LLM to generate a short thumbnail hook + accent keyword.

    Returns ``(hook_text, accent_keyword)`` tuple.
    *clip_transcript* is the actual spoken text of the clip segment.
    Falls back to a truncated *clip_content* if the LLM call fails.
    """
    parts = [f"Video title: {video_title}"]
    if clip_transcript:
        # Limit transcript to ~800 chars to avoid token waste
        t = clip_transcript[:800]
        if len(clip_transcript) > 800:
            t = t.rsplit(" ", 1)[0] + " …"
        parts.append(f"Transcript:\n{t}")
    parts.append(f"Content summary: {clip_content}")
    parts.append(f"Language: {language}")
    user_msg = "\n\n".join(parts)

    try:
        raw = _call_llm(TITLE_HOOK_PROMPT, user_msg, temperature=0.7)
        lines = [l.strip().strip('"').strip("'").strip('\u201c').strip('\u201d')
                 for l in raw.strip().splitlines() if l.strip()]
        hook = lines[0] if lines else ""
        keyword = lines[1] if len(lines) > 1 else ""
        if len(hook) > 40:
            hook = hook[:37] + "\u2026"
        if hook:
            # Validate keyword is actually in the hook
            if keyword.upper() not in hook.upper():
                # Pick the longest word as accent
                keyword = max(hook.split(), key=len) if hook.split() else ""
            return hook, keyword
    except Exception as exc:
        print(f"[TitleHook] LLM call failed ({exc}); using fallback.")

    # Fallback: first 30 chars of clip_content
    fallback = clip_content[:30].rsplit(" ", 1)[0] if len(clip_content) > 30 else clip_content
    kw = max(fallback.split(), key=len) if fallback.split() else ""
    return fallback, kw


if __name__ == "__main__":
    print(GetHighlight(User))
