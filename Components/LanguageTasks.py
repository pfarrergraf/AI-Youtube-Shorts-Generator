from dotenv import load_dotenv
import json as _json
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
You are a viral-content editor. The input is a timestamped transcription of a talk, sermon, or comedy set (may be German or English).

The transcription has TWO entry types:
- **Speech segments**: timestamped text from the speaker.
- **Audience reaction markers**: `[AUDIENCE REACTION — loud, 2.3s]` with timestamps.

Your job: divide the transcription into ALL self-contained episodes that would each make a good short-form video clip.

HOW TO IDENTIFY SEQUENCES:
1. **Read the ENTIRE transcription first.** Map out where each distinct topic, story, or joke begins and ends.
2. **A new sequence starts** where the speaker introduces a new premise or topic. Signals: topic shift, new setup phrase ("Also...", "Und dann...", "Stellt euch vor..."), or simply a different subject after a pause/reaction.
3. **A sequence ends** when:
   - The final audience reaction for this topic has COMPLETELY finished (use the END timestamp of the last reaction)
   - The speaker is about to start a DIFFERENT topic
   - There is a clear pause/transition
4. **Each clip MUST include the COMPLETE setup.** The start time must be where the speaker FIRST introduces this topic — not partway through. A viewer who hasn't seen the rest of the video must understand the clip on its own.
5. **Each clip MUST include ALL punchlines and reactions.** If the speaker escalates after a reaction (same topic), include the escalation. Only stop when the topic truly changes.

CRITICAL RULES:
- **Find ALL highlights.** A 5-minute video has 1-3. A 60-minute video has 15-30.
- **No overlaps.** Clips must not overlap in time.
- **Complete** means: setup + build-up + punchline + audience reaction. Missing the setup = unusable clip.
- **Stop before new content**: Do NOT include the start of the next topic.
- **Duration**: 30-180s each. Too long is far better than too short.
- **Complete sentences only**: never start or end mid-sentence.
- **Quality bar**: Only genuinely engaging moments. Skip flat/boring passages.

Return ONLY a JSON array (no markdown fences). Each element:
{
    "start": <start seconds — EXACT timestamp where this topic's setup begins>,
    "end": <end seconds — after the last reaction for this topic, before the next topic>,
    "content": "<brief summary of this clip>",
    "quality": "<high|medium>"
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
You are a careful transcript cleanup editor.

You receive numbered transcript segments from a spoken recording. Your job is to
correct only the transcript text so it reads well as subtitles.

Rules:
- Preserve the same segment indexes, timestamps, and order.
- Do NOT merge, split, reorder, drop, or invent segments.
- Do NOT summarize or rewrite for style.
- Make conservative fixes only: punctuation, casing, obvious ASR mistakes,
  broken sentence boundaries, and strongly inferable proper nouns / vocabulary.
- If a segment is uncertain, keep it close to the original.
- Leave bracketed markers such as audience reactions unchanged.
- Return ONLY JSON.

Return format:
[
  {"index": 0, "text": "Corrected text"},
  {"index": 1, "text": "Corrected text"}
]
"""


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

            if cleaned[idx][0] != new_text:
                cleaned[idx][0] = new_text
                updated += 1
                applied += 1

        print(f"  [Cleanup] Chunk {chunk_idx}/{len(chunks)} updated {applied} segment(s).")

    print(f"[Cleanup] Updated {updated} segment(s) total.")
    return cleaned


def GetAllHighlights(Transcription):
    """Analyze the full transcription and return ALL highlight-worthy segments.

    Returns a list of dicts: [{"start": float, "end": float, "content": str, "quality": str}, ...]
    Segments are sorted by start time and de-overlapped.
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
                    all_highlights.append({
                        "start": s,
                        "end": e,
                        "content": item.get("content", ""),
                        "quality": item.get("quality", "medium"),
                    })

        # Sort by start time
        all_highlights.sort(key=lambda h: h["start"])

        # Remove overlaps: if two clips overlap, keep the higher-quality or longer one
        cleaned = []
        for h in all_highlights:
            if cleaned and h["start"] < cleaned[-1]["end"]:
                # Overlap — keep the one with higher quality, or longer duration
                prev = cleaned[-1]
                prev_dur = prev["end"] - prev["start"]
                cur_dur = h["end"] - h["start"]
                quality_rank = {"high": 2, "medium": 1}
                prev_q = quality_rank.get(prev.get("quality"), 1)
                cur_q = quality_rank.get(h.get("quality"), 1)
                if cur_q > prev_q or (cur_q == prev_q and cur_dur > prev_dur):
                    cleaned[-1] = h
                # else keep previous
            else:
                cleaned.append(h)

        print(f"\n{'='*60}")
        print(f"FOUND {len(cleaned)} HIGHLIGHT(S):")
        for i, h in enumerate(cleaned):
            dur = h['end'] - h['start']
            print(f"  {i+1}. [{h['start']:.1f}s - {h['end']:.1f}s] ({dur:.0f}s) [{h['quality']}] {h['content'][:80]}")
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


if __name__ == "__main__":
    print(GetHighlight(User))
