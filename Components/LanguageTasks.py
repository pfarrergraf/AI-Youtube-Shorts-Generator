from dotenv import load_dotenv
import hashlib as _hashlib
import json as _json
from difflib import SequenceMatcher
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
import urllib.error

load_dotenv()

try:
    from utils.highlight_schema import normalise_highlight_candidate
except Exception:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from utils.highlight_schema import normalise_highlight_candidate

def _first_env(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _registry_model_for_role(role):
    """Resolve the serving name from the shared role registry when available."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_registry.json"))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            registry = _json.load(handle)
        model_key = registry.get("roles", {}).get(role)
        model = registry.get("models", {}).get(model_key, {})
        return model.get("served_model_name") or model.get("model")
    except (OSError, ValueError, TypeError):
        return None


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
model_role = _first_env("LLM_ROLE", default="highlight_analysis_de")
model_name = _first_env(
    "VLLM_MODEL",
    "LOCAL_LLM_MODEL",
    "LLM_MODEL",
    default=_registry_model_for_role(model_role) or "qwen2.5-72b-instruct",
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
2. **A sequence starts** where the speaker introduces a new topic, premise, or story. Look for topic shifts: "Also...", "Und dann...", "Ich war mal...", "Stellt euch vor...", "Wenn du...", "...wir müssen uns fragen...", "die Frage ist..." or "Wenn wir nicht lernen..." or simply a new subject after a pause/reaction.
3. **A sequence is pre-introduced or announced by the speaker explicitly: **"Hör mir gut zu...:" or "Lasst mich mal ein Beispiel geben..." or "über eine Sache im klaren...:" or "Ich weiß noch, wie ich mal...". If the strong line depends on a setup sentence or question right before it, include that setup. Do NOT start at the apparent "best sentence" if that sentence lands better with 1-3 earlier sentences of framing.
4. **A rhetorical device like in Lausbergs docs/Stilfiguren_Lausberg/stilfiguren.json: 1. `antonomasie`
2. `evidentia`
3. `gedankenfiguren`
4. `geminatio`
5. `gradatio`
6. `homoeoteleuton`
7. `lausberg-ironia-rev1`
8. `isocolon`
9. `litotes`
10. `onomasiologische_distinctio`
11. `paromoeosis`
12. `periphrase`
13. `permissio`
14. `reduplicatio`
15. `regressio`
16. `anaphora`
17. `annominatio`
18. `antitheton`
19. `asyndeton`
20. `brachylogia`
21. `commoratio`
22. `communicatio`
23. `commutatio`
24. `derivatio`
25. `detractio`
26. `disiunctio`
27. `epiphora`
28. `metapher_lausberg`
29. `metonymie`
30. `reflexio`
31. `reticentia`
32. `tropus_synecdoche_001`
33. `traductio`
etc.** can signal the start of a sequence. If the speaker uses one of these devices to build up to a punchline or insight, include the whole device until the payoff.
5. **A sequence ends** when:
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
- **Duration target**: 15-90 seconds. This is a target, not a hard ceiling — a
  complete, finished thought always outranks hitting the target. Too long is
  better than too short.
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
A rhetorical device like in Lausbergs docs/Stilfiguren_Lausberg/stilfiguren.json: 1. `antonomasie`
2. `evidentia`
3. `gedankenfiguren`
4. `geminatio`
5. `gradatio`
6. `homoeoteleuton`
7. `lausberg-ironia-rev1`
8. `isocolon`
9. `litotes`
10. `onomasiologische_distinctio`
11. `paromoeosis`
12. `periphrase`
13. `permissio`
14. `reduplicatio`
15. `regressio`
16. `anaphora`
17. `annominatio`
18. `antitheton`
19. `asyndeton`
20. `brachylogia`
21. `commoratio`
22. `communicatio`
23. `commutatio`
24. `derivatio`
25. `detractio`
26. `disiunctio`
27. `epiphora`
28. `metapher_lausberg`
29. `metonymie`
30. `reflexio`
31. `reticentia`
32. `tropus_synecdoche_001`
33. `traductio`
4. **Self-contained meaning** — A first-time viewer who has NEVER seen the full video must fully understand the clip. No dangling references, no "as I said earlier".
5. **Psychological hooks** — Stories with conflict, unexpected turns, relatable situations, or statements that challenge assumptions.
6. **Theological/intellectual precision** — Interesting biblical insights, counter-intuitive interpretations, connections the audience hasn't heard before.
7. Humor — jokes, witty remarks, playful language, or funny situations. Humor is a strong engagement driver but must be complete with the punchline and reaction.

HOW TO IDENTIFY COMPLETE SEQUENCES:
1. **Read the ENTIRE transcription first.** Map out where each distinct topic, story, joke or argument begins and ends.
2. **A sequence starts** where the speaker introduces the premise that makes the payoff meaningful. Signals: topic shift, setup phrase ("Also...", "Und dann...", "Stellt euch vor...", "Ich weiß noch..."), ("Wenn du...") or a new subject after a pause/reaction.
    - Start EARLY enough that a first-time viewer understands who or what the speaker is talking about.
    - If the strong line depends on a setup sentence or question right before it, include that setup.
    - Do NOT start at the apparent "best sentence" if that sentence lands better with 1-3 earlier sentences of framing.
3. **A sequence starts** where the speaker introduces a new topic, premise, or story. Look for topic shifts: "Also...", "Und dann...", "Ich war mal...", "Stellt euch vor...", "Wenn du...", "...wir müssen uns fragen...", "die Frage ist..." or "Wenn wir nicht lernen..." or simply a new subject after a pause/reaction.
    4. **A sequence is pre-introduced or announced by the speaker explicitly: **"Hör mir gut zu...:" or "Lasst mich mal ein Beispiel geben..." or "über eine Sache im klaren...:" or "Ich weiß noch, wie ich mal...". If the strong line depends on a setup sentence or question right before it, include that setup. Do NOT start at the apparent "best sentence" if that sentence lands better with 1-3 earlier sentences of framing.
5. **A sequence MUST end AFTER the payoff.** This is the most critical rule:
   - If a story leads to a funny moment → include the laughter/reaction COMPLETELY
   - If an argument builds to a conclusion → include the conclusion sentence
   - If there's an audience reaction → your end time MUST be AFTER the END timestamp of the last reaction
   - If someone says something witty and the audience laughs → that laugh IS the ending, don't cut before it
   - If we have a story with multiple punchlines → include them all until the story truly ends
    - If the speaker adds a final clarifying or sharpening sentence immediately after the main punchline, include that too.
6. **Never end a clip during setup.** If the story is "X happened, and then Y said Z" — you MUST include what Z said and how the audience reacted.

OPENING AND ENDING QUALITY CHECK:
- Before returning a clip, ask: "Would a new viewer instantly understand why this starts here?"
- If the opening feels abrupt, move the start earlier.
- Ask: "Does the final sentence feel like a real ending, not a truncation?"
- If the ending feels cut off, include the next resolving sentence or reaction.
- Prefer a slightly longer clip with a meaningful opening and a complete ending over a shorter clip with a sharper but confusing cut.
- A forward-looking sentence is NOT an ending. Phrases such as "jetzt wird es
  interessant", "gleich zeige ich euch", "die Frage ist", "aber jetzt kommt",
  or any announcement of what follows are setup for the NEXT payoff. Extend
  through that payoff or reject the candidate.
- Reject excerpts that are merely sensible sermon material but lack a distinct
  turn, payoff, memorable formulation, resolved tension, or actionable insight.
  "The speaker says something true" is not enough to make a highlight.

CRITICAL RULES:
- **The payoff is NON-NEGOTIABLE.** A 90-second clip that includes the punchline beats a 45-second clip that cuts before it. If the punchline is at second 88 of a story that starts at second 0, the clip is 88+ seconds. So be it.
- **Find ALL highlights.** A 5-minute video may have 1-2. A 60-minute sermon has 10-25.
- **No overlaps.** Clips must not overlap in time.
- **Duration target**: 15-90 seconds. This is a target, not a hard ceiling.
  Shorter is fine if the payoff is strong and the story is already complete at
  that point. Longer is fine — even well past 90s — if that's what a finished
  thought actually needs. Never cut a clip short just to hit the target; a
  complete 110-second clip beats an incomplete 80-second one every time.
- **Start at a paragraph/topic boundary, never mid-explanation.** The first
  sentence of a clip must be where the speaker actually opens that thought —
  a new topic, a new premise, a new illustration — not a sentence pulled out
  of the middle of an argument that only makes sense because of something
  said earlier. If you can't summarise the clip's opening line without
  reaching further back in the transcript, your start is too late.
- **End only when the thought is fully resolved, never at the first available
  period.** A clip must end on a sentence that completes the point being
  made — the argument has landed, the story has paid off, the question has
  been answered. Ending on a grammatically-complete sentence is not enough if
  the very next sentence is clearly still part of the same thought (e.g. it
  continues a list, restates the point, or delivers the conclusion the
  previous sentence was building toward). When in doubt, extend to the next
  sentence that actually closes the topic.
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
    "title": "<short 4-8 word clip title>",
    "hook": "<very short scroll-stopping hook phrase>",
    "content": "<1-2 sentence summary: what happens AND what the payoff/punchline is>",
    "impact": <1-10 integer — honest audience impact score>,
    "confidence": <0.00-1.00 confidence that this stands alone as a clip>,
    "opening_complete": <true only if no prior sentence is needed>,
    "ending_complete": <true only if the thought/payoff has actually landed>,
    "cliffhanger": <true if the ending announces, asks for, or sets up what comes next>,
    "payoff_excerpt": "<short verbatim final line that supplies the payoff>",
    "why": "<brief explanation of what makes this clip compelling>",
    "speaker": "<speaker name if known, else null>",
    "transcript_excerpt": "<short verbatim excerpt from the transcript for this clip>",
    "suggested_caption": "<1 sentence social caption suggestion>",
    "suggested_thumbnail_concept": "<short thumbnail text or image concept>",
    "platform_fit": ["youtube_shorts", "instagram_reels", "tiktok"],
    "risk_notes": ["context risk, transcript uncertainty, or doctrinal nuance if relevant"]
}

Return [] if no good clips exist.

You may also receive a REFERENCE EXCERPT from a manuscript or service script.
Use it only to recognise names, theological terms, likely topic flow, and fragile wording.
The actual spoken transcript always wins if it differs from the reference."""

_MULTI_HIGHLIGHT_PROMPT_VERSION = _hashlib.md5(
    MULTI_HIGHLIGHT_PROMPT.encode(), usedforsecurity=False
).hexdigest()[:12]


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
You are an expert German subtitle editor specialising in christian, biblical, theological, and rhetorical speech.

You receive numbered subtitle segments from an ASR-transcribed sermon. Your ONLY job is to
fix obvious ASR recognition errors — nothing else. Treat every segment as a faithful
transcription of what the speaker actually said, including their rhetorical devices, repetitions, mistakes and self-corrections.

Sometimes you also receive a REFERENCE TEXT and/or protected sermon terms. The reference is
supporting context from a preparation manuscript or service script. Use it only to recognise
names, places, theological wording, likely numbers, and likely sermon flow. NEVER force the
reference wording onto the transcript if the speaker clearly digressed or said something else.

PRIMARY RULE — FAITHFULNESS ABOVE ALL:
- If in doubt, output the ORIGINAL text unchanged.
- Minimal edits only. Never rewrite. Never paraphrase. Never condense.
- Do NOT remove, merge, reorder, or silence any segment.

==== WHAT YOU MAY FIX (ASR errors only) ====

1. FILLER WORDS — remove ONLY these standalone spoken fillers that add zero meaning:
   „äh", „ähm", „eh", „hm"
   Do NOT remove „also" (= „therefore"), „ja" (= „yes" or rhetorical affirmation),
   „ne", „gell", „halt" UNLESS they are clearly mid-sentence noise with no semantic role.

2. OBVIOUS MISRECOGNITIONS — fix when the correction is unambiguous from the context:
   - Theological names & places: Elisa→Elischa, Joasch, Aram, Paulus, Golgatha, etc.
   - Biblical book names, psalm numbers, verse references
   - German compound words split or fused incorrectly by ASR
   - Numbers and dates that are clearly wrong
   - Known German idioms mangled: „mit allem Drum und Ranner"→„mit allem Drum und Dran"

3. PUNCTUATION — add or correct punctuation and capitalise sentence starts.
   Do NOT restructure sentences to add punctuation — work with the existing word order.

==== WHAT YOU MUST NEVER DO ====

- NEVER remove repeated sentences or phrases.
  Preachers deliberately repeat sentences for rhetorical emphasis (anaphora, epistrophe).
  If the same sentence appears twice (or three times), keep ALL occurrences exactly as they are.

- NEVER replace content words with alternatives, even if the alternative „makes more sense".
  Example of FORBIDDEN behaviour:
    Original: „Denn seinen Freunden gibt er es im Schlaf."
    Forbidden: changing „Freunden" to any other word, removing the sentence, or paraphrasing it.

- NEVER remove words because they seem redundant or repetitive — the speaker said them.

- NEVER summarise, shorten, or condense segments.

- NEVER change register: keep spoken German as spoken German.
  „Da hab ich gesagt" stays — do not change to „Da habe ich gesagt".

- NEVER invent content that might not have been said.

- NEVER alter bracketed markers like [Applaus] or [Lachen] — leave them unchanged.

IF UNCERTAIN:
  Return the segment text UNCHANGED. Err strongly on the side of keeping the original.

==== EMPHASIS ====

For every segment also name the word the speaker is driving at — the one a
listener would remember, and that a caption should shout. Rules:

- 0 to 2 words per segment, fewer is better. Return [] when nothing stands out.
- Each word must appear VERBATIM in that segment's text (same spelling, no
  punctuation, no inflection changes). A word that is not in the text is
  discarded.
- Never pick function words (und, der, die, das, ist, nicht, aber, ich, du, …).
- Prefer the noun or verb carrying the claim over an adjective decorating it.

Return ONLY JSON (no markdown, no comments):
[
  {"index": 0, "text": "Corrected text or original text", "emphasis": ["Wort"]},
  {"index": 1, "text": "Corrected text or original text", "emphasis": []}
]
"""

# Short hash of the prompt — used to invalidate subtitle cleanup caches when
# the prompt is changed. Recomputed lazily so editing the prompt above is all
# that's needed; no manual version bump required.
_TRANSCRIPT_CLEANUP_PROMPT_VERSION = _hashlib.md5(
    TRANSCRIPT_CLEANUP_PROMPT.encode(), usedforsecurity=False
).hexdigest()[:8]


# Conservative safety rails for cleanup application.
# Goal: reject semantic drift / hallucinated rewrites while still allowing
# punctuation, filler removal, and minor obvious ASR typo fixes.
# Raised similarity floor (was 0.72) and lowered new-token budget (was 0.28)
# to prevent the LLM from replacing whole words/phrases.
_CLEANUP_MIN_SIMILARITY = 0.78
_CLEANUP_MAX_NEW_TOKEN_RATIO = 0.18
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


def _validated_emphasis(raw, segment_text: str) -> list[str]:
    """Keep only emphasis picks that really occur in the segment.

    The model occasionally returns an inflected or invented form; a word the
    caption renderer cannot find would silently produce no emphasis at all,
    so drop it here and let the heuristic take over instead.
    """
    if not isinstance(raw, list):
        return []
    present = {
        "".join(ch for ch in token.lower() if ch.isalnum())
        for token in str(segment_text).split()
    }
    picked: list[str] = []
    for candidate in raw[:2]:
        token = "".join(ch for ch in str(candidate).lower() if ch.isalnum())
        if token and token in present and token not in picked:
            picked.append(token)
    return picked


def CleanTranscriptSegments(
    transcriptions,
    language="de",
    rejected_log_path=None,
    reference_text=None,
    protected_terms=None,
    prompt_hints=None,
    speaker_profile_text=None,
    collect_emphasis=False,
):
    """Conservatively clean segment text for subtitle rendering.

    Keeps segment timing untouched and returns the same ``[[text, start, end], ...]``
    shape as the input.  This is intentionally segment-level cleanup so it can be
    used before subtitle generation without remapping word timestamps.

    If *rejected_log_path* is given, every change that was blocked by the
    safety rails is written there as JSON for later inspection.
    """
    if not transcriptions:
        return []

    cleaned = [[str(text), float(start), float(end)] for text, start, end in transcriptions]
    emphasis_map: dict[int, list[str]] = {}
    chunks = _chunk_segments_for_cleanup(cleaned)
    if not chunks:
        return (cleaned, emphasis_map) if collect_emphasis else cleaned

    print(f"Cleaning transcript text in {len(chunks)} chunk(s)...")
    updated = 0
    rejected = 0
    rejected_entries: list[dict] = []

    for chunk_idx, chunk in enumerate(chunks, start=1):
        lines = []
        for item in chunk:
            lines.append(
                f"[{item['index']}] {item['start']:.2f} - {item['end']:.2f} | {item['text']}"
            )

        user_content = (
            f"Language hint: {language}\n"
            "Correct these subtitle transcript segments conservatively.\n"
            "Return only JSON.\n"
        )
        if reference_text:
            user_content += (
                "\nReference sermon manuscript / service script excerpt:\n"
                "Use this only as NON-AUTHORITATIVE context for names, key terms, and the main flow.\n"
                "The spoken sermon may digress, paraphrase, or differ from the manuscript. Never force manuscript wording over actual speech.\n\n"
                + str(reference_text).strip()
                + "\n"
            )
        if speaker_profile_text:
            user_content += (
                "\nSpeaker profile context:\n"
                "Use this only to preserve speaker-specific terms, rhetorical habits, and likely highlight-worthy wording.\n"
                "Do not invent wording that is not present in the transcript.\n\n"
                + str(speaker_profile_text).strip()
                + "\n"
            )
        if protected_terms:
            user_content += "\nProtected sermon terms and spellings:\n- " + "\n- ".join(str(item).strip() for item in protected_terms if str(item).strip()) + "\n"
        if prompt_hints:
            user_content += "\nExact correction hints:\n- " + "\n- ".join(str(item).strip() for item in prompt_hints if str(item).strip()) + "\n"
        user_content += "\n"
        user_content += "\n".join(lines)

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
                    rejected_entries.append({
                        "index": idx,
                        "original": old_text,
                        "proposed": new_text,
                        "reason": "safety_rail",
                    })
                    continue
                cleaned[idx][0] = new_text
                updated += 1
                applied += 1

            if collect_emphasis:
                picked = _validated_emphasis(item.get("emphasis"), cleaned[idx][0])
                if picked:
                    emphasis_map[idx] = picked

        print(f"  [Cleanup] Chunk {chunk_idx}/{len(chunks)} updated {applied} segment(s).")

    print(f"[Cleanup] Updated {updated} segment(s) total.")
    if rejected:
        print(f"[Cleanup] Rejected {rejected} risky rewrite(s); kept original ASR wording.")
        if rejected_log_path:
            try:
                with open(rejected_log_path, "w", encoding="utf-8") as fh:
                    _json.dump(rejected_entries, fh, ensure_ascii=False, indent=2)
                print(f"[Cleanup] Rejected rewrites logged to: {rejected_log_path}")
            except Exception as exc:
                print(f"[Cleanup] Could not write rejected log: {exc}")
    if collect_emphasis:
        print(f"[Cleanup] Marked emphasis words in {len(emphasis_map)} segment(s).")
        return cleaned, emphasis_map
    return cleaned


def GetAllHighlights(Transcription, reference_text=None, speaker_profile_text=None):
    """Analyze the full transcription and return ALL highlight-worthy segments.

    Returns a list of dicts sorted by impact score (highest first):
    [{"start": float, "end": float, "content": str, "impact": int, "why": str, ...}, ...]
    """
    try:
        chunks = _chunk_transcription(Transcription)
        print(f"Analyzing transcription in {len(chunks)} chunk(s) for ALL highlights...")

        highlight_prompt = MULTI_HIGHLIGHT_PROMPT
        override_path = os.getenv("HIGHLIGHT_SYSTEM_PROMPT_FILE")
        if override_path:
            if os.path.isfile(override_path):
                with open(override_path, "r", encoding="utf-8") as f:
                    highlight_prompt = f.read()
                print(f"  Using HIGHLIGHT_SYSTEM_PROMPT_FILE override: {override_path}")
            else:
                print(f"  Warning: HIGHLIGHT_SYSTEM_PROMPT_FILE not found: {override_path}")

        all_highlights = []

        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")

            user_content = chunk
            if reference_text:
                user_content = (
                    "Reference sermon manuscript / service script excerpt:\n"
                    "Use this as NON-AUTHORITATIVE context for names, key ideas, and topic flow only.\n"
                    "The spoken sermon may differ from the manuscript, so the transcript always wins when there is tension.\n\n"
                    + str(reference_text).strip()
                    + "\n\n=== Transcript ===\n"
                    + chunk
                )
            if speaker_profile_text:
                user_content = (
                    "Speaker profile context:\n"
                    "Use this only to preserve speaker-specific theological vocabulary, recurring rhetoric, and highlight patterns.\n"
                    "The actual transcript always wins when the profile and transcript differ.\n\n"
                    + str(speaker_profile_text).strip()
                    + "\n\n=== Transcript ===\n"
                    + (user_content if user_content is not chunk else chunk)
                )
            text = _call_llm(highlight_prompt, user_content, temperature=0.5)
            try:
                parsed = _json.loads(text)
            except _json.JSONDecodeError:
                print(f"  Warning: Could not parse LLM response for chunk {i+1}")
                continue

            if not isinstance(parsed, list):
                parsed = [parsed]

            for item in parsed:
                if not isinstance(item, dict):
                    continue
                def _explicit_true(value):
                    return value is True or str(value).strip().lower() in {"true", "yes", "1"}

                opening_complete = _explicit_true(item.get("opening_complete"))
                ending_complete = _explicit_true(item.get("ending_complete"))
                cliffhanger = _explicit_true(item.get("cliffhanger"))
                payoff_excerpt = str(item.get("payoff_excerpt") or "").strip()
                if not opening_complete or not ending_complete or cliffhanger or not payoff_excerpt:
                    print(
                        "  Rejected incomplete highlight candidate: "
                        f"{item.get('title') or item.get('hook') or '<untitled>'}"
                    )
                    continue
                normalised = normalise_highlight_candidate(item)
                if not normalised:
                    continue
                if normalised["impact"] < 7 or normalised["confidence"] < 0.75:
                    print(
                        "  Rejected weak highlight candidate: "
                        f"{normalised.get('title') or '<untitled>'} "
                        f"(impact={normalised['impact']}, confidence={normalised['confidence']:.2f})"
                    )
                    continue
                # 15-90s is the LLM's target, not a hard ceiling (see prompt);
                # only reject genuinely broken candidates here. The upper
                # bound is set well above _expand_highlights_to_segment_boundaries's
                # own ~140s completion cap in shorts_bridge.py so a highlight
                # that's already complete just past the target isn't thrown
                # away outright, only to fall back to the sentence-unaware
                # heuristic path if too many candidates get rejected this way.
                if normalised["duration"] < 15 or normalised["duration"] > 150:
                    continue
                all_highlights.append(normalised)

        # Sort by start time first for de-overlap pass
        all_highlights.sort(key=lambda h: h["start"])

        # Remove overlaps: keep the higher-impact clip
        cleaned = []
        for h in all_highlights:
            if cleaned and h["start"] < cleaned[-1]["end"]:
                prev = cleaned[-1]
                if (
                    h["impact"] > prev["impact"]
                    or (
                        h["impact"] == prev["impact"]
                        and float(h.get("confidence", 0.0)) > float(prev.get("confidence", 0.0))
                    )
                ):
                    cleaned[-1] = h
                # else keep previous (higher or equal impact)
            else:
                cleaned.append(h)

        # Final sort by impact score (best first), then confidence, then duration.
        cleaned.sort(
            key=lambda h: (
                int(h.get("impact", 0) or 0),
                float(h.get("confidence", 0.0) or 0.0),
                float(h.get("duration", 0.0) or 0.0),
            ),
            reverse=True,
        )

        print(f"\n{'='*60}")
        print(f"FOUND {len(cleaned)} HIGHLIGHT(S) (ranked by impact):")
        for i, h in enumerate(cleaned):
            dur = h['end'] - h['start']
            print(f"  {i+1}. [{h['start']:.1f}s - {h['end']:.1f}s] ({dur:.0f}s) "
                  f"[impact={h['impact']}] {h.get('title') or h['content'][:70]}")
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
- LANGUAGE: If the transcript is German, the hook MUST be in German. No exceptions. Do not mix languages.
- NEVER describe the speaker or sermon. NEVER write "The speaker...", "Der Sprecher...", "Er erklärt...", "This clip...". Write the MESSAGE ITSELF as a headline or quote.
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


_HOOK_META_PREFIXES = (
    "the speaker", "der sprecher", "er erklärt", "sie erklärt",
    "this clip", "dieses video", "in this", "in diesem", "the pastor",
    "the preacher", "der prediger", "he draws", "she draws",
)
_HOOK_VISUAL_PROMPTS = (
    "gesichtsausdruck",
    "gesicht",
    "smartphone",
    "handy",
    "selfie",
    "kamera",
    "foto",
    "bild",
    "portrait",
    "screenshot",
    "thumbnail",
    "close-up",
    "hintergrund",
    "verwirrt",
    "verwirrter",
    "bildbeschreibung",
    "look",
    "expression",
)
_ENGLISH_META_WORDS = {
    "the", "speaker", "uses", "use", "this", "clip", "message", "explains",
    "shows", "pastor", "preacher", "metaphor", "perspective", "why", "what",
}
_GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "oder", "aber", "den", "dem", "des", "ein", "eine",
    "einer", "einem", "einen", "ist", "sind", "war", "waren", "wir", "ihr", "du", "sie",
    "er", "es", "ich", "man", "nicht", "noch", "schon", "auch", "nur", "für", "mit",
    "von", "aus", "bei", "auf", "im", "in", "am", "an", "zu", "zum", "zur", "dass",
    "wenn", "weil", "wie", "was", "wer", "wo", "heute", "jetzt", "immer", "ganz",
    "einfach", "mal", "also", "doch", "dann", "hier", "dort", "haben", "hat", "hatte",
    "wird", "werden", "kann", "können", "soll", "sollen", "muss", "müssen",
    "uns", "euch", "ihnen", "dein", "deine", "deinen", "deiner", "mein", "meine",
    "meinen", "unser", "unsere", "unserem", "unseren",
}


def _hook_words(text):
    return re.findall(r"[A-Za-zÄÖÜäöüß']+", str(text or ""))


def _transcript_word_set(text):
    return {word.lower() for word in _hook_words(text) if len(word) >= 4}


def _is_invalid_hook(hook, keyword, clip_transcript="", language="de"):
    hook = " ".join(str(hook or "").split()).strip()
    keyword = " ".join(str(keyword or "").split()).strip()
    if not hook:
        return True

    lowered = hook.lower()
    if any(lowered.startswith(prefix) for prefix in _HOOK_META_PREFIXES):
        return True
    if any(marker in lowered for marker in _HOOK_VISUAL_PROMPTS):
        return True

    words = _hook_words(hook)
    if not 2 <= len(words) <= 5:
        return True
    if keyword and keyword.upper() not in hook.upper():
        return True

    transcript_words = _transcript_word_set(clip_transcript)
    if transcript_words and not any(word.lower() in transcript_words for word in words):
        return True

    if str(language or "").lower().startswith("de"):
        english_hits = sum(1 for word in words if word.lower() in _ENGLISH_META_WORDS)
        if english_hits >= max(2, len(words) - 1):
            return True
    return False


def _fallback_title_hook(clip_content, clip_transcript="", language="de"):
    title_text = ""
    if isinstance(language, tuple):
        language, title_text = language
    transcript_words = _hook_words(clip_transcript)
    cleaned_title = str(title_text or "").strip()
    if cleaned_title:
        title_segments = [seg.strip() for seg in re.split(r"[|｜]+", cleaned_title) if seg.strip()]
        primary = title_segments[0] if title_segments else cleaned_title
        dash_segments = [seg.strip() for seg in re.split(r"\s[-–:]\s", primary) if seg.strip()]
        primary = max(dash_segments, key=len) if dash_segments else primary
        if primary:
            candidate = re.sub(r"\b(move|church|wiesbaden|predigt|predigtclip|kanzelclips)\b", "", primary, flags=re.IGNORECASE)
            candidate = " ".join(candidate.split()).upper()
            if 5 <= len(candidate) <= 28 and len(_hook_words(candidate)) <= 5:
                keyword = max(_hook_words(candidate), key=len)
                return candidate, keyword

    if str(language or "").lower().startswith("de") and transcript_words:
        filtered = [word for word in transcript_words if len(word) >= 4 and word.lower() not in _GERMAN_STOPWORDS]
        if filtered:
            ranked = sorted(filtered, key=lambda word: (-len(word), transcript_words.index(word)))
            anchor = ranked[0]
            anchor_idx = next((idx for idx, word in enumerate(transcript_words) if word.lower() == anchor.lower()), 0)
            window = []
            for word in transcript_words[max(0, anchor_idx - 2): anchor_idx + 2]:
                if len(word) >= 3 and word.lower() not in _GERMAN_STOPWORDS:
                    window.append(word.upper())
            if not window:
                window = [anchor.upper()]
            hook = " ".join(window[:4]).strip()
            if len(hook) > 30:
                hook = hook[:30].rsplit(" ", 1)[0].strip()
            if hook:
                keyword = max(hook.split(), key=len)
                return hook, keyword

    fallback = clip_content[:30].rsplit(" ", 1)[0] if len(clip_content) > 30 else clip_content
    fallback = " ".join(str(fallback or "").split()).upper()
    kw = max(fallback.split(), key=len) if fallback.split() else ""
    return fallback, kw


def _fallback_thumbnail_background_prompt(
    clip_content,
    *,
    video_title="",
    speaker_name="",
    brand_label="",
    background_style="clean_gradient",
):
    style = str(background_style or "clean_gradient").strip().lower()
    style_map = {
        "clean_gradient": "clean editorial sermon background, soft depth, calm gradient, subtle stage light",
        "strong_contrast": "strong cinematic sermon background, dark contrast, bright rim light, moody atmosphere",
        "emotion_focus": "emotion-focused sermon background, warm atmosphere, gentle bloom, concentrated light",
    }
    theme_source = " ".join(str(value or "").strip() for value in (video_title, clip_content))
    theme_tokens = [word for word in _hook_words(theme_source) if len(word) >= 4]
    theme = " ".join(dict.fromkeys(token.lower() for token in theme_tokens))[:120]
    parts = [
        "cinematic sermon background, portrait orientation",
        style_map.get(style, style_map["clean_gradient"]),
    ]
    if theme:
        parts.append(f"inspired by {theme}")
    if speaker_name:
        parts.append(f"for {speaker_name}")
    if brand_label:
        parts.append(f"branded for {brand_label}")
    parts.append("no people, no text, no watermark, no logo, soft depth")
    return ", ".join(parts)


_DEFAULT_BACKGROUND_NEGATIVE = (
    "text, letters, caption, watermark, logo, subtitles, screenshot, UI, "
    "people, face, hands, phone, smartphone, camera, blurry, distorted, clutter"
)

_THUMBNAIL_POSE_IDS = {
    "empathic_open",
    "battle_ready",
    "point_to_heaven",
    "compassion_near_tears",
    "righteous_anger",
    "urgent_warning",
    "joyful_breakthrough",
    "astonished_revelation",
    "prayerful_surrender",
    "direct_challenge",
    "protective_pastor",
    "hopeful_invitation",
}

_THUMBNAIL_STORY_ASSET_IDS = {
    "daniel_lions_den",
    "david_goliath",
    "heavenly_banquet",
    "storm_boat",
    "prodigal_road_home",
    "empty_tomb",
    "broken_chains",
    "lion_foreground",
    "mustard_seed_tree",
    "oil_lamp_darkness",
}

_THUMBNAIL_LAYER_ORDER = [
    "background_plate",
    "atmosphere_back",
    "story_midground",
    "title_back",
    "speaker",
    "foreground_eye_catcher",
    "title_front",
    "light_wrap",
    "final_grade",
]


THUMBNAIL_BRIEF_PROMPT = """\
You are the thumbnail director for portrait smartphone sermon highlight clips.

You receive:
- Video title
- Transcript excerpt
- Content summary
- Language
- Optional speaker name
- Optional brand label

Your job:
Return EXACTLY one JSON object with these keys:
- hook_text
- accent_keyword
- social_title
- story_concept
- curiosity_gap
- emotion_target
- pose_id
- story_asset_ids
- speaker_side
- background_style
- background_prompt
- background_negative_prompt
- layer_plan
- light_direction
- palette
- art_direction
- brand_label

Hard constraints:
- The thumbnail is portrait 9:16 for smartphone-first sermon shorts.
- The same design is used both as the uploaded thumbnail image and as the first frame / opening card.
- hook_text must be 2-5 words, max 28 characters.
- hook_text must be truthful, concrete, and curiosity-driving without cheap clickbait.
- social_title is the cross-platform upload title. It must begin with hook_text,
  then add useful context or the promise of the clip; max 90 characters.
- The image, hook_text and social_title must complement one another instead of
  repeating the same information three times.
- story_concept states the concrete visual story in one sentence.
- curiosity_gap states the unanswered question that makes the viewer want the clip's payoff.
- Never add a biblical person, miracle or danger unless it is grounded in the
  supplied title, transcript or content summary. Do not fabricate a Bible story.
- If the transcript is German, the hook must be German.
- Do not describe the speaker. Write the message itself.
- Keep typography clean and easy to read in under 1 second.
- Avoid red default typography. Prefer ivory text with one restrained accent.
- Leave the lower caption zone visually calm; prioritize the upper third.
- background_prompt must be a positive SDXL background prompt for the image only.
- background_negative_prompt must be a concise negative prompt that excludes text, watermark, logo, people, face, hands, and UI clutter.
- pose_id must be one of: empathic_open, battle_ready, point_to_heaven,
  compassion_near_tears, righteous_anger, urgent_warning, joyful_breakthrough,
  astonished_revelation, prayerful_surrender, direct_challenge,
  protective_pastor, hopeful_invitation.
- story_asset_ids is an array with 0-3 grounded ids chosen from:
  daniel_lions_den, david_goliath, heavenly_banquet, storm_boat,
  prodigal_road_home, empty_tomb, broken_chains, lion_foreground,
  mustard_seed_tree, oil_lamp_darkness. Use [] when none is grounded.
- layer_plan is an ordered subset of: background_plate, atmosphere_back,
  story_midground, title_back, speaker, foreground_eye_catcher, title_front,
  light_wrap, final_grade. Speaker must never be behind every title line.
- light_direction names one physically coherent key-light direction shared by
  background, speaker rim light and foreground assets.
- palette contains exactly three concise colour names, including one accent.
- art_direction must be one of: modern_cinematic, renaissance_chiaroscuro,
  baroque_drama, classical_tableau. Historic modes are occasional accents, not defaults.

speaker_side must be one of: left, right, center_low, auto
background_style must be one of: clean_gradient, strong_contrast, emotion_focus

Return JSON only. No markdown fences."""


# This profile is deliberately additive: production still uses
# ``GenerateThumbnailBrief`` until its text-only A/B evaluation is reviewed.
SERMON_THUMBNAIL_HEADLINE_V2_PROFILE = "sermon_thumbnail_headline_v2"
SERMON_THUMBNAIL_HEADLINE_V2_PROMPT = """\
You are the headline and thumbnail-hook editor for a high-quality Christian sermon and short-form video channel.

Your job is NOT to summarize the transcript. Identify the strongest emotional,
theological, rhetorical, narrative or surprising idea in the supplied clip and
turn it into highly memorable German thumbnail hooks and social-media titles.
The transcript is the source of truth. Never invent a claim, promise, event,
quotation, miracle, theological statement or emotional conclusion that the
transcript does not support.

PRIMARY GOAL: make a viewer stop scrolling because the text creates relevance,
surprise, tension, a strong formulation or a legitimate question. Do not
optimize for traditional sermon titles; optimize for Shorts, Reels and TikTok.

THUMBNAIL HOOK: German; ideally 2–5 words, at most 6 words and roughly 38
characters; understandable in under one second; one dominant thought; strong
rhythm; no unnecessary punctuation, speaker/church name, hashtags or generic
category descriptions. Prefer verbs, concrete nouns, contrast, active voice,
spoken German, emotional precision and controlled incompleteness.

SOCIAL TITLE: German; normally 5–12 words and preferably under 75 characters;
understandable without sermon context; may contain a question, contrast or a
mini-open-loop. Do not normally generate the speaker name.

Generate genuinely different candidates where supported: contradiction/reversal,
direct challenge, open loop, identity, pain/human tension, biblical drama,
quotable sentence, question, contrast and specific surprise. The three final
thumbnail hooks must not be minor rewrites: prefer one direct, one curiosity
and one rhetorical/emotional version.

Avoid abstract church language, explanatory subtitles, empty superlatives,
fake shock, vague motivational clichés and unsupported clickbait. In particular
avoid wording such as "Die Bedeutung von", "Die Rolle von", "Wie X uns hilft",
"Eine Geschichte über", "Gedanken zu", "inspirierend", "kraftvoll",
"lebensverändernd", "unglaublich", "mächtig" and "bahnbrechend" unless the
literal source requires it. Christian content may name Jesus, Gott, Sünde or
Gnade when grounded, but do not insert terminology just to sound religious.

TRUTHFULNESS GATE: reject a candidate when the transcript does not support it,
it promises a payoff the clip never gives, attributes unsupported words to
Jesus/Gott, turns a possibility into certainty, sensationalizes suffering,
changes the speaker's theology, or needs unavailable context.

Internally identify central payoff, strongest sentence/image, emotional tension
and possible reversal; generate at least 12 distinct candidates, eliminate weak
ones and score the rest for stop power, clarity, tension, emotion, specificity,
rhythm, visual fit and source fidelity. Do not expose private reasoning.

Return valid JSON only, exactly this structure:
{
  "core_message": "one concise sentence",
  "emotional_tension": "one concise sentence",
  "strongest_angle": "short description",
  "thumbnail_hooks": [
    {"text": "HOOK", "angle": "direct_challenge", "score": 94},
    {"text": "HOOK", "angle": "open_loop", "score": 91},
    {"text": "HOOK", "angle": "quotable", "score": 89}
  ],
  "social_titles": [
    {"text": "Social title", "score": 93},
    {"text": "Social title", "score": 90},
    {"text": "Social title", "score": 87}
  ]
}
Do not return markdown. Do not add explanations outside the JSON."""


def _parse_json_object_response(raw):
    """Parse a local-model JSON object even when it wrapped it in Markdown."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    payload = _json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object")
    return payload


def GenerateSermonThumbnailHeadlineV2(
    clip_content,
    *,
    clip_transcript="",
    highlight=None,
    video_title="",
    language="de",
    temperature=0.5,
    retries=3,
):
    """Text-only candidate generator for the headline A/B harness.

    It intentionally receives full highlight metadata (not just the first 800
    transcript characters), returns raw provenance for evaluation, and has no
    production caller yet.
    """
    highlight = highlight if isinstance(highlight, dict) else {}
    context = {
        "video_title": str(video_title or ""),
        "language": str(language or "de"),
        "content_summary": str(clip_content or highlight.get("content") or ""),
        "highlight_title": str(highlight.get("title") or ""),
        "existing_hook": str(highlight.get("hook") or ""),
        "payoff_excerpt": str(highlight.get("payoff_excerpt") or ""),
        "suggested_caption": str(highlight.get("suggested_caption") or ""),
        "clip_transcript": str(clip_transcript or highlight.get("transcript_excerpt") or ""),
    }
    user_msg = "Supplied clip context (source material):\n" + _json.dumps(
        context, ensure_ascii=False, indent=2
    )
    failures = []
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            raw = _call_llm(
                SERMON_THUMBNAIL_HEADLINE_V2_PROMPT,
                user_msg,
                temperature=float(temperature),
                _retries=1,
            )
            payload = _parse_json_object_response(raw)
            hooks = payload.get("thumbnail_hooks")
            titles = payload.get("social_titles")
            if not isinstance(hooks, list) or not isinstance(titles, list):
                raise ValueError("missing thumbnail_hooks or social_titles arrays")
            return {
                "ok": True,
                "profile": SERMON_THUMBNAIL_HEADLINE_V2_PROFILE,
                "attempt": attempt,
                "payload": payload,
                "raw_response": raw,
                "failures": failures,
            }
        except Exception as exc:
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    return {
        "ok": False,
        "profile": SERMON_THUMBNAIL_HEADLINE_V2_PROFILE,
        "payload": None,
        "raw_response": "",
        "failures": failures,
    }


SERMON_THUMBNAIL_HEADLINE_V3_PROFILE = "sermon_thumbnail_headline_v3"
_HEADLINE_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
SERMON_THUMBNAIL_HEADLINE_V3_PROMPT_PATH = os.path.join(
    _HEADLINE_PROMPT_DIR, "sermon_thumbnail_headline_v3.txt"
)
SERMON_THUMBNAIL_HEADLINE_SELECTOR_V3_PROMPT_PATH = os.path.join(
    _HEADLINE_PROMPT_DIR, "sermon_thumbnail_headline_selector_v3.txt"
)
_HEADLINE_V3_ANGLES = (
    "source_quote", "question", "contrast", "direct", "concrete_detail",
    "tension", "payoff", "ultra_short",
)
_HEADLINE_META_PATTERNS = (
    r"\bdie bedeutung von\b", r"\bdie rolle von\b", r"\beine geschichte über\b",
    r"\bgedanken zu\b", r"\bdiese predigt\b", r"\bdieser clip\b",
    r"\bin diesem (?:clip|video)\b", r"\bpredigt über\b",
)
_HEADLINE_BRAND_NAMES = (
    "kanzelclips", "move church", "st. martini", "icf zürich", "icf zurich",
)
_SENSITIVE_TOPIC_TERMS = {
    "holocaust_genocide": ("holocaust", "auschwitz", "genozid", "völkermord"),
    "suicide": ("suizid", "selbstmord", "selbsttötung"),
    "abortion": ("abtreibung", "abtreibungsversuch", "abtreiben", "abgetrieben"),
    "abuse": ("missbrauch", "misshandelt", "misshandlung"),
    "death": ("tod", "tödlich", "gestorben", "sterben", "grab"),
    "severe_illness": ("krebs", "krebsdiagnose", "tumor", "schwere krankheit"),
    "mental_health": (
        "psychische erkrankung", "depression", "panikattacke", "panikattacken",
        "psychose", "schizophrenie",
    ),
    "trauma": ("trauma", "traumatisch", "traumatisiert"),
}


def _load_headline_prompt(path):
    with open(path, "r", encoding="utf-8") as handle:
        prompt = handle.read().strip()
    if not prompt:
        raise ValueError(f"empty headline prompt: {path}")
    return prompt


def BuildSermonHeadlineSourceV3(
    clip_content,
    *,
    clip_transcript="",
    highlight=None,
    video_title="",
    current_thumbnail_title="",
    language="de",
):
    """Build the complete text-only source packet shared by generator/selector."""
    highlight = highlight if isinstance(highlight, dict) else {}
    return {
        "language": str(language or "de"),
        "video_title": str(video_title or ""),
        "content_summary": str(clip_content or highlight.get("content") or ""),
        "highlight_title": str(highlight.get("title") or ""),
        "highlight_hook": str(highlight.get("hook") or ""),
        "payoff_excerpt": str(highlight.get("payoff_excerpt") or ""),
        "suggested_caption": str(highlight.get("suggested_caption") or ""),
        "current_thumbnail_title": str(current_thumbnail_title or ""),
        "clip_transcript": str(
            clip_transcript or highlight.get("transcript_excerpt") or ""
        ),
    }


def DetectSensitiveHeadlineTopicsV3(source_material):
    """Conservatively label sensitive subjects without suppressing them."""
    source_text = " ".join(str(value or "") for value in source_material.values()).casefold()
    topics = []
    for topic, terms in _SENSITIVE_TOPIC_TERMS.items():
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", source_text) for term in terms):
            topics.append(topic)
    return {"sensitive_topic": bool(topics), "sensitive_topics": topics}


def _headline_words_v3(value):
    return re.findall(r"[0-9A-Za-zÄÖÜäöüß]+(?:['’][0-9A-Za-zÄÖÜäöüß]+)?", str(value or ""))


def _headline_contains_identity_v3(title, identities):
    folded = str(title or "").casefold()
    for identity in identities:
        identity_words = [word.casefold() for word in _headline_words_v3(identity)]
        if not identity_words:
            continue
        if " ".join(identity_words) in folded:
            return True
        if len(identity_words) > 1 and any(
            re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", folded)
            for word in identity_words if len(word) >= 4
        ):
            return True
    return False


def ValidateSermonHeadlineCandidateV3(
    candidate,
    *,
    source_text="",
    speaker="",
    channel="",
    require_anchor=True,
    require_accent=True,
):
    """Apply non-negotiable gates before a candidate can reach the selector."""
    candidate = candidate if isinstance(candidate, dict) else {}
    title = " ".join(str(candidate.get("title") or "").split())
    accent = " ".join(str(candidate.get("accent_word") or "").split())
    anchor = " ".join(str(candidate.get("source_anchor") or "").split())
    angle = str(candidate.get("angle") or "").strip()
    reasons = []
    title_words = _headline_words_v3(title)

    if not title:
        reasons.append("empty")
    if title and len(title_words) < 2:
        reasons.append("too_few_words")
    if len(title_words) > 5:
        reasons.append("too_many_words")
    if len(title) > 28:
        reasons.append("too_long")
    if title.endswith(("…", "...", "-", "–", "—", "/")):
        reasons.append("truncated_word")
    if any(
        char == "\ufffd" or unicodedata.category(char) in {"Cc", "Cs", "Co", "So"}
        for char in title
    ):
        reasons.append("unsafe_unicode")
    if "#" in title or any(char == "#" for char in accent):
        reasons.append("hashtag")
    if any(re.search(pattern, title.casefold()) for pattern in _HEADLINE_META_PATTERNS):
        reasons.append("meta_phrase")
    if _headline_contains_identity_v3(title, [speaker]):
        reasons.append("speaker_name")
    if _headline_contains_identity_v3(title, list(_HEADLINE_BRAND_NAMES) + [channel]):
        reasons.append("brand_name")
    if require_accent and accent not in title_words:
        reasons.append("accent_missing")
    if require_anchor and not anchor:
        reasons.append("anchor_empty")
    if require_anchor and anchor and anchor.casefold() not in str(source_text or "").casefold():
        reasons.append("anchor_not_found")
    if require_anchor and angle not in _HEADLINE_V3_ANGLES:
        reasons.append("angle_invalid")

    normalized = dict(candidate)
    normalized.update({"title": title, "accent_word": accent, "angle": angle, "source_anchor": anchor})
    return {"valid": not reasons, "reasons": reasons, "candidate": normalized}


def _sensitive_headline_reasons_v3(title, source_text, sensitive_topics):
    """Block a small set of known high-risk semantic transformations."""
    folded = str(title or "").casefold()
    source_folded = str(source_text or "").casefold()
    topics = set(sensitive_topics or [])
    reasons = []
    if "severe_illness" in topics and "krebs" in folded and "freude" in folded:
        if not any(marker in folded for marker in ("trotz", "woher", "wieso")):
            reasons.append("sensitive_causality_risk")
    if "holocaust_genocide" in topics:
        risky = (
            ("asche" in folded and ("neugeburt" in folded or "neu geboren" in folded)),
            ("holocaust" in folded and any(term in folded for term in ("überwand", "überwunden", "besiegt"))),
        )
        if any(risky):
            reasons.append("sensitive_sensationalism")
    loaded_terms = ("verrat", "schuld", "heilung", "wunder")
    if topics and any(term in folded and term not in source_folded for term in loaded_terms):
        reasons.append("sensitive_ungrounded_claim")
    return reasons


def GenerateHeadlineCandidatesV3(
    clip_content,
    *,
    clip_transcript="",
    highlight=None,
    video_title="",
    current_thumbnail_title="",
    speaker="",
    channel="",
    language="de",
    temperature=0.5,
    max_retries=2,
):
    """Generate eight challengers, repair invalid sets, and preserve audit data."""
    source_material = BuildSermonHeadlineSourceV3(
        clip_content,
        clip_transcript=clip_transcript,
        highlight=highlight,
        video_title=video_title,
        current_thumbnail_title=current_thumbnail_title,
        language=language,
    )
    source_text = "\n".join(str(value or "") for value in source_material.values())
    sensitive = DetectSensitiveHeadlineTopicsV3(source_material)
    prompt = _load_headline_prompt(SERMON_THUMBNAIL_HEADLINE_V3_PROMPT_PATH)
    base_user_msg = "Quellmaterial:\n" + _json.dumps(source_material, ensure_ascii=False, indent=2)
    attempts = []
    repair_reasons = []
    max_calls = 1 + max(0, int(max_retries))

    for attempt_number in range(1, max_calls + 1):
        user_msg = base_user_msg
        if repair_reasons:
            user_msg += (
                "\n\nREPAIR: Der vorige Versuch wurde verworfen. Erzeuge alle acht Kandidaten neu "
                "und behebe diese deterministischen Fehler:\n- " + "\n- ".join(repair_reasons)
            )
        try:
            raw = _call_llm(prompt, user_msg, temperature=float(temperature), _retries=1)
            payload = _parse_json_object_response(raw)
            raw_candidates = payload.get("candidates")
            if not isinstance(raw_candidates, list):
                raise ValueError("missing candidates array")
            validations = []
            seen_angles = set()
            seen_titles = set()
            for index, item in enumerate(raw_candidates, 1):
                candidate = dict(item) if isinstance(item, dict) else {}
                candidate["id"] = f"v3_{index}"
                validation = ValidateSermonHeadlineCandidateV3(
                    candidate,
                    source_text=source_text,
                    speaker=speaker,
                    channel=channel,
                )
                validation["reasons"].extend(_sensitive_headline_reasons_v3(
                    validation["candidate"].get("title", ""), source_text,
                    sensitive["sensitive_topics"],
                ))
                angle = validation["candidate"].get("angle")
                title_key = validation["candidate"].get("title", "").casefold()
                if angle in seen_angles:
                    validation["reasons"].append("duplicate_angle")
                if title_key and title_key in seen_titles:
                    validation["reasons"].append("duplicate_title")
                validation["valid"] = not validation["reasons"]
                seen_angles.add(angle)
                seen_titles.add(title_key)
                validations.append(validation)
            if len(raw_candidates) != 8:
                repair_reasons = [f"candidate_count:{len(raw_candidates)} (required:8)"]
            else:
                repair_reasons = [
                    f"{entry['candidate'].get('id')}:{','.join(entry['reasons'])}"
                    for entry in validations if not entry["valid"]
                ]
            attempts.append({
                "attempt": attempt_number,
                "raw_response": raw,
                "raw_candidates": [entry["candidate"] for entry in validations],
                "rejected": [entry for entry in validations if not entry["valid"]],
                "valid_candidates": [entry["candidate"] for entry in validations if entry["valid"]],
                "repair_reasons": list(repair_reasons),
            })
            if not repair_reasons:
                break
        except Exception as exc:
            repair_reasons = [f"parse_failure:{type(exc).__name__}:{exc}"]
            attempts.append({
                "attempt": attempt_number, "raw_response": "", "raw_candidates": [],
                "rejected": [], "valid_candidates": [], "repair_reasons": list(repair_reasons),
            })

    final = attempts[-1]
    return {
        "ok": bool(final["valid_candidates"]),
        "profile": SERMON_THUMBNAIL_HEADLINE_V3_PROFILE,
        "source_material": source_material,
        **sensitive,
        "attempts": attempts,
        "generator_calls": len(attempts),
        "retries": max(0, len(attempts) - 1),
        "raw_candidates": final["raw_candidates"],
        "rejected": final["rejected"],
        "valid_candidates": final["valid_candidates"],
        "parse_failures": sum(
            any(reason.startswith("parse_failure:") for reason in item["repair_reasons"])
            for item in attempts
        ),
    }


def SelectHeadlineWinnerV3(
    *,
    source_material,
    current_champion=None,
    valid_challengers=None,
    sensitive_topic=False,
    speaker="",
    channel="",
    retries=2,
):
    """Select conservatively from already validated titles at temperature zero."""
    source_text = "\n".join(str(value or "") for value in source_material.values())
    candidates = []
    champion_validation = None
    if isinstance(current_champion, dict):
        champion = dict(current_champion)
        champion["id"] = "current_champion"
        champion_validation = ValidateSermonHeadlineCandidateV3(
            champion,
            source_text=source_text,
            speaker=speaker,
            channel=channel,
            require_anchor=False,
            require_accent=False,
        )
        if champion_validation["valid"]:
            candidates.append({**champion_validation["candidate"], "source": "current_champion"})
    for challenger in valid_challengers or []:
        candidates.append({**challenger, "source": "v3_challenger"})
    if not candidates:
        return {
            "ok": False, "payload": None, "failures": ["no validated candidates"],
            "current_champion_validation": champion_validation, "selector_calls": 0,
        }

    allowed = {candidate["id"]: candidate for candidate in candidates}
    prompt = _load_headline_prompt(SERMON_THUMBNAIL_HEADLINE_SELECTOR_V3_PROMPT_PATH)
    base_user_msg = "Auswahlpaket:\n" + _json.dumps({
        "source_material": source_material,
        "sensitive_topic": bool(sensitive_topic),
        "current_champion": next((item for item in candidates if item["source"] == "current_champion"), None),
        "valid_challengers": [item for item in candidates if item["source"] == "v3_challenger"],
    }, ensure_ascii=False, indent=2)
    failures = []
    calls = 0
    for attempt in range(1, max(1, int(retries)) + 1):
        calls += 1
        try:
            user_msg = base_user_msg
            if failures:
                user_msg += (
                    "\n\nREPAIR: Die vorige Antwort war technisch unzulässig. Behebe: "
                    + failures[-1]
                )
            raw = _call_llm(prompt, user_msg, temperature=0.0, _retries=1)
            payload = _parse_json_object_response(raw)
            winner_id = str(payload.get("winner_id") or "")
            runner_up_id = str(payload.get("runner_up_id") or "")
            if winner_id not in allowed:
                raise ValueError("winner_id is not a validated candidate")
            expected_source = allowed[winner_id]["source"]
            if payload.get("winner_source") != expected_source:
                raise ValueError("winner_source does not match winner_id")
            if runner_up_id and runner_up_id not in allowed:
                raise ValueError("runner_up_id is not a validated candidate")
            social_title = " ".join(str(payload.get("social_title") or "").split())
            if not social_title or len(social_title) > 90:
                raise ValueError("social_title must contain 1-90 characters")
            if "#" in social_title:
                raise ValueError("social_title must not contain hashtags")
            if any(
                char == "\ufffd" or unicodedata.category(char) in {"Cc", "Cs", "Co", "So"}
                for char in social_title
            ):
                raise ValueError("social_title contains unsafe Unicode or emoji")
            if _headline_contains_identity_v3(social_title, [speaker]):
                raise ValueError("social_title must not contain the speaker name")
            if _headline_contains_identity_v3(
                social_title, list(_HEADLINE_BRAND_NAMES) + [channel]
            ):
                raise ValueError("social_title must not contain a brand name")
            sensitive_reasons = _sensitive_headline_reasons_v3(
                social_title, source_text,
                DetectSensitiveHeadlineTopicsV3(source_material)["sensitive_topics"]
                if sensitive_topic else [],
            )
            if sensitive_reasons:
                raise ValueError(
                    "social_title failed sensitive guard: " + ",".join(sensitive_reasons)
                )
            payload["social_title"] = social_title
            payload["winner_title"] = allowed[winner_id]["title"]
            return {
                "ok": True, "payload": payload, "raw_response": raw,
                "failures": failures, "selector_calls": calls,
                "current_champion_validation": champion_validation,
            }
        except Exception as exc:
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    return {
        "ok": False, "payload": None, "raw_response": "", "failures": failures,
        "selector_calls": calls, "current_champion_validation": champion_validation,
    }


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

    for _attempt in range(3):
        try:
            raw = _call_llm(TITLE_HOOK_PROMPT, user_msg, temperature=0.7)
            lines = [l.strip().strip('"').strip("'").strip('\u201c').strip('\u201d')
                     for l in raw.strip().splitlines() if l.strip()]
            hook = lines[0] if lines else ""
            keyword = lines[1] if len(lines) > 1 else ""
            if len(hook) > 40:
                hook = hook[:37] + "\u2026"
            if hook:
                if keyword.upper() not in hook.upper():
                    keyword = max(hook.split(), key=len) if hook.split() else ""
                if _is_invalid_hook(hook, keyword, clip_transcript=clip_transcript, language=language):
                    print(f"[TitleHook] Rejected weak/meta hook: {hook!r} — retrying")
                    continue
                return hook, keyword
        except Exception as exc:
            print(f"[TitleHook] LLM call failed ({exc}); using fallback.")
            break

    return _fallback_title_hook(
        clip_content,
        clip_transcript=clip_transcript,
        language=(language, video_title),
    )


THUMBNAIL_ANGLES_PROMPT = """\
You write short, punchy thumbnail hooks for sermon short-form videos.

Return ONLY a JSON array of objects, no prose, no code fences. Each object:
  {"hook": "...", "accent_line": 0, "emotion": "...", "scene": "..."}

Hard rules:
- "hook" is 2-4 words, in the requested language, ALL-CAPS-friendly, max 22 characters.
- Ground every hook in the supplied transcript. Never invent claims, quotes or
  teachings that are not in the source material.
- The hooks must be meaningfully DIFFERENT from one another. Vary the angle
  across tension, promise, curiosity, urgency, transformation, challenge,
  comfort, revelation and breakthrough. No near-duplicates.
- "accent_line" is the 0-based index of the line to highlight in colour, or null.
  A hook renders as 1-3 stacked lines, so use 0, 1 or 2.
- "emotion" is one word. "scene" is a short visual idea, max 12 words, no text in it.
"""


def GenerateThumbnailAngles(
    clip_content,
    clip_transcript="",
    video_title="",
    language="de",
    speaker_name="",
    n_angles=10,
):
    """N distinct thumbnail hook angles, grounded in the transcript.

    Mirrors the sermon-agent brief: many different angles on one sermon rather
    than one hook. Returns [] on any failure so callers can fall back to the
    single-hook path — the Sunday pipeline must never block on the LLM.
    """
    parts = [f"Video title: {video_title}"]
    if clip_transcript:
        t = clip_transcript[:2000]
        if len(clip_transcript) > 2000:
            t = t.rsplit(" ", 1)[0] + " ..."
        parts.append(f"Transcript:\n{t}")
    parts.append(f"Content summary: {clip_content}")
    parts.append(f"Language: {language}")
    if speaker_name:
        parts.append(f"Speaker name: {speaker_name}")
    parts.append(f"Produce exactly {n_angles} distinct angles.")

    try:
        raw = _call_llm(THUMBNAIL_ANGLES_PROMPT, "\n\n".join(parts), temperature=0.8)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        payload = _json.loads(cleaned)
        if not isinstance(payload, list):
            return []
    except Exception as exc:
        print(f"[ThumbnailAngles] LLM call failed ({exc}); no angles.")
        return []

    angles = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        hook = " ".join(str(item.get("hook") or "").split())[:22]
        if not hook or len(hook.split()) > 4:
            continue
        key = hook.upper()
        if key in seen:  # the prompt asks for variety; enforce it too
            continue
        seen.add(key)
        accent = item.get("accent_line")
        accent = int(accent) if isinstance(accent, (int, float)) and 0 <= int(accent) <= 2 else None
        angles.append({
            "hook": hook,
            "accent_line": accent,
            "emotion": " ".join(str(item.get("emotion") or "").split())[:24],
            "scene": " ".join(str(item.get("scene") or "").split())[:80],
        })
    return angles[:n_angles]


def GenerateThumbnailBrief(
    clip_content,
    clip_transcript="",
    video_title="",
    language="de",
    speaker_name="",
    brand_label="",
    n_angles=1,
):
    """Ask the LLM for a compact portrait-thumbnail brief.

    With `n_angles > 1` the result also carries an ``angles`` list of distinct
    hook variants (see :func:`GenerateThumbnailAngles`). The single-hook keys are
    unchanged, so the existing v2/v2_test callers are unaffected.
    """
    parts = [f"Video title: {video_title}"]
    if clip_transcript:
        t = clip_transcript[:800]
        if len(clip_transcript) > 800:
            t = t.rsplit(" ", 1)[0] + " ..."
        parts.append(f"Transcript:\n{t}")
    parts.append(f"Content summary: {clip_content}")
    parts.append(f"Language: {language}")
    if speaker_name:
        parts.append(f"Speaker name: {speaker_name}")
    if brand_label:
        parts.append(f"Brand label: {brand_label}")
    user_msg = "\n\n".join(parts)

    hook_fallback, accent_fallback = GenerateTitleHook(
        clip_content,
        clip_transcript=clip_transcript,
        video_title=video_title,
        language=language,
    )
    fallback = {
        "hook_text": hook_fallback,
        "accent_keyword": accent_fallback,
        "social_title": " – ".join(
            value for value in (hook_fallback, " ".join(str(video_title or "").split())) if value
        )[:90],
        "story_concept": " ".join(str(clip_content or "").split())[:180],
        "curiosity_gap": "",
        "emotion_target": "conviction",
        "pose_id": "direct_challenge",
        "story_asset_ids": [],
        "speaker_side": "auto",
        "background_style": "clean_gradient",
        "background_prompt": _fallback_thumbnail_background_prompt(
            clip_content,
            video_title=video_title,
            speaker_name=speaker_name,
            brand_label=brand_label,
            background_style="clean_gradient",
        ),
        "background_negative_prompt": _DEFAULT_BACKGROUND_NEGATIVE,
        "layer_plan": list(_THUMBNAIL_LAYER_ORDER),
        "light_direction": "upper_left",
        "palette": ["deep navy", "warm gold", "ivory"],
        "art_direction": "modern_cinematic",
        "brand_label": brand_label or "",
        "speaker_name": speaker_name or "",
        "angles": [],
    }

    try:
        raw = _call_llm(THUMBNAIL_BRIEF_PROMPT, user_msg, temperature=0.5)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        payload = _json.loads(cleaned)
        if not isinstance(payload, dict):
            return fallback
        hook_text = " ".join(str(payload.get("hook_text") or "").split())[:28]
        accent_keyword = " ".join(str(payload.get("accent_keyword") or "").split())
        if not hook_text:
            return fallback
        if accent_keyword.upper() not in hook_text.upper():
            accent_keyword = accent_fallback or max(hook_text.split(), key=len)
        if _is_invalid_hook(hook_text, accent_keyword, clip_transcript=clip_transcript, language=language):
            return fallback
        speaker_side = str(payload.get("speaker_side") or "auto").strip().lower()
        if speaker_side not in {"left", "right", "center_low", "auto"}:
            speaker_side = "auto"
        background_style = str(payload.get("background_style") or "clean_gradient").strip().lower()
        if background_style not in {"clean_gradient", "strong_contrast", "emotion_focus"}:
            background_style = "clean_gradient"
        background_prompt = " ".join(str(payload.get("background_prompt") or "").split())[:240]
        if not background_prompt:
            background_prompt = _fallback_thumbnail_background_prompt(
                clip_content,
                video_title=video_title,
                speaker_name=speaker_name,
                brand_label=brand_label,
                background_style=background_style,
            )
        background_negative_prompt = " ".join(str(payload.get("background_negative_prompt") or "").split())[:240]
        if not background_negative_prompt:
            background_negative_prompt = _DEFAULT_BACKGROUND_NEGATIVE
        social_title = " ".join(str(payload.get("social_title") or "").split())[:90]
        if not social_title:
            social_title = fallback["social_title"]
        if hook_text.upper() not in social_title.upper():
            social_title = f"{hook_text} – {social_title}"[:90]
        pose_id = str(payload.get("pose_id") or "").strip().lower()
        if pose_id not in _THUMBNAIL_POSE_IDS:
            pose_id = fallback["pose_id"]
        raw_asset_ids = payload.get("story_asset_ids")
        story_asset_ids = []
        if isinstance(raw_asset_ids, list):
            for item in raw_asset_ids:
                asset_id = str(item or "").strip().lower()
                if asset_id in _THUMBNAIL_STORY_ASSET_IDS and asset_id not in story_asset_ids:
                    story_asset_ids.append(asset_id)
        raw_layer_plan = payload.get("layer_plan")
        layer_plan = []
        if isinstance(raw_layer_plan, list):
            for item in raw_layer_plan:
                layer = str(item or "").strip().lower()
                if layer in _THUMBNAIL_LAYER_ORDER and layer not in layer_plan:
                    layer_plan.append(layer)
        if "speaker" not in layer_plan or "final_grade" not in layer_plan:
            layer_plan = list(_THUMBNAIL_LAYER_ORDER)
        raw_palette = payload.get("palette")
        palette = [" ".join(str(item or "").split())[:24] for item in raw_palette] if isinstance(raw_palette, list) else []
        palette = [item for item in palette if item][:3]
        if len(palette) != 3:
            palette = list(fallback["palette"])
        art_direction = str(payload.get("art_direction") or "").strip().lower()
        if art_direction not in {"modern_cinematic", "renaissance_chiaroscuro", "baroque_drama", "classical_tableau"}:
            art_direction = "modern_cinematic"
        return {
            "hook_text": hook_text,
            "accent_keyword": accent_keyword,
            "social_title": social_title,
            "story_concept": " ".join(str(payload.get("story_concept") or clip_content or "").split())[:180],
            "curiosity_gap": " ".join(str(payload.get("curiosity_gap") or "").split())[:140],
            "emotion_target": " ".join(str(payload.get("emotion_target") or "conviction").split())[:40],
            "pose_id": pose_id,
            "story_asset_ids": story_asset_ids[:3],
            "speaker_side": speaker_side,
            "background_style": background_style,
            "background_prompt": background_prompt,
            "background_negative_prompt": background_negative_prompt,
            "layer_plan": layer_plan,
            "light_direction": " ".join(str(payload.get("light_direction") or "upper_left").split())[:40],
            "palette": palette,
            "art_direction": art_direction,
            "brand_label": " ".join(str(payload.get("brand_label") or brand_label or "").split())[:28],
            "speaker_name": speaker_name or "",
            "angles": _thumbnail_angles_or_empty(
                clip_content, clip_transcript, video_title, language, speaker_name, n_angles
            ),
        }
    except Exception as exc:
        print(f"[ThumbnailBrief] LLM call failed ({exc}); using fallback.")
        return fallback


def _thumbnail_angles_or_empty(clip_content, clip_transcript, video_title, language, speaker_name, n_angles):
    if not n_angles or n_angles < 2:
        return []
    return GenerateThumbnailAngles(
        clip_content,
        clip_transcript=clip_transcript,
        video_title=video_title,
        language=language,
        speaker_name=speaker_name,
        n_angles=n_angles,
    )


if __name__ == "__main__":
    pass
