Captions & Word Timing — Format, Retiming, and Debugging

Overview

This document describes the expected word-timing formats used by the Shorts pipeline, how cleaned subtitle text is retimed back to ASR word timestamps, the conservative safeguards in place, and how to use the generated diagnostics to iterate or roll back changes.

Cache format requirements

- The transcription cache next to each video must be a JSON dict with two keys:
  - `segments`: list of `[text, start_sec, end_sec]` entries (segment-level ASR output).
  - `words`: list of `{"text": str, "start": float, "end": float}` entries containing per-word timings from the ASR engine.

- If the cache is a legacy bare list of segments (no `words` key), per-word karaoke highlighting will not work. Delete the `.transcription.json` file and re-run transcription to regenerate `words`.

Retiming summary

- Function: `cli/shorts_bridge.py::_retime_cleaned_word_timestamps()`
- Purpose: given raw ASR transcriptions and cleaned (LLM-edited) segment texts, approximate per-word start/end times for the cleaned tokens so ASS karaoke highlighting can be applied.
- Strategies used (in order):
  1. Direct 1:1 mapping when the cleaned token count equals the original word count.
  2. Fuzzy anchoring: attempt to match cleaned tokens to original words by stripped/prefix matches.
  3. Distributed interpolation: if anchors are insufficient, distribute tokens across the segment proportionally by token length.

Conservative fallback and thresholds

- Global sanity check: if the total cleaned token count across the clip differs from the total original ASR word count by more than 25% (|cleaned - orig| / orig > 0.25), retiming is aborted and the pipeline falls back to chunked subtitles (no per-word timings).
- Fuzzy matching requires at least 40% of tokens to be anchored for the algorithm to be considered useful (existing behavior).

Diagnostics and logs

- On every retiming attempt, a JSON diagnostics file is written beside the CLI script:
  - Path: `<repo_root>/cli/shorts_bridge.retiming_debug.json`
  - Contents include: timestamp, global counts, per-segment diagnostics entries with `segment` (start time), `strategy` (`direct`/`fuzzy`/`fuzzy_failed`/`distributed`), and per-segment `orig_count` and `cleaned_count`.

How to interpret diagnostics

- If `reason` is `global_count_mismatch`, the retiming was aborted. Inspect `total_orig` vs `total_cleaned` and the `diagnostics` list to see which segments diverged.
- `fuzzy_failed` indicates the fuzzy matching threshold was not met for that segment.
- `distributed` shows segments where interpolation was used; verify timings near sentence boundaries and audience reactions.

Rollback and iteration workflow

1. If retiming aborts frequently for a clip, consider:
   - Inspecting the cleaned transcript (`.subtitle_cleanup.json`) to see if LLM rewrites introduced/removed many tokens.
   - Running the pipeline with `--force-raw-word-subtitles` (or set `force_raw_word_subtitles=True` in code) to use raw ASR timings while iterating.
2. To re-generate word timestamps: delete the video-level `.transcription.json` and re-run the ASR step.
3. To collect reproducible diagnostics for a failing file, run the CLI with the same input and then share the generated `cli/shorts_bridge.retiming_debug.json` file.

Next steps for improvement

- If you see many `fuzzy_failed` segments, we can log the actual token-to-word matching attempts and increase the matching heuristics (edit-distance, stemming, or bilingual normalization for German compound words).
- For stubborn divergences, consider aligning on sentence-level anchors (e.g., anchor cleaned segment boundaries to nearest ASR segment boundaries), then interpolate within sentences.

Contact and notes

- File updated by automation: `cli/shorts_bridge.py` now writes `shorts_bridge.retiming_debug.json` for every retiming run.
- For help reproducing failures, attach the `.transcription.json` and `.subtitle_cleanup.json` files for the clip in question.
