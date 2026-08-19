---
name: create-thumbnail
description: Generate Move Church style thumbnails from a sermon video with a built-in design→critique→iterate loop. Use when the user types /create-thumbnail or asks to "create a thumbnail", "thumbnail erstellen", "generate thumbnails", or wants cover images for a short/video. Wraps generate_thumbnail.py (8 templates), reads thumbnail_report.json quality metrics, and iterates on weak scores instead of accepting the first render.
---

This skill turns thumbnail generation into a quality loop: **generate → score → critique → iterate → deliver**. Never hand the user the first render without reading its report.

## Production standard (August 2026)

For normal production and backlog repair, use `--thumbnail-mode v2`. V2 selects a
reviewed, text-free AI hero from `assets/speaker_references/` for known speakers,
places the exact headline on the empty side, and hard-rejects candidates whose
text panel intersects the expanded face-safe zone. The image model must never
render the headline itself. `legacy`, `v2_test`, and `epic` are comparison modes.

For finished clips waiting in the upload folders, use the non-destructive batch
tool; it backs up existing thumbnail siblings before replacing them:

```bash
python tools/regenerate_pending_thumbnails.py
python -m cli.parakeet cloud-sync --brand latzel --direct-r2
python -m cli.parakeet cloud-sync --brand movechurch --direct-r2
```

The supported upload path is direct R2 queueing. The former local watchfolder
publishing daemons are retired and must not be restarted.

## Two renderers — pick the right one

| | Legacy (below, 8 templates) | `epic` (reference-look) |
|---|---|---|
| Look | Depth-layered mega-stack, drop shadow + glow + extrude | Huge per-line-filled type, no contour, one light source, chiaroscuro |
| Gate | Scalar score only, no pass/fail | `Components/ThumbnailReferenceGate.py`, bands fitted to `thumbnail_ideal_examples/` |
| Speaker | rembg/birefnet cutout | ComfyUI `PersonMaskUltra V2` (falls back to rembg when ComfyUI is down) |
| CLI | `generate_thumbnail.py` | `thumbnail_lab.py` (sweep) or `--thumbnail-mode epic` in `parakeet shorts` |

Default to **`epic`** for a fresh "make this look better" ask — that is the gap the user actually measured (references at ~250 peak-luma / 11% blown highlights vs. our old ~165 / 1%). Use the legacy 8-template path only when the user explicitly wants one of those named styles, or is iterating on an existing legacy render.

### `epic` quick start

```bash
python thumbnail_lab.py --video "VIDEO.mp4" --hooks 8 --title "Sermon title" --all-variants --contact-sheet
```

Sweeps every mood (`warm_shaft, gold_burst, cool_door, cyan_split, red_alert, white_stage`) x the real-speaker render paths (`frame_cinematic, real_procedural, real_relight, ai_plate`), writes a labelled `thumbnail_contact_sheet.png`, and a `thumbnail_lab_report.json` with each variant's gate verdict. The lab defaults to the `closeup` speaker layout; use `--speaker-layout portrait --text-anchor bottom` for the reference-like face-above/title-below composition. `--hooks N` calls `LanguageTasks.GenerateThumbnailAngles` for N grounded, distinct hooks; without it, pass `--hook "..." --accent-line N` directly.

The synthetic `ai_hero` path is intentionally excluded from the sweep: SDXL invents a generic or distorted face instead of preserving the real speaker. Do not use it for named real people.

Read `thumbnail_lab_report.json`'s `gate.out_of_band` before delivering — a variant with entries there is outside the reference distribution on that metric (type too small, no highlight, wrong dark/light balance, etc.).

## Setup

```bash
cd /home/benjamin_graf/parakeet_uv/ai-youtube-shorts-generator
source /home/benjamin_graf/parakeet_uv/.venv/bin/activate
```

## Design system (the fixed vocabulary — do not invent outside it)

8 templates in `Components/ThumbnailMoveChurch.py` (`PALETTES` + `TEMPLATE_FONTS`):

| Template | Feel | Font |
|---|---|---|
| `navy_dark` | Brand default, blue beams, rings | Barlow Condensed Black |
| `energy_orange` | High energy, italic | Barlow Condensed Black Italic |
| `warm_gold` | Warm, spotlight, elegant | Playfair Display Black (serif) |
| `cinematic_dark` | Film noir, gold separator, grain | Playfair Display Black (serif) |
| `fire_red` | Urgent, red radial, diagonal lines | Barlow Condensed Black Italic |
| `heaven_blue` | Light from above, peaceful | Barlow Condensed Black |
| `bold_minimal` | Pure typography, huge back words | Bebas Neue |
| `sunset_warm` | Intimate, purple→amber | Barlow SemiCondensed Black |

Hard constraints (from CLAUDE.md): **never VHS/glitch/retro effects**, 9:16 is the default format, depth layering (back words behind speaker at 0.88 opacity, front word in front) is the signature look.

Signature rendering (≥5 layers, marketing-grade):
1. **Background** — gradient + glow orbs + rings/swirl arcs + light rays + atmosphere
2. **Keylight bloom** — multi-pass silhouette glow (`_add_subject_keylight`): wide wash → mid bloom → hot near-white core; reads like a strong stage keylight behind the speaker
3. **Back text** — mega-stack: every line fills the text column (per-word fill-width sizing, clamped 0.68×–1.50×), −6° diagonal, glow + drop shadow + extrude
4. **Speaker** — palette rim light + edge glow baked on the cutout
5. **Front text** — same mega-stack treatment, punch word in front of the body

Template choice by sermon tone: bold statement → `bold_minimal`/`cinematic_dark`; evangelistic/urgent → `fire_red`/`energy_orange`; comfort/hope → `heaven_blue`/`sunset_warm`; testimony/personal → `sunset_warm`/`warm_gold`.

## Workflow

### 1. Interview (one AskUserQuestion call, skip what's already in the message)

- **Title split** — propose back/front split of the title (last word(s) in front, e.g. "EINE WIE" / "KEINE"). Front zone should be the punch word.
- **Template** — offer "All 8 (auto-pick best)" (recommended), plus 2–3 tone-matched single templates.
- **Logo** — yes/no (`--logo`).

### 2. Generate

```bash
# All templates, 3 best frames, auto-pick (recommended):
python generate_thumbnail.py --source "VIDEO.mp4" --back "EINE WIE" --front "KEINE" \
    --all-templates --provider rembg

# Single template:
python generate_thumbnail.py --source "VIDEO.mp4" --back "..." --front "..." \
    --template fire_red --provider rembg

# Quick visual comparison of all 8 templates (no video needed):
python generate_thumbnail.py --preview-all
```

Run with `run_in_background: true` (full batch ≈ 2–4 min). `--all-templates` writes 24 variants to `thumbnail_variants/`, the winner to `thumbnail_best.png`, and metrics to `thumbnail_report.json`.

### 3. Critique loop (read `thumbnail_report.json`, act on weak metrics)

Check `selected_variant.metrics` (and per-variant `metrics`). Thresholds → actions:

| Metric | Bad when | Action |
|---|---|---|
| `readability_score` | < 0.45 | Different template (more contrast) or shorter words |
| `face_overlap_ratio` | > 0.15 | `--timestamp` for a different frame, or shorter front word |
| `back_text_max_occlusion_ratio` | > 0.30 | Move words: fewer back words, or different frame |
| `clipping_penalty` | > 0.05 | Title too long — shorten or re-split back/front |
| `subject_coverage` | < 0.05 | Cutout failed — try `--provider birefnet` or `--timestamp` |
| `frame_selection.selected.rejected_reason` | non-empty | Pass an explicit `--timestamp` from a good moment |

Iterate at most 2 rounds; report what changed and why. If `provider_used` is `grabcut_local`, the cutout is low quality — flag it and retry with `rembg`.

### 4. Deliver

- Read the final PNG(s) to visually verify: no clipped text, face visible, depth effect intact.
- Send the winner (and up to 2 runners-up) to the user via SendUserFile with the score in the caption.
- Name outputs after the video stem. The backlog tool preserves existing files in
  `_thumbnail_backups/<date>/` before updating a sibling thumbnail.

## Testing after any code change to the thumbnail modules

1. `python generate_thumbnail.py --preview-all` — all 8 must render, grid saved.
2. Palette completeness: every `PALETTES` entry needs `bg_top, bg_bottom, glow_a, glow_b, accent, text, ray_color (RGBA), rings, italic, logo_color`.
3. API stability: `quick_generate()`, `batch_generate()`, `generate_move_church_thumbnail()` signatures must not break — `cli/shorts_bridge.py` and external callers depend on them.

## Gotchas

- rembg needs `onnxruntime` in the venv (`uv pip install rembg onnxruntime`).
- Fonts live in `~/.local/share/fonts/mc_thumbnails/`; missing template fonts fall back to Barlow Condensed Black silently — run `python install_fonts.py --list` if a template looks wrong.
- `bold_minimal` works best with 1–2 words per zone; long titles will be width-clamped.
- Video paths often contain fullwidth `｜` characters **and non-breaking spaces (U+00A0)** — always quote them, and when a "file not found" looks impossible, resolve the path with a glob: `--source "$(ls /path/Stem*Suffix.mp4)"`.
