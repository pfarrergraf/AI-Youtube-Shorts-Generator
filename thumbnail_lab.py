#!/usr/bin/env python3
"""
Thumbnail lab — sweep hooks x moods x speaker-render variants, then compare.
============================================================================

Renders the "epic" look across every axis and writes a labelled contact sheet
with each variant's gate verdict, so the choice can be made by looking rather
than by guessing.

    python thumbnail_lab.py --video sermon.mp4 --all-variants --contact-sheet
    python thumbnail_lab.py --video sermon.mp4 --hook "DAS VOLLE PAKET" --accent-line 2
    python thumbnail_lab.py --video sermon.mp4 --hooks 6          # LLM hook angles

Everything except `--speaker-render {real_relight,ai_plate,ai_hero}` and
`--hooks` runs with no server. Those degrade to the offline path and say so in
the report rather than failing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Components.ThumbnailEffects import estimate_face_box  # noqa: E402
from Components.ThumbnailEpic import (  # noqa: E402
    CANVAS_9X16,
    MOODS,
    SPEAKER_LAYOUTS,
    SPEAKER_RENDERS,
    compose,
)
from Components.ThumbnailMatting import extract_subject, upscale_subject  # noqa: E402
from Components.ThumbnailMoveChurch import extract_best_frame, extract_frame_at  # noqa: E402


def _load_frame(args) -> np.ndarray:
    if args.frame:
        return np.load(args.frame) if args.frame.endswith(".npy") else _bgr_from_path(args.frame)
    if args.timestamp is not None:
        return extract_frame_at(args.video, args.timestamp)
    return extract_best_frame(args.video, n_candidates=args.frame_candidates)


def _bgr_from_path(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def _derive_hooks(args) -> list[tuple[str, int | None]]:
    """Hooks to render: explicit, LLM-generated, or a title fallback."""
    if args.hook:
        return [(args.hook, args.accent_line)]
    if args.hooks:
        try:
            from Components.LanguageTasks import GenerateThumbnailBrief  # noqa: PLC0415

            brief = GenerateThumbnailBrief(
                args.title or "",
                clip_transcript=args.transcript or "",
                video_title=args.title or "",
                speaker_name=args.speaker or "",
                n_angles=args.hooks,
            )
            angles = brief.get("angles") or []
            if angles:
                return [(a["hook"], a.get("accent_line")) for a in angles[: args.hooks]]
            print("  ! LLM returned no angles — falling back to the title", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - any LLM failure must not stop the sweep
            print(f"  ! hook generation unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
    title = args.title or Path(args.video or "thumbnail").stem
    return [(title, None)]


def render_sweep(args) -> list[dict]:
    frame = _load_frame(args)
    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")

    matte = extract_subject(frame, isolate_primary=not args.keep_all_people)
    print(f"Matting: {matte.provider}  coverage={matte.coverage:.3f}")
    subject = matte.subject_rgba
    if args.upscale:
        subject, up_info = upscale_subject(subject)
        print(f"Upscale: {up_info}")
    face_box = estimate_face_box(subject)

    hooks = _derive_hooks(args)
    moods = list(MOODS) if args.all_variants else [args.mood]
    renders = list(SPEAKER_RENDERS) if args.all_variants else [args.speaker_render]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for h_i, (hook, accent) in enumerate(hooks):
        for mood in moods:
            for render in renders:
                res = compose(
                    hook,
                    subject_rgba=subject,
                    subject_face_box=face_box,
                    frame_bgr=frame,
                    mood=mood,
                    speaker_render=render,
                    speaker_layout=args.speaker_layout,
                    text_anchor=args.text_anchor,
                    accent_line=accent,
                    theme=args.title or "",
                    gate_tier=args.gate,
                    seed=args.seed + h_i,
                    subject_height_ratio=args.subject_height,
                    speaker_bottom_ratio=args.speaker_bottom_ratio,
                )
                name = f"epic_{h_i:02d}_{mood}_{render}.png"
                path = out_dir / name
                res.image.save(path)
                metrics = res.metrics()
                metrics["path"] = str(path)
                results.append(metrics)
                gate = res.gate
                flag = "OK " if (gate.passed and not gate.out_of_band) else (
                    "GATE" if not gate.passed else f"{len(gate.out_of_band)}ob"
                )
                print(f"  [{flag}] {name}  {hook!r}")
    return results


def contact_sheet(results: list[dict], path: Path, *, cols: int = 5, cell_w: int = 300) -> Path:
    """Labelled grid of every variant with its gate verdict."""
    if not results:
        raise ValueError("nothing to tile")
    cell_h = int(cell_w * CANVAS_9X16[1] / CANVAS_9X16[0])
    label_h = 46
    rows = (len(results) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for i, item in enumerate(results):
        cx, cy = (i % cols) * cell_w, (i // cols) * (cell_h + label_h)
        sheet.paste(Image.open(item["path"]).resize((cell_w, cell_h), Image.LANCZOS), (cx, cy))
        gate = item.get("gate", {})
        n_ob = len(gate.get("out_of_band", {}))
        ok = gate.get("passed") and n_ob == 0
        colour = (120, 230, 140) if ok else ((240, 200, 90) if gate.get("passed") else (240, 110, 110))
        verdict = "in band" if ok else (f"{n_ob} out of band" if gate.get("passed") else "GATE FAIL")
        draw.text((cx + 6, cy + cell_h + 4), f"{item['mood']} / {item['speaker_render']}", fill=(225, 225, 230), font=font)
        draw.text((cx + 6, cy + cell_h + 22), verdict, fill=colour, font=font)

    sheet.save(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="source video")
    src.add_argument("--frame", help="pre-extracted frame (.npy or image)")
    ap.add_argument("--timestamp", type=float, default=None, help="pick the frame at this second")
    ap.add_argument("--frame-candidates", type=int, default=24)

    ap.add_argument("--hook", help="explicit hook text; skips the LLM")
    ap.add_argument("--accent-line", type=int, default=None, help="0-based line to colour")
    ap.add_argument("--hooks", type=int, default=0, help="generate N LLM hook angles")
    ap.add_argument("--title", default="", help="video title / theme, used for hooks and prompts")
    ap.add_argument("--transcript", default="", help="clip transcript for hook grounding")
    ap.add_argument("--speaker", default="", help="speaker name")

    ap.add_argument("--mood", default="warm_shaft", choices=sorted(MOODS))
    ap.add_argument("--speaker-render", default="real_procedural", choices=SPEAKER_RENDERS)
    ap.add_argument("--all-variants", action="store_true", help="sweep every mood x speaker-render")
    ap.add_argument(
        "--speaker-layout",
        default="closeup",
        choices=sorted(SPEAKER_LAYOUTS),
        help="speaker crop: balanced, closeup, or portrait (default: closeup)",
    )
    ap.add_argument("--subject-height", type=float, default=None,
                    help="override the layout's speaker height ratio")
    ap.add_argument("--speaker-bottom-ratio", type=float, default=None,
                    help="override the speaker's bottom anchor (0..1)")
    ap.add_argument("--text-anchor", choices=["top", "bottom"], default="top",
                    help="place the giant hook at the top or lower half")
    ap.add_argument("--keep-all-people", action="store_true", help="skip primary-subject isolation")
    ap.add_argument("--upscale", action="store_true", help="RealESRGAN the cutout first")

    ap.add_argument("--gate", default="normal", choices=["off", "normal", "strict"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="thumbnail_lab_out")
    ap.add_argument("--contact-sheet", action="store_true")
    args = ap.parse_args()

    results = render_sweep(args)

    out_dir = Path(args.output_dir)
    report = out_dir / "thumbnail_lab_report.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report}")

    if args.contact_sheet:
        sheet = contact_sheet(results, out_dir / "thumbnail_contact_sheet.png")
        print(f"Contact sheet: {sheet}")

    clean = [r for r in results if r.get("gate", {}).get("passed") and not r["gate"]["out_of_band"]]
    print(f"{len(clean)}/{len(results)} variants fully in band")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
