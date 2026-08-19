#!/usr/bin/env python3
"""Thin CLI over the ComfyUI bridge (Components/ComfyUIBackground.py).

Subcommands:
  status      Ping the running ComfyUI server and report version.
  background  Text->image sermon background (auto-wired into thumbnails).
  music       Text->music instrumental generation with ACE-Step 1.5.
  edit        img2img restyle/edit of an existing frame.
  run         Run any API-format workflow JSON, with optional node overrides.
  list        Scan a folder and report which workflows are API vs UI format.

The bridge talks to a running ComfyUI server (default http://127.0.0.1:8188).
Only API-format workflows (flat {node_id: {class_type, inputs}}) can run via
the server's /prompt endpoint; UI-graph exports cannot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Components.ComfyUIBackground import (
    check_server,
    detect_workflow_format,
    generate_background_image,
    generate_edited_image,
    run_api_workflow,
)
from Components.ComfyUIMusic import generate_music


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _apply_overrides(workflow: dict, overrides: list[str]) -> None:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node_id, _, field = key.partition(".")
        if not (node_id and field and _):
            raise SystemExit(f"--set expects NODE.input=value, got: {item!r}")
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise SystemExit(f"--set node {node_id!r} not found in workflow")
        node.setdefault("inputs", {})[field] = _coerce(raw)


def _cmd_status(args) -> int:
    info = check_server(args.base_url)
    _emit(info)
    return 0 if info.get("up") else 1


def _cmd_background(args) -> int:
    image, info = generate_background_image(
        title=args.title,
        template=args.template,
        speaker_name=args.speaker,
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        comfyui_root=args.comfyui_root,
        base_url=args.base_url,
    )
    image.convert("RGB").save(args.out)
    _emit({"out": args.out, **info})
    return 0


def _cmd_edit(args) -> int:
    image, info = generate_edited_image(
        input_image=args.input,
        prompt=args.prompt,
        negative_prompt=args.negative,
        denoise=args.denoise,
        steps=args.steps,
        cfg=args.cfg,
        base_url=args.base_url,
    )
    image.convert("RGB").save(args.out)
    _emit({"out": args.out, **info})
    return 0


def _cmd_music(args) -> int:
    output, info = generate_music(
        tags=args.prompt,
        output_path=args.out,
        duration_sec=args.duration,
        bpm=args.bpm,
        seed=args.seed,
        keyscale=args.key,
        language=args.language,
        lyrics=args.lyrics,
        generate_audio_codes=args.audio_codes,
        prepare_loop=not args.no_loop_prep,
        crossfade_sec=args.crossfade,
        base_url=args.base_url,
        timeout_sec=args.timeout,
    )
    _emit({"out": str(output), **info})
    return 0


def _cmd_run(args) -> int:
    workflow_path = Path(args.workflow).expanduser()
    fmt = detect_workflow_format(workflow_path)
    if fmt != "api":
        raise SystemExit(
            f"{workflow_path} is {fmt}-format; /prompt needs API format. "
            "In the ComfyUI UI use 'Save (API Format)' to export a runnable copy."
        )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    _apply_overrides(workflow, args.set)
    image, info = run_api_workflow(workflow, base_url=args.base_url, timeout_sec=args.timeout)
    image.convert("RGB").save(args.out)
    _emit({"out": args.out, "workflow": str(workflow_path), **info})
    return 0


def _cmd_list(args) -> int:
    root = Path(args.dir).expanduser()
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")
    rows = []
    for path in sorted(root.glob("*.json")):
        rows.append({"file": path.name, "format": detect_workflow_format(path)})
    api = [r["file"] for r in rows if r["format"] == "api"]
    _emit({"dir": str(root), "total": len(rows), "api_runnable": api, "workflows": rows})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ComfyUI bridge CLI for parakeet_uv")
    parser.add_argument("--base-url", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Ping the ComfyUI server")
    p_status.set_defaults(func=_cmd_status)

    p_bg = sub.add_parser("background", help="Text->image sermon background")
    p_bg.add_argument("--title", required=True)
    p_bg.add_argument("--template", default="bold_minimal")
    p_bg.add_argument("--speaker", default=None)
    p_bg.add_argument("--prompt", default=None, help="Optional explicit positive prompt for the background image")
    p_bg.add_argument("--negative", default=None, help="Optional explicit negative prompt for the background image")
    p_bg.add_argument("--width", type=int, default=832)
    p_bg.add_argument("--height", type=int, default=1472)
    p_bg.add_argument("--comfyui-root", default=None)
    p_bg.add_argument("--out", required=True)
    p_bg.set_defaults(func=_cmd_background)

    p_edit = sub.add_parser("edit", help="img2img restyle of an existing frame")
    p_edit.add_argument("--input", required=True)
    p_edit.add_argument("--prompt", required=True)
    p_edit.add_argument("--negative", default=None)
    p_edit.add_argument("--denoise", type=float, default=0.6)
    p_edit.add_argument("--steps", type=int, default=6)
    p_edit.add_argument("--cfg", type=float, default=1.0)
    p_edit.add_argument("--out", required=True)
    p_edit.set_defaults(func=_cmd_edit)

    p_music = sub.add_parser("music", help="Generate instrumental music with ACE-Step 1.5")
    p_music.add_argument("--prompt", required=True, help="Genre, instruments, mood and mix description")
    p_music.add_argument("--duration", type=float, default=45.0)
    p_music.add_argument("--bpm", type=int, default=78)
    p_music.add_argument("--key", default="D major")
    p_music.add_argument("--language", default="en")
    p_music.add_argument("--lyrics", default="[Instrumental]")
    p_music.add_argument(
        "--audio-codes",
        action="store_true",
        help="Enable LM-planned audio codes (better for structured songs than continuous beds)",
    )
    p_music.add_argument("--no-loop-prep", action="store_true", help="Keep the raw generated ending")
    p_music.add_argument("--crossfade", type=float, default=1.5, help="Loop crossfade in seconds")
    p_music.add_argument("--seed", type=int, default=None)
    p_music.add_argument("--timeout", type=int, default=600)
    p_music.add_argument("--out", required=True, help="Output .flac or .mp3 path")
    p_music.set_defaults(func=_cmd_music)

    p_run = sub.add_parser("run", help="Run an API-format workflow JSON")
    p_run.add_argument("--workflow", required=True)
    p_run.add_argument("--set", action="append", default=[], help="Override NODE.input=value (repeatable)")
    p_run.add_argument("--timeout", type=int, default=300)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="Report API vs UI format of workflows in a folder")
    p_list.add_argument("--dir", required=True)
    p_list.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
