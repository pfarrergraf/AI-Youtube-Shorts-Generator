#!/usr/bin/env python3
"""Create a review candidate with Azure Foundry GPT-image-2.

Examples:
  python tools/azure_image_hero.py --speaker olaf_latzel --prompt-file prompt.txt
  python tools/azure_image_hero.py --speaker antonio_weil --prompt "..." --no-references
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Components.AzureImageGeneration import generate_image, speaker_source_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, help="manifest key, e.g. olaf_latzel")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", action="append", type=Path, help="override identity source; repeatable")
    parser.add_argument("--no-references", action="store_true")
    parser.add_argument("--size", default="1024x1536")
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("assets/speaker_references") / args.speaker / "jobs" / f"azure_{stamp}.png"
    references = [] if args.no_references else (args.source or speaker_source_paths(args.speaker))
    result = generate_image(
        prompt,
        output,
        reference_images=references,
        size=args.size,
        quality=args.quality,
    )
    print(f"Candidate written: {result}")
    print("Review it manually before copying it into heroes/ and marking approved in manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
