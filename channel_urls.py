"""Export all public YouTube channel watch URLs to a UTF-8 text file.

Usage examples:
    .venv\Scripts\python.exe .\channel_urls.py "https://www.youtube.com/@olaflatzel/videos"
    .venv\Scripts\python.exe .\channel_urls.py "https://www.youtube.com/@olaflatzel" --output .\latzel_urls.txt
    .venv\Scripts\python.exe .\channel_urls.py "https://www.youtube.com/@olaflatzel" --tabs videos --dry-run
"""

import argparse
import io
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from pytubefix import Channel


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


TAB_LOADERS = {
    "videos": lambda channel: channel.videos,
    "streams": lambda channel: channel.live,
    "shorts": lambda channel: channel.shorts,
}


def safe_print(message: str):
    print(message, flush=True)


def sanitize_filename(value: str) -> str:
    value = value.strip().lstrip("@")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "channel"


def build_default_output_path(repo_root: Path, channel: Channel) -> Path:
    channel_slug = (channel.channel_uri or "").strip("/").split("/")[-1]
    return repo_root / f"{sanitize_filename(channel_slug)}_urls.txt"


def collect_tab_urls(channel_url: str, tab: str) -> List[str]:
    channel = Channel(channel_url)
    items: Iterable = TAB_LOADERS[tab](channel)
    urls: List[str] = []
    seen = set()
    for item in items:
        watch_url = getattr(item, "watch_url", None)
        if not watch_url or watch_url in seen:
            continue
        seen.add(watch_url)
        urls.append(watch_url)
    return urls


def collect_channel_urls(channel_url: str, tabs: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    merged_urls: List[str] = []
    global_seen = set()
    counts: Dict[str, int] = {}

    for tab in tabs:
        safe_print(f"Loading channel tab: {tab}")
        tab_urls = collect_tab_urls(channel_url, tab)
        counts[tab] = len(tab_urls)
        safe_print(f"Found {len(tab_urls)} URLs in {tab}")
        for url in tab_urls:
            if url in global_seen:
                continue
            global_seen.add(url)
            merged_urls.append(url)

    return merged_urls, counts


def write_urls(path: Path, urls: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for url in urls:
            handle.write(f"{url}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all public watch URLs from a YouTube channel to a text file."
    )
    parser.add_argument(
        "channel_url",
        help="YouTube channel URL, handle URL, or a channel tab URL such as /videos",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .txt file path. Default: .\\<channel>_urls.txt",
    )
    parser.add_argument(
        "--tabs",
        nargs="+",
        choices=tuple(TAB_LOADERS.keys()),
        default=["videos", "streams", "shorts"],
        help="Public channel tabs to scan. Default: videos streams shorts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered URLs without writing a file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    channel = Channel(args.channel_url)
    output_path = Path(args.output) if args.output else build_default_output_path(repo_root, channel)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    urls, counts = collect_channel_urls(args.channel_url, args.tabs)
    safe_print(
        "Tab counts: " + ", ".join(f"{tab}={counts.get(tab, 0)}" for tab in args.tabs)
    )
    safe_print(f"Unique URLs across selected tabs: {len(urls)}")

    if args.dry_run:
        for index, url in enumerate(urls, start=1):
            safe_print(f"{index:04d}. {url}")
        return 0

    write_urls(output_path, urls)
    safe_print(f"Wrote {len(urls)} URLs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
