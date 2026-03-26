import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from pytubefix import Playlist


def _configure_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


_configure_stdio()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_print(message: str):
    print(message, flush=True)


def load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {
            "version": 1,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "playlist_url": "",
            "videos": {},
        }
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state_path: Path, state: Dict):
    state["updated_at"] = now_iso()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def build_default_state_path(repo_root: Path, playlist: Playlist) -> Path:
    playlist_id = getattr(playlist, "playlist_id", None) or "playlist"
    return repo_root / "automation" / f"playlist_{playlist_id}.json"


def fetch_playlist_urls(playlist_url: str) -> List[str]:
    playlist = Playlist(playlist_url)
    urls = list(playlist.video_urls)
    safe_print(
        f"Loaded playlist: {playlist.title or playlist.playlist_id} ({len(urls)} videos)"
    )
    return urls


def should_skip_video(state: Dict, url: str, retry_failures: bool) -> bool:
    entry = state["videos"].get(url)
    if not entry:
        return False
    if entry.get("status") == "success":
        return True
    if entry.get("status") == "failed" and not retry_failures:
        return True
    return False


def mark_video(state: Dict, url: str, status: str, **extra):
    entry = state["videos"].get(url, {})
    entry.update(extra)
    entry["status"] = status
    entry["updated_at"] = now_iso()
    state["videos"][url] = entry


def run_video(repo_root: Path, python_exe: str, url: str, auto_approve: bool, log_path: Path) -> int:
    command = [python_exe, str(repo_root / "main.py"), url]
    if auto_approve:
        command.append("--auto-approve")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    safe_print(f"Processing {url}")
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{now_iso()}] START {url}\n")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        process.wait()
        log_handle.write(f"[{now_iso()}] END {url} returncode={process.returncode}\n")
        return process.returncode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run main.py over every video in a YouTube playlist."
    )
    parser.add_argument("playlist_url", help="YouTube playlist URL")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Pass --auto-approve through to main.py",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry videos that failed in a previous run",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based starting index inside the playlist. Default: 0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of playlist entries to process. Default: 0 (all)",
    )
    parser.add_argument(
        "--sleep-between-videos",
        type=float,
        default=2.0,
        help="Delay between playlist items in seconds. Default: 2.0",
    )
    parser.add_argument(
        "--state-file",
        default="",
        help="Optional JSON state file for resume/dedupe",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file for subprocess output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List playlist URLs without processing them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    playlist = Playlist(args.playlist_url)
    urls = list(playlist.video_urls)
    playlist_title = playlist.title or playlist.playlist_id or "playlist"
    safe_print(f"Loaded playlist: {playlist_title} ({len(urls)} videos)")

    if args.start_index < 0 or args.start_index >= len(urls):
        raise SystemExit(
            f"--start-index {args.start_index} is out of range for {len(urls)} playlist items"
        )

    selected_urls = urls[args.start_index:]
    if args.limit > 0:
        selected_urls = selected_urls[:args.limit]

    state_path = Path(args.state_file) if args.state_file else build_default_state_path(repo_root, playlist)
    log_path = Path(args.log_file) if args.log_file else repo_root / "automation" / f"playlist_{playlist.playlist_id or 'playlist'}.log"
    if not log_path.is_absolute():
        log_path = repo_root / log_path
    if not state_path.is_absolute():
        state_path = repo_root / state_path

    state = load_state(state_path)
    state["playlist_url"] = args.playlist_url
    save_state(state_path, state)

    if args.dry_run:
        for index, url in enumerate(selected_urls, start=args.start_index):
            safe_print(f"{index:03d}. {url}")
        return 0

    successes = 0
    failures = 0
    skipped = 0

    for index, url in enumerate(selected_urls, start=args.start_index):
        if should_skip_video(state, url, args.retry_failures):
            skipped += 1
            safe_print(f"Skipping previously processed item {index}: {url}")
            continue

        mark_video(state, url, "running", index=index)
        save_state(state_path, state)

        return_code = run_video(repo_root, python_exe, url, args.auto_approve, log_path)
        if return_code == 0:
            successes += 1
            mark_video(state, url, "success", index=index, returncode=return_code)
        else:
            failures += 1
            mark_video(state, url, "failed", index=index, returncode=return_code)

        save_state(state_path, state)
        safe_print(
            f"Progress: success={successes} failed={failures} skipped={skipped}"
        )
        time.sleep(args.sleep_between_videos)

    safe_print(
        f"Finished playlist run: success={successes} failed={failures} skipped={skipped}"
    )
    safe_print(f"State file: {state_path}")
    safe_print(f"Log file: {log_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
