"""Search YouTube and process top results through the shorts pipeline.

Usage:
    .venv\\Scripts\\python.exe search_runner.py "funny sermon moments" --limit 100 --auto-approve
    .venv\\Scripts\\python.exe search_runner.py "hilarious pastor comedy" --limit 50 --min-length 120 --max-length 3600
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from pytubefix import Search

from Components.History import is_short_created, is_downloaded, get_stats


def _configure_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


_configure_stdio()


def safe_print(msg):
    print(msg, flush=True)


def collect_search_results(query, limit, min_length, max_length):
    """Search YouTube with pagination, return up to *limit* unique video URLs.
    
    *query* can contain multiple queries separated by ``|`` — results from all
    queries are merged and deduplicated.
    """
    queries = [q.strip() for q in query.split("|") if q.strip()]
    seen = set()
    results = []

    for q in queries:
        if len(results) >= limit:
            break
        safe_print(f"Searching YouTube: \"{q}\"")
        s = Search(q)

        def add_batch(videos):
            for v in videos:
                if len(results) >= limit:
                    return
                try:
                    url = v.watch_url
                except Exception:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                # v.length triggers a full metadata fetch that can raise
                # BotDetection — skip length filtering at search time.
                # main.py will handle videos of any length.
                try:
                    title = v.title
                except Exception:
                    title = url
                try:
                    length = v.length or 0
                except Exception:
                    length = 0
                results.append({"url": url, "title": title, "length": length})

        try:
            add_batch(s.videos)
        except Exception as e:
            safe_print(f"  Search failed for \"{q}\": {e}")
            continue

        # Paginate until we have enough
        max_pages = (limit // 15) + 5
        for _ in range(max_pages):
            if len(results) >= limit:
                break
            try:
                s.get_next_results()
                add_batch(s.videos)
            except Exception:
                break

    safe_print(f"Found {len(results)} videos matching criteria")
    return results


def run_video(repo_root, python_exe, url, auto_approve):
    """Run main.py for a single video URL. Returns exit code."""
    cmd = [python_exe, str(repo_root / "main.py"), url]
    if auto_approve:
        cmd.append("--auto-approve")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()
    return process.returncode


def parse_args():
    p = argparse.ArgumentParser(description="Search YouTube and create shorts from results.")
    p.add_argument("query", help="YouTube search query")
    p.add_argument("--limit", type=int, default=100, help="Max videos to process (default: 100)")
    p.add_argument("--min-length", type=int, default=60, help="Min video length in seconds (default: 60)")
    p.add_argument("--max-length", type=int, default=7200, help="Max video length in seconds (default: 7200)")
    p.add_argument("--auto-approve", action="store_true", help="Auto-approve highlight selection")
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds between videos (default: 2)")
    p.add_argument("--skip-downloaded", action="store_true", help="Skip videos already in download history")
    p.add_argument("--dry-run", action="store_true", help="List search results without processing")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    results = collect_search_results(args.query, args.limit, args.min_length, args.max_length)

    if not results:
        safe_print("No results found.")
        return 1

    # Filter already-processed
    to_process = []
    skipped = 0
    for r in results:
        url = r["url"]
        if is_short_created(url):
            skipped += 1
            continue
        if args.skip_downloaded and is_downloaded(url):
            skipped += 1
            continue
        to_process.append(r)

    if skipped:
        safe_print(f"Skipping {skipped} already-processed videos")

    safe_print(f"\nWill process {len(to_process)} videos\n")

    if args.dry_run:
        for i, r in enumerate(to_process):
            mins = r["length"] // 60
            secs = r["length"] % 60
            safe_print(f"  {i+1:3d}. [{mins}:{secs:02d}] {r['title']}")
            safe_print(f"       {r['url']}")
        stats = get_stats()
        safe_print(f"\nHistory: {stats['downloaded']} downloaded, {stats['shorts_created']} shorts, {stats['failed']} failed")
        return 0

    successes = 0
    failures = 0

    for i, r in enumerate(to_process):
        safe_print(f"\n{'='*60}")
        safe_print(f"[{i+1}/{len(to_process)}] {r['title']}")
        safe_print(f"  {r['url']}  ({r['length']}s)")
        safe_print(f"{'='*60}")

        try:
            rc = run_video(repo_root, python_exe, r["url"], args.auto_approve)
            if rc == 0:
                successes += 1
            else:
                failures += 1
        except Exception as e:
            safe_print(f"ERROR: {e}")
            failures += 1

        safe_print(f"Progress: {successes} success, {failures} failed, {len(to_process)-i-1} remaining")

        if i < len(to_process) - 1:
            time.sleep(args.sleep)

    stats = get_stats()
    safe_print(f"\n{'='*60}")
    safe_print(f"SEARCH RUN COMPLETE")
    safe_print(f"  This run: {successes} success, {failures} failed")
    safe_print(f"  All-time: {stats['downloaded']} downloaded, {stats['shorts_created']} shorts")
    safe_print(f"{'='*60}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
