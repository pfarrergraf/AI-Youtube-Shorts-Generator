import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pytubefix import Search
from pytubefix.contrib.search import Filter


def _configure_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


_configure_stdio()


SERMON_PRESET = [
    "lustige predigt deutsch",
    "predigt humor deutsch",
    "christliche predigt deutsch live",
    "predigt gemeinde deutsch",
    "evangelium predigt deutsch",
    "bibelarbeit humorvoll deutsch",
]

FUNNY_PRESET = [
    "stand up comedy deutsch live",
    "kabarett deutsch live",
    "lustiger vortrag deutsch live",
    "lustige rede deutsch live",
    "comedy programm deutsch live",
    "humor buhne deutsch live",
]

PRESETS = {
    "sermons": SERMON_PRESET,
    "funny": FUNNY_PRESET,
    "mixed": SERMON_PRESET + FUNNY_PRESET,
}

SORT_OPTIONS = {
    "view_count": Filter.SortBy.VIEW_COUNT,
    "upload_date": Filter.SortBy.UPLOAD_DATE,
    "relevance": Filter.SortBy.RELEVANCE,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Candidate:
    video_id: str
    url: str
    title: str
    author: str
    length_seconds: int
    views: int
    publish_date: Optional[str]
    query: str
    sort_by: str


def _fmt_views(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _check_view_threshold(
    views: int,
    publish_date: Optional[datetime],
    min_views: int,
    trending_vph: int,
) -> Tuple[bool, str]:
    """Return (passes, reason) based on view count and trending rate."""
    if views >= min_views:
        return True, ""

    # Trending exception: high views-per-hour for recent uploads
    if publish_date and trending_vph > 0:
        try:
            now = datetime.now(timezone.utc)
            if publish_date.tzinfo is None:
                publish_date = publish_date.replace(tzinfo=timezone.utc)
            age_hours = max((now - publish_date).total_seconds() / 3600, 1)
            vph = views / age_hours
            if vph >= trending_vph:
                return True, f"trending ({vph:.0f} v/h)"
        except Exception:
            pass

    return False, f"< {_fmt_views(min_views)}"


def safe_print(message: str):
    print(message, flush=True)


def load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {
            "version": 1,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "videos": {},
        }
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state_path: Path, state: Dict):
    state["updated_at"] = now_iso()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def load_queries(args) -> List[str]:
    queries: List[str] = []

    if args.preset:
        queries.extend(PRESETS[args.preset])

    if args.query:
        queries.extend(args.query)

    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    queries.append(stripped)

    deduped: List[str] = []
    seen = set()
    for query in queries:
        normalized = query.casefold()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(query)

    return deduped


def make_search(sort_name: str, query: str) -> Search:
    filters = (
        Filter.create()
        .type(Filter.Type.VIDEO)
        .sort_by(SORT_OPTIONS[sort_name])
    )
    return Search(query, filters=filters)


def video_status_entry(state: Dict, video_id: str) -> Optional[Dict]:
    return state["videos"].get(video_id)


def should_skip_video(state: Dict, video_id: str, retry_failures: bool) -> bool:
    entry = video_status_entry(state, video_id)
    if not entry:
        return False
    if entry.get("status") == "success":
        return True
    if entry.get("status") == "failed" and not retry_failures:
        return True
    return False


def extract_candidates(args, state: Dict) -> List[Candidate]:
    queries = load_queries(args)
    if not queries:
        raise ValueError("No queries configured. Use --preset, --query, or --query-file.")

    safe_print(f"Loaded {len(queries)} search queries")

    desired_candidates = max(args.target_count * args.candidate_pool_factor, args.target_count)
    candidates: List[Candidate] = []
    queued_ids = set()

    for query in queries:
        for sort_name in args.sorts:
            safe_print(f"Searching YouTube: query='{query}' sort='{sort_name}'")
            search = make_search(sort_name, query)
            fetches = 0

            while True:
                if fetches == 0:
                    videos = search.videos
                else:
                    search.get_next_results()
                    videos = search.videos
                fetches += 1

                current_batch = videos[-args.results_per_fetch:]
                if not current_batch:
                    break

                for video in current_batch:
                    video_id = getattr(video, "video_id", None)
                    if not video_id:
                        continue
                    if video_id in queued_ids:
                        continue
                    if should_skip_video(state, video_id, args.retry_failures):
                        continue

                    try:
                        length_seconds = int(getattr(video, "length", 0) or 0)
                    except Exception:
                        length_seconds = 0
                    if length_seconds < args.min_duration:
                        continue
                    if args.max_duration and length_seconds > args.max_duration:
                        continue

                    title = getattr(video, "title", "").strip()
                    author = getattr(video, "author", "").strip()
                    title_lc = title.casefold()
                    if any(marker in title_lc for marker in ("shorts", "#shorts", "clip", "best of shorts")):
                        continue

                    try:
                        view_count = int(getattr(video, "views", 0) or 0)
                    except Exception:
                        view_count = 0

                    try:
                        pub_date = getattr(video, "publish_date", None)
                        pub_date_str = pub_date.isoformat() if pub_date else None
                    except Exception:
                        pub_date = None
                        pub_date_str = None

                    passes, reason = _check_view_threshold(
                        view_count, pub_date, args.min_views, args.trending_vph,
                    )
                    if not passes:
                        safe_print(
                            f"  skipped (views): {title[:60]} "
                            f"[{_fmt_views(view_count)} views] {reason}"
                        )
                        continue

                    candidate = Candidate(
                        video_id=video_id,
                        url=getattr(video, "watch_url", f"https://youtube.com/watch?v={video_id}"),
                        title=title,
                        author=author,
                        length_seconds=length_seconds,
                        views=view_count,
                        publish_date=pub_date_str,
                        query=query,
                        sort_by=sort_name,
                    )
                    candidates.append(candidate)
                    queued_ids.add(video_id)

                    safe_print(
                        f"  queued {len(candidates)}/{desired_candidates}: "
                        f"{candidate.title[:60]} [{candidate.length_seconds}s, "
                        f"{_fmt_views(candidate.views)} views]"
                    )
                    if len(candidates) >= desired_candidates:
                        return candidates

                if fetches >= args.max_fetches_per_query:
                    break
                if len(current_batch) < args.results_per_fetch:
                    break

                time.sleep(args.search_sleep_seconds)

    return candidates


def mark_video(state: Dict, candidate: Candidate, status: str, **extra):
    entry = asdict(candidate)
    entry["status"] = status
    entry["updated_at"] = now_iso()
    entry.update(extra)
    state["videos"][candidate.video_id] = entry


def run_candidate(repo_root: Path, candidate: Candidate, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    command = [
        sys.executable,
        str(repo_root / "main.py"),
        candidate.url,
        "--auto-approve",
    ]

    safe_print(f"Processing {candidate.url}")
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[{now_iso()}] START {candidate.video_id} {candidate.url} {candidate.title}\n"
        )
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
        log_handle.write(
            f"[{now_iso()}] END {candidate.video_id} returncode={process.returncode}\n"
        )
        return process.returncode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search YouTube and run main.py on discovered long-form videos all night."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="mixed",
        help="Built-in search preset. Default: mixed",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Additional search query. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--query-file",
        help="Path to a UTF-8 text file with one search query per line.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Stop after this many successful shorts. Default: 100",
    )
    parser.add_argument(
        "--candidate-pool-factor",
        type=int,
        default=3,
        help="Discover this many times more candidates than the success target. Default: 3",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=8 * 60,
        help="Minimum source video length in seconds. Default: 480",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=3 * 60 * 60,
        help="Maximum source video length in seconds. Default: 10800",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=200_000,
        help="Minimum view count to accept a video. Default: 200000",
    )
    parser.add_argument(
        "--trending-vph",
        type=int,
        default=100,
        help="Views-per-hour threshold for recent uploads to bypass --min-views. Default: 100",
    )
    parser.add_argument(
        "--results-per-fetch",
        type=int,
        default=20,
        help="How many search results to inspect per fetch page. Default: 20",
    )
    parser.add_argument(
        "--max-fetches-per-query",
        type=int,
        default=4,
        help="How many search result pages to fetch per query/sort pair. Default: 4",
    )
    parser.add_argument(
        "--sorts",
        nargs="+",
        choices=sorted(SORT_OPTIONS.keys()),
        default=["view_count", "upload_date"],
        help="Search sorts to combine. Default: view_count upload_date",
    )
    parser.add_argument(
        "--state-file",
        default=str(Path("automation") / "overnight_state.json"),
        help="JSON state file used for dedupe and resume.",
    )
    parser.add_argument(
        "--log-file",
        default=str(Path("automation") / "overnight_run.log"),
        help="Log file for subprocess output.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry videos that previously failed.",
    )
    parser.add_argument(
        "--search-sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between paginated search fetches. Default: 1.0",
    )
    parser.add_argument(
        "--sleep-between-videos",
        type=float,
        default=2.0,
        help="Delay between processed videos. Default: 2.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover candidates and print them without running main.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    state_path = repo_root / args.state_file
    log_path = repo_root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    candidates = extract_candidates(args, state)
    safe_print(f"Discovered {len(candidates)} candidates")

    if args.dry_run:
        for index, candidate in enumerate(candidates, start=1):
            safe_print(
                f"{index:03d}. {candidate.url} | {candidate.length_seconds}s | "
                f"{candidate.author} | {candidate.title}"
            )
        return 0

    successes = 0
    failures = 0

    for candidate in candidates:
        if successes >= args.target_count:
            break

        mark_video(state, candidate, "running")
        save_state(state_path, state)

        return_code = run_candidate(repo_root, candidate, log_path)
        if return_code == 0:
            successes += 1
            mark_video(state, candidate, "success", returncode=return_code)
        else:
            failures += 1
            mark_video(state, candidate, "failed", returncode=return_code)

        save_state(state_path, state)
        safe_print(
            f"Progress: success={successes} failed={failures} target={args.target_count}"
        )
        time.sleep(args.sleep_between_videos)

    safe_print(
        f"Finished overnight run: success={successes} failed={failures} "
        f"log={log_path} state={state_path}"
    )
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
