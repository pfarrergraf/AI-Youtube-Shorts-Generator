"""Persistent history of downloaded videos and created shorts.

Stores a JSON file at ``automation/history.json`` relative to the repo root.
Each entry is keyed by YouTube video ID (or local file path hash for local
files) and tracks download status, short creation status, and timestamps.
"""

import json
import os
import hashlib
import re
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_DIR = os.path.join(_REPO_ROOT, "automation")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _extract_video_id(url_or_path):
    """Extract a YouTube video ID from a URL, or hash a local path."""
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url_or_path)
    if m:
        return m.group(1)
    # Local file — use a stable hash of the absolute path
    abs_path = os.path.abspath(url_or_path)
    return "local_" + hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]


def load_history():
    """Load history from disk. Returns a dict."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):
    """Write history to disk."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def is_downloaded(url_or_path):
    """Return True if this video was already downloaded."""
    vid_id = _extract_video_id(url_or_path)
    history = load_history()
    entry = history.get(vid_id)
    return entry is not None and entry.get("downloaded", False)


def is_short_created(url_or_path):
    """Return True if a short was already created from this video."""
    vid_id = _extract_video_id(url_or_path)
    history = load_history()
    entry = history.get(vid_id)
    return entry is not None and entry.get("short_created", False)


def mark_downloaded(url_or_path, title=None, local_path=None):
    """Record that a video was downloaded."""
    vid_id = _extract_video_id(url_or_path)
    history = load_history()
    entry = history.get(vid_id, {})
    entry["downloaded"] = True
    entry["download_time"] = _now_iso()
    if title:
        entry["title"] = title
    if local_path:
        entry["local_path"] = local_path
    entry["source"] = url_or_path
    history[vid_id] = entry
    save_history(history)


def mark_short_created(url_or_path, output_path=None):
    """Record that a short was created from this video.  Supports multiple
    shorts per video — each output_path is appended to a list."""
    vid_id = _extract_video_id(url_or_path)
    history = load_history()
    entry = history.get(vid_id, {})
    entry["short_created"] = True
    entry["short_time"] = _now_iso()
    # Track all created shorts (not just the last one)
    shorts_list = entry.get("shorts", [])
    if output_path and output_path not in shorts_list:
        shorts_list.append(output_path)
    entry["shorts"] = shorts_list
    if output_path:
        entry["short_path"] = output_path
    entry["short_count"] = len(shorts_list)
    history[vid_id] = entry
    save_history(history)


def mark_failed(url_or_path, error=None):
    """Record that processing failed for this video."""
    vid_id = _extract_video_id(url_or_path)
    history = load_history()
    entry = history.get(vid_id, {})
    entry["last_status"] = "failed"
    entry["fail_time"] = _now_iso()
    if error:
        entry["last_error"] = str(error)[:500]
    history[vid_id] = entry
    save_history(history)


def get_stats():
    """Return summary stats."""
    history = load_history()
    total = len(history)
    downloaded = sum(1 for e in history.values() if e.get("downloaded"))
    shorts = sum(1 for e in history.values() if e.get("short_created"))
    failed = sum(1 for e in history.values() if e.get("last_status") == "failed")
    return {"total": total, "downloaded": downloaded, "shorts_created": shorts, "failed": failed}
