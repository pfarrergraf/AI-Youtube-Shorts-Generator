"""Run `main.py` for every URL listed in a UTF-8 text file (one URL per line).

Usage examples:
    .venv/Scripts/python.exe ./urls_runner.py --file urls.txt --auto-approve
    .venv/Scripts/python.exe ./urls_runner.py --file urls.txt --dry-run

The script is intentionally minimal: it calls `main.py` as a subprocess for
each URL in the file, streams output, and waits for each run to finish before
continuing. Use `--sleep-between` to add a pause between runs.
"""
import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

# Force stdout to UTF-8 so we never crash on non-ASCII subprocess output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)


def read_url_file(path: Path):
    urls = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            urls.append(s)
    return urls


def run_url(url: str, auto_approve: bool, cwd: Path) -> int:
    command = [sys.executable, str(cwd / "main.py"), url]
    if auto_approve:
        command.append("--auto-approve")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    return proc.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run main.py for each URL in a text file.")
    parser.add_argument("--file", required=True, help="Path to UTF-8 text file with one URL per line.")
    parser.add_argument("--auto-approve", action="store_true", help="Pass --auto-approve to main.py")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs and exit")
    parser.add_argument("--sleep-between", type=float, default=2.0, help="Seconds to sleep between runs")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"URL file not found: {path}")
        return 2

    urls = read_url_file(path)
    if not urls:
        print("No URLs found in file.")
        return 0

    if args.dry_run:
        for i, u in enumerate(urls, start=1):
            print(f"{i:03d}. {u}")
        return 0

    repo_root = Path(__file__).resolve().parent
    for i, u in enumerate(urls, start=1):
        print(f"\n==> ({i}/{len(urls)}) Processing: {u}\n", flush=True)
        rc = run_url(u, args.auto_approve, repo_root)
        print(f"==> returncode={rc}\n", flush=True)
        time.sleep(args.sleep_between)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
