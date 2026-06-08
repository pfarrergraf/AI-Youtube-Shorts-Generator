"""
YoutubeDownloader — downloads YouTube videos as H.264/MP4.

Uses yt-dlp when available (preferred: faster, more reliable, H.264 codec selection).
Falls back to pytubefix + ffmpeg re-encode when yt-dlp is not installed.

Why H.264?  YouTube now serves AV1 as "highest quality", but AV1 is not
supported by OpenCV's bundled decoder. H.264 works everywhere.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path


# ── yt-dlp format selector: prefer H.264 (avc1), fall back to any mp4 ──────
_YTDLP_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]"
    "+bestaudio[ext=m4a]"
    "/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
    "/bestvideo[height<=1080]+bestaudio"
    "/best[height<=1080]"
)


def _sanitize(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|｜]', "-", title).strip()


def download_youtube_video(
    url: str,
    output_dir: str = "videos",
    prefer_h264: bool = True,
) -> str | None:
    """
    Download a YouTube video to output_dir.
    Returns the output file path, or None on failure.

    Args:
        url:          YouTube URL
        output_dir:   Directory to save the file
        prefer_h264:  Force H.264 codec (avoids AV1/VP9 which OpenCV can't decode)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Try yt-dlp first ────────────────────────────────────────────────────
    if shutil.which("yt-dlp"):
        fmt = _YTDLP_FORMAT if prefer_h264 else "bestvideo+bestaudio/best"
        out_template = str(Path(output_dir) / "%(title)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--format", fmt,
            "--merge-output-format", "mp4",
            "--output", out_template,
            "--no-playlist",
            "--print", "after_move:filepath",
            url,
        ]
        print(f"[YoutubeDownloader] yt-dlp downloading: {url}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            # Find the most recently modified mp4 in output_dir
            mp4s = sorted(Path(output_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4s:
                print(f"[YoutubeDownloader] Saved: {mp4s[0]}")
                return str(mp4s[0])
        print("[YoutubeDownloader] yt-dlp failed, trying pytubefix fallback.")

    # ── Fallback: pytubefix + ffmpeg re-encode to H.264 ────────────────────
    try:
        from pytubefix import YouTube
        import ffmpeg as ffmpeg_python

        yt = YouTube(url)
        safe_title = _sanitize(yt.title)
        print(f"\n[YoutubeDownloader] pytubefix: {yt.title}")

        video_streams = yt.streams.filter(type="video").order_by("resolution").desc()
        audio_stream  = yt.streams.filter(only_audio=True).first()

        # Prefer H.264 streams (codec starts with "avc")
        h264_streams = [s for s in video_streams if s.codecs and s.codecs[0].startswith("avc")]
        selected = h264_streams[0] if h264_streams else video_streams[0]

        print(f"  Video: {selected.resolution} | codec: {selected.codecs}")

        video_file = selected.download(output_path=output_dir, filename_prefix="video_")

        if not selected.is_progressive:
            audio_file  = audio_stream.download(output_path=output_dir, filename_prefix="audio_")
            output_file = str(Path(output_dir) / f"{safe_title}.mp4")
            v = ffmpeg_python.input(video_file)
            a = ffmpeg_python.input(audio_file)
            ffmpeg_python.output(
                v, a, output_file,
                vcodec="libx264", acodec="aac",
                strict="experimental",
            ).run(overwrite_output=True, quiet=True)
            os.remove(video_file)
            os.remove(audio_file)
        else:
            output_file = video_file

        print(f"[YoutubeDownloader] Saved: {output_file}")
        return output_file

    except Exception as exc:
        print(f"[YoutubeDownloader] Error: {exc}")
        print("Install yt-dlp for best results:  uv pip install yt-dlp")
        return None


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else input("YouTube URL: ")
    out = sys.argv[2] if len(sys.argv) > 2 else "videos"
    download_youtube_video(url, output_dir=out)
