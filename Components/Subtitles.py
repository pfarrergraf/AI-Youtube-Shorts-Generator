import os
import shutil
import subprocess
import tempfile

import cv2


MAX_WORDS_PER_CAPTION = 4
MAX_LINES_PER_CAPTION = 2

NVENC_FLAGS = [
    "-c:v",
    "h264_nvenc",
    "-preset",
    "p7",
    "-rc",
    "constqp",
    "-qp",
    "18",
    "-b:v",
    "0",
    "-gpu",
    "0",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
]


def _run_ffmpeg(command, description, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"{description} failed: {stderr}")


def _seconds_to_ass_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _escape_ass_text(text):
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _format_caption_lines(words):
    if len(words) <= 3 or MAX_LINES_PER_CAPTION == 1:
        return " ".join(words)

    split_index = min(3, max(2, (len(words) + 1) // 2))
    first_line = " ".join(words[:split_index])
    second_line = " ".join(words[split_index:])

    if not second_line:
        return first_line
    return f"{first_line}\n{second_line}"


def _chunk_transcriptions(transcriptions):
    chunked = []

    for text, start, end in transcriptions:
        words = text.split()
        if not words:
            continue

        total_words = len(words)
        duration = max(0.01, end - start)

        for index in range(0, total_words, MAX_WORDS_PER_CAPTION):
            chunk_words = words[index:index + MAX_WORDS_PER_CAPTION]
            chunk_start = start + (index / total_words) * duration
            chunk_end = start + (
                min(index + MAX_WORDS_PER_CAPTION, total_words) / total_words
            ) * duration
            chunked.append(
                [_format_caption_lines(chunk_words), chunk_start, chunk_end]
            )

    return chunked


def _read_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for subtitle burn: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    return width, height, duration


def _write_ass_file(subtitle_path, video_width, video_height, chunks):
    font_size = max(28, int(video_height * 0.038))
    margin_v = max(80, int(video_height * 0.12))
    margin_h = max(50, int(video_width * 0.10))
    outline = max(2, int(video_height * 0.003))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Default,Arial,"
            f"{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
            f"1,0,0,0,100,100,0,0,1,{outline},0,2,{margin_h},{margin_h},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for text, start, end in chunks:
        safe_text = _escape_ass_text(text.strip())
        if not safe_text:
            continue
        lines.append(
            "Dialogue: 0,"
            f"{_seconds_to_ass_time(start)},"
            f"{_seconds_to_ass_time(end)},"
            f"Default,,0,0,0,,{safe_text}"
        )

    with open(subtitle_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def add_subtitles_to_video(input_video, output_video, transcriptions, video_start_time=0):
    """
    Add subtitles to video based on transcription segments.

    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        transcriptions: List of [text, start, end] from transcribeAudio
        video_start_time: Start time offset if video was cropped
    """
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)

    video_width, video_height, video_duration = _read_video_metadata(input_video)

    relevant_transcriptions = []
    for text, start, end in transcriptions:
        adjusted_start = start - video_start_time
        adjusted_end = end - video_start_time

        if adjusted_end <= 0:
            continue
        if video_duration > 0 and adjusted_start >= video_duration:
            continue

        adjusted_start = max(0, adjusted_start)
        if video_duration > 0:
            adjusted_end = min(video_duration, adjusted_end)

        stripped_text = text.strip()
        if stripped_text and not stripped_text.startswith("["):
            relevant_transcriptions.append([stripped_text, adjusted_start, adjusted_end])

    if not relevant_transcriptions:
        print("No transcriptions found for this video segment")
        shutil.copyfile(input_video, output_video)
        return

    chunked = _chunk_transcriptions(relevant_transcriptions)
    if not chunked:
        print("No subtitle chunks generated for this video segment")
        shutil.copyfile(input_video, output_video)
        return

    subtitle_dir = os.path.dirname(output_video) or os.getcwd()
    subtitle_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ass",
            prefix="captions_",
            dir=subtitle_dir,
            delete=False,
        ) as handle:
            subtitle_path = handle.name

        _write_ass_file(subtitle_path, video_width, video_height, chunked)

        print(
            f"Adding {len(chunked)} subtitle segments to video with FFmpeg NVENC..."
        )
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-vf",
            f"subtitles={os.path.basename(subtitle_path)}",
            "-an",
            *NVENC_FLAGS,
            output_video,
        ]
        _run_ffmpeg(command, "subtitle burn", cwd=subtitle_dir)
        print(f"Subtitles added successfully -> {output_video}")
    finally:
        if subtitle_path and os.path.exists(subtitle_path):
            os.remove(subtitle_path)
