import subprocess

from moviepy.editor import VideoFileClip


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


def _run_ffmpeg(command, description):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"{description} failed: {stderr}")


def extractAudio(video_path, audio_path="audio.wav"):
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        print(f"Extracted audio to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
        return None


def crop_video(input_file, output_file, start_time, end_time):
    with VideoFileClip(input_file) as video:
        max_time = video.duration - 0.1
        if end_time > max_time:
            print(
                f"Warning: Requested end time ({end_time}s) exceeds video duration "
                f"({video.duration}s). Capping to {max_time}s"
            )
            end_time = max_time

    if end_time <= start_time:
        raise ValueError(
            f"Invalid crop range: start_time={start_time}, end_time={end_time}"
        )

    duration = end_time - start_time
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        input_file,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *NVENC_FLAGS,
        "-c:a",
        "aac",
        output_file,
    ]
    print("  Trimming clip with FFmpeg NVENC...")
    _run_ffmpeg(command, "clip extraction")


def crop_video_with_cuts(input_file, output_file, start_time, end_time, cuts):
    # In-between cut removal is intentionally disabled. The selected highlight
    # should stay continuous from start to end so no important setup or payoff
    # is removed by aggressive jump cuts.
    if cuts:
        print(
            "  In-between cut-outs are disabled; keeping the full continuous "
            "highlight range."
        )
    crop_video(input_file, output_file, start_time, end_time)
    return [(start_time, end_time)]


if __name__ == "__main__":
    input_file = r"Example.mp4"
    output_file = "Short.mp4"
    start_time = 31.92
    end_time = 49.2

    crop_video(input_file, output_file, start_time, end_time)
