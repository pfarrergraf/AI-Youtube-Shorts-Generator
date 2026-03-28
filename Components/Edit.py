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


HWACCEL_FLAGS = [
    "-hwaccel",
    "cuda",
]


def _run_ffmpeg(command, description):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"{description} failed: {stderr}")


def _probe_input_video_codec(input_file):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip().lower()


def _build_crop_command(input_file, output_file, start_time, duration, *, use_hwaccel):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
    ]
    if use_hwaccel:
        command.extend(HWACCEL_FLAGS)
    command.extend(
        [
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
    )
    return command


def _is_hwaccel_decode_error(message):
    lowered = str(message).lower()
    needles = (
        "no device available for decoder",
        "hardware device setup failed for decoder",
        "device type cuda needed for codec",
        "failed setup for decoder",
        "hardware accelerated av1 decoding",
    )
    return any(needle in lowered for needle in needles)


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
    codec_name = _probe_input_video_codec(input_file)
    use_hwaccel = codec_name != "av1"
    command = _build_crop_command(
        input_file,
        output_file,
        start_time,
        duration,
        use_hwaccel=use_hwaccel,
    )
    print("  Trimming clip with FFmpeg NVENC...")
    if not use_hwaccel and codec_name:
        print(f"  Input codec {codec_name}: using software decode for clip extraction.")
    try:
        _run_ffmpeg(command, "clip extraction")
    except RuntimeError as exc:
        if use_hwaccel and _is_hwaccel_decode_error(exc):
            print("  CUDA decode unavailable for this input; retrying with software decode...")
            fallback_command = _build_crop_command(
                input_file,
                output_file,
                start_time,
                duration,
                use_hwaccel=False,
            )
            _run_ffmpeg(fallback_command, "clip extraction")
            return
        raise


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
