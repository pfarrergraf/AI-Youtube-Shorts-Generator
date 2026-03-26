import cv2
import numpy as np
import os
import random
import subprocess


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


def _start_nvenc_writer(output_video_path, width, height, fps):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        *NVENC_FLAGS,
        os.path.abspath(output_video_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_nvenc_writer(process, description):
    stderr = ""
    try:
        if process.stdin:
            process.stdin.close()
        if process.stderr:
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    finally:
        return_code = process.wait()
        if process.stderr:
            process.stderr.close()

    if return_code != 0:
        raise RuntimeError(f"{description} failed: {stderr}")


def _configure_dnn_backend(net):
    if not hasattr(cv2, "cuda"):
        print("Using DNN face detector on CPU (OpenCV CUDA module not present)")
        return

    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    except cv2.error:
        cuda_devices = 0

    if cuda_devices <= 0:
        print("Using DNN face detector on CPU (OpenCV build has no CUDA support)")
        return

    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("Using DNN face detector on CUDA")
    except cv2.error as exc:
        print(f"Using DNN face detector on CPU (CUDA backend unavailable: {exc})")


def _ease_in_out(t):
    """Smooth ease-in-out curve (Hermite / smoothstep)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _ease_out(t):
    """Deceleration curve — fast start, gentle stop."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# ======================================================================
# Camera effect types
# ======================================================================
# Each effect is (type, start_frame, end_frame, params_dict)
#
# Types:
#   "slow_push"   — cinematic slow zoom toward the face over several seconds
#   "slow_pull"   — slow zoom out back to normal over several seconds
#   "jump_close"  — instant jump cut to a close-up (held steady, no movement)
#   "jump_mid"    — instant jump cut to a mid-shot (moderate zoom, held steady)
#   "drift"       — slow horizontal Ken Burns drift with slight zoom
# ======================================================================

def _plan_camera_effects(total_frames, fps):
    """Plan a sequence of camera effects spread across the clip."""
    duration_sec = total_frames / fps
    effects = []

    # Minimum gap (in frames) between effects
    min_gap = int(fps * 2.0)
    # Don't place effects in first or last 1.5s
    margin = int(fps * 1.5)
    safe_start = margin
    safe_end = total_frames - margin

    if safe_end - safe_start < int(fps * 4):
        return []  # clip too short for effects

    # Build a pool of candidate effects
    candidates = []

    # --- Slow push-ins: 3-5 seconds, zoom 1.0 → 1.15–1.25 ---
    push_dur = random.uniform(3.0, 5.0)
    push_frames = int(push_dur * fps)
    candidates.append(("slow_push", push_frames, {"zoom_end": random.uniform(1.15, 1.25)}))

    # --- Slow pull-outs: 2-4 seconds, zoom 1.20 → 1.0 ---
    pull_dur = random.uniform(2.0, 4.0)
    pull_frames = int(pull_dur * fps)
    candidates.append(("slow_pull", pull_frames, {"zoom_start": random.uniform(1.15, 1.22)}))

    # --- Jump cut close-ups: instant zoom held for 2-4 seconds ---
    close_dur = random.uniform(2.0, 4.0)
    close_frames = int(close_dur * fps)
    candidates.append(("jump_close", close_frames, {"zoom": random.uniform(1.30, 1.50)}))

    # --- Jump cut mid-shots: instant moderate zoom held for 1.5-3 seconds ---
    mid_dur = random.uniform(1.5, 3.0)
    mid_frames = int(mid_dur * fps)
    candidates.append(("jump_mid", mid_frames, {"zoom": random.uniform(1.12, 1.22)}))

    # --- Ken Burns drift: slow horizontal pan + slight zoom, 4-6 seconds ---
    drift_dur = random.uniform(4.0, 6.0)
    drift_frames = int(drift_dur * fps)
    drift_dir = random.choice([-1, 1])
    candidates.append(("drift", drift_frames, {
        "zoom": random.uniform(1.05, 1.12),
        "drift_px": drift_dir * random.randint(20, 60),
    }))

    # For a second slow push (different params)
    push_dur2 = random.uniform(3.0, 4.5)
    push_frames2 = int(push_dur2 * fps)
    candidates.append(("slow_push", push_frames2, {"zoom_end": random.uniform(1.12, 1.20)}))

    # Shuffle and place without overlaps
    random.shuffle(candidates)

    # Determine how many effects to place (~1 every 6-10 seconds)
    max_effects = max(1, int(duration_sec / random.uniform(6, 10)))
    placed_ranges = []  # list of (start, end) already used

    for etype, edur, eparams in candidates:
        if len(effects) >= max_effects:
            break
        # Try a few random placements
        for _attempt in range(15):
            es = random.randint(safe_start, max(safe_start, safe_end - edur))
            ee = es + edur
            if ee > safe_end:
                continue
            # Check overlap with existing + min_gap
            overlap = False
            for ps, pe in placed_ranges:
                if not (ee + min_gap < ps or es > pe + min_gap):
                    overlap = True
                    break
            if not overlap:
                effects.append((etype, es, ee, eparams))
                placed_ranges.append((es, ee))
                break

    effects.sort(key=lambda e: e[1])
    return effects


def _apply_zoom_crop(frame, x_pos, zoom, vertical_width, vertical_height,
                     original_width, original_height, x_offset=0):
    """Crop a zoomed region centered on face position, return resized to output dims."""
    crop_w = int(vertical_width / zoom)
    crop_h = int(vertical_height / zoom)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    zx = x_pos + (vertical_width - crop_w) // 2 + x_offset
    zy = (original_height - crop_h) // 2
    zx = max(0, min(zx, original_width - crop_w))
    zy = max(0, min(zy, original_height - crop_h))

    cropped = frame[zy:zy + crop_h, zx:zx + crop_w]
    return cv2.resize(cropped, (vertical_width, vertical_height), interpolation=cv2.INTER_LANCZOS4)


def crop_to_vertical(input_video_path, output_video_path):
    """Crop video to vertical 9:16 format with professional camera tracking and effects."""
    # Use DNN face detector (more accurate than Haar cascade) if model files exist
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prototxt_path = os.path.join(script_dir, "models", "deploy.prototxt")
    model_path = os.path.join(script_dir, "models", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    use_dnn = os.path.exists(prototxt_path) and os.path.exists(model_path)
    if use_dnn:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        _configure_dnn_backend(net)
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Using Haar cascade face detector (DNN model not found)")

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)
    vertical_width = vertical_width - (vertical_width % 2)
    vertical_height = vertical_height - (vertical_height % 2)
    print(f"Output dimensions: {vertical_width}x{vertical_height}")

    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return

    # ------------------------------------------------------------------
    # PASS 1: Detect faces and build smooth tracking path
    # ------------------------------------------------------------------
    print("Pass 1/2: Detecting faces and planning camera moves...")
    detect_interval = max(1, int(fps / 3))

    face_detections = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % detect_interval == 0:
            face_center_x = None
            if use_dnn:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                best_confidence = 0.5
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > best_confidence:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        face_center_x = (x1 + x2) // 2
                        best_confidence = confidence
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=8, minSize=(30, 30))
                if len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = best_face
                    face_center_x = fx + fw // 2
            if face_center_x is not None:
                face_detections[frame_idx] = face_center_x
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  Detected {frame_idx}/{total_frames} frames")

    center_x = (original_width - vertical_width) // 2

    def clamp_target(cx):
        return max(0, min(cx - vertical_width // 2, original_width - vertical_width))

    keyframes = []
    if face_detections:
        for fi in sorted(face_detections.keys()):
            keyframes.append((fi, clamp_target(face_detections[fi])))
    else:
        keyframes.append((0, center_x))
        keyframes.append((total_frames - 1, center_x))

    if keyframes[0][0] != 0:
        keyframes.insert(0, (0, keyframes[0][1]))
    if keyframes[-1][0] < total_frames - 1:
        keyframes.append((total_frames - 1, keyframes[-1][1]))

    # Smooth interpolation between keyframes
    targets = np.empty(total_frames, dtype=np.float64)
    for seg in range(len(keyframes) - 1):
        f0, x0 = keyframes[seg]
        f1, x1 = keyframes[seg + 1]
        span = max(f1 - f0, 1)
        for f in range(f0, f1 + 1 if seg == len(keyframes) - 2 else f1):
            t = _ease_in_out((f - f0) / span)
            targets[f] = x0 + (x1 - x0) * t

    # Gaussian smoothing for buttery-smooth panning
    kernel_size = int(fps * 1.5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size >= 3:
        from scipy.ndimage import gaussian_filter1d
        sigma = kernel_size / 4.0
        targets = gaussian_filter1d(targets, sigma=sigma, mode='nearest')
    np.clip(targets, 0, original_width - vertical_width, out=targets)

    # ------------------------------------------------------------------
    # Plan camera effects
    # ------------------------------------------------------------------
    effects = _plan_camera_effects(total_frames, fps)
    if effects:
        labels = []
        for etype, es, ee, _ in effects:
            t_start = es / fps
            t_end = ee / fps
            labels.append(f"    {etype} @ {t_start:.1f}s–{t_end:.1f}s")
        print(f"  Planned {len(effects)} camera effects:\n" + "\n".join(labels))
    else:
        print("  No camera effects (clip too short)")

    # ------------------------------------------------------------------
    # PASS 2: Write frames with tracking + effects
    # ------------------------------------------------------------------
    print("Pass 2/2: Writing cropped video with camera effects...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("  Encoding cropped video with FFmpeg NVENC...")
    writer = _start_nvenc_writer(output_video_path, vertical_width, vertical_height, fps)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x_pos = int(round(targets[frame_count]))

        # Check if any effect is active on this frame
        active_effect = None
        for etype, es, ee, eparams in effects:
            if es <= frame_count <= ee:
                active_effect = (etype, es, ee, eparams)
                break

        if active_effect:
            etype, es, ee, eparams = active_effect
            progress = (frame_count - es) / max(ee - es, 1)

            if etype == "slow_push":
                # Cinematic slow zoom toward the face
                zoom = 1.0 + (eparams["zoom_end"] - 1.0) * _ease_in_out(progress)
                cropped = _apply_zoom_crop(frame, x_pos, zoom,
                                           vertical_width, vertical_height,
                                           original_width, original_height)

            elif etype == "slow_pull":
                # Slow zoom out from close to normal
                zoom_start = eparams["zoom_start"]
                zoom = zoom_start + (1.0 - zoom_start) * _ease_in_out(progress)
                cropped = _apply_zoom_crop(frame, x_pos, zoom,
                                           vertical_width, vertical_height,
                                           original_width, original_height)

            elif etype == "jump_close":
                # Jump to close-up with smooth ease-in/out over 6 frames
                zoom_target = eparams["zoom"]
                ramp = min(6, (ee - es) // 4)
                if ramp > 0 and (frame_count - es) < ramp:
                    t = _ease_in_out((frame_count - es) / ramp)
                    zoom = 1.0 + (zoom_target - 1.0) * t
                elif ramp > 0 and (ee - frame_count) < ramp:
                    t = _ease_in_out((ee - frame_count) / ramp)
                    zoom = 1.0 + (zoom_target - 1.0) * t
                else:
                    zoom = zoom_target
                cropped = _apply_zoom_crop(frame, x_pos, zoom,
                                           vertical_width, vertical_height,
                                           original_width, original_height)

            elif etype == "jump_mid":
                # Jump to mid-shot with smooth ease-in/out over 6 frames
                zoom_target = eparams["zoom"]
                ramp = min(6, (ee - es) // 4)
                if ramp > 0 and (frame_count - es) < ramp:
                    t = _ease_in_out((frame_count - es) / ramp)
                    zoom = 1.0 + (zoom_target - 1.0) * t
                elif ramp > 0 and (ee - frame_count) < ramp:
                    t = _ease_in_out((ee - frame_count) / ramp)
                    zoom = 1.0 + (zoom_target - 1.0) * t
                else:
                    zoom = zoom_target
                cropped = _apply_zoom_crop(frame, x_pos, zoom,
                                           vertical_width, vertical_height,
                                           original_width, original_height)

            elif etype == "drift":
                # Ken Burns: slow horizontal drift with slight zoom
                zoom = eparams["zoom"]
                drift_px = eparams["drift_px"]
                offset = int(drift_px * _ease_in_out(progress))
                cropped = _apply_zoom_crop(frame, x_pos, zoom,
                                           vertical_width, vertical_height,
                                           original_width, original_height,
                                           x_offset=offset)
            else:
                # Fallback: normal crop
                x_start = max(0, min(x_pos, original_width - vertical_width))
                cropped = frame[:, x_start:x_start + vertical_width]
        else:
            # Normal tracking — no effect active
            x_start = max(0, min(x_pos, original_width - vertical_width))
            cropped = frame[:, x_start:x_start + vertical_width]
            if cropped.shape[0] != vertical_height or cropped.shape[1] != vertical_width:
                cropped = cv2.resize(cropped, (vertical_width, vertical_height),
                                     interpolation=cv2.INTER_LANCZOS4)

        # Safety: ensure frame matches writer dimensions
        if cropped.shape[1] != vertical_width or cropped.shape[0] != vertical_height:
            cropped = cv2.resize(cropped, (vertical_width, vertical_height),
                                 interpolation=cv2.INTER_LANCZOS4)
        cropped = np.ascontiguousarray(cropped)
        if writer.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is not available")
        writer.stdin.write(cropped.tobytes())
        frame_count += 1
        if frame_count >= total_frames:
            break

        if frame_count % 200 == 0:
            print(f"  Written {frame_count}/{total_frames} frames")

    cap.release()
    _finish_nvenc_writer(writer, "cropped video encode")
    print(f"Cropping complete. Processed {frame_count} frames -> {output_video_path}")



def combine_videos(video_with_audio, video_without_audio, output_filename):
    try:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            os.path.abspath(video_without_audio),
            "-i",
            os.path.abspath(video_with_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            os.path.abspath(output_filename),
        ]
        _run_ffmpeg(command, "audio/video mux")
        print(f"Combined video saved successfully as {output_filename}")
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")



if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = 'Croped_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    crop_to_vertical(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)



