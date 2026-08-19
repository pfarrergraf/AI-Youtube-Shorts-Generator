"""Crop video with multiple speakers visible using face tracking + dynamic zoom-out."""

import cv2
import numpy as np
import os
import subprocess
from scipy.ndimage import gaussian_filter1d


NVENC_FLAGS = [
    "-c:v", "h264_nvenc", "-preset", "p7", "-rc", "constqp",
    "-qp", "18", "-b:v", "0", "-gpu", "0",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
]


def _configure_dnn_backend(net):
    """Enable CUDA for DNN backend if available."""
    if not hasattr(cv2, "cuda"):
        return
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    except cv2.error:
        cuda_devices = 0
    if cuda_devices <= 0:
        return
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    except cv2.error:
        pass


def _start_nvenc_writer(output_video_path, width, height, fps):
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}", "-i", "-", "-an", *NVENC_FLAGS,
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


def crop_dual_speaker(
    input_video_path,
    output_video_path,
    layout_mode="side-by-side",
    target_height=1920,
):
    """Crop video showing both speakers with face tracking.

    Args:
        input_video_path: Source landscape video
        output_video_path: Output 9:16 vertical video
        layout_mode: "split-screen" (left/right) or "side-by-side" (both scaled smaller)
        target_height: Output height in pixels (width will be 9/16 * height)

    Returns:
        List of camera effects (or empty list on failure)
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prototxt_path = os.path.join(script_dir, "models", "deploy.prototxt")
    model_path = os.path.join(script_dir, "models", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    use_dnn = os.path.exists(prototxt_path) and os.path.exists(model_path)
    if use_dnn:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        _configure_dnn_backend(net)
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[DualSpeakerCrop] Error: Could not open video.")
        return []

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output dimensions (9:16 vertical)
    out_h = int(target_height)
    out_w = int(out_h * 9 / 16)
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)

    print(f"[DualSpeakerCrop] Input: {original_width}x{original_height} -> Output: {out_w}x{out_h}")
    print(f"[DualSpeakerCrop] Layout: {layout_mode}")

    # ──────────────────────────────────────────────────────────────────
    # PASS 1: Detect faces and build smooth tracking paths
    # ──────────────────────────────────────────────────────────────────
    print("[DualSpeakerCrop] Pass 1/2: Detecting faces and planning camera moves...")
    detect_interval = max(1, int(fps / 3))

    face_tracks = {}  # {face_id: [(frame_idx, x_center, y_center), ...]}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % detect_interval == 0:
            detected_faces = []
            if use_dnn:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0)
                )
                net.setInput(blob)
                detections = net.forward()
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        detected_faces.append((cx, cy, confidence))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    cx = x + w // 2
                    cy = y + h // 2
                    detected_faces.append((cx, cy, 1.0))

            # Sort by x position (left to right) for consistent IDs
            detected_faces.sort(key=lambda f: f[0])

            for face_id, (cx, cy, conf) in enumerate(detected_faces[:2]):  # Track top 2
                if face_id not in face_tracks:
                    face_tracks[face_id] = []
                face_tracks[face_id].append((frame_idx, cx, cy, conf))

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  Detected {frame_idx}/{total_frames} frames")

    cap.release()

    # Build smooth X-position tracks for each speaker
    speaker_x_tracks = {}  # {speaker_id: [x_pos_per_frame]}

    for speaker_id in range(2):
        if speaker_id not in face_tracks or len(face_tracks[speaker_id]) == 0:
            speaker_x_tracks[speaker_id] = [original_width // 2] * total_frames
            continue

        keyframes = []
        for frame_idx, cx, cy, conf in face_tracks[speaker_id]:
            keyframes.append((frame_idx, cx))

        if not keyframes:
            speaker_x_tracks[speaker_id] = [original_width // 2] * total_frames
            continue

        # Ensure start/end keyframes
        if keyframes[0][0] != 0:
            keyframes.insert(0, (0, keyframes[0][1]))
        if keyframes[-1][0] < total_frames - 1:
            keyframes.append((total_frames - 1, keyframes[-1][1]))

        # Smooth interpolation
        targets = np.empty(total_frames, dtype=np.float64)
        for seg in range(len(keyframes) - 1):
            f0, x0 = keyframes[seg]
            f1, x1 = keyframes[seg + 1]
            span = max(f1 - f0, 1)
            for f in range(f0, f1 + 1 if seg == len(keyframes) - 2 else f1):
                t = min(1.0, (f - f0) / span)
                targets[f] = x0 + (x1 - x0) * t

        # Gaussian smoothing
        kernel_size = int(fps * 1.5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size >= 3:
            sigma = kernel_size / 4.0
            targets = gaussian_filter1d(targets, sigma=sigma, mode='nearest')

        np.clip(targets, 0, original_width, out=targets)
        speaker_x_tracks[speaker_id] = targets

    # ──────────────────────────────────────────────────────────────────
    # PASS 2: Write frames with dual-speaker layout
    # ──────────────────────────────────────────────────────────────────
    print("[DualSpeakerCrop] Pass 2/2: Writing cropped video...")
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)

    print("  Encoding cropped video with FFmpeg NVENC...")
    writer = _start_nvenc_writer(output_video_path, out_w, out_h, fps)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if layout_mode == "side-by-side":
            cropped = _crop_side_by_side(
                frame, speaker_x_tracks, original_width, original_height,
                out_w, out_h, frame_count
            )
        else:  # split-screen
            cropped = _crop_split_screen(
                frame, speaker_x_tracks, original_width, original_height,
                out_w, out_h, frame_count
            )

        # Ensure dimensions match
        if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
            cropped = cv2.resize(cropped, (out_w, out_h),
                                interpolation=cv2.INTER_LANCZOS4)

        cropped = np.ascontiguousarray(cropped)
        if writer.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is not available")
        writer.stdin.write(cropped.tobytes())
        frame_count += 1

        if frame_count % 200 == 0:
            print(f"  Written {frame_count}/{total_frames} frames")

    cap.release()
    _finish_nvenc_writer(writer, "dual-speaker crop encode")
    print(f"[DualSpeakerCrop] Complete. Processed {frame_count} frames -> {output_video_path}")

    return []


def _crop_side_by_side(frame, speaker_x_tracks, src_w, src_h, out_w, out_h, frame_idx=0):
    """Both speakers scaled smaller, side by side vertically."""
    # Each speaker gets half the height
    speaker_h = out_h // 2
    speaker_w = int(speaker_h * 9 / 16)
    speaker_w = speaker_w - (speaker_w % 2)

    # Get current X positions for this frame
    track_0 = speaker_x_tracks.get(0, np.array([src_w // 2] * 10000))
    track_1 = speaker_x_tracks.get(1, np.array([src_w // 2] * 10000))
    x_pos_0 = int(track_0[min(frame_idx, len(track_0) - 1)])
    x_pos_1 = int(track_1[min(frame_idx, len(track_1) - 1)])

    # Crop regions for each speaker (narrow vertical strips)
    crop_h = src_h
    crop_w = int(crop_h * 9 / 16)

    # Speaker 0 (left side of source)
    x0_start = max(0, min(x_pos_0 - crop_w // 2, src_w - crop_w))
    crop_0 = frame[0:crop_h, x0_start:x0_start + crop_w]
    crop_0 = cv2.resize(crop_0, (speaker_w, speaker_h), interpolation=cv2.INTER_LANCZOS4)

    # Speaker 1 (right side of source)
    x1_start = max(0, min(x_pos_1 - crop_w // 2, src_w - crop_w))
    crop_1 = frame[0:crop_h, x1_start:x1_start + crop_w]
    crop_1 = cv2.resize(crop_1, (speaker_w, speaker_h), interpolation=cv2.INTER_LANCZOS4)

    # Stack vertically with padding
    padding_w = (out_w - speaker_w) // 2
    padding_h = (out_h - 2 * speaker_h) // 2

    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Place speaker 0 top-center
    y0_start = padding_h
    x_center = (out_w - speaker_w) // 2
    result[y0_start:y0_start + speaker_h, x_center:x_center + speaker_w] = crop_0

    # Place speaker 1 bottom-center
    y1_start = padding_h + speaker_h + padding_h
    result[y1_start:y1_start + speaker_h, x_center:x_center + speaker_w] = crop_1

    return result


def _crop_split_screen(frame, speaker_x_tracks, src_w, src_h, out_w, out_h, frame_idx=0):
    """Left/right split screen, each speaker full height."""
    half_w = out_w // 2
    half_w = half_w - (half_w % 2)

    # Get current X positions for this frame
    track_0 = speaker_x_tracks.get(0, np.array([src_w // 2] * 10000))
    track_1 = speaker_x_tracks.get(1, np.array([src_w // 2] * 10000))
    x_pos_0 = int(track_0[min(frame_idx, len(track_0) - 1)])
    x_pos_1 = int(track_1[min(frame_idx, len(track_1) - 1)])

    crop_h = src_h
    crop_w_each = int(crop_h * 9 / 16)

    # Speaker 0 (left side of source -> left side of output)
    x0_start = max(0, min(x_pos_0 - crop_w_each // 2, src_w - crop_w_each))
    crop_0 = frame[0:crop_h, x0_start:x0_start + crop_w_each]
    crop_0 = cv2.resize(crop_0, (half_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    # Speaker 1 (right side of source -> right side of output)
    x1_start = max(0, min(x_pos_1 - crop_w_each // 2, src_w - crop_w_each))
    crop_1 = frame[0:crop_h, x1_start:x1_start + crop_w_each]
    crop_1 = cv2.resize(crop_1, (half_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    # Side-by-side horizontal
    result = np.hstack([crop_0, crop_1])
    return result
