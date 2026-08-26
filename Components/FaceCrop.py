import cv2
import json
import numpy as np
import os
import random
import subprocess


_DEFAULT_VERTICAL_ANCHOR_RATIO = 0.5

_DEFAULT_CROP_PROFILE = {
    # -- detection --
    "face_confidence_threshold": None,
    "detection_search_band_ratio": None,
    # -- stability / damping --
    "stillness_lock": False,
    "stillness_threshold_ratio": 0.10,
    "stillness_window_sec": 2.5,
    "lock_transition_ease_sec": 0.4,
    "pan_smoothing_sigma_sec": None,
    # -- framing / headroom --
    "base_zoom": None,
    "vertical_anchor_ratio": None,
    # -- shot switching --
    "shot_switch_mode": "random",
    "mid_zoom_range": [1.10, 1.22],
    "close_zoom_range": [1.28, 1.45],
    "xcu_zoom_range": [1.55, 1.85],
    "mid_duration_range_sec": [1.2, 2.4],
    "close_duration_range_sec": [1.6, 3.0],
    "xcu_duration_range_sec": [1.2, 2.2],
    "effect_interval_range_sec": [4.0, 6.0],
}


def _merge_profile_dict(profile, overrides, source_label):
    """Merge one JSON file's keys into `profile` in place.

    Keys starting with "_" are treated as human-readable comments (e.g.
    "_erklaerung") and intentionally ignored — they exist so a config file
    documents itself when someone opens it, without being a profile key.
    """
    applied = [k for k in overrides if k in _DEFAULT_CROP_PROFILE]
    unknown = [k for k in overrides if k not in _DEFAULT_CROP_PROFILE and not k.startswith("_")]
    profile.update({k: overrides[k] for k in applied})
    if applied:
        print(f"  {source_label}: applied {', '.join(applied)}")
    if unknown:
        print(f"  Warning: {source_label} has unknown keys, ignored: {', '.join(unknown)}")


def _load_crop_profile_override():
    """Optional per-run camera-crop profile, mirroring HIGHLIGHT_SYSTEM_PROMPT_FILE.

    Two ways to opt in (every other caller/brand is unaffected since both
    env vars are unset for them; never raises — falls back to the built-in
    defaults, identical to today's behaviour, on any missing/invalid file):

    - CAMERA_CROP_PROFILE_FILE: a single JSON file of _DEFAULT_CROP_PROFILE keys.
    - CAMERA_CROP_CONFIG_DIR: a directory of thematically-named *.json files
      (e.g. detection.json, stability.json, framing.json, shot_switching.json).
      All of them are loaded and merged, in filename order, so a later file
      can override an earlier one on a shared key. This is the recommended
      way to configure a lane with many knobs instead of one large file.

    If both are set, the single file loads first, then the directory's files
    layer on top.
    """
    profile = dict(_DEFAULT_CROP_PROFILE)

    single_path = os.getenv("CAMERA_CROP_PROFILE_FILE")
    if single_path:
        if os.path.isfile(single_path):
            try:
                with open(single_path, "r", encoding="utf-8") as f:
                    _merge_profile_dict(profile, json.load(f), f"CAMERA_CROP_PROFILE_FILE ({single_path})")
            except (OSError, ValueError) as exc:
                print(f"  Warning: could not read CAMERA_CROP_PROFILE_FILE ({exc}); using defaults")
        else:
            print(f"  Warning: CAMERA_CROP_PROFILE_FILE not found: {single_path}")

    config_dir = os.getenv("CAMERA_CROP_CONFIG_DIR")
    if config_dir:
        if os.path.isdir(config_dir):
            json_files = sorted(
                name for name in os.listdir(config_dir) if name.lower().endswith(".json")
            )
            if not json_files:
                print(f"  Warning: CAMERA_CROP_CONFIG_DIR has no *.json files: {config_dir}")
            for name in json_files:
                path = os.path.join(config_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        _merge_profile_dict(profile, json.load(f), name)
                except (OSError, ValueError) as exc:
                    print(f"  Warning: could not read {name} ({exc}); skipped")
        else:
            print(f"  Warning: CAMERA_CROP_CONFIG_DIR not found: {config_dir}")

    return profile


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
    """Smooth pan interpolation only; jump cuts remain instantaneous."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _clamp_search_region(anchor_x, band_width, frame_width):
    """Clamp a [anchor - band/2, anchor + band/2] window to the frame bounds."""
    x0 = max(0, int(anchor_x - band_width / 2))
    x1 = min(frame_width, int(anchor_x + band_width / 2))
    if x1 - x0 < 10:
        return None
    return x0, x1


def _detect_dnn_face_center(net, frame, confidence_threshold, region=None):
    """Run the SSD face detector, optionally restricted to a horizontal region.

    A face that is small relative to the *whole* frame (a wide static stage
    shot) shrinks to only a handful of pixels once resized into the
    detector's 300x300 input, which tanks its confidence score — cropping to
    a narrower band around the last known position first, before that same
    300x300 resize, gives the same real face several times the effective
    input resolution (measured on real footage: mean confidence 0.25 -> 0.88
    for an identical, position-agreeing detection).

    Returns (face_center_x_in_full_frame_coords, confidence) or (None, None).
    """
    frame_h, frame_w = frame.shape[:2]
    if region is not None:
        x0, x1 = region
        crop = frame[:, x0:x1]
    else:
        x0 = 0
        crop = frame
    crop_h, crop_w = crop.shape[:2]
    if crop_h < 10 or crop_w < 10:
        return None, None
    blob = cv2.dnn.blobFromImage(cv2.resize(crop, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    best_confidence = confidence_threshold
    face_center_x = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > best_confidence:
            box = detections[0, 0, i, 3:7] * np.array([crop_w, crop_h, crop_w, crop_h])
            (bx1, _by1, bx2, _by2) = box.astype("int")
            face_center_x = x0 + (bx1 + bx2) // 2
            best_confidence = confidence
    if face_center_x is None:
        return None, None
    return face_center_x, float(best_confidence)


def _motion_speaker_center(previous_gray, current_frame):
    """Return the horizontal centre of a large moving person-like region.

    The face model is frontal-face biased and can miss a preacher who turns
    sideways. On a static stage camera the speaker is normally the largest
    tall moving region, while signs and lecterns stay fixed. This fallback is
    deliberately conservative: it rejects shot-wide changes and small moving
    audience regions.
    """
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (9, 9), 0)
    if previous_gray is None or previous_gray.shape != current_gray.shape:
        return None, current_gray

    height, width = current_gray.shape[:2]
    delta = cv2.absdiff(current_gray, previous_gray)
    _, mask = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
    changed_ratio = float(np.count_nonzero(mask)) / float(mask.size or 1)
    if changed_ratio > 0.45:
        return None, current_gray

    close_size = max(7, int(round(width * 0.009)))
    if close_size % 2 == 0:
        close_size += 1
    dilate_size = max(5, int(round(width * 0.005)))
    if dilate_size % 2 == 0:
        dilate_size += 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    mask = cv2.dilate(
        mask, np.ones((dilate_size, dilate_size), np.uint8), iterations=2
    )

    candidates = []
    min_area = 0.002 * width * height
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area < min_area or box_h < 0.22 * height:
            continue
        if box_w > 0.65 * width or box_h > 0.95 * height:
            continue
        candidates.append((area, x + box_w // 2))
    if not candidates:
        return None, current_gray
    return int(max(candidates)[1]), current_gray


# ======================================================================
# Camera effect types
# ======================================================================
# Each effect is (type, start_frame, end_frame, params_dict)
#
# Types:
#   "jump_close"  — instant jump cut to a close-up (held steady, no movement)
#   "jump_mid"    — instant jump cut to a mid-shot (held steady, no movement)
# ======================================================================

def _plan_camera_effects(total_frames, fps):
    """Plan a sequence of attention-grabbing jump cuts across the clip."""
    duration_sec = total_frames / fps
    effects = []

    # Minimum gap (in frames) between effects
    min_gap = int(fps * 1.2)
    # Don't place effects in first or last 1.5s
    margin = int(fps * 1.5)
    safe_start = margin
    safe_end = total_frames - margin

    if safe_end - safe_start < int(fps * 4):
        return []  # clip too short for effects

    # Build a pool of jump-cut candidates.
    candidates = []

    for _ in range(4):
        close_dur = random.uniform(1.6, 3.0)
        close_frames = int(close_dur * fps)
        candidates.append(("jump_close", close_frames, {"zoom": random.uniform(1.28, 1.45)}))

    for _ in range(4):
        mid_dur = random.uniform(1.2, 2.4)
        mid_frames = int(mid_dur * fps)
        candidates.append(("jump_mid", mid_frames, {"zoom": random.uniform(1.10, 1.22)}))

    # Shuffle and place without overlaps
    random.shuffle(candidates)

    # Determine how many cuts to place (~1 every 4-6 seconds)
    max_effects = max(1, int(duration_sec / random.uniform(4, 6)))
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


def _classify_stillness(targets, fps, vertical_width, window_sec, threshold_ratio):
    """Per-frame bool array: True where the speaker holds still (e.g. a pulpit).

    "Still" means the pan target's range over a sliding window stays under
    threshold_ratio * vertical_width — small face-detection jitter on an
    otherwise static speaker, not real movement across the stage.
    """
    total_frames = len(targets)
    window = max(1, int(round(window_sec * fps)))
    threshold_px = threshold_ratio * vertical_width
    is_static = np.zeros(total_frames, dtype=bool)
    for start in range(0, total_frames, window):
        end = min(total_frames, start + window)
        segment = targets[start:end]
        if segment.size == 0:
            continue
        is_static[start:end] = (segment.max() - segment.min()) < threshold_px
    return is_static


def _lock_static_runs(targets, is_static, ease_frames=0):
    """Freeze the pan target to its run median wherever _classify_stillness said still.

    Applied after the existing ease + Gaussian smoothing pass, so moving
    segments keep today's smooth-pan behaviour untouched and static segments
    get a perfectly flat crop window instead of residual sub-pixel jitter.

    ease_frames > 0 cross-fades in/out of the flat lock over that many frames
    at each end of a static run (an S-curve blend against the surrounding
    smoothed path) instead of snapping straight to the median — the snap
    itself was a visible micro-jump exactly at every lock boundary.
    """
    locked = targets.copy()
    n = len(targets)
    i = 0
    while i < n:
        if not is_static[i]:
            i += 1
            continue
        j = i
        while j < n and is_static[j]:
            j += 1
        run_len = j - i
        median = np.median(targets[i:j])
        ef = min(max(0, ease_frames), run_len // 2)
        for k in range(i, j):
            if ef > 0 and k < i + ef:
                t = _ease_in_out((k - i) / ef)
                locked[k] = targets[k] * (1 - t) + median * t
            elif ef > 0 and k >= j - ef:
                t = _ease_in_out((k - (j - ef)) / ef)
                locked[k] = median * (1 - t) + targets[k] * t
            else:
                locked[k] = median
        i = j
    return locked


def _plan_camera_effects_movement_aware(
    total_frames, fps, is_static,
    mid_zoom_range=(1.10, 1.22),
    close_zoom_range=(1.28, 1.45),
    xcu_zoom_range=(1.55, 1.85),
    mid_duration_range_sec=(1.2, 2.4),
    close_duration_range_sec=(1.6, 3.0),
    xcu_duration_range_sec=(1.2, 2.2),
    effect_interval_range_sec=(4.0, 6.0),
):
    """Like _plan_camera_effects, but close-ups/extreme-close-ups only land
    inside static (pulpit-still) stretches; a moving speaker only ever gets
    a mid-shot. All ranges come from the Oberlahnstein-lane camera-crop
    profile (shot_switching.json) — every default above matches the
    original hardcoded _plan_camera_effects values.
    """
    duration_sec = total_frames / fps
    effects = []

    min_gap = int(fps * 1.2)
    margin = int(fps * 1.5)
    safe_start = margin
    safe_end = total_frames - margin

    if safe_end - safe_start < int(fps * 4):
        return []

    def _range_is_static(start, end):
        end = min(end, total_frames)
        return bool(is_static[start:end].all()) if end > start else False

    candidates = []
    for _ in range(3):
        close_dur = random.uniform(*close_duration_range_sec)
        candidates.append(("jump_close", int(close_dur * fps), {"zoom": random.uniform(*close_zoom_range)}, True))
    for _ in range(2):
        xcu_dur = random.uniform(*xcu_duration_range_sec)
        candidates.append(("jump_xcu", int(xcu_dur * fps), {"zoom": random.uniform(*xcu_zoom_range)}, True))
    for _ in range(4):
        mid_dur = random.uniform(*mid_duration_range_sec)
        candidates.append(("jump_mid", int(mid_dur * fps), {"zoom": random.uniform(*mid_zoom_range)}, False))

    random.shuffle(candidates)
    max_effects = max(1, int(duration_sec / random.uniform(*effect_interval_range_sec)))
    placed_ranges = []

    for etype, edur, eparams, requires_static in candidates:
        if len(effects) >= max_effects:
            break
        for _attempt in range(15):
            es = random.randint(safe_start, max(safe_start, safe_end - edur))
            ee = es + edur
            if ee > safe_end:
                continue
            if requires_static and not _range_is_static(es, ee):
                continue
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


def _resolve_crop_window(
    x_pos,
    zoom,
    vertical_width,
    vertical_height,
    original_width,
    original_height,
    x_offset=0,
    vertical_anchor_ratio=_DEFAULT_VERTICAL_ANCHOR_RATIO,
):
    zoom = max(1.0, float(zoom or 1.0))
    anchor = max(0.0, min(1.0, float(vertical_anchor_ratio)))
    crop_w = int(vertical_width / zoom)
    crop_h = int(vertical_height / zoom)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    zx = int(x_pos + (vertical_width - crop_w) // 2 + x_offset)
    zy_slack = max(0, original_height - crop_h)
    zy = int(round(zy_slack * anchor))
    zx = max(0, min(zx, original_width - crop_w))
    zy = max(0, min(zy, original_height - crop_h))
    return zx, zy, crop_w, crop_h


def _effect_vertical_anchor_ratio(
    base_anchor_ratio,
    *,
    effect_type=None,
    zoom=1.0,
):
    """Lift jump-cut close-ups upward so the head sits lower in frame.

    Smaller anchor ratios move the crop window higher in the source frame,
    which restores headroom in aggressive close-ups.
    """
    anchor = max(0.0, min(1.0, float(base_anchor_ratio)))
    zoom = max(1.0, float(zoom or 1.0))
    if effect_type == "jump_xcu":
        lift = min(0.20, 0.10 + (zoom - 1.0) * 0.20)
        return max(0.14, anchor - lift)
    if effect_type == "jump_close":
        lift = min(0.16, 0.08 + (zoom - 1.0) * 0.20)
        return max(0.18, anchor - lift)
    if effect_type == "jump_mid":
        lift = min(0.10, 0.04 + (zoom - 1.0) * 0.12)
        return max(0.22, anchor - lift)
    return anchor


def _apply_zoom_crop(frame, x_pos, zoom, vertical_width, vertical_height,
                     original_width, original_height, x_offset=0,
                     out_w=None, out_h=None,
                     vertical_anchor_ratio=_DEFAULT_VERTICAL_ANCHOR_RATIO):
    """Crop a zoomed region centered on face position, return resized to output dims."""
    zx, zy, crop_w, crop_h = _resolve_crop_window(
        x_pos,
        zoom,
        vertical_width,
        vertical_height,
        original_width,
        original_height,
        x_offset=x_offset,
        vertical_anchor_ratio=vertical_anchor_ratio,
    )

    cropped = frame[zy:zy + crop_h, zx:zx + crop_w]
    rw = out_w if out_w else vertical_width
    rh = out_h if out_h else vertical_height
    return cv2.resize(cropped, (rw, rh), interpolation=cv2.INTER_LANCZOS4)


def crop_to_vertical(input_video_path, output_video_path, enable_camera_effects=True,
                     target_height=1920, base_zoom=1.0,
                     vertical_anchor_ratio=_DEFAULT_VERTICAL_ANCHOR_RATIO):
    """Crop video to vertical 9:16 format with professional camera tracking and effects.

    Face tracking runs at native resolution for accuracy.  The final output is
    always scaled to *target_height* (default 1920) with a 9:16 aspect ratio,
    i.e. 1080x1920 by default.

    Returns:
        list of (etype, start_sec, end_sec) tuples describing camera effects,
        or an empty list if the function fails.
    """
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
        return []

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    crop_profile = _load_crop_profile_override()
    if crop_profile["base_zoom"] is not None:
        base_zoom = crop_profile["base_zoom"]
    if crop_profile["vertical_anchor_ratio"] is not None:
        vertical_anchor_ratio = crop_profile["vertical_anchor_ratio"]
    base_zoom = max(1.0, float(base_zoom or 1.0))
    vertical_anchor_ratio = max(0.0, min(1.0, float(vertical_anchor_ratio)))
    face_confidence_threshold = float(crop_profile["face_confidence_threshold"] or 0.5)
    search_band_ratio = crop_profile["detection_search_band_ratio"]

    # Native crop dimensions (face tracking at source resolution)
    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)
    vertical_width = vertical_width - (vertical_width % 2)
    vertical_height = vertical_height - (vertical_height % 2)

    # Final output dimensions (always target_height, 9:16)
    out_h = int(target_height)
    out_w = int(out_h * 9 / 16)
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)
    print(f"Output dimensions: {out_w}x{out_h} (native crop: {vertical_width}x{vertical_height})")
    if base_zoom > 1.001 or abs(vertical_anchor_ratio - _DEFAULT_VERTICAL_ANCHOR_RATIO) > 0.001:
        print(
            f"Applying base reframe: zoom={base_zoom:.3f}, "
            f"vertical_anchor={vertical_anchor_ratio:.2f}"
        )

    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return []

    # ------------------------------------------------------------------
    # PASS 1: Detect faces and build smooth tracking path
    # ------------------------------------------------------------------
    print("Pass 1/2: Detecting faces and planning camera moves...")
    detect_interval = max(1, int(fps / 3))

    face_detections = {}
    face_hits = 0
    motion_hits = 0
    band_hits = 0
    previous_detection_gray = None
    detection_anchor_x = None
    band_width = search_band_ratio * vertical_width if search_band_ratio else None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % detect_interval == 0:
            face_center_x = None
            if use_dnn:
                w = frame.shape[1]
                region = None
                if band_width and detection_anchor_x is not None:
                    region = _clamp_search_region(detection_anchor_x, band_width, w)
                face_center_x, _conf = _detect_dnn_face_center(
                    net, frame, face_confidence_threshold, region=region
                )
                if face_center_x is not None and region is not None:
                    band_hits += 1
                elif face_center_x is None and region is not None:
                    # Lost inside the narrow band (she moved further than
                    # expected) — re-acquire on the full frame before falling
                    # back to the noisy motion heuristic.
                    face_center_x, _conf = _detect_dnn_face_center(
                        net, frame, face_confidence_threshold, region=None
                    )
                if face_center_x is not None:
                    detection_anchor_x = face_center_x
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=8, minSize=(30, 30))
                if len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = best_face
                    face_center_x = fx + fw // 2
            if face_center_x is not None:
                face_hits += 1
                # Keep the motion baseline current even when the frontal face
                # path succeeds, so a later side turn can fall back cleanly.
                _, previous_detection_gray = _motion_speaker_center(
                    previous_detection_gray, frame
                )
            else:
                face_center_x, previous_detection_gray = _motion_speaker_center(
                    previous_detection_gray, frame
                )
                if face_center_x is not None:
                    motion_hits += 1
                    if band_width:
                        detection_anchor_x = face_center_x
            if face_center_x is not None:
                face_detections[frame_idx] = face_center_x
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  Detected {frame_idx}/{total_frames} frames")

    band_note = f" ({band_hits} via search-band crop)" if band_width else ""
    print(f"  Tracking detections: {face_hits} face{band_note}, {motion_hits} motion fallback")

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
    pan_sigma_sec = crop_profile["pan_smoothing_sigma_sec"]
    kernel_size = int(fps * (pan_sigma_sec * 4 if pan_sigma_sec else 1.5))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size >= 3:
        from scipy.ndimage import gaussian_filter1d
        sigma = pan_sigma_sec * fps if pan_sigma_sec else kernel_size / 4.0
        targets = gaussian_filter1d(targets, sigma=sigma, mode='nearest')
    np.clip(targets, 0, original_width - vertical_width, out=targets)

    # A speaker planted at a fixed pulpit still produces a few pixels of
    # frame-to-frame detection jitter that the Gaussian pass alone doesn't
    # fully remove; lock those stretches dead flat when the profile asks
    # for it (opt-in — untouched for every render that doesn't set it).
    is_static = _classify_stillness(
        targets, fps, vertical_width,
        crop_profile["stillness_window_sec"],
        crop_profile["stillness_threshold_ratio"],
    )
    if crop_profile["stillness_lock"]:
        static_frames = int(is_static.sum())
        if static_frames:
            print(f"  Stillness lock: {static_frames}/{total_frames} frames held flat")
        ease_frames = int(round(crop_profile["lock_transition_ease_sec"] * fps))
        targets = _lock_static_runs(targets, is_static, ease_frames=ease_frames)

    # ------------------------------------------------------------------
    # Plan camera effects
    # ------------------------------------------------------------------
    if not enable_camera_effects:
        effects = []
    elif crop_profile["shot_switch_mode"] == "movement_aware":
        effects = _plan_camera_effects_movement_aware(
            total_frames, fps, is_static,
            mid_zoom_range=crop_profile["mid_zoom_range"],
            close_zoom_range=crop_profile["close_zoom_range"],
            xcu_zoom_range=crop_profile["xcu_zoom_range"],
            mid_duration_range_sec=crop_profile["mid_duration_range_sec"],
            close_duration_range_sec=crop_profile["close_duration_range_sec"],
            xcu_duration_range_sec=crop_profile["xcu_duration_range_sec"],
            effect_interval_range_sec=crop_profile["effect_interval_range_sec"],
        )
    else:
        effects = _plan_camera_effects(total_frames, fps)
    if effects:
        labels = []
        for etype, es, ee, _ in effects:
            t_start = es / fps
            t_end = ee / fps
            labels.append(f"    {etype} @ {t_start:.1f}s–{t_end:.1f}s")
        print(f"  Planned {len(effects)} camera effects:\n" + "\n".join(labels))
    elif enable_camera_effects:
        print("  No camera effects (clip too short)")
    else:
        print("  Camera effects disabled for this render stage")

    # ------------------------------------------------------------------
    # PASS 2: Write frames with tracking + effects
    # ------------------------------------------------------------------
    print("Pass 2/2: Writing cropped video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("  Encoding cropped video with FFmpeg NVENC...")
    writer = _start_nvenc_writer(output_video_path, out_w, out_h, fps)

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

            if etype in ("jump_close", "jump_xcu"):
                # Hold an instant close-up/extreme-close-up — no ramp, no slow zoom.
                zoom_target = max(base_zoom, eparams["zoom"])
                anchor_ratio = _effect_vertical_anchor_ratio(
                    vertical_anchor_ratio,
                    effect_type=etype,
                    zoom=zoom_target,
                )
                cropped = _apply_zoom_crop(frame, x_pos, zoom_target,
                                           vertical_width, vertical_height,
                                           original_width, original_height,
                                           out_w=out_w, out_h=out_h,
                                           vertical_anchor_ratio=anchor_ratio)

            elif etype == "jump_mid":
                # Hold an instant mid-shot — no ramp, no slow zoom.
                zoom_target = max(base_zoom, eparams["zoom"])
                anchor_ratio = _effect_vertical_anchor_ratio(
                    vertical_anchor_ratio,
                    effect_type=etype,
                    zoom=zoom_target,
                )
                cropped = _apply_zoom_crop(frame, x_pos, zoom_target,
                                           vertical_width, vertical_height,
                                           original_width, original_height,
                                           out_w=out_w, out_h=out_h,
                                           vertical_anchor_ratio=anchor_ratio)
            else:
                # Fallback: normal crop
                if base_zoom > 1.001 or abs(vertical_anchor_ratio - _DEFAULT_VERTICAL_ANCHOR_RATIO) > 0.001:
                    cropped = _apply_zoom_crop(
                        frame,
                        x_pos,
                        base_zoom,
                        vertical_width,
                        vertical_height,
                        original_width,
                        original_height,
                        out_w=out_w,
                        out_h=out_h,
                        vertical_anchor_ratio=vertical_anchor_ratio,
                    )
                else:
                    x_start = max(0, min(x_pos, original_width - vertical_width))
                    cropped = frame[:, x_start:x_start + vertical_width]
                    cropped = cv2.resize(cropped, (out_w, out_h),
                                         interpolation=cv2.INTER_LANCZOS4)
        else:
            # Normal tracking — no effect active
            if base_zoom > 1.001 or abs(vertical_anchor_ratio - _DEFAULT_VERTICAL_ANCHOR_RATIO) > 0.001:
                cropped = _apply_zoom_crop(
                    frame,
                    x_pos,
                    base_zoom,
                    vertical_width,
                    vertical_height,
                    original_width,
                    original_height,
                    out_w=out_w,
                    out_h=out_h,
                    vertical_anchor_ratio=vertical_anchor_ratio,
                )
            else:
                x_start = max(0, min(x_pos, original_width - vertical_width))
                cropped = frame[:, x_start:x_start + vertical_width]
                cropped = cv2.resize(cropped, (out_w, out_h),
                                     interpolation=cv2.INTER_LANCZOS4)

        # Safety: ensure frame matches writer dimensions
        if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
            cropped = cv2.resize(cropped, (out_w, out_h),
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

    # Return camera effects as (type, start_sec, end_sec) for SFX integration
    return [(etype, es / fps, ee / fps) for etype, es, ee, _ in effects]


def _get_video_duration(video_path):
    """Return duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 60.0


def combine_videos(video_with_audio, video_without_audio, output_filename,
                   speech_gain_db=0.0, bg_music_path=None, music_gain_db=None,
                   target_lufs=None):
    """*target_lufs*, when set, takes over speech leveling from *speech_gain_db*.

    A single fixed-gain ``volume=`` filter can't safely bring typical sermon
    speech (high peak-to-average ratio) up to a modern streaming loudness
    target without clipping — the peak headroom runs out long before the
    integrated loudness does (verified: raising -20 LUFS speech to -14 LUFS by
    flat gain alone pushed measured true peaks to +5 dBTP on real clips).
    ``loudnorm`` solves this with adaptive gain + limiting in one pass instead
    of a single multiplier.
    """
    try:
        if not speech_gain_db and not bg_music_path and not target_lufs:
            # Fast path: stream-copy when no audio processing needed
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
        else:
            # Audio processing path: gain + optional background music
            total_dur = _get_video_duration(video_without_audio)

            speech_filter = "[1:a]aresample=48000"
            if target_lufs is not None:
                # ``loudnorm`` keeps roughly three seconds of look-ahead.  If
                # its output is consumed by ``amix`` and the mux is bounded by
                # ``-shortest``, FFmpeg can discard that buffered tail instead
                # of flushing it.  This removed the last 3.1 s of real speech
                # from an ICF Zürich render while the music continued.  Pad
                # before normalization to force the flush, then trim back to
                # the exact body duration.  The padding is never audible.
                speech_filter += (
                    f",apad=pad_dur=4.0,"
                    f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11:print_format=none,"
                    f"atrim=0:{total_dur:.3f},asetpts=N/SR/TB"
                )
            elif speech_gain_db:
                speech_filter += f",volume={speech_gain_db:.1f}dB"
            speech_filter += "[speech]"

            inputs = [
                "-i", os.path.abspath(video_without_audio),
                "-i", os.path.abspath(video_with_audio),
            ]

            if bg_music_path and os.path.isfile(bg_music_path):
                inputs += ["-i", os.path.abspath(bg_music_path)]
                fade_start = max(0, total_dur - 5.0)
                _music_db = music_gain_db if music_gain_db is not None else -15
                music_filter = (
                    f"[2:a]aloop=loop=-1:size=2e+09,"
                    f"atrim=0:{total_dur:.2f},"
                    f"volume={_music_db}dB,"
                    f"afade=t=out:st={fade_start:.2f}:d=5.0[music]"
                )
                filter_complex = (
                    f"{speech_filter};"
                    f"{music_filter};"
                    # NOT duration=first: combined with loudnorm on [speech],
                    # ffmpeg's amix mis-tracks "first" input duration and
                    # truncates the mix ~1.3-2.9s early, cutting the tail of
                    # every clip with background music (verified via isolated
                    # ffmpeg repro, 2026-08-19). Music is already atrim'd to
                    # total_dur above, so "longest" gives the identical result
                    # without the bug.
                    f"[speech][music]amix=inputs=2:duration=longest:normalize=0[a]"
                )
            else:
                filter_complex = speech_filter.replace("[speech]", "[a]")

            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "0:v:0",
                "-map", "[a]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                os.path.abspath(output_filename),
            ]
            _run_ffmpeg(command, "audio/video mux with background music")
        print(f"Combined video saved successfully as {output_filename}")
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")



if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = 'Croped_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    crop_to_vertical(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)
