"""Detect single vs. multiple speakers in a video for routing to appropriate cropper."""

import cv2
import numpy as np
import os


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


def detect_speakers_in_video(video_path, sample_interval=10, confidence_threshold=0.5):
    """Analyze video to determine if it has one or multiple visible speakers.

    Args:
        video_path: Path to video file
        sample_interval: Analyze every Nth frame (default 10 = ~3 fps @ 30fps source)
        confidence_threshold: Min confidence for face detection (0.0-1.0)

    Returns:
        {
            'speaker_count': 1 or 2+ (int),
            'is_multi_speaker': bool,
            'face_detections': [{frame_idx, count, faces}, ...],
            'confidence': float (0-1, how confident in the detection),
        }
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

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[SpeakerDetection] Error: Could not open {video_path}")
        return {
            'speaker_count': 1,
            'is_multi_speaker': False,
            'face_detections': [],
            'confidence': 0.0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    face_detections = []
    frame_count = 0
    multi_speaker_count = 0
    single_speaker_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
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
                    confidence = float(detections[0, 0, i, 2])
                    if confidence > confidence_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        detected_faces.append({
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'confidence': confidence,
                        })
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    detected_faces.append({
                        'x1': int(x),
                        'y1': int(y),
                        'x2': int(x + w),
                        'y2': int(y + h),
                        'confidence': 1.0,
                    })

            if len(detected_faces) >= 2:
                multi_speaker_count += 1
            elif len(detected_faces) == 1:
                single_speaker_count += 1

            face_detections.append({
                'frame_idx': frame_count,
                'timestamp_sec': frame_count / fps,
                'count': len(detected_faces),
                'faces': detected_faces,
            })

        frame_count += 1

    cap.release()

    total_samples = len(face_detections)
    if total_samples == 0:
        return {
            'speaker_count': 1,
            'is_multi_speaker': False,
            'face_detections': face_detections,
            'confidence': 0.0,
        }

    # Determine speaker count based on prevalence
    multi_speaker_ratio = multi_speaker_count / total_samples
    single_speaker_ratio = single_speaker_count / total_samples

    if multi_speaker_ratio > 0.3:
        speaker_count = 2
        is_multi = True
        confidence = multi_speaker_ratio
    elif single_speaker_ratio > 0.5:
        speaker_count = 1
        is_multi = False
        confidence = single_speaker_ratio
    else:
        speaker_count = 1
        is_multi = False
        confidence = 0.5

    print(
        f"[SpeakerDetection] {video_path}: "
        f"{speaker_count} speaker(s) detected "
        f"(single: {single_speaker_ratio:.1%}, multi: {multi_speaker_ratio:.1%}, "
        f"confidence: {confidence:.1%})"
    )

    return {
        'speaker_count': speaker_count,
        'is_multi_speaker': is_multi,
        'face_detections': face_detections,
        'confidence': confidence,
    }
