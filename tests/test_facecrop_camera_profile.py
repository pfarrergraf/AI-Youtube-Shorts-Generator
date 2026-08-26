"""
Offline tests for the Oberlahnstein-lane camera-crop profile switch in
FaceCrop.py: the CAMERA_CROP_PROFILE_FILE override, the stillness
lock, and movement-aware shot-type placement. No video/GPU needed —
these exercise the pure numpy/JSON helpers directly.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Components import FaceCrop as fc  # noqa: E402


def test_default_profile_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("CAMERA_CROP_PROFILE_FILE", raising=False)
    profile = fc._load_crop_profile_override()
    assert profile == fc._DEFAULT_CROP_PROFILE


def test_override_file_merges_over_defaults(monkeypatch, tmp_path):
    override = tmp_path / "profile.json"
    override.write_text(json.dumps({"stillness_lock": True, "base_zoom": 1.25}), encoding="utf-8")
    monkeypatch.setenv("CAMERA_CROP_PROFILE_FILE", str(override))

    profile = fc._load_crop_profile_override()

    assert profile["stillness_lock"] is True
    assert profile["base_zoom"] == 1.25
    # Untouched keys keep their defaults.
    assert profile["shot_switch_mode"] == "random"


def test_missing_override_file_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CAMERA_CROP_PROFILE_FILE", str(tmp_path / "missing.json"))
    profile = fc._load_crop_profile_override()
    assert profile == fc._DEFAULT_CROP_PROFILE


def test_classify_stillness_flags_flat_stretch_and_not_the_moving_one():
    fps = 25
    vertical_width = 1000
    # 4s flat (pulpit) + 4s of a real sweep across the frame.
    still = np.full(int(4 * fps), 500.0)
    moving = np.linspace(500.0, 900.0, int(4 * fps))
    targets = np.concatenate([still, moving])

    is_static = fc._classify_stillness(
        targets, fps, vertical_width, window_sec=1.0, threshold_ratio=0.012,
    )

    assert is_static[: int(4 * fps)].all()
    assert not is_static[int(4 * fps) :].any()


def test_lock_static_runs_flattens_only_static_frames():
    targets = np.array([10.0, 12.0, 9.0, 11.0, 50.0, 60.0, 70.0], dtype=np.float64)
    is_static = np.array([True, True, True, True, False, False, False])

    locked = fc._lock_static_runs(targets, is_static)

    assert np.all(locked[:4] == np.median(targets[:4]))
    assert np.array_equal(locked[4:], targets[4:])


def test_config_dir_merges_multiple_thematic_files(monkeypatch, tmp_path):
    monkeypatch.delenv("CAMERA_CROP_PROFILE_FILE", raising=False)
    (tmp_path / "detection.json").write_text(
        json.dumps({"face_confidence_threshold": 0.15, "_erklaerung": "ignored comment"}),
        encoding="utf-8",
    )
    (tmp_path / "framing.json").write_text(
        json.dumps({"base_zoom": 1.5}), encoding="utf-8",
    )
    monkeypatch.setenv("CAMERA_CROP_CONFIG_DIR", str(tmp_path))

    profile = fc._load_crop_profile_override()

    assert profile["face_confidence_threshold"] == 0.15
    assert profile["base_zoom"] == 1.5
    # Untouched keys keep their defaults.
    assert profile["shot_switch_mode"] == "random"


def test_config_dir_later_file_overrides_earlier_on_shared_key(monkeypatch, tmp_path):
    monkeypatch.delenv("CAMERA_CROP_PROFILE_FILE", raising=False)
    (tmp_path / "a_first.json").write_text(json.dumps({"base_zoom": 1.1}), encoding="utf-8")
    (tmp_path / "b_second.json").write_text(json.dumps({"base_zoom": 1.9}), encoding="utf-8")
    monkeypatch.setenv("CAMERA_CROP_CONFIG_DIR", str(tmp_path))

    profile = fc._load_crop_profile_override()

    assert profile["base_zoom"] == 1.9  # alphabetically later file wins


def test_missing_config_dir_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CAMERA_CROP_PROFILE_FILE", raising=False)
    monkeypatch.setenv("CAMERA_CROP_CONFIG_DIR", str(tmp_path / "does_not_exist"))
    profile = fc._load_crop_profile_override()
    assert profile == fc._DEFAULT_CROP_PROFILE


def test_lock_static_runs_with_easing_ramps_instead_of_snapping():
    # A run with small jitter around 500 (not perfectly flat, like real
    # detection noise on an otherwise still speaker), between two moving
    # stretches.
    rng = np.random.default_rng(0)
    static_run = 500.0 + rng.uniform(-4, 4, size=40)
    targets = np.concatenate([
        np.linspace(0.0, 100.0, 20),
        static_run,
        np.linspace(500.0, 700.0, 20),
    ])
    is_static = np.zeros(80, dtype=bool)
    is_static[20:60] = True

    locked = fc._lock_static_runs(targets, is_static, ease_frames=5)
    median = np.median(targets[20:60])

    # Continuous at the very entry: no snap the instant the lock starts.
    assert locked[20] == targets[20]
    # The core of the run (away from both edges) is fully flat at the median.
    assert np.all(locked[28:52] == median)
    # Partway through the ease-in, it's a genuine blend: between the raw
    # (jittery) value and the flat median, not equal to either endpoint.
    mid_ease = locked[22]
    lo, hi = sorted([targets[22], median])
    assert lo < mid_ease < hi
    # No infinite/NaN weirdness anywhere.
    assert np.all(np.isfinite(locked))


def test_clamp_search_region_centers_band_on_anchor():
    region = fc._clamp_search_region(anchor_x=1000, band_width=400, frame_width=3840)
    assert region == (800, 1200)


def test_clamp_search_region_clips_to_frame_bounds():
    left = fc._clamp_search_region(anchor_x=50, band_width=400, frame_width=3840)
    assert left == (0, 250)
    right = fc._clamp_search_region(anchor_x=3800, band_width=400, frame_width=3840)
    assert right == (3600, 3840)


def test_clamp_search_region_rejects_degenerate_band():
    assert fc._clamp_search_region(anchor_x=10, band_width=1, frame_width=20) is None


def test_movement_aware_effects_never_place_close_ups_on_moving_frames():
    random.seed(0)
    fps = 25
    total_frames = int(20 * fps)
    is_static = np.zeros(total_frames, dtype=bool)
    # Only frames 5s-12s are "pulpit-still"; the rest is the speaker moving.
    is_static[int(5 * fps) : int(12 * fps)] = True

    effects = fc._plan_camera_effects_movement_aware(total_frames, fps, is_static)

    assert effects, "expected at least one planned effect"
    for etype, es, ee, _params in effects:
        if etype in ("jump_close", "jump_xcu"):
            assert is_static[es:ee].all(), (
                f"{etype} at [{es},{ee}) overlaps a moving stretch"
            )
