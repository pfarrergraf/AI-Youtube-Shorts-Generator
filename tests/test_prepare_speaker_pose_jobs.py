from __future__ import annotations

import json

import pytest

from tools.prepare_speaker_pose_jobs import build_prompt, execute_jobs, load_matrix


def test_pose_catalog_has_at_least_ten_distinct_thumbnail_poses():
    _manifest, poses = load_matrix()
    ids = [pose["id"] for pose in poses]
    assert len(ids) >= 10
    assert len(ids) == len(set(ids))
    assert {"empathic_open", "battle_ready", "point_to_heaven", "compassion_near_tears", "righteous_anger"} <= set(ids)


def test_pose_prompt_locks_identity_and_excludes_baked_text():
    _manifest, poses = load_matrix()
    prompt = build_prompt("Thomas Herrmann", poses[0])
    assert "strict identity references" in prompt
    assert "same recognizable face" in prompt
    assert "text-free" in prompt
    assert "captions" in prompt


def test_paid_execution_requires_exact_count_confirmation(tmp_path):
    job_path = tmp_path / "job.json"
    payload = {
        "references": [],
        "jobs": [{"prompt_path": "one.prompt.txt", "output": "one.png"}],
    }
    (tmp_path / "one.prompt.txt").write_text("test", encoding="utf-8")
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="--confirm-count 1"):
        execute_jobs([{"job_path": job_path, "payload": payload}], quality="low", size="1024x1536", confirm_count=None)
