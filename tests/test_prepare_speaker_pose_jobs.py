from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

GENERATOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GENERATOR_ROOT))

import Components.AzureImageGeneration as azure_generation
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


def test_three_generated_candidates_remain_unapproved(monkeypatch, tmp_path):
    jobs = []
    for index in range(3):
        prompt_name = f"pose_{index}.prompt.txt"
        output_name = f"candidate_{index}.png"
        (tmp_path / prompt_name).write_text(f"prompt {index}", encoding="utf-8")
        jobs.append({
            "prompt_path": prompt_name,
            "output": output_name,
            "status": "prepared",
            "approved": False,
        })
    payload = {
        "speaker_key": "guest_speaker",
        "display_name": "Guest Speaker",
        "status": "prepared",
        "references": [],
        "jobs": jobs,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    def _fake_generate(_prompt, output, **_kwargs):
        output.write_bytes(b"candidate")
        return output

    monkeypatch.setattr(azure_generation, "generate_image", _fake_generate)

    completed = execute_jobs(
        [{"job_path": job_path, "payload": payload}],
        quality="high",
        size="1024x1536",
        confirm_count=3,
    )

    saved = json.loads(job_path.read_text(encoding="utf-8"))
    assert completed == 3
    assert saved["status"] == "manual_review_required"
    assert saved["approved"] is False
    assert all(job["status"] == "manual_review_required" for job in saved["jobs"])
    assert all(job["approved"] is False for job in saved["jobs"])
    assert not (tmp_path / "heroes").exists()
