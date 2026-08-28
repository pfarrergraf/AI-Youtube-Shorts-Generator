from __future__ import annotations

import json
from pathlib import Path
import sys

GENERATOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GENERATOR_ROOT))

from tools.review_speaker_hero_candidates import (
    create_intake_packet,
    handoff_review,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _job(tmp_path: Path) -> Path:
    jobs = []
    for index in range(3):
        output = tmp_path / f"candidate_{index}.png"
        output.write_bytes(b"image")
        jobs.append({"output": output.name, "approved": False, "status": "manual_review_required"})
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps({
            "speaker_key": "stefan_kreszis",
            "display_name": "Stefan Kreszis",
            "status": "manual_review_required",
            "jobs": jobs,
        }),
        encoding="utf-8",
    )
    return job_path


def test_intake_scaffolds_review_state_without_manifest_approval(tmp_path):
    packet_path = create_intake_packet(
        tmp_path,
        speaker_key="Stefan Kreszis",
        display_name="Stefan Kreszis",
        public_references=[{
            "url": "https://example.invalid/source",
            "retrieved_at": "2026-08-27",
            "intended_role": "identity reference",
            "attribution": "Example congregation",
        }],
        video_frames=[tmp_path / "frame.png"],
        stamp="2026-08-27",
    )

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    speaker_dir = tmp_path / "stefan_kreszis"
    assert packet["identity_status"] == "manual_review_required"
    assert packet["manifest_updated"] is False
    assert (speaker_dir / "source").is_dir()
    assert (speaker_dir / "jobs").is_dir()
    assert (speaker_dir / "heroes").is_dir()
    assert not (tmp_path / "manifest.json").exists()


def test_missing_whatsapp_configuration_leaves_recoverable_packet(monkeypatch, tmp_path):
    monkeypatch.delenv("WHATSAPP_DEFAULT_TARGET", raising=False)
    packet_path = handoff_review(_job(tmp_path), send_whatsapp=True)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["candidate_count"] == 3
    assert packet["review_status"] == "manual_review_required"
    assert packet["approved"] is False
    assert packet["delivery"] == {
        "channel": "whatsapp",
        "status": "pending",
        "error": "target_not_configured",
    }


def test_failed_whatsapp_healthcheck_does_not_send_or_expose_target(tmp_path):
    requests = []

    def _opener(request, timeout):
        requests.append((request.full_url, timeout))
        return _Response({"connected": False})

    packet_path = handoff_review(
        _job(tmp_path),
        send_whatsapp=True,
        target="4917696422064",
        opener=_opener,
    )

    packet_text = packet_path.read_text(encoding="utf-8")
    packet = json.loads(packet_text)
    assert requests == [("http://127.0.0.1:8787/health", 3.0)]
    assert packet["delivery"]["status"] == "pending"
    assert packet["delivery"]["error"] == "bridge_not_connected"
    assert "4917696422064" not in packet_text


def test_connected_bridge_records_message_ids_but_not_phone(tmp_path):
    sent_payloads = []

    def _opener(request, timeout):
        if request.full_url.endswith("/health"):
            return _Response({"connected": True})
        sent_payloads.append(json.loads(request.data.decode("utf-8")))
        return _Response({"ok": True, "sent": {"id": f"message-{len(sent_payloads)}"}})

    packet_path = handoff_review(
        _job(tmp_path),
        send_whatsapp=True,
        target="4917696422064",
        opener=_opener,
    )

    packet_text = packet_path.read_text(encoding="utf-8")
    packet = json.loads(packet_text)
    assert len(sent_payloads) == 3
    assert packet["delivery"]["status"] == "sent_pending_response"
    assert packet["delivery"]["message_ids"] == ["message-1", "message-2", "message-3"]
    assert packet["approved"] is False
    assert "4917696422064" not in packet_text
