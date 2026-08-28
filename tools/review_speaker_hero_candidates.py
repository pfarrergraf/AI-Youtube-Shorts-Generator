#!/usr/bin/env python3
"""Create recoverable speaker-intake and hero-candidate review packets.

This is an explicit human-review step. It does not search the web, compare
faces, approve identities, call Azure, or promote files into ``heroes/``.
WhatsApp delivery is opt-in and attempted only when the local bridge reports
that it is connected and ``WHATSAPP_DEFAULT_TARGET`` (or ``--target``) exists.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_BRIDGE_URL = "http://127.0.0.1:8787"


def normalise_speaker_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not key:
        raise ValueError("Speaker key must contain letters or digits")
    return key


def create_intake_packet(
    repertoire_root: Path,
    *,
    speaker_key: str,
    display_name: str,
    public_references: list[dict[str, str]],
    video_frames: list[Path],
    stamp: str | None = None,
) -> Path:
    """Scaffold an unapproved identity intake without changing manifest.json."""
    key = normalise_speaker_key(speaker_key)
    speaker_dir = repertoire_root / key
    for name in ("source", "jobs", "heroes"):
        (speaker_dir / name).mkdir(parents=True, exist_ok=True)

    required = {"url", "retrieved_at", "intended_role", "attribution"}
    for index, reference in enumerate(public_references, start=1):
        missing = sorted(required - set(reference))
        if missing:
            raise ValueError(f"Public reference {index} is missing: {', '.join(missing)}")
    frames = [str(path.resolve()) for path in video_frames]
    packet_dir = speaker_dir / "jobs" / f"{stamp or date.today().isoformat()}_identity_intake"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet_path = packet_dir / "identity_review.json"
    packet = {
        "schema_version": 1,
        "kind": "speaker_identity_intake",
        "speaker_key": key,
        "display_name": display_name.strip() or key,
        "identity_status": "manual_review_required",
        "manifest_updated": False,
        "public_references": public_references,
        "video_frames": frames,
        "review_instructions": (
            "A human must verify that the attributable public references and video frames depict "
            "the same speaker before adding an unapproved manifest entry or generating candidates."
        ),
    }
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return packet_path


def _read_job(job_path: Path, *, expected_count: int = 3) -> tuple[dict[str, Any], list[Path]]:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or len(jobs) != expected_count:
        raise ValueError(f"Hero review requires exactly {expected_count} candidates")
    candidates: list[Path] = []
    for job in jobs:
        output = job_path.parent / str(job.get("output") or "")
        if not output.is_file():
            raise ValueError(f"Missing generated candidate: {output.name or '<unnamed>'}")
        if job.get("approved") is True:
            raise ValueError("Candidate job is already approved; review handoff must remain unapproved")
        candidates.append(output.resolve())
    return payload, candidates


def write_review_packet(job_path: Path, *, expected_count: int = 3) -> Path:
    payload, candidates = _read_job(job_path, expected_count=expected_count)
    packet_path = job_path.parent / "hero_review.json"
    packet = {
        "schema_version": 1,
        "kind": "speaker_hero_review",
        "speaker_key": payload.get("speaker_key"),
        "display_name": payload.get("display_name") or payload.get("speaker_key"),
        "review_status": "manual_review_required",
        "approved": False,
        "candidate_count": len(candidates),
        "candidates": [str(path) for path in candidates],
        "delivery": {"channel": "manual", "status": "pending"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instructions": "Review identity, hands, text-free output, composition, and quality; approve/promote separately.",
    }
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return packet_path


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 3.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def handoff_review(
    job_path: Path,
    *,
    send_whatsapp: bool = False,
    target: str | None = None,
    bridge_url: str = DEFAULT_BRIDGE_URL,
    timeout: float = 3.0,
    opener: Callable[..., Any] = urlopen,
) -> Path:
    """Always leave a packet; WhatsApp errors are recorded without target data."""
    packet_path = write_review_packet(job_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    configured_target = str(target or os.environ.get("WHATSAPP_DEFAULT_TARGET") or "").strip()
    if not send_whatsapp:
        return packet_path
    if not configured_target:
        packet["delivery"] = {"channel": "whatsapp", "status": "pending", "error": "target_not_configured"}
    else:
        message_ids: list[str] = []
        try:
            health = _request_json(f"{bridge_url.rstrip('/')}/health", timeout=timeout, opener=opener)
            if health.get("connected") is not True:
                raise RuntimeError("bridge_not_connected")
            total = packet["candidate_count"]
            for index, candidate in enumerate(packet["candidates"], start=1):
                response = _request_json(
                    f"{bridge_url.rstrip('/')}/send/number",
                    payload={
                        "phone": configured_target,
                        "media_path": candidate,
                        "caption": (
                            f"Hero candidate {index}/{total} for {packet['display_name']} — "
                            "reply with approve/reject."
                        ),
                    },
                    timeout=timeout,
                    opener=opener,
                )
                sent = response.get("sent") if isinstance(response.get("sent"), dict) else {}
                message_id = response.get("message_id") or response.get("id") or sent.get("id")
                if not message_id:
                    raise RuntimeError("bridge_returned_no_message_id")
                message_ids.append(str(message_id))
            packet["delivery"] = {
                "channel": "whatsapp",
                "status": "sent_pending_response",
                "message_ids": message_ids,
            }
        except (OSError, URLError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            error_code = str(exc) if str(exc) in {
                "bridge_not_connected",
                "bridge_returned_no_message_id",
            } else "bridge_request_failed"
            packet["delivery"] = {"channel": "whatsapp", "status": "pending", "error": error_code}
            if message_ids:
                packet["delivery"]["message_ids"] = message_ids
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return packet_path


def _load_reference_json(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Reference JSON must be a list of metadata objects")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    intake = subparsers.add_parser("intake", help="scaffold an unapproved identity-review packet")
    intake.add_argument("--repertoire-root", type=Path, required=True)
    intake.add_argument("--speaker-key", required=True)
    intake.add_argument("--display-name", required=True)
    intake.add_argument("--public-references-json", type=Path)
    intake.add_argument("--video-frame", action="append", type=Path, default=[])
    intake.add_argument("--stamp")

    handoff = subparsers.add_parser("handoff", help="write a three-candidate review packet")
    handoff.add_argument("--job", type=Path, required=True)
    handoff.add_argument("--send-whatsapp", action="store_true")
    handoff.add_argument("--target", help="defaults to WHATSAPP_DEFAULT_TARGET; never written to the packet")
    handoff.add_argument("--bridge-url", default=os.environ.get("WHATSAPP_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    handoff.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    if args.command == "intake":
        output = create_intake_packet(
            args.repertoire_root,
            speaker_key=args.speaker_key,
            display_name=args.display_name,
            public_references=_load_reference_json(args.public_references_json),
            video_frames=args.video_frame,
            stamp=args.stamp,
        )
    else:
        output = handoff_review(
            args.job,
            send_whatsapp=args.send_whatsapp,
            target=args.target,
            bridge_url=args.bridge_url,
            timeout=args.timeout,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
