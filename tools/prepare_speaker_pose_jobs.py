#!/usr/bin/env python3
"""Prepare or execute a reviewed speaker-pose candidate matrix.

Preparation is free and writes deterministic job manifests and prompt files.
``--execute`` calls the configured Azure GPT-image-2 deployment. Generated
images remain review candidates and are never added to the approved repertoire.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "assets" / "speaker_references" / "manifest.json"
POSES_PATH = PROJECT_ROOT / "assets" / "thumbnail_strategy" / "pose_archetypes.json"


def load_matrix() -> tuple[dict, list[dict]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    poses = json.loads(POSES_PATH.read_text(encoding="utf-8")).get("poses", [])
    if len(poses) < 10:
        raise ValueError("Pose catalog must contain at least ten archetypes")
    return manifest, poses


def build_prompt(display_name: str, pose: dict) -> str:
    best_for = ", ".join(str(value) for value in pose.get("best_for", []))
    palette = ", ".join(str(value) for value in pose.get("palette", []))
    return (
        "Use case: identity-preserve.\n"
        "Asset type: reusable text-free 9:16 sermon speaker pose candidate.\n"
        f"Primary request: Create {display_name} in the pose archetype {pose['id']}.\n"
        "Input images: all supplied images are strict identity references; preserve the "
        "same recognizable face, age, glasses if present, hair, skin tone and body proportions.\n"
        f"Expression: {pose['expression']}.\n"
        f"Gesture: {pose['gesture']}.\n"
        f"Composition/framing: {pose['framing']}; rule of thirds; keep one clean side for later typography.\n"
        f"Lighting/mood: {pose['lighting']}.\n"
        f"Color palette: {palette}.\n"
        f"Story compatibility: {best_for}.\n"
        "Style/medium: photorealistic editorial portrait, cinematic depth, realistic skin, "
        "anatomically correct hands, believable handheld microphone when natural.\n"
        "Constraints: exactly one speaker; text-free; background remains simple enough for later "
        "layer compositing; light direction must be visually explicit and physically coherent.\n"
        "Avoid: written words, captions, lettering, logo, watermark, lower third, audience, second "
        "person, duplicated limbs, deformed hands, synthetic plastic skin, halo, wings, costume."
    )


def prepare_jobs(
    *, speaker_keys: list[str] | None = None, pose_ids: list[str] | None = None, stamp: str | None = None
) -> list[dict]:
    manifest, poses = load_matrix()
    speakers = manifest.get("speakers", {})
    selected_speakers = speaker_keys or list(speakers)
    selected_poses = [pose for pose in poses if not pose_ids or pose.get("id") in pose_ids]
    unknown_speakers = sorted(set(selected_speakers) - set(speakers))
    if unknown_speakers:
        raise ValueError(f"Unknown speaker keys: {', '.join(unknown_speakers)}")
    if pose_ids and len(selected_poses) != len(set(pose_ids)):
        known = {str(pose.get('id')) for pose in poses}
        raise ValueError(f"Unknown pose ids: {', '.join(sorted(set(pose_ids) - known))}")

    job_stamp = stamp or date.today().isoformat()
    prepared: list[dict] = []
    repertoire_root = MANIFEST_PATH.parent
    for speaker_key in selected_speakers:
        speaker = speakers[speaker_key]
        job_dir = repertoire_root / speaker_key / "jobs" / f"{job_stamp}_pose_matrix"
        job_dir.mkdir(parents=True, exist_ok=True)
        references = [
            str((repertoire_root / str(item["path"])).resolve())
            for item in speaker.get("source_images", [])
            if isinstance(item, dict) and item.get("path")
        ]
        jobs = []
        for pose in selected_poses:
            prompt = build_prompt(str(speaker.get("display_name") or speaker_key), pose)
            prompt_path = job_dir / f"{pose['id']}.prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            jobs.append({
                "pose_id": pose["id"],
                "prompt_path": prompt_path.name,
                "output": f"{speaker_key}_{pose['id']}_candidate_v1.png",
                "status": "prepared",
                "approved": False,
            })
        payload = {
            "schema_version": 1,
            "provider": "azure_openai_api",
            "created": job_stamp,
            "speaker_key": speaker_key,
            "display_name": speaker.get("display_name") or speaker_key,
            "status": "prepared",
            "references": references,
            "jobs": jobs,
        }
        job_path = job_dir / "job.json"
        job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prepared.append({"job_path": job_path, "payload": payload})
    return prepared


def execute_jobs(prepared: list[dict], *, quality: str, size: str, confirm_count: int | None) -> int:
    all_jobs = [(entry, job) for entry in prepared for job in entry["payload"]["jobs"]]
    if confirm_count != len(all_jobs):
        raise ValueError(
            f"Execution would create {len(all_jobs)} paid images; pass --confirm-count {len(all_jobs)}"
        )
    from Components.AzureImageGeneration import generate_image

    completed = 0
    for entry, job in all_jobs:
        job_path = entry["job_path"]
        job_dir = job_path.parent
        prompt = (job_dir / job["prompt_path"]).read_text(encoding="utf-8")
        output = job_dir / job["output"]
        references = [Path(value) for value in entry["payload"]["references"]]
        generate_image(prompt, output, reference_images=references, size=size, quality=quality)
        job["status"] = "manual_review_required"
        completed += 1
        job_path.write_text(json.dumps(entry["payload"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return completed


def _split_csv(value: str) -> list[str] | None:
    values = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return None if not values or values == ["all"] else values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakers", default="all", help="comma-separated manifest keys or all")
    parser.add_argument("--poses", default="all", help="comma-separated pose ids or all")
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-count", type=int)
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    parser.add_argument("--size", default="1024x1536")
    args = parser.parse_args()

    prepared = prepare_jobs(
        speaker_keys=_split_csv(args.speakers), pose_ids=_split_csv(args.poses), stamp=args.stamp
    )
    count = sum(len(entry["payload"]["jobs"]) for entry in prepared)
    print(f"Prepared {count} pose jobs for {len(prepared)} speaker(s).")
    if args.execute:
        completed = execute_jobs(
            prepared, quality=args.quality, size=args.size, confirm_count=args.confirm_count
        )
        print(f"Generated {completed} review candidates; none were approved automatically.")
    else:
        print("Preparation only. Add --execute with the exact --confirm-count to call Azure GPT-image-2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
