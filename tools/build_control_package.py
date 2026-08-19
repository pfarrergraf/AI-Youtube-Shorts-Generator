from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path("/home/benjamin_graf/parakeet_uv")
OUT = ROOT / "automation/control_package_2026-08-14"


def main() -> None:
    for path in (
        OUT / "thumbnails/olaf_latzel",
        OUT / "thumbnails/movechurch",
        OUT / "videos_by_speaker",
        OUT / "archives",
    ):
        path.mkdir(parents=True, exist_ok=True)

    for source, destination in (
        (Path("/home/benjamin_graf/upload_to_social_media"), OUT / "thumbnails/olaf_latzel"),
        (Path("/home/benjamin_graf/upload_to_social_media_movechurch"), OUT / "thumbnails/movechurch"),
    ):
        for image in sorted(source.glob("*_thumb.jpg")):
            shutil.copy2(image, destination / image.name)

    state = json.loads((ROOT / "automation/movechurch_overnight_state.json").read_text())
    groups: dict[str, list[Path]] = {}
    for record in state.get("processed_videos", {}).values():
        speaker = record.get("speaker") or "Unbekannt"
        for source in map(Path, record.get("generated_shorts", [])):
            if speaker == "Unknown Speaker":
                match = re.search(r"｜-([^｜]+)-｜-move-church", source.name)
                speaker_name = match.group(1).strip() if match else speaker
                speaker = "Leo & Susanna Bigger" if speaker_name.lower() == "leo & susanna bigger" else speaker_name.title()
            groups.setdefault(speaker, []).append(source)

    olaf_sources = sorted(Path("/home/benjamin_graf/upload_to_social_media/synced_to_cloud").glob("*.mp4"))
    groups["Pastor Olaf Latzel"] = olaf_sources

    manifest = []
    for speaker, sources in sorted(groups.items()):
        unique = []
        for source in sources:
            if source.exists() and source not in unique:
                unique.append(source)
        destination = OUT / "videos_by_speaker" / (re.sub(r"[^a-z0-9]+", "_", speaker.lower()).strip("_") or "unknown")
        destination.mkdir(parents=True, exist_ok=True)
        selected = []
        for index, source in enumerate(unique[:2], 1):
            target = destination / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            selected.append({"source": str(source), "file": str(target.relative_to(OUT)), "bytes": target.stat().st_size})
        manifest.append({"speaker": speaker, "selected_count": len(selected), "videos": selected})

    thumbnail_count = len(list((OUT / "thumbnails").rglob("*_thumb.jpg")))
    metadata = {"created": "2026-08-14", "thumbnail_count": thumbnail_count, "speakers": manifest}
    (OUT / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    lines = ["Kontrollpaket 2026-08-14", "", f"Thumbnails: {thumbnail_count}"]
    for entry in manifest:
        lines.append(f"{entry['speaker']}: {entry['selected_count']} Videos")
        lines.extend(f"  - {item['file']}" for item in entry["videos"])
    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"thumbnails": thumbnail_count, "speakers": [(x["speaker"], x["selected_count"]) for x in manifest]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
