from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from Components.ThumbnailStudioBridge import (
    discover_thumbnail_artifacts,
    build_canva_review_packet,
    build_comfyui_job_spec,
    export_thumbnail_studio_bundle_from_output_dir,
    find_best_thumbnail_artifact,
    export_thumbnail_studio_bundle,
)


def _make_png(path: Path, color: tuple[int, int, int] = (24, 24, 28)) -> Path:
    image = Image.new("RGBA", (256, 256), (*color, 255))
    image.save(path, "PNG")
    return path


def _write_report(path: Path, **payload) -> Path:
    report = {
        "output": str(path.with_suffix(".png")),
        "title": "GOTT IST TREU",
        "template": "warm_gold",
        "format": "9x16",
        "effect_profile": "premium",
        "score": 91.2,
        "provider_used": "birefnet",
        "source_kind": "path",
        "source_path": "/tmp/source.mp4",
        **payload,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_comfyui_job_spec_infers_workflow_and_reference_assets(tmp_path):
    thumbnail = _make_png(tmp_path / "thumbnail.png")
    report = _write_report(tmp_path / "thumbnail.thumbnail_report.json")
    comfyui_root = tmp_path / "ComfyUI"
    (comfyui_root / "blueprints").mkdir(parents=True)
    (comfyui_root / "blueprints" / "Image Edit (Flux.2 Dev).json").write_text("{}", encoding="utf-8")
    references = tmp_path / "references"
    references.mkdir()
    _make_png(references / "ref_a.png", (60, 40, 20))
    _make_png(references / "ref_b.png", (20, 40, 60))

    spec = build_comfyui_job_spec(
        thumbnail,
        report_path=report,
        source_path="/tmp/source.mp4",
        comfyui_root=comfyui_root,
        reference_dir=references,
        notes="Use warm, editorial typography",
    )

    assert spec["kind"] == "comfyui_thumbnail_job"
    assert spec["workflow"]["name"] == "thumbnail_premium"
    assert any(node["node_id"] == "ParakeetThumbnailJobSpec" for node in spec["custom_nodes"])
    assert len(spec["reference_thumbnails"]) == 2
    assert spec["notes"] == "Use warm, editorial typography"
    assert any(bp["exists"] for bp in spec["workflow"]["blueprints"])


def test_export_thumbnail_studio_bundle_writes_bundle_and_review_packets(tmp_path):
    thumbnail = _make_png(tmp_path / "thumbnail.png")
    report = _write_report(tmp_path / "thumbnail.thumbnail_report.json")
    references = tmp_path / "references"
    references.mkdir()
    _make_png(references / "ref_a.png")

    created = export_thumbnail_studio_bundle(
        thumbnail,
        report_path=report,
        source_path="/tmp/source.mp4",
        output_dir=tmp_path / "exports",
        reference_dir=references,
        notes="Bundle export smoke test",
    )

    bundle = json.loads(created["bundle"].read_text(encoding="utf-8"))
    comfyui = json.loads(created["comfyui"].read_text(encoding="utf-8"))
    canva = json.loads(created["canva"].read_text(encoding="utf-8"))

    assert bundle["kind"] == "thumbnail_studio_bundle"
    assert comfyui["workflow"]["name"] == "thumbnail_premium"
    assert canva["kind"] == "canva_review_packet"
    assert canva["brand_kit_available"] is False
    assert created["bundle"].exists()
    assert created["comfyui"].exists()
    assert created["canva"].exists()


def test_build_canva_review_packet_defaults_to_manual_review(tmp_path):
    thumbnail = _make_png(tmp_path / "thumbnail.png")
    report = _write_report(tmp_path / "thumbnail.thumbnail_report.json", score=80.0, effect_profile="editorial")

    packet = build_canva_review_packet(thumbnail, report_path=report, notes="Manual Canva polish")

    assert packet["review_mode"] == "manual_review"
    assert packet["effect_profile"] == "editorial"
    assert len(packet["review_goals"]) == 3
    assert packet["notes"] == "Manual Canva polish"


def test_discover_thumbnail_artifacts_picks_highest_scoring_report(tmp_path):
    best_thumb = _make_png(tmp_path / "best.png", (120, 30, 30))
    best_report = tmp_path / "best.thumbnail_report.json"
    best_report.write_text(
        json.dumps(
            {
                "output": str(best_thumb),
                "score": 93.5,
                "effect_profile": "premium",
                "provider_used": "birefnet",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    other_dir = tmp_path / "nested"
    other_dir.mkdir()
    other_thumb = _make_png(other_dir / "other.png", (30, 120, 30))
    other_report = other_dir / "thumbnail_report.json"
    other_report.write_text(
        json.dumps(
            {
                "selected": "other.png",
                "selected_score": 80.0,
                "selected_provider": "rembg",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    discovered = discover_thumbnail_artifacts(tmp_path)
    best = find_best_thumbnail_artifact(tmp_path)

    assert discovered[0]["thumbnail"] == str(best_thumb)
    assert best["report"] == str(best_report)
    assert len(discovered) == 2
    assert other_thumb.exists()


def test_export_thumbnail_studio_bundle_from_output_dir_uses_best_report(tmp_path):
    best_thumb = _make_png(tmp_path / "thumbnail_best.png", (44, 44, 100))
    report = tmp_path / "thumbnail_report.json"
    report.write_text(
        json.dumps(
            {
                "selected": "thumbnail_best.png",
                "selected_score": 89.1,
                "selected_provider": "rembg",
                "selected_effect_profile": "editorial",
                "template": "warm_gold",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    created = export_thumbnail_studio_bundle_from_output_dir(
        tmp_path,
        output_dir=tmp_path / "studio",
        notes="Output-dir export smoke test",
    )

    assert created["bundle"].exists()
    assert created["comfyui"].exists()
    assert created["canva"].exists()
    payload = json.loads(created["bundle"].read_text(encoding="utf-8"))
    assert payload["workflow"] in {"thumbnail_editorial", "thumbnail_base", "thumbnail_premium"}
    assert best_thumb.exists()
