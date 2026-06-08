from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_BLUEPRINTS = (
    ("thumbnail_base", "blueprints/Remove Background (BiRefNet).json"),
    ("thumbnail_editorial", "blueprints/Image Edit (Flux.2 Dev).json"),
    ("thumbnail_premium", "blueprints/Image to Layers(Qwen-Image-Layered).json"),
    ("thumbnail_cleanup", "blueprints/Film Grain.json"),
    ("thumbnail_prompt", "blueprints/Prompt Enhance.json"),
)

_CUSTOM_NODE_CATALOG = (
    {
        "node_id": "ParakeetThumbnailJobSpec",
        "display_name": "Parakeet Thumbnail Job Spec",
        "role": "normalize thumbnail inputs into a reproducible JSON contract",
    },
    {
        "node_id": "ParakeetThumbnailWorkflowPreset",
        "display_name": "Parakeet Thumbnail Workflow Preset",
        "role": "select the recommended ComfyUI blueprint and finishing path",
    },
    {
        "node_id": "ParakeetCanvaReviewPacket",
        "display_name": "Parakeet Canva Review Packet",
        "role": "prepare a manual-review bundle for Canva or collaborators",
    },
)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return candidate.resolve(strict=False)


def resolve_comfyui_root(explicit_root: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    explicit = _resolve_existing_path(explicit_root)
    if explicit and explicit.exists():
        return explicit

    for env_name in ("PARAKEET_COMFYUI_ROOT", "COMFYUI_ROOT"):
        env_value = _resolve_existing_path(os.environ.get(env_name))
        if env_value and env_value.exists():
            return env_value

    workspace = _workspace_root()
    candidates.extend(
        [
            workspace / "ComfyUI",
            workspace.parent / "ComfyUI",
            Path.home() / "ComfyUI",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return explicit if explicit is not None else None


def load_thumbnail_report(report_path: str | Path | None) -> dict[str, Any]:
    if report_path is None:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_report_thumbnail(report: dict[str, Any], report_path: Path) -> Path | None:
    candidates: list[str] = []
    for key in ("output", "selected", "thumbnail", "thumbnail_path"):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (report_path.parent / path).resolve(strict=False)
        if path.exists():
            return path
    best = report_path.parent / "thumbnail_best.png"
    if best.exists():
        return best.resolve()
    return None


def discover_thumbnail_artifacts(search_root: str | Path) -> list[dict[str, Any]]:
    root = Path(search_root).expanduser().resolve(strict=False)
    if not root.exists():
        return []

    report_paths = sorted(
        {path.resolve() for path in root.rglob("thumbnail_report.json")}
        | {path.resolve() for path in root.rglob("*.thumbnail_report.json")}
    )
    discovered: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = load_thumbnail_report(report_path)
        thumbnail = _resolve_report_thumbnail(report, report_path)
        if thumbnail is None:
            continue
        discovered.append(
            {
                "thumbnail": str(thumbnail),
                "report": str(report_path),
                "score": float(report.get("score") or report.get("selected_score") or 0.0),
                "effect_profile": report.get("effect_profile") or report.get("selected_effect_profile") or "classic",
                "provider_used": report.get("provider_used") or report.get("selected_provider") or "unknown",
                "template": report.get("template") or report.get("selected_template") or "",
                "mtime": max(report_path.stat().st_mtime, thumbnail.stat().st_mtime),
            }
        )

    discovered.sort(key=lambda item: (float(item.get("score") or 0.0), float(item.get("mtime") or 0.0)), reverse=True)
    return discovered


def find_best_thumbnail_artifact(search_root: str | Path) -> dict[str, Any] | None:
    discovered = discover_thumbnail_artifacts(search_root)
    return discovered[0] if discovered else None


def collect_reference_thumbnails(reference_dir: str | Path | None = None, *, limit: int = 6) -> list[str]:
    root = Path(reference_dir).expanduser() if reference_dir else _workspace_root() / "examples" / "thumbnails"
    if not root.exists():
        return []
    files = [
        path.resolve()
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    return [str(path) for path in files[: max(0, int(limit))]]


def _infer_workflow(report: dict[str, Any]) -> dict[str, Any]:
    profile = str(report.get("effect_profile") or "classic").strip().lower()
    score = float(report.get("score") or 0.0)
    provider = str(report.get("provider_used") or "unknown")
    if profile in {"premium", "poster"} or score >= 88.0:
        workflow_name = "thumbnail_premium"
        rationale = "high-drama rendering with extra separation and finish"
    elif profile in {"editorial", "halo"} or score >= 75.0:
        workflow_name = "thumbnail_editorial"
        rationale = "clean editorial look with readability-first layout"
    else:
        workflow_name = "thumbnail_base"
        rationale = "safe default pipeline with robust cutout fallback"
    if provider in {"birefnet", "rembg", "grabcut_local"} and workflow_name != "thumbnail_base":
        rationale += f"; cutout provider={provider}"
    return {
        "name": workflow_name,
        "rationale": rationale,
        "blueprints": [
            {
                "name": preset_name,
                "path": str(path),
                "exists": path.exists(),
            }
            for preset_name, path in _iter_blueprints(resolve_comfyui_root(report.get("comfyui_root")))
        ],
    }


def _iter_blueprints(comfyui_root: Path | None) -> list[tuple[str, Path]]:
    if comfyui_root is None:
        comfyui_root = resolve_comfyui_root()
    root = comfyui_root if comfyui_root is not None else _workspace_root()
    return [(name, root / relative_path) for name, relative_path in _DEFAULT_BLUEPRINTS]


def build_comfyui_job_spec(
    thumbnail_path: str | Path,
    *,
    report_path: str | Path | None = None,
    source_path: str | Path | None = None,
    comfyui_root: str | Path | None = None,
    reference_dir: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    thumbnail = Path(thumbnail_path).expanduser().resolve(strict=False)
    report = load_thumbnail_report(report_path or thumbnail.with_suffix(".thumbnail_report.json"))
    comfyui_root_path = resolve_comfyui_root(comfyui_root)
    source_value = source_path or report.get("source_path") or report.get("source")
    source_path_resolved = str(_resolve_existing_path(source_value)) if source_value else None
    workflow = _infer_workflow(report)
    workflow["blueprints"] = [
        {
            "name": preset_name,
            "path": str(path),
            "exists": path.exists(),
        }
        for preset_name, path in _iter_blueprints(comfyui_root_path)
    ]
    references = collect_reference_thumbnails(reference_dir)
    return {
        "schema_version": "1.0",
        "kind": "comfyui_thumbnail_job",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thumbnail": str(thumbnail),
        "thumbnail_report": str(Path(report_path).expanduser().resolve(strict=False)) if report_path else str(thumbnail.with_suffix(".thumbnail_report.json")),
        "source_path": source_path_resolved,
        "source_kind": report.get("source_kind"),
        "title": report.get("title") or report.get("metrics", {}).get("title"),
        "template": report.get("template"),
        "format": report.get("format"),
        "effect_profile": report.get("effect_profile"),
        "score": report.get("score"),
        "provider_used": report.get("provider_used"),
        "workflow": workflow,
        "custom_nodes": list(_CUSTOM_NODE_CATALOG),
        "reference_thumbnails": references,
        "notes": notes or "",
    }


def build_canva_review_packet(
    thumbnail_path: str | Path,
    *,
    report_path: str | Path | None = None,
    reference_dir: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    thumbnail = Path(thumbnail_path).expanduser().resolve(strict=False)
    report = load_thumbnail_report(report_path or thumbnail.with_suffix(".thumbnail_report.json"))
    references = collect_reference_thumbnails(reference_dir)
    return {
        "schema_version": "1.0",
        "kind": "canva_review_packet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thumbnail": str(thumbnail),
        "thumbnail_report": str(Path(report_path).expanduser().resolve(strict=False)) if report_path else str(thumbnail.with_suffix(".thumbnail_report.json")),
        "score": report.get("score"),
        "effect_profile": report.get("effect_profile"),
        "provider_used": report.get("provider_used"),
        "review_mode": "manual_review",
        "brand_kit_available": False,
        "review_goals": [
            "Check readability at mobile size",
            "Check face safety and crop balance",
            "Check whether typography matches the reference thumbnails",
        ],
        "reference_thumbnails": references,
        "notes": notes or "Canva template search is currently a manual review path.",
    }


def export_thumbnail_studio_bundle(
    thumbnail_path: str | Path,
    *,
    report_path: str | Path | None = None,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    comfyui_root: str | Path | None = None,
    reference_dir: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Path]:
    thumbnail = Path(thumbnail_path).expanduser().resolve(strict=False)
    if not thumbnail.exists():
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail}")

    report = Path(report_path).expanduser().resolve(strict=False) if report_path else thumbnail.with_suffix(".thumbnail_report.json")
    bundle_dir = Path(output_dir).expanduser().resolve(strict=False) if output_dir else thumbnail.parent
    bundle_dir.mkdir(parents=True, exist_ok=True)

    comfyui_spec = build_comfyui_job_spec(
        thumbnail,
        report_path=report,
        source_path=source_path,
        comfyui_root=comfyui_root,
        reference_dir=reference_dir,
        notes=notes,
    )
    canva_packet = build_canva_review_packet(
        thumbnail,
        report_path=report,
        reference_dir=reference_dir,
        notes=notes,
    )

    bundle = {
        "schema_version": "1.0",
        "kind": "thumbnail_studio_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thumbnail": str(thumbnail),
        "thumbnail_report": str(report),
        "comfyui_spec": str(bundle_dir / f"{thumbnail.stem}.comfyui_job.json"),
        "canva_packet": str(bundle_dir / f"{thumbnail.stem}.canva_review.json"),
        "workflow": comfyui_spec["workflow"]["name"],
        "notes": notes or "",
    }

    bundle_path = bundle_dir / f"{thumbnail.stem}.thumbnail_studio.json"
    comfyui_path = bundle_dir / f"{thumbnail.stem}.comfyui_job.json"
    canva_path = bundle_dir / f"{thumbnail.stem}.canva_review.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    comfyui_path.write_text(json.dumps(comfyui_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    canva_path.write_text(json.dumps(canva_packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "bundle": bundle_path,
        "comfyui": comfyui_path,
        "canva": canva_path,
    }


def export_thumbnail_studio_bundle_from_output_dir(
    search_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    comfyui_root: str | Path | None = None,
    reference_dir: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Path]:
    best = find_best_thumbnail_artifact(search_root)
    if best is None:
        raise FileNotFoundError(f"No thumbnail reports found under {Path(search_root).expanduser().resolve(strict=False)}")

    target_dir = Path(output_dir).expanduser().resolve(strict=False) if output_dir else Path(best["report"]).parent / "_studio"
    return export_thumbnail_studio_bundle(
        best["thumbnail"],
        report_path=best["report"],
        output_dir=target_dir,
        comfyui_root=comfyui_root,
        reference_dir=reference_dir,
        notes=notes or f"Exported from {best['report']}",
    )
