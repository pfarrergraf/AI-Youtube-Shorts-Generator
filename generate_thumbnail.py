"""
generate_thumbnail.py
=====================
Lokaler Thumbnail-Generator für Move Church.
Kein Server, kein n8n, kein Docker — direkt ausführbar.

Beispiele:

  # Einfachste Nutzung — Auto-Split des Titels:
  python generate_thumbnail.py --source predigt.mp4 --title "GOTT NUTZT WEN ER WILL"

  # Explizite Layer-Aufteilung (hinten / vorne):
  python generate_thumbnail.py \\
      --source predigt.mp4 \\
      --back "GOTT NUTZT WEN ER" \\
      --front "WILL" \\
      --template energy_orange \\
      --logo

  # Alle 3 Varianten auf einmal:
  python generate_thumbnail.py \\
      --source predigt.mp4 \\
      --title "EINE WIE KEINE" \\
      --all-templates \\
      --fmt 9x16

  # Mit Zeitstempel für besten Frame + Symbol + Rahmen:
  python generate_thumbnail.py \\
      --source predigt.mp4 \\
      --title "ES WIRD ZEIT" \\
      --timestamp 620 \\
      --template warm_gold \\
      --symbol cross \\
      --frame \\
      --logo

  # YouTube 16:9 mit rembg (bessere Freistellung):
  python generate_thumbnail.py \\
      --source predigt.mp4 \\
      --title "WARUM GOTT NICHT SCHWEIGT" \\
      --back "WARUM GOTT" \\
      --front "NICHT SCHWEIGT" \\
      --accent NICHT \\
      --template warm_gold \\
      --fmt 16x9 \\
      --provider rembg \\
      --logo

  # Direkt aus Pydantic-AI / eigenem Code:
  from generate_thumbnail import quick_generate
  path = quick_generate("predigt.mp4", title="GOTT IST TREU", template="navy_dark")

  # Schnellvergleich aller Templates (ohne Video, mit Platzhalter-Silhouette):
  python generate_thumbnail.py --preview-all

Verfügbare Templates:
  navy_dark      – Dunkelblau, blaue Lichtstrahlen, konzentrische Ringe
  energy_orange  – Orange/Violett, kursiver Text, Energie-Linien
  warm_gold      – Gold/Warmton, Scheinwerfer, Gold-Akzente (Serif)
  cinematic_dark – Film-Noir, Spotlight, Serif, Gold-Trennlinie, Filmkorn
  fire_red       – Rot/Schwarz radial, Energie-Linien, kursiv, intensiv
  heaven_blue    – Himmelblaues Licht von oben, Ringe, weiße Linie
  bold_minimal   – Rein typografisch, riesige Back-Words, Orange vorn
  sunset_warm    – Lila/Amber-Verlauf, warmes Spotlight, Gold-Akzente

Verfügbare Symbole:
  cross, bible, fire, dove, star, heart, anchor, crown, arrow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Components.ThumbnailMoveChurch import (
    FORMATS,
    EFFECT_PROFILES,
    SYMBOLS,
    TEMPLATES,
    generate_all_variants,
    generate_move_church_thumbnail,
    auto_split_title,
    extract_best_frame,
    extract_frame_at,
)
from Components.ThumbnailStudioBridge import export_thumbnail_studio_bundle


def _load_layer_specs(value: str | None) -> list[dict] | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("@"):
        payload = Path(raw[1:]).read_text(encoding="utf-8")
    else:
        candidate = Path(raw)
        payload = candidate.read_text(encoding="utf-8") if candidate.exists() else raw
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("layer specs JSON must be a list of layer dictionaries")
    return data


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE API (für Pydantic AI / direkte Script-Integration)
# ════════════════════════════════════════════════════════════════════════════

def quick_generate(
    source: str,
    *,
    title: str,
    template: str = "navy_dark",
    fmt: str = "9x16",
    show_logo: bool = False,
    accent: str | None = None,
    symbol: str | None = None,
    effect_profile: str = "classic",
    badge_text: str | None = None,
    badge_mode: str = "arc",
    badge_position: str = "top_right",
    layer_specs: list[dict] | None = None,
    output_dir: str = ".",
    provider: str = "auto",
    timestamp: float | None = None,
) -> str:
    """
    One-liner API for generating a single thumbnail.
    Returns the output file path.

    Example:
        path = quick_generate("predigt.mp4", title="GOTT NUTZT WEN ER WILL")
    """
    stem = Path(source).stem
    out  = str(Path(output_dir) / f"{stem}_{template}_{fmt}.png")

    src = source
    if timestamp is not None and Path(source).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        src = extract_best_frame(source, prefer_timestamp=timestamp)

    generate_move_church_thumbnail(
        src,
        title=title,
        template=template,
        fmt=fmt,
        show_logo=show_logo,
        accent_word=accent,
        symbol=symbol,
        effect_profile=effect_profile,
        badge_text=badge_text,
        badge_mode=badge_mode,
        badge_position=badge_position,
        layer_specs=layer_specs,
        output_path=out,
        bg_removal_provider=provider,
    )
    return out


def batch_generate(
    videos: list[dict],
    *,
    output_dir: str = ".",
    provider: str = "auto",
) -> list[dict]:
    """
    Generate thumbnails for a list of videos.

    videos: list of dicts with keys:
        source    (str)            – video or image path
        title     (str)            – full title (auto-split)
        back      (str, optional)  – explicit back words
        front     (str, optional)  – explicit front words
        template  (str, optional)  – defaults to "navy_dark"
        fmt       (str, optional)  – "9x16" or "16x9"
        logo      (bool, optional) – show logo
        accent    (str, optional)  – accent word
        symbol    (str, optional)  – symbol name
        timestamp (float, optional)– preferred video timestamp

    Returns list of result dicts with 'source', 'output', 'template', 'fmt'.

    Example:
        results = batch_generate([
            {"source": "predigt1.mp4", "title": "GOTT NUTZT WEN ER WILL",
             "template": "energy_orange", "logo": True},
            {"source": "predigt2.mp4", "title": "EINE WIE KEINE",
             "template": "navy_dark"},
        ])
    """
    results = []
    for spec in videos:
        try:
            source   = spec["source"]
            title    = spec.get("title", "")
            back     = spec.get("back", "")
            front    = spec.get("front", "")
            template = spec.get("template", "navy_dark")
            fmt      = spec.get("fmt", "9x16")
            logo     = spec.get("logo", False)
            accent   = spec.get("accent")
            symbol   = spec.get("symbol")
            ts       = spec.get("timestamp")
            effect_profile = spec.get("effect_profile", "classic")
            badge_text = spec.get("badge_text")
            badge_mode = spec.get("badge_mode", "arc")
            badge_position = spec.get("badge_position", "top_right")
            layer_specs = spec.get("layer_specs")

            stem = Path(source).stem
            out  = str(Path(output_dir) / f"{stem}_{template}_{fmt}.png")

            src = source
            if ts is not None:
                src = extract_best_frame(source, prefer_timestamp=ts)

            generate_move_church_thumbnail(
                src,
                title=title,
                title_back=back,
                title_front=front,
                template=template,
                fmt=fmt,
                show_logo=logo,
                accent_word=accent,
                symbol=symbol,
                effect_profile=effect_profile,
                badge_text=badge_text,
                badge_mode=badge_mode,
                badge_position=badge_position,
                layer_specs=layer_specs,
                output_path=out,
                bg_removal_provider=provider,
            )
            results.append({
                "source": source,
                "output": out,
                "template": template,
                "fmt": fmt,
                "success": True,
            })
            print(f"✓ {Path(source).name} → {out}")
        except Exception as exc:
            print(f"✗ {spec.get('source', '?')}: {exc}")
            results.append({
                "source": spec.get("source", ""),
                "output": None,
                "success": False,
                "error": str(exc),
            })
    return results


# ════════════════════════════════════════════════════════════════════════════
# TEMPLATE PREVIEW GRID
# ════════════════════════════════════════════════════════════════════════════

def _placeholder_subject(width: int = 540, height: int = 1080) -> dict:
    """Solid grey speaker silhouette so the preview works without a video."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    grey = (118, 120, 128, 255)
    cx = width // 2
    head_r = width // 5
    head_cy = height // 6
    # Head
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=grey)
    # Neck
    draw.rectangle([cx - head_r // 2, head_cy + head_r - 8, cx + head_r // 2, head_cy + head_r + height // 16], fill=grey)
    # Shoulders/torso
    draw.ellipse(
        [cx - int(width * 0.46), head_cy + head_r + height // 18,
         cx + int(width * 0.46), height + int(height * 0.45)],
        fill=grey,
    )
    return {
        "speaker_rgba": img,
        "face_box": (cx - head_r, head_cy - head_r, head_r * 2, head_r * 2),
        "coverage": 0.30,
        "provider_used": "placeholder",
        "removal_attempts": [],
        "caption_cleanup": {},
    }


def generate_template_preview(
    output: str = "template_preview_grid.png",
    *,
    back: str = "EINE WIE",
    front: str = "KEINE",
) -> str:
    """Render all templates with a placeholder silhouette into a 3x3 grid PNG."""
    import numpy as np
    from PIL import Image, ImageDraw

    from Components.ThumbnailMoveChurch import _load_mc_font

    cell_w, cell_h = 360, 640
    cols, rows = 3, 3
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (14, 14, 18))
    label_font = _load_mc_font(34)

    frame = np.full((1920, 1080, 3), 18, dtype=np.uint8)  # dark dummy frame (BGR)
    subject = _placeholder_subject()

    for i, tmpl in enumerate(TEMPLATES):
        print(f"[Preview] {tmpl} …")
        img = generate_move_church_thumbnail(
            frame,
            title_back=back,
            title_front=front,
            template=tmpl,
            fmt="9x16",
            _precomputed_subject=subject,
        )
        cell = img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(cell)
        # Label bar at the bottom of each cell
        draw.rectangle([0, cell_h - 44, cell_w, cell_h], fill=(0, 0, 0))
        draw.text((10, cell_h - 40), tmpl, font=label_font, fill=(255, 255, 255))
        grid.paste(cell, ((i % cols) * cell_w, (i // cols) * cell_h))

    grid.save(output, "PNG", optimize=True)
    print(f"\n✓ Template-Vorschau gespeichert: {output}")
    return output


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Move Church Thumbnail Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", default=None,
                        help="Video oder Bild-Pfad (nicht nötig bei --preview-all)")
    parser.add_argument("--preview-all", action="store_true",
                        help="3x3-Vorschau-Grid aller Templates erzeugen "
                             "(mit Platzhalter-Silhouette, kein Video nötig)")
    parser.add_argument("--title", default=None,
                        help='Voller Titel (Auto-Split). Z.B. "EINE WIE KEINE"')
    parser.add_argument("--back", default="",
                        help='Wörter HINTER dem Prediger. Z.B. "EINE WIE"')
    parser.add_argument("--front", default="",
                        help='Wörter VOR dem Prediger. Z.B. "KEINE"')
    parser.add_argument("--template", default="navy_dark",
                        choices=TEMPLATES)
    parser.add_argument("--fmt", default="9x16",
                        choices=list(FORMATS))
    parser.add_argument("--output", default=None,
                        help="Ausgabe-PNG-Pfad (default: <source>_<template>_<fmt>.png)")
    parser.add_argument("--logo", action="store_true",
                        help="Move Church Logo anzeigen")
    parser.add_argument("--accent", default=None,
                        help="Akzent-Wort (in Orange/Gold)")
    parser.add_argument("--symbol", default=None,
                        choices=list(SYMBOLS) + ["none"],
                        help="Symbol")
    parser.add_argument("--frame", action="store_true",
                        help="Dekorativer Rahmen")
    parser.add_argument("--arrow", action="store_true",
                        help="Pfeil-Element")
    parser.add_argument("--all-templates", action="store_true",
                        help="Alle 3 Templates generieren")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "birefnet", "rmbg", "rembg", "grabcut_local"],
                        help="Hintergrund-Entfernung")
    parser.add_argument("--outline", default="palette_rim",
                        choices=["palette_rim", "creator_white", "creator_blue", "sermon_gold"],
                        help="Sprecher-Outline: palette_rim = Template-farbiges Rim-Light (Default)")
    parser.add_argument("--effect-profile", default="classic",
                        choices=EFFECT_PROFILES,
                        help="High-end render preset")
    parser.add_argument("--badge-text", default=None,
                        help="Optionaler Badge-Text für Path-/Arc-Text")
    parser.add_argument("--badge-mode", default="arc",
                        choices=["arc", "circle"],
                        help="Badge-Text als Bogen oder Kreis")
    parser.add_argument("--badge-position", default="top_right",
                        choices=["top_left", "top_right", "upper_left", "bottom_left"],
                        help="Badge-Position auf dem Thumbnail")
    parser.add_argument("--layers-json", default=None,
                        help="JSON string or @path to a list of custom layer specs")
    parser.add_argument("--timestamp", type=float, default=None,
                        help="Exakter Video-Zeitstempel in Sekunden (erzwingt diesen Frame)")
    parser.add_argument("--export-studio-bundle", action="store_true",
                        help="Thumbnail + Report als ComfyUI/Canva-Studio-Bundle exportieren")
    parser.add_argument("--comfyui-root", default=None,
                        help="Pfad zum lokalen ComfyUI-Checkout (default: auto-detect)")
    parser.add_argument("--studio-notes", default="",
                        help="Freitext für den ComfyUI-/Canva-Handoff")
    parser.add_argument("--relight", action="store_true",
                        help="IC-Light Relighting des Sprechers (optional, lädt ~6 GB Gewichte beim ersten Lauf)")
    parser.add_argument("--output-dir", default=".",
                        help="Ausgabe-Verzeichnis (bei --all-templates)")

    args = parser.parse_args()
    layer_specs = _load_layer_specs(args.layers_json)

    if args.preview_all:
        generate_template_preview(
            "template_preview_grid.png",
            back=args.back or "EINE WIE",
            front=args.front or "KEINE",
        )
        return

    if not args.source:
        parser.error("--source ist erforderlich (außer mit --preview-all)")

    source = args.source
    if args.timestamp is not None:
        ext = Path(source).suffix.lower()
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            print(f"Extrahiere Frame bei {args.timestamp}s …")
            source = extract_frame_at(source, args.timestamp)

    sym = args.symbol if args.symbol != "none" else None

    if args.all_templates:
        title = args.title or f"{args.back} {args.front}".strip() or "PREDIGT"
        out_dir = args.output_dir or (str(Path(args.output).parent) if args.output else ".")
        results = generate_all_variants(
            source,
            title=title,
            fmt=args.fmt,
            show_logo=args.logo,
            symbol=sym,
            output_dir=out_dir,
            bg_removal_provider=args.provider,
            outline_preset=args.outline,
            effect_profile=args.effect_profile,
            badge_text=args.badge_text,
            badge_mode=args.badge_mode,
            badge_position=args.badge_position,
            layer_specs=layer_specs,
            title_back=args.back,
            title_front=args.front,
            accent_word=args.accent,
            relight=args.relight,
        )
        print("\nGenerierte Thumbnails:")
        for tmpl, path in results.items():
            print(f"  {tmpl:20s} → {path}")
        if args.export_studio_bundle:
            for tmpl, path in results.items():
                export_thumbnail_studio_bundle(
                    path,
                    source_path=args.source,
                    comfyui_root=args.comfyui_root,
                    notes=args.studio_notes or f"Generated from template={tmpl}, effect_profile={args.effect_profile}",
                )
    else:
        out_path = args.output
        if not out_path:
            stem = Path(args.source).stem
            out_path = str(Path(args.output_dir or ".") / f"{stem}_{args.template}_{args.fmt}.png")

        generate_move_church_thumbnail(
            source,
            title=args.title,
            title_back=args.back,
            title_front=args.front,
            template=args.template,
            fmt=args.fmt,
            show_logo=args.logo,
            accent_word=args.accent,
            symbol=sym,
            show_frame=args.frame,
            show_arrow=args.arrow,
            output_path=out_path,
            bg_removal_provider=args.provider,
            outline_preset=args.outline,
            effect_profile=args.effect_profile,
            badge_text=args.badge_text,
            badge_mode=args.badge_mode,
            badge_position=args.badge_position,
            layer_specs=layer_specs,
            relight=args.relight,
        )
        if args.export_studio_bundle:
            export_thumbnail_studio_bundle(
                out_path,
                source_path=args.source,
                comfyui_root=args.comfyui_root,
                notes=args.studio_notes or f"Generated with template={args.template}, effect_profile={args.effect_profile}",
            )
        print(f"\n✓ Gespeichert: {out_path}")


if __name__ == "__main__":
    _cli()
