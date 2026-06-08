"""
For each installed font slug, temporarily set TEMPLATE_FONTS entries to use that font file
and render a template preview grid named font_previews/preview_<slug>.png

Usage:
    source .venv/bin/activate
    python tools/batch_font_template_preview.py --slugs "anton,bangers,..."
"""
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from generate_thumbnail import generate_template_preview
import Components.ThumbnailMoveChurch as TMC

FONT_DIR = Path.home() / ".local/share/fonts/mc_thumbnails"
OUT = ROOT / "font_previews"
OUT.mkdir(parents=True, exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--slugs', default='')
args = parser.parse_args()
slugs = [s.strip() for s in args.slugs.split(',') if s.strip()]

for slug in slugs:
    # find a ttf file for slug
    candidates = list(FONT_DIR.rglob(f"{slug}/*.ttf")) + list(FONT_DIR.rglob(f"*{slug}*.ttf"))
    if not candidates:
        print(f"No font file found for slug: {slug}")
        continue
    fontfile = candidates[0]
    filename = fontfile.name
    print(f"Using font {fontfile} for preview {slug}")

    # Patch TEMPLATE_FONTS mapping to use this filename for all templates
    orig = dict(TMC.TEMPLATE_FONTS)
    try:
        for tmpl in TMC.TEMPLATES:
            TMC.TEMPLATE_FONTS[tmpl] = (filename,)
        out = OUT / f"preview_{slug}.png"
        generate_template_preview(output=str(out), back="EINE WIE", front="KEINE")
        print(f"Saved preview for {slug} → {out}")
    finally:
        # restore
        TMC.TEMPLATE_FONTS.clear()
        TMC.TEMPLATE_FONTS.update(orig)

print("Done.")
