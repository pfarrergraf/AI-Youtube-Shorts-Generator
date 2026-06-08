"""
Render simple preview images for installed font families under FONT_DIR.
Usage:
    source .venv/bin/activate
    python tools/font_preview.py --slugs "anton,bangers,unbounded,work-sans" 

Outputs: font_previews/<slug>.png and font_previews/grid.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

FONT_DIR = Path.home() / ".local/share/fonts/mc_thumbnails"
OUT_DIR = Path(__file__).parent.parent / "font_previews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXT = "EINE WIE KEINE"


def find_font_for_slug(slug: str) -> Path | None:
    # Look under FONT_DIR/slug for TTF files
    d = FONT_DIR / slug
    if d.exists():
        ttf = sorted(d.glob("*.ttf"))
        if ttf:
            return ttf[0]
    # Fallback: search any ttf with slug in the path/name
    for f in FONT_DIR.rglob("*.ttf"):
        if slug.replace("-", "").lower() in f.stem.replace("-", "").lower():
            return f
    return None


def render_preview(font_path: Path | None, slug: str, out_path: Path):
    w, h = 1080, 360
    img = Image.new("RGB", (w, h), (18, 18, 20))
    draw = ImageDraw.Draw(img)
    if font_path and font_path.exists():
        try:
            # Choose a large size that fits height
            size = int(h * 0.5)
            font = ImageFont.truetype(str(font_path), size=size)
            # shrink until it fits
            bbox = draw.textbbox((0,0), SAMPLE_TEXT, font=font)
            while bbox[2] - bbox[0] > w - 200 and size > 18:
                size = int(size * 0.9)
                font = ImageFont.truetype(str(font_path), size=size)
                bbox = draw.textbbox((0,0), SAMPLE_TEXT, font=font)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # draw label background bar
    draw.rectangle([0, h-48, w, h], fill=(0,0,0))
    # center text
    bbox = draw.textbbox((0,0), SAMPLE_TEXT, font=font)
    tx = (w - (bbox[2]-bbox[0])) // 2
    ty = (h - (bbox[3]-bbox[1])) // 2 - 10
    draw.text((tx, ty), SAMPLE_TEXT, font=font, fill=(255,255,255))
    # label slug
    label_font = ImageFont.load_default()
    draw.text((10, h-36), slug, font=label_font, fill=(200,200,200))
    img.save(out_path, "PNG", optimize=True)


def make_grid(images: list[Path], out: Path):
    if not images:
        return
    cols = 2
    thumb_w, thumb_h = 1080, 360
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (14,14,18))
    for i, p in enumerate(images):
        img = Image.open(p).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h
        grid.paste(img, (x, y))
    grid.save(out, "PNG", optimize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slugs', default='', help='Comma-separated slugs to preview')
    args = parser.parse_args()
    slugs = [s.strip() for s in args.slugs.split(',') if s.strip()]
    outputs = []
    for slug in slugs:
        font_path = find_font_for_slug(slug)
        out = OUT_DIR / f"{slug}.png"
        render_preview(font_path, slug, out)
        outputs.append(out)
        print(f"Saved preview: {out} (font: {font_path})")
    grid_out = OUT_DIR / 'grid.png'
    make_grid(outputs, grid_out)
    print(f"Saved grid: {grid_out}")
