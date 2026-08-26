"""
Font Installer for Move Church Thumbnail Generator
===================================================
Strategie:
  1. jsDelivr CDN (fontsource-Pakete) → woff2 download → TTF via fonttools
  2. apt für System-Fonts (Roboto, Open Sans, etc. bereits installiert)

Voraussetzung:
    uv pip install fonttools brotli

Run:
    python install_fonts.py           # alles installieren
    python install_fonts.py --list    # Status
    python install_fonts.py --test    # PIL-Test
    python install_fonts.py --force   # neu herunterladen
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

try:
    from PIL import ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

FONT_DIR = Path.home() / ".local/share/fonts/mc_thumbnails"
CDN = "https://cdn.jsdelivr.net/npm/@fontsource/{slug}/files/{slug}-latin-{weight}-{style}.woff2"
UA  = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


# ────────────────────────────────────────────────────────────────────────────
# FONT DEFINITIONS
# slug    = npm package name slug (lowercase, hyphenated)
# weights = list of (weight_int, style, output_suffix) tuples
# ────────────────────────────────────────────────────────────────────────────

FONTS: list[dict] = [

    # ── MOVE CHURCH HAUPTSTIL ────────────────────────────────────────────────
    {"slug": "barlow-condensed",      "category": "condensed", "use": "Move Church Haupttitel ★★★", "weights": [
        (900, "normal", "Black"), (800, "normal", "ExtraBold"), (700, "normal", "Bold"),
        (900, "italic", "BlackItalic"), (800, "italic", "ExtraBoldItalic"),
    ]},
    {"slug": "barlow-semi-condensed", "category": "condensed", "use": "Etwas breiter, auch sehr gut ★★", "weights": [
        (900, "normal", "Black"), (800, "normal", "ExtraBold"), (700, "normal", "Bold"),
    ]},

    # ── YOUTUBE THUMBNAIL KLASSIKER ──────────────────────────────────────────
    {"slug": "bebas-neue",            "category": "display",   "use": "THE YouTube-Schrift ★★★", "weights": [
        (400, "normal", "Regular"),
    ]},
    {"slug": "anton",                 "category": "display",   "use": "Breit + schwer, sehr lesbar ★★★", "weights": [
        (400, "normal", "Regular"),
    ]},
    {"slug": "oswald",                "category": "condensed", "use": "Klassisch kondensiert ★★", "weights": [
        (700, "normal", "Bold"), (600, "normal", "SemiBold"),
        (400, "normal", "Regular"), (200, "normal", "ExtraLight"),
    ]},
    {"slug": "teko",                  "category": "condensed", "use": "Sehr eng, modern ★★", "weights": [
        (700, "normal", "Bold"), (600, "normal", "SemiBold"), (500, "normal", "Medium"),
    ]},

    # ── TECH / THEMATISCH ───────────────────────────────────────────────────
    {"slug": "rajdhani",              "category": "condensed", "use": "Tech/Apokalypse-Stil", "weights": [
        (700, "normal", "Bold"), (600, "normal", "SemiBold"),
    ]},
    {"slug": "chakra-petch",          "category": "condensed", "use": "Futuristisch, sci-fi", "weights": [
        (700, "normal", "Bold"), (700, "italic", "BoldItalic"),
    ]},
    {"slug": "saira-condensed",       "category": "condensed", "use": "Sauber kondensiert", "weights": [
        (900, "normal", "Black"), (800, "normal", "ExtraBold"), (700, "normal", "Bold"),
    ]},
    {"slug": "exo-2",                 "category": "condensed", "use": "Rounded tech", "weights": [
        (900, "normal", "Black"), (800, "normal", "ExtraBold"), (900, "italic", "BlackItalic"),
    ]},

    # ── SANS / CLEAN ─────────────────────────────────────────────────────────
    {"slug": "montserrat",            "category": "sans",      "use": "Schwer, clean ★★", "weights": [
        (900, "normal", "Black"), (800, "normal", "ExtraBold"),
        (900, "italic", "BlackItalic"), (800, "italic", "ExtraBoldItalic"),
    ]},
    {"slug": "nunito",                "category": "sans",      "use": "Rund, freundlich", "weights": [
        (900, "normal", "Black"), (800, "normal", "ExtraBold"),
    ]},
    {"slug": "paytone-one",           "category": "display",   "use": "Fett und rund", "weights": [
        (400, "normal", "Regular"),
    ]},

    # ── SERIF / GOLD-VARIANTE ────────────────────────────────────────────────
    {"slug": "playfair-display",      "category": "serif",     "use": "Elegant, Gold-Variante ★★", "weights": [
        (900, "normal", "Black"), (700, "normal", "Bold"),
        (900, "italic", "BlackItalic"), (700, "italic", "BoldItalic"),
    ]},
    {"slug": "merriweather",          "category": "serif",     "use": "Stark und lesbar", "weights": [
        (900, "normal", "Black"), (700, "normal", "Bold"),
    ]},

    # ── AKZENTE / HANDSCHRIFT ────────────────────────────────────────────────
    {"slug": "permanent-marker",      "category": "handwriting","use": "Marker-Stil für Akzente", "weights": [
        (400, "normal", "Regular"),
    ]},
    {"slug": "covered-by-your-grace", "category": "handwriting","use": "Locker, informal", "weights": [
        (400, "normal", "Regular"),
    ]},
    {"slug": "mrs-saint-delafield",   "category": "handwriting","use": "Elegante Signatur-Kursive (Caption font_mix Punch-Wort)", "weights": [
        (400, "normal", "Regular"),
    ]},
    # Additional preview / requested fonts (may fail if not available on fontsource)
    {"slug": "bangers",               "category": "display",   "use": "Comic-Display (Bangers)", "weights": [
        (400, "normal", "Regular"),
    ]},
    {"slug": "unbounded",             "category": "display",   "use": "Neutral variable display (Unbounded)", "weights": [
        (400, "normal", "Regular"), (700, "normal", "Bold"),
    ]},
    {"slug": "holtwood-one-sc",       "category": "display",   "use": "Holtwood One SC (Small Caps)", "weights": [
        (400, "normal", "Regular"),
    ]},
    {"slug": "work-sans",             "category": "sans",      "use": "Work Sans (clean)", "weights": [
        (400, "normal", "Regular"), (700, "normal", "Bold"),
    ]},
    {"slug": "rethink-sans",          "category": "sans",      "use": "Rethink Sans (third-party, may fail)", "weights": [
        (400, "normal", "Regular"),
    ]},
]


# ────────────────────────────────────────────────────────────────────────────
# WOFF2 → TTF CONVERSION
# ────────────────────────────────────────────────────────────────────────────

def _woff2_to_ttf(woff2_data: bytes) -> bytes | None:
    """Convert woff2 bytes to TTF bytes using fonttools."""
    try:
        from fontTools.ttLib.woff2 import decompress
        from fontTools.ttLib import TTFont
        inp = io.BytesIO(woff2_data)
        font = TTFont(inp)
        out = io.BytesIO()
        font.save(out)
        return out.getvalue()
    except Exception as e:
        # Try alternative: write to temp files and use CLI
        try:
            with tempfile.TemporaryDirectory() as d:
                w2 = Path(d) / "font.woff2"
                ttf = Path(d) / "font.ttf"
                w2.write_bytes(woff2_data)
                r = subprocess.run(
                    ["python", "-m", "fonttools", "ttLib.woff2", "decompress",
                     str(w2), "-o", str(ttf)],
                    capture_output=True
                )
                if r.returncode == 0 and ttf.exists():
                    return ttf.read_bytes()
        except Exception:
            pass
        return None


def _check_fonttools() -> bool:
    try:
        import fontTools  # noqa
        return True
    except ImportError:
        return False


# ────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ────────────────────────────────────────────────────────────────────────────

def _fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()
    except Exception:
        return None


def _font_filename(slug: str, weight: int, style: str, suffix: str) -> str:
    # Convert slug to CamelCase family name
    family = "".join(w.capitalize() for w in slug.split("-"))
    return f"{family}-{suffix}.ttf"


def _install_font_weight(
    slug: str, weight: int, style: str, suffix: str,
    family_dir: Path, *, force: bool = False
) -> bool:
    filename = _font_filename(slug, weight, style, suffix)
    target = family_dir / filename
    if target.exists() and not force:
        return True  # already installed

    url = CDN.format(slug=slug, weight=weight, style=style)
    woff2 = _fetch(url)
    if not woff2 or len(woff2) < 1000:
        return False

    ttf = _woff2_to_ttf(woff2)
    if not ttf:
        return False

    target.write_bytes(ttf)
    return True


def _install_family(font: dict, *, force: bool = False) -> tuple[int, int]:
    slug = font["slug"]
    family_dir = FONT_DIR / slug
    family_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for weight, style, suffix in font["weights"]:
        if _install_font_weight(slug, weight, style, suffix, family_dir, force=force):
            ok += 1
        else:
            fail += 1
    return ok, fail


def install_all(force: bool = False) -> None:
    if not _check_fonttools():
        print("✗ fonttools fehlt! Installieren mit:  uv pip install fonttools brotli")
        print("  Dann nochmal: python install_fonts.py")
        sys.exit(1)

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nLade {sum(len(f['weights']) for f in FONTS)} Font-Dateien via jsDelivr...\n")

    total_ok = total_fail = 0
    for font in FONTS:
        ok, fail = _install_family(font, force=force)
        total_ok += ok
        total_fail += fail
        sym = "✓" if fail == 0 else ("~" if ok > 0 else "✗")
        print(f"  {sym} {font['slug']:35s}  {ok}/{ok+fail} weights  [{font['category']}]")

    print(f"\nFont-Cache aktualisieren...")
    subprocess.run(["fc-cache", "-f", str(FONT_DIR)], capture_output=True)

    n = len(list(FONT_DIR.rglob("*.ttf")))
    print(f"\n{'='*60}")
    print(f"  Installiert: {n} TTF-Dateien")
    print(f"  Fehler:      {total_fail}")
    print(f"  Verzeichnis: {FONT_DIR}")


# ────────────────────────────────────────────────────────────────────────────
# LIST / TEST
# ────────────────────────────────────────────────────────────────────────────

def list_fonts() -> None:
    total = len(list(FONT_DIR.rglob("*.ttf")))
    print(f"\n{total} Fonts installiert in {FONT_DIR}\n")
    for font in FONTS:
        fdir = FONT_DIR / font["slug"]
        installed = sorted(fdir.glob("*.ttf")) if fdir.exists() else []
        sym = "✓" if len(installed) == len(font["weights"]) else ("~" if installed else "·")
        names = ", ".join(f.stem.split("-")[-1] for f in installed) if installed else "—"
        print(f"  {sym} {font['slug']:35s}  {names}  [{font['use']}]")


def test_fonts() -> None:
    if not PIL_AVAILABLE:
        print("PIL nicht verfügbar"); return
    files = sorted(FONT_DIR.rglob("*.ttf"))
    print(f"\nTeste {len(files)} Fonts:\n")
    ok = fail = 0
    for f in files:
        try:
            ImageFont.truetype(str(f), 48)
            print(f"  ✓ {f.parent.name}/{f.name}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {f.parent.name}/{f.name}  ({e})")
            fail += 1
    print(f"\nOK: {ok}  Fehler: {fail}")


# ────────────────────────────────────────────────────────────────────────────
# UPDATE ThumbnailMoveChurch.py
# ────────────────────────────────────────────────────────────────────────────

PRIORITY_SLUGS = [
    "barlow-condensed",
    "bebas-neue",
    "anton",
    "teko",
    "oswald",
    "barlow-semi-condensed",
    "saira-condensed",
    "rajdhani",
]
PRIORITY_SUFFIXES = ["Black", "ExtraBold", "Bold"]


def update_thumbnail_module() -> None:
    mc_path = Path(__file__).parent / "Components" / "ThumbnailMoveChurch.py"
    if not mc_path.exists():
        print("ThumbnailMoveChurch.py nicht gefunden"); return

    all_ttf = sorted(FONT_DIR.rglob("*.ttf"))

    # Build priority list
    priority, rest = [], []
    for slug in PRIORITY_SLUGS:
        for suffix in PRIORITY_SUFFIXES:
            for f in all_ttf:
                if f.parent.name == slug and f.stem.endswith(suffix):
                    priority.append(f)
                    break

    in_priority = {str(f) for f in priority}
    rest = [f for f in all_ttf if str(f) not in in_priority]

    # Add system condensed fonts found by apt
    sys_fonts = []
    for d in [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]:
        for f in d.rglob("*.ttf"):
            if any(k in f.name for k in ["Condensed", "Narrow", "Bold", "Heavy"]):
                sys_fonts.append(f)

    lines = (
        "[\n"
        + "".join(f'    r"{p}",\n' for p in priority)
        + "".join(f'    r"{p}",\n' for p in rest[:30])
        + "    # System fonts (apt)\n"
        + "".join(f'    r"{p}",\n' for p in sorted(set(sys_fonts))[:20])
        + '    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",\n'
        + '    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",\n'
        + '    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",\n'
        + "]"
    )

    content = mc_path.read_text(encoding="utf-8")
    new, n = re.subn(
        r"_FONT_CANDIDATES_BOLD_CONDENSED\s*=\s*\[.*?\]",
        f"_FONT_CANDIDATES_BOLD_CONDENSED = {lines}",
        content, flags=re.DOTALL
    )
    if n:
        mc_path.write_text(new, encoding="utf-8")
        print(f"✓ ThumbnailMoveChurch.py: {len(priority)} Priority-Fonts + {len(rest[:30])} weitere eingetragen")
    else:
        print("⚠ Konnte ThumbnailMoveChurch.py nicht patchen")


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC API (für Import aus anderen Modulen)
# ────────────────────────────────────────────────────────────────────────────

def get_font_path(slug: str, suffix: str = "Black") -> str | None:
    fdir = FONT_DIR / slug
    if not fdir.exists(): return None
    for f in fdir.glob(f"*{suffix}.ttf"):
        return str(f)
    # fallback: any weight
    files = sorted(fdir.glob("*.ttf"))
    return str(files[-1]) if files else None


def get_fonts_by_category(category: str) -> list[str]:
    result = []
    for font in FONTS:
        if font["category"] == category:
            fdir = FONT_DIR / font["slug"]
            result.extend(str(f) for f in sorted(fdir.glob("*.ttf"))) if fdir.exists() else None
    return result


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Font installer")
    p.add_argument("--list",      action="store_true")
    p.add_argument("--test",      action="store_true")
    p.add_argument("--force",     action="store_true")
    p.add_argument("--no-update", action="store_true")
    args = p.parse_args()

    if args.list:  list_fonts(); sys.exit(0)
    if args.test:  test_fonts(); sys.exit(0)

    install_all(force=args.force)
    if not args.no_update:
        print()
        update_thumbnail_module()
    print("\nFertig! Starte generate_thumbnail.py neu.")
