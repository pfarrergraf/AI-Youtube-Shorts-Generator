"""
Fit the reference gate's metric bands from thumbnail_ideal_examples/.
=====================================================================

Run::

    python tools/calibrate_reference_gate.py            # fit + write bands
    python tools/calibrate_reference_gate.py --report   # fit + print per-image table
    python tools/calibrate_reference_gate.py --check    # verify every reference passes

The bands are derived, never hand-written. That is the whole point: a
hand-written band such as ``peak_luma >= 240`` rejects reference 9 (202.7), and
a gate that rejects its own ground truth is worthless.

TITLE_BOXES below are hand-annotated once: the title block as fractions of
(width, height), plus the line count. Both are needed —

* glyph coverage has to be measured *inside* the type area, because globally a
  light shaft reads exactly like white type;
* the line count turns block height into cap height, which is the metric that
  actually separates the reference look from ours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Components.ThumbnailReferenceGate import (  # noqa: E402
    CALIBRATED_METRICS,
    cap_height_ratio,
    derive_bands,
    image_fingerprint,
    image_metrics,
    type_block_ratio,
    type_ink_ratio_from_box,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "thumbnail_ideal_examples"
BANDS_PATH = REPO_ROOT / "Components" / "reference_gate_bands.json"

# Hand-annotated title blocks: (x1, y1, x2, y2) as fractions of the frame,
# plus the line count. Keyed by the trailing "(n)" in the ChatGPT filenames.
TITLE_BOXES: dict[str, tuple[float, float, float, float]] = {
    "1": (0.19, 0.03, 0.83, 0.49),   # DAS / VOLLE / PAKET
    "2": (0.07, 0.07, 0.53, 0.50),   # NICHT / NUR / EIN TEIL
    "3": (0.19, 0.11, 0.82, 0.38),   # ICH BIN / BEREIT
    "4": (0.07, 0.12, 0.93, 0.38),   # MEHR ALS / EIN GEFÜHL
    "5": (0.55, 0.06, 0.97, 0.55),   # DEINE / GABE / WARTET
    "6": (0.05, 0.04, 0.49, 0.48),   # KRAFT / FÜR / HEUTE
    "7": (0.13, 0.61, 0.88, 0.95),   # GOTT / WILL / DURCH / DICH
    "8": (0.07, 0.02, 0.93, 0.33),   # KEINE ANGST / VOR MEHR
    "9": (0.05, 0.22, 0.96, 0.30),   # EMPFANGEN. GEBEN.
    "10": (0.03, 0.65, 0.97, 0.97),  # WAS HÄLT / DICH ZURÜCK?
}

TITLE_LINES: dict[str, int] = {
    "1": 3, "2": 3, "3": 2, "4": 2, "5": 3,
    "6": 3, "7": 4, "8": 2, "9": 1, "10": 2,
}


def _key_for(path: Path) -> str | None:
    stem = path.stem
    if "(" in stem and stem.endswith(")"):
        return stem.rsplit("(", 1)[1].rstrip(")")
    return None


def collect_references(reference_dir: Path = REFERENCE_DIR) -> list[tuple[str, Path]]:
    """Distinct reference images, deduplicated by file hash.

    The supplied set has 11 files but only 10 distinct images — the two files
    ending in "(10)" are byte-identical. Without the dedupe that image would be
    weighted twice when the bands are fitted.
    """
    seen: dict[str, Path] = {}
    for path in sorted(reference_dir.glob("*.png")):
        digest = image_fingerprint(path)
        if digest in seen:
            continue
        seen[digest] = path
    out: list[tuple[str, Path]] = []
    for path in seen.values():
        key = _key_for(path)
        if key is None or key not in TITLE_BOXES:
            print(f"  ! no title-box annotation for {path.name} — skipped", file=sys.stderr)
            continue
        out.append((key, path))
    return sorted(out, key=lambda kv: int(kv[0]))


def measure(
    path: Path,
    box_frac: tuple[float, float, float, float],
    n_lines: int,
) -> dict[str, float]:
    image = Image.open(path)
    w, h = image.size
    box = (
        int(box_frac[0] * w),
        int(box_frac[1] * h),
        int(box_frac[2] * w),
        int(box_frac[3] * h),
    )
    metrics = image_metrics(image)
    metrics["type_ink_ratio"] = type_ink_ratio_from_box(image, box)
    metrics["type_block_ratio"] = type_block_ratio(box, (w, h))
    metrics["cap_height_ratio"] = cap_height_ratio(box, n_lines, (w, h))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    ap.add_argument("--out", default=str(BANDS_PATH))
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--report", action="store_true", help="print the per-image metric table")
    ap.add_argument("--check", action="store_true", help="verify every reference lands in band")
    args = ap.parse_args()

    refs = collect_references(Path(args.reference_dir))
    if not refs:
        print("no reference images found", file=sys.stderr)
        return 1
    print(f"Calibrating on {len(refs)} distinct references")

    per_image = {
        key: measure(path, TITLE_BOXES[key], TITLE_LINES[key]) for key, path in refs
    }

    samples: dict[str, list[float]] = {name: [] for name in CALIBRATED_METRICS}
    for metrics in per_image.values():
        for name in CALIBRATED_METRICS:
            if name in metrics:
                samples[name].append(metrics[name])

    bands = derive_bands(samples, margin=args.margin)

    if args.report:
        names = list(CALIBRATED_METRICS) + ["type_block_ratio"]
        print()
        print("ref  " + "".join(f"{n[:15]:>17}" for n in names))
        print("-" * (5 + 17 * len(names)))
        for key, metrics in per_image.items():
            row = "".join(f"{metrics.get(n, float('nan')):>17.4f}" for n in names)
            print(f"{key:<5}{row}")
        print()
        for name, band in bands.items():
            print(f"  {name:<18} band [{band['low']:.4f}, {band['high']:.4f}]  "
                  f"observed [{band['observed_min']:.4f}, {band['observed_max']:.4f}]  n={band['n']}")

    if args.check:
        failures = 0
        for key, metrics in per_image.items():
            for name, band in bands.items():
                val = metrics.get(name)
                if val is None:
                    continue
                if val < band["low"] or val > band["high"]:
                    print(f"  FAIL ref {key}: {name}={val:.4f} outside "
                          f"[{band['low']:.4f}, {band['high']:.4f}]", file=sys.stderr)
                    failures += 1
        if failures:
            print(f"{failures} reference metric(s) outside band — bands are wrong", file=sys.stderr)
            return 1
        print("  all references land inside their bands")

    payload = {
        "source": str(Path(args.reference_dir).name),
        "n_references": len(refs),
        "margin": args.margin,
        "bands": bands,
        "per_image": {k: {m: round(v, 5) for m, v in met.items()} for k, met in per_image.items()},
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
