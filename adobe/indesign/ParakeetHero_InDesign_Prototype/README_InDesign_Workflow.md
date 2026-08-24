# Parakeet Hero InDesign Prototype

> Status: editable prototype source. The document dimensions, named layers and
> linked assets are defined, but a clean final export from the current
> workstation still needs visual approval. Generated and superseded exports are
> intentionally not versioned.

This package proves one editable hero design in two digital formats:

- `1080 × 1920` for the Parakeet 9:16 opening still
- `1280 × 720` for a conventional YouTube thumbnail

## Build automatically on this workstation

Run in Windows PowerShell from this folder:

```powershell
.\Run_InDesign_Prototype.ps1
```

The runner starts Adobe InDesign 2026 through COM, executes
`ParakeetHero_BUILD.jsx`, verifies the expected files, and closes the automation
instance. It refuses to run while InDesign is already open.

## Build from the InDesign Scripts panel

Keep `assets/` beside `ParakeetHero_BUILD.jsx`. Copy or link the whole project
folder into the InDesign Scripts Panel folder, then double-click
`ParakeetHero_BUILD.jsx`. Verified files appear under `output_final/`.

The JSX creates movable objects on named layers, keeps all visible typography
as live text, links the approved hero and SVG atmosphere assets, explicitly
removes accidental strokes, saves native INDD documents, exports IDML, and
exports exact-size PNG previews at 72 ppi.

## Fonts

The build requests locally installed `Montserrat` (`Black` and `SemiBold`). It
falls back to `Arial Bold` only if Montserrat is unavailable. Font files are not
included or redistributed.

## Content and asset policy

The speaker image is the approved, text-free `astonished_revelation` Leo Bigger
hero from Parakeet's `assets/speaker_references/manifest.json`. No new synthetic
identity was created for this proof. `ICF ZÜRICH` deliberately preserves the
approved visible brand spelling.

## Editable structure

See `project-spec.json` for dimensions, typography, colors and linked assets.
See `object-map.json` for stable InDesign names/labels used by future automation.
