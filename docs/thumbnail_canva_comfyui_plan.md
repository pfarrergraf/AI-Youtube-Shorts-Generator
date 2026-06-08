# Canva + ComfyUI Thumbnail Plan

## Summary

Build a thumbnail pipeline that uses the local `ai-youtube-shorts-generator` code as the
deterministic control plane, ComfyUI as the image-production engine, and Canva as the review
and finishing layer.

The target look is the same family as `examples/thumbnails/`: dark atmospheric stage
backgrounds, one strong speaker subject, warm key light, blue/orange split lighting, large
stacked typography, hard black outlines, and very little visual clutter.

## Proposed Workflow

1. Select a strong frame from the source video.
2. Extract the speaker with a robust cutout provider.
3. Generate 2-4 thumbnail variants with distinct layout and lighting profiles.
4. Score the variants locally for readability, face safety, crop quality, and brand fit.
5. Send the best PNGs into Canva for manual review, comparison, and final polish.
6. Export the final thumbnail back into the local thumbnail folder and report metadata.

## ComfyUI Integration

Use the local ComfyUI checkout at `/home/benjamin_graf/ComfyUI` as the rendering backend.
The repo already contains the right building blocks:

- `blueprints/Remove Background (BiRefNet).json`
- `blueprints/Text to Image (Flux.1 Dev).json`
- `blueprints/Image Edit (Flux.2 Dev).json`
- `blueprints/Image to Layers(Qwen-Image-Layered).json`
- `blueprints/Prompt Enhance.json`
- `blueprints/Film Grain.json`
- `custom_nodes/comfyui_essentials/*`

### First workflows to create

- `thumbnail_base`
  - Fast cutout + dark background + text plate + logo.
  - Lowest latency, safest fallback.
- `thumbnail_editorial`
  - Strong subject crop, more negative space, cleaner typography, subtle grain.
  - Best for sermon/news-style clips.
- `thumbnail_premium`
  - More dramatic lighting, stronger rim light, layered glow, richer background separation.
  - Best for high-intensity highlight clips.
- `thumbnail_cleanup`
  - Post-process only: contrast, curves, grain, sharpen, halo cleanup, final readability pass.

### Custom nodes to build

Create a dedicated package under `ComfyUI/custom_nodes/parakeet_thumbnails/` using the
native `ComfyNode` pattern from `custom_nodes/example_node.py.example`.

Recommended nodes:

- `ThumbnailJobSpec`
  - Inputs: source path, title, speaker name, template, target format, provider preferences.
  - Output: a normalized job spec for all downstream nodes.
- `ThumbnailFrameRanker`
  - Scores candidate frames by face visibility, expression, head angle, and composition.
  - Emits the selected frame plus top-N alternatives.
- `ThumbnailCutoutRouter`
  - Tries BiRefNet first, then local fallback providers.
  - Emits cutout, mask, face box, provider used, and rejection reasons.
- `ThumbnailLayoutPlanner`
  - Turns the title into back/front word groups and safe text regions.
  - Chooses layout variants based on face position and title length.
- `ThumbnailReadabilityScore`
  - Scores the rendered image at 320px width.
  - Penalizes face overlap, clipping, and low contrast.
- `ThumbnailExportBundle`
  - Writes PNG, JSON report, and optional Canva upload/export metadata.

### Why custom nodes, not just workflows

Workflows alone are good for orchestration, but the project needs reusable logic for:

- frame selection rules
- face-safe layout decisions
- cutout fallback logic
- variant scoring
- report generation

Those are better as nodes so they can be recomposed across different thumbnail styles.

## Canva Integration

Use Canva as the human review and presentation layer, not as the core render engine.

Planned Canva use:

- compare several generated variants side by side
- add manual polish or final text tweaks when needed
- maintain a visual reference board for the style family
- hand off final review assets to non-technical collaborators

Current limitation:

- Canva brand-template search requires a paid Canva plan.
- The connector currently has no brand kits available.

So the initial integration should assume manual Canva review, with optional brand-template
usage later if the account is upgraded.

## Acceptance Criteria

- A source clip can produce several thumbnail candidates locally.
- The best variant is selected automatically and saved as `thumbnail_best.png`.
- The report includes provider used, score, and layout metadata.
- The selected thumbnail remains readable when shrunk to 320px width.
- Canva can be used to inspect and finish the final asset without breaking the local pipeline.
- New ComfyUI nodes/workflows stay additive and do not break the existing thumbnail CLI/API.

## Suggested Implementation Order

1. Add the ComfyUI workflow contract and a local dry-run adapter.
2. Implement the first custom nodes for job spec, cutout routing, and layout planning.
3. Wire the thumbnail generator to export a ComfyUI job spec and ingest the returned asset.
4. Add Canva review/export handoff for the selected variants.
5. Add smoke tests for local-only, ComfyUI-backed, and Canva-handoff paths.

## Notes

- The existing thumbnail renderer already has strong local scoring and should remain the
  fallback path.
- The ComfyUI repo already includes useful blueprint examples for background removal, image
  editing, layering, grain, and prompt enhancement.
- The first implementation should be conservative: no new mandatory runtime dependencies for
  the default thumbnail path.
