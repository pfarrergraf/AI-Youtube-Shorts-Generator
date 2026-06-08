# Thumbnail Workflow Blueprints

This document turns the current thumbnail direction into a concrete ComfyUI plan.

## Target look

- dark stage-like backgrounds
- one dominant speaker cutout
- warm orange and cool blue split lighting
- large stacked headline typography
- hard black outline and readable glow
- minimal clutter, high contrast, mobile-first readability

## Initial workflows

### `thumbnail_base`

- Purpose: safe default for most sermons and talk clips
- Inputs: source frame, title, speaker cutout, template, effect profile
- Path:
  1. select best frame
  2. remove background
  3. apply safe editorial layout
  4. render typography and finish
- Use when:
  - the clip needs a conservative look
  - the subject is clear but not dramatic

### `thumbnail_editorial`

- Purpose: cleaner layout with stronger text hierarchy
- Path:
  1. frame ranking
  2. robust cutout
  3. face-safe composition
  4. tighter typography and lighter finish
- Use when:
  - the title is short
  - readability matters more than dramatic lighting

### `thumbnail_premium`

- Purpose: higher-end look for highlight clips
- Path:
  1. select strongest expressive frame
  2. cutout and rim-light the subject
  3. add halo, glow, and richer separation
  4. finish with contrast and grain
- Use when:
  - the clip has a strong emotional hook
  - the reference style should feel more premium

### `thumbnail_cleanup`

- Purpose: post-process and final polish
- Path:
  1. sharpen or smooth edges
  2. reduce cutout artifacts
  3. check readability at 320px
  4. write final export bundle

## Custom nodes to build

- `ParakeetThumbnailJobSpec`
  - Normalize the job into a reproducible JSON contract.
- `ParakeetThumbnailFrameRanker`
  - Score candidate frames by face size, expression, and composition.
- `ParakeetThumbnailCutoutRouter`
  - Route between BiRefNet, rembg, and local fallbacks.
- `ParakeetThumbnailLayoutPlanner`
  - Split title into back/front groups and choose a safe layout.
- `ParakeetThumbnailReadabilityScore`
  - Score the render at mobile size.
- `ParakeetThumbnailWorkflowPreset`
  - Pick the best workflow family for the current clip.
- `ParakeetCanvaReviewPacket`
  - Produce the manual Canva handoff package.

## Handoff flow

1. Generate the thumbnail locally.
2. Select the best variant.
3. Export the ComfyUI job spec.
4. Export the Canva review packet.
5. Review the result against the reference thumbnails.
6. Iterate the workflow or custom node logic if the result misses the target style.

## Current implementation status

- Local thumbnail generation is already the baseline.
- Studio export now writes ComfyUI and Canva JSON bundles.
- The ComfyUI repo has a local `parakeet_thumbnails` custom-node package.
- Canva is currently a manual review layer because brand-template automation is limited in this account.
