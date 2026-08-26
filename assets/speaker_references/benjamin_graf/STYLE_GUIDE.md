# Benjamin Graf — Hero Style Guide

Standing rules for every future `tools/azure_image_hero.py` prompt for this speaker.
Established 2026-08-26 after reviewing the first `2026-08-26_pose_matrix` batch (8 poses,
all superseded — every one of them accidentally carried glasses over from the
`graf_benjamin.jpg` identity reference, several also had wardrobe/contrast problems).
Bake these into every new prompt file's Expression/Clothing/Lighting fields — don't
re-derive them per pose.

## Never
- **No eyeglasses, ever.** Even though `graf_benjamin.jpg` (one of the 8 identity source
  photos) shows him wearing glasses, every generated hero must be glasses-free. State this
  explicitly in the prompt's Expression or Avoid field — the model otherwise carries them
  over from that one reference.
- No jackets/blazers/denim as the outer layer (rejected: denim jacket in
  `passionate_open_arms_candidate_v1`, blazer in `urgent_lean_forward_candidate_v1`).

## Always
- **Wardrobe — exactly one of two outer layers**: either the black clergy robe (**Talar**)
  with white clerical collar tabs (Beffchen), or a fine-knit **cashmere pullover** (crew or
  quarter-zip) worn **over a collared dress shirt (Hemd)** with the shirt's collar visibly
  showing at the neckline. Never a bare shirt with no pullover, never a jacket/blazer.
- **Hemd color varies per pose** — stylish, not corporate-plain: navy, teal/petrol, light
  blue, burgundy, olive, white-with-contrast-trim. Never re-use the same shirt color twice
  in one batch. Pullover color varies too (navy, oatmeal/beige, charcoal, grey) — pick a
  pullover/shirt pair that reads as a coordinated outfit, not clashing.
  Reference silhouettes (colors/fits, not identity): the 5 lookbook-style images the user
  attached 2026-08-26 (navy crew-neck over patterned white collar; oatmeal quarter-zip over
  white collar; petrol/teal dress shirt; navy dress shirt; white dress shirt with navy
  contrast trim).
- **Skin tone**: healthy, warm, sun-kissed golden undertone — "like he just came back from
  summer vacation," not pale/sickly. `pointing_upward_conviction_candidate_v1` from the
  first batch got this right and is the reference to match going forward.
- **Contrast**: his clothing must read clearly against the background — no dark-on-dark
  (e.g. the rejected pointing_upward_conviction had a blue shirt nearly disappearing into a
  near-black backdrop) and no light-on-light. Pick a background tone that sits clearly
  apart from the outfit color for that specific pose.
- Preserve the same recognizable face, age, hair, body proportions from the identity photos
  in `source/` — everything in this guide is about clothing/eyewear/color grading, not
  identity.

## Physical appearance — RETRACTED (added 2026-08-26, retracted same day)
Originally added: somewhat older, 3-day stubble beard, subtle grey hair strands. **Retracted
after the `2026-08-26_quintilian_pathos` batch** — the user's exact words: "the beard and age
change made a fictitious brother, not me." Combining an age change with new facial hair gave
the model too much room to drift off real identity; asking for two simultaneous physical
deviations from the reference photos apparently exceeds what "identity-preserve" mode can hold
onto reliably. **Current rule: match his exact current age, face, skin, and hair color from the
reference photos — no beard, no stubble, no grey — the only permitted deviation is removing
eyeglasses.** If an older/bearded look is wanted again later, test ONE deviation at a time
(e.g. stubble alone, on a single pose) and verify identity fidelity before batching 13 prompts
on top of it — don't repeat combining multiple physical deviations at once.

## Talar — corrected garment description (added 2026-08-26)
Every prior Talar prompt used a generic "black clergy robe with white clerical collar tabs"
description, which every image-gen run rendered as a fitted robe with a stiff white clerical
collar band — wrong. The user supplied a real reference photo
(`source/Benjamin Graf in Talar (2).jpg`, added to manifest.json as `talar_garment_reference`)
showing his actual garment. **Use this exact description in every future Talar prompt's
Clothing field, not the old generic one:**

> black clergy robe (Talar) in the traditional Lutheran cut — dramatically wide, bell-shaped
> bishop sleeves gathered into pleats at the shoulder seam; a gathered, vertically pleated
> fabric yoke across the chest below a flat black turn-down collar (NOT a stiff white clerical
> collar band); two plain white rectangular Beffchen tabs hanging from beneath that black
> collar at the neckline, roughly hand-length, not overly long; matte black fabric with a
> subtle sheen; a voluminous, floor-length, robe-like silhouette rather than a tailored fit.

The 12 Talar-based heroes approved before this correction (5 in `2026-08-26_rhetorical_matrix`,
6 in `2026-08-26_quintilian_pathos_v2`, 1 in `2026-08-26_striding_movement`) were NOT
retroactively redone — the user asked to get it right going forward, not to redo what's already
approved. Only regenerate one of those if the user asks for that specific pose again.

## Quintilian/pathos batch (2026-08-26_quintilian_pathos, added 2026-08-26)
- User supplied 4 fully-written ChatGPT-style thumbnail concepts (topics: doubt/faith crisis,
  end-times/revelation, miracles/healing, church frustration) with complete image-gen prompts —
  explicitly **"work with this only as ideas"**, i.e. do not reuse those prompts or topics
  directly. What WAS carried over as legitimate technique/mood inspiration: the cinematographic
  variety (shallow-DOF 85-105mm close-ups, ultra-wide 16mm low-angle drama, golden-hour lens
  flare, teal-neon editorial lighting) and the general "high-stakes, life-changing urgency" mood.
- The actual pose vocabulary for this batch comes from **Quintilian's classical rhetorical
  gestures** (Institutio Oratoria, Book XI) — a named canon of oratorical hand/body gestures,
  not generic poses — explicitly requested by the user.
- Every pose in this batch is pushed toward maximum **pathos**: "as if he has to say the words
  that will change your life, the moment you hear this very sermon" — burning urgency and
  conviction, not calm teaching. Push expression/lighting drama further than earlier batches.

## Persona (added after first rhetorical-matrix batch started, 2026-08-26)
- **Overall character**: a strong preacher, visionary, commanding presence — someone people
  are drawn to follow, apostolic gravitas ("Paul the Apostle" energy). Not aggressive or
  cold — grounded, magnetic authority, not a corporate headshot. Bake a version of this
  into every pose's Expression/Style field, calibrated to that pose's specific emotion
  (a "caring" pose still has this gravitas underneath the warmth; a "commanding" pose
  leads with it).
- **Hair**: neatly styled, groomed, a deliberate cut — not flat or messy. Apply in every
  prompt's Expression or Clothing field explicitly ("hair neatly styled/groomed").
- **Clothing must be crisp and pressed, never wrinkled** — the rejected
  `walking_toward_calling_candidate_v1` had a visibly creased shirt; every prompt should
  say so explicitly (e.g. "crisp, freshly pressed, no wrinkles or creases").

## Pose direction notes
- When a pose calls for a downward/humble head tilt, make it a *real*, clearly visible tilt
  — the first `conducting_energy_declaration` attempt under-shot this and read as barely
  tilted at all.
- "Suppressing tears" / restrained emotion poses: glassy, slightly moist eyes are fine and
  wanted — subtle and obvious at the same time, never full melodramatic crying.

## Marketing audit: scroll-stop gap (added 2026-08-26, `2026-08-26_marketing_stage_context`)
Reviewed as a growth-marketer/thumbnail-stopping-power problem, not a portrait-quality problem,
after two pieces of direct client feedback: (1) `grave_seriousness_talar_wide` rejected — *"we
want to catch people with passion, not strictness and severeness"*; (2)
`warm_laughter_cashmere_burgundy` — *"may be nice, but we need a preacher with at least a
lavalier mic or a real wireless mic, ... on stage or at least in front of the altar or on the
pulpit. This is for YouTube and TikTok to stop people from scrolling!"*

Audit of all 42 poses across the four live batches (`rhetorical_matrix` 19 approved,
`quintilian_pathos_v2` 13 approved, `striding_movement` 2 candidates, `mixed_expressions` 8
candidates; `pose_matrix` excluded — fully superseded):
- **Mic**: 0 / 42 render a visible mic. (The discarded `pose_matrix` predecessor had an
  optional "handheld or lavalier microphone if natural" clause in 2 of its 8 prompts — too
  weak to render reliably, and moot since that whole batch was superseded for the glasses bug.)
- **Setting**: 2 / 42 (`declaring_word_at_pulpit_v2`, `low_angle_authority`) show a wooden
  pulpit/lectern edge entering frame — the rest are plain seamless-studio gradients. Several
  prompts contain the word "stage" (`energetic_step_forward`, `advancing_conviction_walk`,
  `striding_stage_command`), but only as lighting-mood flavor text ("dark stage
  surroundings") — no rig, no architecture, no set dressing was ever actually specified, so
  none of those renders reads as an actual stage or church. Even the 2 pulpit exceptions sit
  against a flat void with no other church/stage context.
- **Expression register**: only 1 / 42 (`grave_seriousness_talar_wide`) was actually rejected
  as stern/severe. The rest already lean warm/urgent/passionate per the existing persona rule
  — so severity is not a systemic problem, but it is a **hard failure mode to guard against
  explicitly**, not just imply via the persona line.
- **Comparison** (`thomas_herrmann`, `olaf_latzel`, `leo_bigger_icf`, `antonio_weil` heroes,
  all approved): every one of them holds a visible handheld wireless mic, stands in a dark
  stage environment with visible rig beams/haze/colored gels (or a bright doorway light
  shaft), and reads as mid-sermon energy, not a studio headshot. That combination — mic +
  legible-but-defocused stage/pulpit/altar setting + energetic expression — is what makes
  those images stop a scroll and is exactly what's missing here.

### Hard rule for every future pose (binding, supersedes nothing above — additive)
1. **Every prompt must specify a visible mic, chosen by garment** (a lavalier reads wrong on
   a Talar collar and risks the model reinterpreting the corrected collar/Beffchen shape —
   never write "lavalier" on a Talar pose):
   - **Talar poses**: a handheld wireless microphone held naturally in the hand that is not
     doing the rhetorical gesture (never both hands full) — optionally paired with a
     pulpit-mounted gooseneck mic as a second visual cue when the pose is at a pulpit.
   - **Cashmere-pullover-over-Hemd poses**: a lavalier mic clipped to the shirt collar edge
     or pullover neckline with a thin visible cable, or a handheld wireless mic for the more
     physically energetic poses (striding, wide gestures).
2. **Every prompt must specify a real preaching setting** — a pulpit/lectern (with wood grain
   or edge visible in frame), a church altar with cross/candles, or a stage with visible
   rig/lighting (beams, haze, gels, trusses) — never a plain seamless studio backdrop. "Dark
   stage surroundings" as mood text alone does not satisfy this; the setting needs its own
   concrete visual noun (pulpit, altar, cross, candle, light rig, beam, truss, stained glass,
   church arch).
3. **Reconcile setting with layer-compositing**: keep the background *legible* but throw it
   out of focus (shallow depth of field) with one side of the frame left dark/low-detail for
   later typography — the same solution the four comparison heroes above already use (haze,
   defocused rig lights, dark drop). Do not fall back to a flat seamless void just to satisfy
   the compositing constraint, and do not add a crowd/congregation — no visible audience,
   ever; the stage read comes from rig/light/architecture, not from other people in frame.
4. **Severe/stern/cold/grim/disapproving/frowning is off-brand** even in serious or urgent
   poses — add these terms to every prompt's Avoid field explicitly, not just to Expression.
   Passion, warmth, and energetic conviction are the target register in every emotional
   register, including grave or urgent ones.
