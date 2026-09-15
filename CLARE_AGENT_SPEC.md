# Clare — Casting & Costume Director

## Purpose and position

Clare is Pass 1B, immediately after Harry has produced and the director has approved a shot breakdown:

`Harry call sheet → Clare conformance → Rachael backdrop/lighting → Prop Master → Continuity → keyframe`

She protects character identity and wardrobe continuity. She does not silently resolve vague direction: ambiguity is a flag. A rejected shot cannot advance to Rachael.

## Existing Harness mapping

| Concern | Existing Harness contract | Clare addition |
|---|---|---|
| Shot identity | `Assets.entry_id` uses the canonical `shot_id` | Clare writes one verdict per shot row; no parallel database. |
| Character identity | `Characters.trigger_id` uses `CHARACTER:name` | A cached identity lock is attached to each character record. |
| Prompt source | Harry provides `description`, `positive_prompt`, `negative_prompt`, `camera_direction`, and `lighting_direction` | Clare validates the resulting per-shot `CHARACTER:` block before downstream prompt merge. |
| Style anchor | `storyboard_v15_element_registry.tsv` makes `thelwell_style_block` global and verbatim for all shots | Clare hard-fails its absence for Pig and Bertie. |
| Resumability | The registry is the production memory and adds stage columns in place | `clare_status` is the checkpoint; per-shot JSON and notes are retained on the same Assets row. |
| Downstream handoff | Specialists produce inspectable JSON | Rachael, Prop Master, and Continuity read Clare's approved/flagged `CHARACTER:` blocks and notes. |

## Proposed storage contract

Add these columns to `storyboard_store.ASSETS_COLUMNS` during implementation:

- `clare_status`: `pending`, `approved`, `flagged`, or `rejected`
- `clare_character_json`: JSON array of the `CHARACTER:` blocks for cast present in the shot
- `clare_checked_at`: UTC timestamp

The identity locks are stored with the relevant `Characters` record, preferably in a new `identity_lock_json` column. This is a **proposed default pending confirmation**: compute/cache Tier 1 once per character, then validate tiers 2–4 per shot.

## Tier 1 — locked identity

| Character | Trigger phrase | Production LoRA | Strength (model, clip) | Base identity / wardrobe |
|---|---|---|---|---|
| Pig | `pig farmer character` | `pig_lora_final.safetensors` | `[0.85, 0.85]` unless a documented exception | Species/base colouring and all approved physical traits; tweed cap, cream linen shirt, leather braces, tweed waistcoat, cord trousers, bare trotters. |
| Bertie | `bertie show cockerel character` | `rooster_lora_p3.safetensors` | `[0.85, 0.85]` unless a documented exception | Species/base colouring and all approved physical traits; locked approved wardrobe and plumage traits. |

For both characters, the full verbatim `thelwell_style_block` from `storyboard_v15_element_registry.tsv` leads every prompt.

## Per-shot `CHARACTER:` contract

One block is required for each character in Harry's cast list:

```json
{
  "name": "Pig | Bertie",
  "trigger_phrase": "exact locked string",
  "lora_file": "locked production filename",
  "lora_strength": [0.85, 0.85],
  "style_block_required": true,
  "wardrobe": {
    "base": "locked default descriptor",
    "variant": "transformation of base, or none"
  },
  "pose_state": "comb/plumage state if applicable",
  "framing": "close-up | medium | full-body | silhouette",
  "out_of_frame_check": "pass | fail",
  "human_visibility_tier": "waist-down | full-figure | face-obscured | n/a",
  "children_appropriate": "pass | fail",
  "status": "approved | rejected | flagged",
  "notes": "drift risk, precedent, or rejection reason"
}
```

`framing` is derived from Harry's `camera_direction`; cast and story state are derived from Harry's shot description, continuity note, and prompt. A wardrobe variant must be a transformation of the locked base, never a redesign. Clare records precedent shot IDs where useful.

## Hard rejects

A shot is `rejected` when any applicable gate fails:

1. Trigger phrase is missing or malformed.
2. LoRA is not the locked production file; any Pig/Rooster pre-production or p2-or-earlier reference fails.
3. Required full Thelwell style block is absent.
4. A close-up describes an out-of-frame body part.
5. A human exceeds the visibility tier specified for the shot: waist-down for crate-loading/capture, full figure where allowed, and face obscured for hammock guy.
6. The image direction is not children-appropriate.
7. Wardrobe damage/state is described as an independent redesign rather than a transformation of the locked base.

Comb-state choice, minor styling uncertainty, and likely trait drift are `flagged`, not rejected.

## Comb-state ownership

**Proposed default pending confirmation:** Harry proposes Bertie's upright/flopped comb as emotional and story direction; Clare validates that the result is on-model and narratively supported. Clare flags contradictions but does not originate story emotion.

## Specialist interface

`specialists.py` already has `clare_character_costume()`, which produces global character dossiers and broad per-shot notes. It must be extended rather than replaced:

1. Preserve its existing reusable dossier output for visual-development work.
2. Add a strict shot-conformance mode after Harry's approved call sheet.
3. Require the `CHARACTER:` blocks above and validate them locally before accepting provider output.
4. Persist outcomes to the shared Assets and Characters registry, not only to `projects/<project>/.../specialists/clare_character_costume.json`.
5. Expose rejection reasons and flags to the Director Harness, Rachael, Prop Master, and Continuity.

## Test gate before integration

Use mocked provider calls and existing v15 material:

1. Select three known-good v14/v15 shots: a close-up, a full-body storm/damage shot, and a crate-loading/capture shot with a human.
2. Assert exact trigger phrases, locked LoRA files, `[0.85, 0.85]`, and the full Thelwell block.
3. Assert correct damaged-base wardrobe handling, close-up out-of-frame behaviour, and human visibility tier.
4. Deliberately remove the style block and separately describe a wardrobe redesign; each must produce `rejected`, never auto-correction.
5. Assert that `rejected` blocks downstream stage eligibility while `flagged` remains eligible with its notes.
6. Assert a second run resumes from `clare_status` without recomputing an unchanged approved shot.

Only after these tests pass should Clare be wired as the real Pass 1B stage.

## Implementation milestones

1. Confirm the two proposed defaults above and the final locked Bertie base wardrobe/physical descriptor cluster.
2. Extend `storyboard_store.py` schema and migration-safe readers/writers.
3. Extend `specialists.py` with strict schema parsing and deterministic hard-gate validation.
4. Add Clare status and notes to the FastAPI Director Harness stage UI.
5. Add fixture-driven tests and run the mock gate.
6. Enable the stage for one reviewed three-shot pilot before production-wide use.
