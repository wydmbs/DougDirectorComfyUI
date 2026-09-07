# ComfyUI Director Harness

A director app to build ComfyUI-based movies. Built for the "Pig and Rooster"
rebaseline production, designed to generalize beyond it.

See `SUITE.md` for the standing design guardrails every module follows.

## Module 1 — Image Build UI

The generate → review → lock loop for base images (character sheets,
backdrops, shot keyframes):

- `app.py` — Gradio shell with a Welcome tab, a Setup tab (point at ComfyUI
  + a workflow exported via "Save (API Format)", map prompt/seed node IDs —
  one-time, works with any workflow), a Build tab, and a Registry tab.
- `comfy_client.py` — generic ComfyUI HTTP API client (queue a prompt, poll
  `/history`, fetch images via `/view`). Assumes nothing about your specific
  node graph.
- `storyboard_store.py` — reads/writes an `.xlsx` asset registry across
  four sheets: **Assets** (one row per entry, schema generalized from
  `Pig_and_Rooster_Storyboard_v14.xlsx`'s Shot List tab), **Panels** (one
  row per panel of a character/backdrop/prop sheet — see below),
  **References** (mood-board images attached before a design locks), and
  **Beats** (beat-level timing against the actual narration audio — the
  Runtime Check equivalent, replaced wholesale each time rather than
  edited row by row, since it's recomputed as a whole from either a
  word-count estimate or a real forced-alignment run). Entries are keyed
  by one `entry_id` convention (`shot_id`, `CHAR:name`, `MASTER:name`,
  `PROP:name`). Locking an entry (or panel) a second time updates the row
  in place and appends to its note log rather than duplicating rows.
- `prepare_ui_assets.py` — one-time local script that resizes the app's
  own UI graphics (logo, tab icons, hero banner, reward seal, celebration
  banner, empty-state illustration, favicon) from their generated source
  size down to what the app actually displays them at, and copies them
  into the tracked `assets/ui/` folder. See "UI graphics" below.

## UI graphics

The app's visual identity (header logo, tab-header icons, Welcome hero
banner, reward-card seal, "That's a wrap" celebration banner, Registry
empty-state illustration, favicon) lives in `assets/ui/`, tracked in git.

**If `assets/ui/` is empty or missing files**, the app still runs
correctly — every usage in `app.py` checks for the file first and falls
back to the previous emoji/CSS-only look. Nothing crashes on a fresh
clone before graphics have been generated.

**To (re)generate the graphics:**
1. The prompts and batch-execution instructions live in
   `asset_manifest.json` (machine-readable) — generate the images
   yourself using those prompts (image-grounded batch: one style-anchor
   image first, then every other asset generated with that anchor
   attached as a style reference, so line weight and palette stay
   consistent across the set).
2. Run `python prepare_ui_assets.py "<folder containing the generated
   PNGs>"` from the repo root. This resizes each asset to its actual
   display size (icons are used small — no need to ship 1024px source art
   in the running app), generates a proper multi-size `favicon.ico`, and
   writes everything into `assets/ui/` with the filenames `app.py`
   expects.
3. `git add assets/ui` and commit — these are tracked app-identity
   assets, not gitignored production/project data.

Locked style contract for any regeneration: flat vector, clean geometric
shapes, subtle flat shading, no gradients, no drop shadows, no text baked
into the art. Palette: navy `#172534`, apricot `#E8AE72`, linen `#F5F1E8`,
slate `#5C6C77`, gold `#F6BE45`.
- `sheet_composer.py` — renders a composite reference sheet for `CHAR:`,
  `MASTER:`, and `PROP:` entries from individually-generated panels,
  against a **fixed** panel template per entry_type. `CHAR` sheets (18
  panels): a full-body turnaround (front/back/left/right), an upper-third/
  bust turnaround (front/back/left/right), six forward-facing emotion
  shots, and four supporting panels (attitude pose, costume detail,
  signature prop, color palette). `MASTER` sheets (5 panels): establishing
  wide, day, night, weather variant, detail close-up. `PROP` sheets (9 panels) — for recurring key objects like a ship, cart,
  farmhouse, or crate: a turnaround (front/back/left/right), an interior
  view and an on-it/surface view (present on every prop sheet, but simply
  left unfilled for objects that don't have one — e.g. a solid decorative
  prop with no interior), a detail/texture close-up, an in-context shot
  for scale reference, and a material palette. This generalizes the old
  storyboard's crate-framing grammar (INTERIOR/EXTERIOR/THROUGH-SLATS/N/A)
  to any prop, not just the crate. Note: a scene-level interior like the
  ship's cargo hold is a `MASTER:` backdrop, not part of the ship's `PROP:`
  sheet — the interior/on-it panels are for the object's own surfaces, not
  spaces large enough to be their own scene. The
  template is fixed on purpose — every sheet has its panels in the same
  position, so downstream motion/animation tooling can rely on that
  consistency.
- `config.py` — one small persisted JSON config, shared/extended by future
  modules rather than each module inventing its own.

### Building a character, backdrop, or prop sheet

Pick `CHAR:`, `MASTER:`, or `PROP:` as the entry type in the Build tab.
It's a two-stage flow:

1. **Concept (mashup)** — explore freely using downloaded reference images
   (attached via the mood-board section) plus text as inspiration, generate
   variants, and lock the one design that becomes *the* character or
   backdrop. This reuses the same Assets row a plain shot would.
2. **Sheet panel** — once a concept is locked, switch stages and build each
   panel of the fixed template (front/back/expressions/etc for characters;
   day/night/weather/etc for backdrops) one at a time. A new panel's prompt
   starts pre-filled from the locked concept's description rather than
   blank, so every pose stays tight to the same design instead of
   reinterpreting it. Switching between panels reloads whatever was last
   locked for that exact panel.

"Render preview" assembles the locked panels into one composite sheet
image, with a small thumbnail of the locked concept in the header
documenting what the panels were built to match. Unfinished panels show as
a placeholder, so you can preview progress at any point. "Save as this
entry's reference sheet" records that composite as the entry's canonical
image in the registry.

### Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Opens at `http://127.0.0.1:7860`. **Mock mode is on by default** — it
generates placeholder images instead of calling ComfyUI, so you can test the
full generate → lock → reload loop with no GPU. Turn it off in the Setup tab
once you've hand-built a workflow in ComfyUI's own UI, exported it via
**Save (API Format)**, and mapped the node IDs.

### Not yet built (intentional scope cuts, not oversights)

- Auto-feeding a locked `MASTER:` plate back in as a reference image for the
  next generation — depends on finalizing which image model (Flux 2 vs. a
  callable API model) since the reference-image mechanism differs between
  them.
- Runtime Check-style automatic duration reconciliation — needs
  word-level audio-narration timing data, which doesn't exist yet for the
  rebaselined story.
- Anything video-side (MiniMax H3 Ref2VA/FL2VA). This module is scoped to
  the image-building loop only; see `SUITE.md` for how a video module should
  plug into the same registry.

## License

Apache-2.0 (see `LICENSE`).
