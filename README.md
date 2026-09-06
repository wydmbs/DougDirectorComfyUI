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
  three sheets: **Assets** (one row per entry, schema generalized from
  `Pig_and_Rooster_Storyboard_v14.xlsx`'s Shot List tab), **Panels** (one
  row per panel of a character/backdrop sheet — see below), and
  **References** (mood-board images attached before a design locks).
  Entries are keyed by one `entry_id` convention (`shot_id`, `CHAR:name`,
  `MASTER:name`, `PROP:name`). Locking an entry (or panel) a second time
  updates the row in place and appends to its note log rather than
  duplicating rows.
- `sheet_composer.py` — renders a composite reference sheet for `CHAR:` and
  `MASTER:` entries from individually-generated panels, against a **fixed**
  panel template per entry_type (front/back/profile/expression-grid/etc for
  characters; day/night/weather/etc for backdrops). The template is fixed
  on purpose — every character sheet has its panels in the same position,
  so downstream motion/animation tooling can rely on that consistency.
- `config.py` — one small persisted JSON config, shared/extended by future
  modules rather than each module inventing its own.

### Building a character or backdrop sheet

Pick `CHAR:` or `MASTER:` as the entry type in the Build tab. It's a
two-stage flow:

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
