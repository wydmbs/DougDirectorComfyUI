# ComfyUI Director Harness

A director app to build ComfyUI-based movies. Built for the "Pig and Rooster"
rebaseline production, designed to generalize beyond it.

See `SUITE.md` for the standing design guardrails every module follows.

## Module 1 — Image Build UI

The generate → review → lock loop for base images (character sheets,
backdrops, shot keyframes):

- `app.py` — Gradio shell, three tabs: **Setup** (point at ComfyUI + a
  workflow exported via "Save (API Format)", map prompt/seed node IDs —
  one-time, works with any workflow), **Build** (pick an `entry_id`, iterate
  prompt/seed, generate N variants, pick a winner, lock it), **Registry**
  (read-only view of everything locked).
- `comfy_client.py` — generic ComfyUI HTTP API client (queue a prompt, poll
  `/history`, fetch images via `/view`). Assumes nothing about your specific
  node graph.
- `storyboard_store.py` — reads/writes an `.xlsx` asset registry. Schema
  generalized from `Pig_and_Rooster_Storyboard_v14.xlsx`'s Shot List tab,
  model-agnostic, keyed by one `entry_id` convention (`shot_id`, `CHAR:name`,
  `MASTER:name`, `PROP:name`). Locking an entry a second time updates the row
  and appends to its note log rather than duplicating rows.
- `config.py` — one small persisted JSON config, shared/extended by future
  modules rather than each module inventing its own.

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
