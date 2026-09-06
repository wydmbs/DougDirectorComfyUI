# SUITE.md — standing guardrails for the ComfyUI Director Harness

Everything built for this project should feel like modules of one coherent
suite, not disconnected scripts that happen to share a folder. Check every
new piece against these before adding it:

1. **One shared data backbone.** `storyboard_store.py`'s Assets registry
   (entry_id / entry_type / prompts / seed / model / reused_from / locked /
   image_path / notes) is the spine. A video-generation module should read
   and write the *same* registry (adding columns as needed, e.g.
   `clip_path`, `motion_prompt`) rather than spinning up a separate tracking
   file. Same for a future QA module — its verdicts land in the same row's
   notes/fields, not a side database.

2. **One shared config.** `config.py`'s pattern (a small persisted JSON,
   edited once via a Setup-style tab) should be reused/extended for new
   integrations (e.g. a MiniMax H3 endpoint) rather than each module
   inventing its own config file.

3. **One shell, more tabs — not more apps.** New capabilities arrive as
   additional tabs in the same Gradio app (or whatever shell we settle on),
   so there's one thing to launch, not a growing list of separate scripts.
   If a capability genuinely needs its own long-running process, it should
   still surface through the same shell rather than a separate UI.

4. **One naming/ID convention.** `shot_id` / `CHAR:` / `MASTER:` / `PROP:`
   is the one convention every module refers to an asset by — video clips,
   QA checks, and any future orchestrator all key off the same entry_id.

5. **Name the suite once, everywhere.** "ComfyUI Director Harness" is the
   suite's name — every module's UI copy, file naming, and README should
   point at this identity rather than needing a rename pass later.

## Module status

- **Module 1 — Image Build UI** (this commit): Welcome / Setup / Build /
  Registry tabs. Two build shapes: a plain SHOT (one image, one prompt) and
  a CHAR/MASTER sheet (a fixed template of panels — front/back/profile/
  expressions/etc for characters, day/night/weather/etc for backdrops —
  each generated and locked independently, then assembled into one
  composite reference image by the tool). Reference-image mood-boards can
  be attached to an entry before its design locks. Mock mode tested
  end-to-end for both shapes: generate → lock → reload → re-lock appends
  note → registry and sheet gallery read back correctly. Not yet wired to
  a live ComfyUI + Flux 2 workflow, and the composite sheet is not yet fed
  back in as a reference image for new generations (still depends on the
  Flux 2 vs. Nano Banana Pro call).
- **Module 2 — not started.** Candidates: video generation (MiniMax H3
  Ref2VA/FL2VA) or a QA/conformance reviewer. Whichever starts first should
  extend the same registry rather than create its own.
