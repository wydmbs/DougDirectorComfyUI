# Legacy Gradio shell

The original `app.py` was a Gradio 4/5 single-file UI. It shipped every
feature this project has, but Gradio 6's breaking changes made it fragile
to keep alive alongside the newer FastAPI shell (`main.py` at the repo
root), and the two shells started drifting apart.

Files here:

- `app.py` — the 2,219-line Gradio Blocks shell.
- `harry_ui.py` — HTML-string helpers that Gradio's `HTML()` component
  rendered inside `app.py`. The FastAPI shell uses server-rendered
  fragments served by `main.py` instead.
- `tests/test_app_smoke.py` — imports `app.py` + `gradio` directly. Only
  useful for validating the legacy shell.

Nothing in the running codebase imports from here. Kept for git-history
convenience: if a decision from the Gradio era needs revisiting, the
code is intact rather than lost to `git log -p`.

To run the legacy shell in isolation you would need `pip install gradio`
plus the imports at the top of `legacy/gradio/app.py` — most of which
still exist at the repo root (`config`, `storyboard_store`,
`sheet_composer`, `harry_advisor`, `project_manager`, `director_engine`,
`draft_prompts`, `model_pipeline`, `reference_conditioning`,
`comfy_client`).
