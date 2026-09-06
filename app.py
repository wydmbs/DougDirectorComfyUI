"""
app.py — ComfyUI Director Harness, Module 1: Image Build UI.

One shell, tabs for capability (see SUITE.md):
  Welcome  — plain-language explanation of what this tool does.
  Setup    — point at a ComfyUI instance + a workflow exported via
             "Save (API Format)". One-time, works with any workflow.
  Build    — the day-to-day workspace. Two shapes, chosen by Type:
               SHOT   — one image, one prompt. Generate, pick, lock.
               CHAR / MASTER — a fixed-template composite sheet (front
               view, expressions, day/night variants, etc — see
               sheet_composer.py). Each panel of the template is its own
               generate/pick/lock cycle; the tool renders the panels
               together into one composite reference image.
  Registry — read-only view of everything locked so far, including
             rendered composite sheets.

Mock mode (default ON) generates placeholder images instead of calling
ComfyUI, so the full loop can be tried with zero setup and no GPU.
"""

import io
import json
import os
import random
import shutil

import gradio as gr
from PIL import Image, ImageDraw

import config as cfgmod
import storyboard_store as store
import sheet_composer as composer
from comfy_client import ComfyClient, ComfyClientError, apply_node_overrides

CFG = cfgmod.load_config()
os.makedirs(CFG.images_dir, exist_ok=True)

SHEET_TYPES = ("CHAR", "MASTER")


# ---------------------------------------------------------------------------
# Mock generation (no GPU / ComfyUI required)
# ---------------------------------------------------------------------------

def _mock_image(label: str, prompt: str, seed: int) -> Image.Image:
    random.seed(seed)
    color = tuple(random.randint(40, 200) for _ in range(3))
    img = Image.new("RGB", (512, 288), color)
    draw = ImageDraw.Draw(img)
    text = f"{label}\nseed {seed}\n{prompt[:60]}"
    draw.multiline_text((12, 12), text, fill=(255, 255, 255))
    return img


def _save_image(img: Image.Image, label: str, seed: int) -> str:
    safe_label = str(label).replace(":", "_").replace("/", "_")
    fname = f"{safe_label}_{seed}.png"
    path = os.path.join(CFG.images_dir, fname)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Setup tab logic
# ---------------------------------------------------------------------------

def setup_save(comfyui_url, workflow_file, pos_node, pos_input, neg_node, neg_input,
                seed_node, seed_input, storyboard_path, mock_mode):
    global CFG
    workflow_path = CFG.workflow_json_path
    if workflow_file is not None:
        workflow_path = workflow_file.name

    nm = cfgmod.NodeMapping(
        positive_prompt_node=pos_node.strip(),
        positive_prompt_input=pos_input.strip() or "text",
        negative_prompt_node=neg_node.strip(),
        negative_prompt_input=neg_input.strip() or "text",
        seed_node=seed_node.strip(),
        seed_input=seed_input.strip() or "seed",
    )
    CFG = cfgmod.ToolchainConfig(
        comfyui_url=comfyui_url.strip() or "http://127.0.0.1:8188",
        workflow_json_path=workflow_path,
        node_mapping=nm,
        storyboard_path=storyboard_path.strip() or "storyboard.xlsx",
        images_dir=CFG.images_dir,
        mock_mode=bool(mock_mode),
    )
    cfgmod.save_config(CFG)
    mode_msg = (
        "✅ Saved. Mock mode is ON — go to the Build tab, nothing here calls ComfyUI yet."
        if CFG.mock_mode else
        "✅ Saved. Mock mode is OFF — Build will now send real requests to ComfyUI at "
        f"{CFG.comfyui_url}."
    )
    return mode_msg


# ---------------------------------------------------------------------------
# Core generation (shared by SHOT and per-panel CHAR/MASTER generation)
# ---------------------------------------------------------------------------

def _generate(label, prompt_positive, prompt_negative, base_seed, n_variants):
    base_seed = int(base_seed) if base_seed not in (None, "") else random.randint(0, 2**31 - 1)
    n_variants = max(1, min(int(n_variants), 8))

    gallery = []
    status_lines = []
    client = None
    workflow = None
    if not CFG.mock_mode:
        if not CFG.workflow_json_path or not os.path.exists(CFG.workflow_json_path):
            return [], "⚠️ No workflow file is set up yet. Go to Setup and either upload one or turn Mock Mode back on."
        with open(CFG.workflow_json_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        client = ComfyClient(CFG.comfyui_url)

    for i in range(n_variants):
        seed = base_seed + i
        if CFG.mock_mode:
            img = _mock_image(label, prompt_positive, seed)
            path = _save_image(img, label, seed)
            gallery.append((path, f"seed {seed}"))
            continue
        try:
            wf = apply_node_overrides(workflow, CFG.node_mapping, prompt_positive, prompt_negative, seed)
            prompt_id = client.queue_prompt(wf)
            hist = client.wait_for_completion(prompt_id)
            refs = client.extract_image_refs(hist)
            if not refs:
                status_lines.append(f"seed {seed}: no images returned")
                continue
            filename, subfolder, folder_type = refs[0]
            img_bytes = client.fetch_image_bytes(filename, subfolder, folder_type)
            img = Image.open(io.BytesIO(img_bytes))
            path = _save_image(img, label, seed)
            gallery.append((path, f"seed {seed}"))
        except ComfyClientError as e:
            status_lines.append(f"seed {seed}: {e}")

    if gallery:
        status = f"✅ Generated {len(gallery)}/{n_variants}. Scroll down: pick your favorite for Step 4."
    else:
        status = "⚠️ Nothing came back."
    if status_lines:
        status += " " + " | ".join(status_lines)
    return gallery, status


CONCEPT_STAGE = "Concept (mashup)"
PANEL_STAGE = "Sheet panel"


def generate_click(entry_id, entry_type, build_stage, panel_key, prompt_positive, prompt_negative,
                    base_seed, n_variants):
    if not entry_id:
        return [], "⚠️ Give this a name first (Step 1) — try 'CHAR:pig' or '1.1'."
    if entry_type in SHEET_TYPES and build_stage == PANEL_STAGE and not panel_key:
        return [], "⚠️ Pick which panel of the sheet you're building first."

    if entry_type not in SHEET_TYPES:
        label = entry_id
    elif build_stage == CONCEPT_STAGE:
        label = f"{entry_id}__concept"
    else:
        label = f"{entry_id}__{panel_key}"
    return _generate(label, prompt_positive, prompt_negative, base_seed, n_variants)


def lock_click(entry_id, entry_type, build_stage, panel_key, beat, description, prompt_positive,
                prompt_negative, winner_path, winner_seed, note):
    if not entry_id:
        return "⚠️ Give this a name first (Step 1)."
    if not winner_path:
        return "⚠️ Generate some variants (Step 3) and tell me which one won (Step 4) before locking."

    if entry_type not in SHEET_TYPES:
        store.lock_entry(
            path=CFG.storyboard_path,
            entry_id=entry_id.strip(),
            entry_type=entry_type,
            beat=beat.strip(),
            description=description.strip(),
            prompt_positive=prompt_positive.strip(),
            prompt_negative_add=prompt_negative.strip(),
            model="mock" if CFG.mock_mode else os.path.basename(CFG.workflow_json_path or ""),
            seed=winner_seed,
            reused_from="",
            image_path=winner_path,
            note=note.strip(),
        )
        return f"🔒 Locked '{entry_id}' with seed {winner_seed}. Check the Registry tab."

    if build_stage == CONCEPT_STAGE:
        # Phase 1 — the mashup/concept design. Uses the same Assets row a
        # plain SHOT would, so this reuses lock_entry() exactly.
        store.lock_entry(
            path=CFG.storyboard_path,
            entry_id=entry_id.strip(),
            entry_type=entry_type,
            beat=beat.strip(),
            description=description.strip(),
            prompt_positive=prompt_positive.strip(),
            prompt_negative_add=prompt_negative.strip(),
            model="mock" if CFG.mock_mode else os.path.basename(CFG.workflow_json_path or ""),
            seed=winner_seed,
            reused_from="",
            image_path=winner_path,
            note=note.strip(),
        )
        return (f"🔒 Locked the concept design for '{entry_id}'. Switch to **Sheet panel** below "
                "to start building poses/expressions — each one will start from this design.")

    # Phase 2 — an individual panel of the fixed sheet template.
    if not panel_key:
        return "⚠️ Pick which panel of the sheet you're building first."
    store.lock_panel(
        path=CFG.storyboard_path,
        entry_id=entry_id.strip(),
        panel_key=panel_key,
        prompt_positive=prompt_positive.strip(),
        prompt_negative_add=prompt_negative.strip(),
        seed=winner_seed,
        image_path=winner_path,
        note=note.strip(),
    )
    template = composer.get_template(entry_type)
    filled = len(store.get_panel_images(CFG.storyboard_path, entry_id))
    return (f"🔒 Locked panel '{panel_key}' for '{entry_id}' ({filled}/{len(template)} panels done). "
            "Render the sheet preview below to see it in context.")


# ---------------------------------------------------------------------------
# Panel-template / context-aware form logic (CHAR / MASTER only)
# ---------------------------------------------------------------------------

def on_entry_type_change(entry_type):
    """Show the two-stage panel section only for CHAR/MASTER; populate the
    panel dropdown from that type's fixed template and reset to the
    Concept stage, since that's always where a new entry starts."""
    if entry_type in SHEET_TYPES:
        template = composer.get_template(entry_type)
        choices = [p["key"] for p in template]
        return (
            gr.update(visible=True),
            gr.update(choices=choices, value=choices[0] if choices else None),
            gr.update(visible=True),
            gr.update(value=CONCEPT_STAGE),
        )
    return (gr.update(visible=False), gr.update(choices=[], value=None),
            gr.update(visible=False), gr.update())


def on_build_stage_change(build_stage):
    """The panel picker only matters in Sheet-panel mode."""
    return gr.update(visible=(build_stage == PANEL_STAGE))


def load_panel_context(entry_id, entry_type, build_stage, panel_key):
    """Whenever the name, stage, or selected panel changes, show what's
    already known instead of a blank form.

    Concept stage: prefill from the entry's own locked concept (Assets row).
    Sheet-panel stage: prefill from that exact panel's last lock if one
    exists; otherwise fall back to the locked concept's description as a
    starting point, since a fresh panel should extend the concept, not
    reinvent it — that's the whole point of doing Phase 1 first."""
    if entry_type not in SHEET_TYPES or not entry_id:
        return gr.update(), gr.update(), ""

    if build_stage == CONCEPT_STAGE:
        entry = store.get_entry(CFG.storyboard_path, entry_id)
        if not entry or not entry.get("image_path"):
            return "", "", ("No concept locked yet for this name — describe the character/backdrop "
                             "using your reference images as a guide, then generate and lock one.")
        return (
            entry.get("prompt_positive") or "",
            entry.get("prompt_negative_add") or "",
            f"📄 Loaded the locked concept (seed {entry.get('seed')}). Generate more variants to "
            "refine it further, or move to **Sheet panel** below to start building poses.",
        )

    # Sheet-panel stage
    if not panel_key:
        return gr.update(), gr.update(), ""
    panels = store.list_panels(CFG.storyboard_path, entry_id)
    match = next((p for p in panels if p.get("panel_key") == panel_key), None)
    if match:
        last_note = (match.get("notes") or "").strip().splitlines()[-1] if match.get("notes") else ""
        return (
            match.get("prompt_positive") or "",
            match.get("prompt_negative_add") or "",
            f"📄 Loaded the last locked prompt for **{panel_key}** (seed {match.get('seed')}). "
            f"{last_note} — edit and regenerate, or lock as-is.",
        )

    entry = store.get_entry(CFG.storyboard_path, entry_id)
    if entry and entry.get("image_path"):
        return (
            entry.get("prompt_positive") or "",
            entry.get("prompt_negative_add") or "",
            f"📄 No **{panel_key}** panel yet — starting from your locked concept design. Keep the "
            "outfit, colors, and silhouette the same; just change the pose or expression described.",
        )
    return "", "", (f"No **{panel_key}** panel yet, and no concept locked either. Consider locking "
                     "a concept first (Concept stage above) so every panel starts from the same design.")


def concept_thumb_refresh(entry_id, entry_type):
    if entry_type not in SHEET_TYPES or not entry_id:
        return None
    entry = store.get_entry(CFG.storyboard_path, entry_id)
    path = entry.get("image_path") if entry else None
    return path if path and os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Reference images (mood board, pre-design)
# ---------------------------------------------------------------------------

def refresh_references(entry_id):
    if not entry_id:
        return []
    refs = store.list_references(CFG.storyboard_path, entry_id)
    return [(r["image_path"], r.get("note") or "") for r in refs if r.get("image_path") and os.path.exists(r["image_path"])]


def add_reference_ui(entry_id, ref_file, ref_note):
    if not entry_id:
        return "⚠️ Enter a name first (Step 1).", refresh_references(entry_id)
    if ref_file is None:
        return "⚠️ Choose an image to upload first.", refresh_references(entry_id)
    dest_dir = os.path.join(CFG.images_dir, "references")
    os.makedirs(dest_dir, exist_ok=True)
    safe_id = entry_id.strip().replace(":", "_")
    fname = os.path.basename(ref_file.name)
    dest = os.path.join(dest_dir, f"{safe_id}_{fname}")
    shutil.copy(ref_file.name, dest)
    store.add_reference(CFG.storyboard_path, entry_id.strip(), dest, (ref_note or "").strip())
    return f"✅ Added a reference image for {entry_id}.", refresh_references(entry_id)


# ---------------------------------------------------------------------------
# Composite sheet rendering
# ---------------------------------------------------------------------------

def render_composite_preview(entry_id, entry_type):
    if not entry_id:
        return None, "⚠️ Enter a name first (Step 1)."
    if entry_type not in SHEET_TYPES:
        return None, "⚠️ Composite sheets are only for CHAR/MASTER entries."
    panel_images = store.get_panel_images(CFG.storyboard_path, entry_id)
    concept_entry = store.get_entry(CFG.storyboard_path, entry_id)
    concept_path = concept_entry.get("image_path") if concept_entry else None
    safe_id = entry_id.strip().replace(":", "_")
    out_dir = os.path.join(CFG.images_dir, "sheets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_id}_sheet.png")
    composer.render_sheet(entry_id, entry_type, panel_images, out_path, concept_image_path=concept_path)
    template = composer.get_template(entry_type)
    filled = sum(1 for p in template if panel_images.get(p["key"]))
    return out_path, f"🖼️ Rendered preview — {filled}/{len(template)} panels filled in."


def save_composite_ui(entry_id, entry_type, composite_path, sheet_note):
    if not composite_path:
        return "⚠️ Render a preview first (the button above)."
    store.set_composite_image(CFG.storyboard_path, entry_id.strip(), entry_type, entry_type,
                               composite_path, (sheet_note or "").strip())
    return f"🔒 Saved as {entry_id}'s reference sheet. Check the Registry tab."


# ---------------------------------------------------------------------------
# Registry tab
# ---------------------------------------------------------------------------

def registry_refresh():
    rows = store.list_entries(CFG.storyboard_path)
    return [
        [r.get("entry_id"), r.get("entry_type"), r.get("beat"), r.get("description"),
         r.get("model"), r.get("seed"), r.get("locked"), r.get("template"), r.get("image_path")]
        for r in rows
    ]


def sheets_gallery_refresh():
    rows = store.list_entries(CFG.storyboard_path)
    items = []
    for r in rows:
        cp = r.get("composite_image_path")
        if cp and os.path.exists(cp):
            items.append((cp, r.get("entry_id")))
    return items


# ---------------------------------------------------------------------------
# Per-field "❓ More info" helper
# ---------------------------------------------------------------------------

def _toggle_help(is_visible):
    new_val = not bool(is_visible)
    return new_val, gr.update(visible=new_val)


# ---------------------------------------------------------------------------
# Theme + styling
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
)

CUSTOM_CSS = """
.step-card {
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-left: 5px solid var(--card-accent, #14b8a6);
}
.step-1 { background: #ecfdf5; border-left-color: #14b8a6; }
.step-2 { background: #eff6ff; border-left-color: #3b82f6; }
.step-3 { background: #fefce8; border-left-color: #eab308; }
.step-4 { background: #fdf4ff; border-left-color: #c026d3; }
.step-5 { background: #f0fdf4; border-left-color: #22c55e; }
.step-6 { background: #fff1f2; border-left-color: #e11d48; }
.step-card h4 { margin-top: 0 !important; }
.help-btn { max-width: 40px !important; }
.help-panel {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 8px 14px !important;
    margin-top: -6px;
    margin-bottom: 10px;
    font-size: 0.92em;
}
#welcome-hero {
    background: linear-gradient(135deg, #ecfeff 0%, #fef9c3 100%);
    border-radius: 16px;
    padding: 24px 28px;
}
"""


# ---------------------------------------------------------------------------
# Welcome tab content
# ---------------------------------------------------------------------------

WELCOME_MARKDOWN = """
# 👋 Welcome — what this tool is for

If you're generating character art, backdrops, or shot images with an AI
image model, you'll usually make several versions before one actually looks
right — and once it does, you want to **remember exactly how you made it**
so you (or a teammate) can reuse or reproduce it later.

This tool is that memory. For a single shot, it's: **generate a batch of
options → pick your favorite → lock it in with a note about why.**

For a **character or a recurring backdrop**, it's two stages. First,
**Concept**: explore freely with reference images as inspiration, generate
variants, and settle on the one design that's *the* character. Then,
**Sheet panels**: build each pose/expression/variant of a fixed template,
each one staying tight to that locked concept rather than reinventing it —
assembled automatically into one composite reference sheet.

### The tabs, in the order you'll actually use them

1. **⚙️ Setup** — tell the tool where your image generator lives. Skip this
   at first — Mock Mode is on by default and lets you try everything with
   fake placeholder images.
2. **🛠️ Build** — name what you're making. A plain shot is one
   generate → pick → lock cycle. A character or backdrop walks you through
   its fixed set of panels one at a time, then renders them into one sheet.
3. **📋 Registry** — everything you've locked so far, including rendered
   sheets.

### The flow, visually
"""

FLOW_SVG = """
<svg viewBox="0 0 900 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:820px;font-family:sans-serif;">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#888"/>
    </marker>
  </defs>
  <g font-size="14" text-anchor="middle">
    <rect x="10" y="50" width="150" height="70" rx="10" fill="#ecfdf5" stroke="#14b8a6" stroke-width="1.5"/>
    <text x="85" y="80" font-weight="bold">1. Name it</text>
    <text x="85" y="100" font-size="12" fill="#555">CHAR:pig, 1.1, etc.</text>

    <rect x="190" y="50" width="150" height="70" rx="10" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="265" y="80" font-weight="bold">2. Describe it</text>
    <text x="265" y="100" font-size="12" fill="#555">prompt + what to avoid</text>

    <rect x="370" y="50" width="150" height="70" rx="10" fill="#fefce8" stroke="#eab308" stroke-width="1.5"/>
    <text x="445" y="80" font-weight="bold">3. Generate</text>
    <text x="445" y="100" font-size="12" fill="#555">a few options at once</text>

    <rect x="550" y="50" width="150" height="70" rx="10" fill="#fdf4ff" stroke="#c026d3" stroke-width="1.5"/>
    <text x="625" y="80" font-weight="bold">4. Pick a winner</text>
    <text x="625" y="100" font-size="12" fill="#555">the one that looks right</text>

    <rect x="730" y="50" width="150" height="70" rx="10" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
    <text x="805" y="80" font-weight="bold">5. Lock it</text>
    <text x="805" y="100" font-size="12" fill="#555">saved + noted, for good</text>
  </g>

  <line x1="160" y1="85" x2="188" y2="85" stroke="#888" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="340" y1="85" x2="368" y2="85" stroke="#888" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="520" y1="85" x2="548" y2="85" stroke="#888" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="700" y1="85" x2="728" y2="85" stroke="#888" stroke-width="2" marker-end="url(#arrow)"/>

  <path d="M805 120 C 805 150, 85 150, 85 120" stroke="#bbb" stroke-width="1.5" fill="none" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <text x="445" y="160" font-size="12" text-anchor="middle" fill="#888">for a character/backdrop, repeat 2-5 per panel, then render the sheet</text>
</svg>
"""

WELCOME_MARKDOWN_2 = """
Once something is locked, it shows up permanently in the **Registry** tab —
that's your project's single source of truth going forward.

Look for a **❓ More info** button next to any field you're unsure about.

**Ready?** Click the **🛠️ Build** tab above and try it — Mock Mode is on, so
this costs nothing and can't break anything.
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ComfyUI Director Harness", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🎬 ComfyUI Director Harness")

    with gr.Tab("👋 Welcome"):
        with gr.Group(elem_id="welcome-hero"):
            gr.Markdown(WELCOME_MARKDOWN)
            gr.HTML(FLOW_SVG)
            gr.Markdown(WELCOME_MARKDOWN_2)

    with gr.Tab("⚙️ Setup"):
        gr.Markdown(
            "### One-time setup\n"
            "You only need this if you're connecting a real image generator "
            "(ComfyUI). **If you just want to try the tool out, skip straight "
            "to the Build tab** — Mock Mode below is on by default and fakes "
            "everything safely."
        )

        with gr.Group(elem_classes=["step-card", "step-1"]):
            gr.Markdown("#### Mock Mode")
            with gr.Row():
                mock_mode = gr.Checkbox(
                    label="Mock Mode (recommended while you're learning the tool)",
                    value=CFG.mock_mode, scale=9,
                )
                mm_btn = gr.Button("❓", scale=0, min_width=36, size="sm", elem_classes=["help-btn"])
            mm_help = gr.Markdown(
                "**When ON:** the Build tab never contacts a real image generator — "
                "it makes simple colored placeholder images instead, instantly, for "
                "free. Use this to learn the flow risk-free.\n\n"
                "**When OFF:** Build sends real requests to the ComfyUI URL and "
                "workflow file below.",
                visible=False, elem_classes=["help-panel"],
            )
            mm_state = gr.State(False)
            mm_btn.click(_toggle_help, inputs=mm_state, outputs=[mm_state, mm_help])

        with gr.Accordion("Real image generator connection (advanced)", open=not CFG.mock_mode):
            gr.Markdown("This section only matters once you turn Mock Mode off.")

            with gr.Group(elem_classes=["step-card", "step-2"]):
                with gr.Row():
                    comfyui_url = gr.Textbox(label="ComfyUI URL", value=CFG.comfyui_url,
                                              info="Where ComfyUI is running.", scale=9)
                    url_btn = gr.Button("❓", scale=0, min_width=36, size="sm", elem_classes=["help-btn"])
                url_help = gr.Markdown(
                    "If ComfyUI is running on this same computer, the default "
                    "`http://127.0.0.1:8188` is almost always correct. On another "
                    "machine, use that machine's IP, e.g. `http://192.168.1.20:8188`.",
                    visible=False, elem_classes=["help-panel"])
                url_state = gr.State(False)
                url_btn.click(_toggle_help, inputs=url_state, outputs=[url_state, url_help])

                gr.Markdown(
                    "In ComfyUI, build and test your image workflow, then use "
                    "**Save (API Format)** to export it as a `.json` file, and "
                    "upload it below."
                )
                workflow_file = gr.File(label="Workflow file", file_types=[".json"])

                gr.Markdown("**Which part of the workflow does what?**")
                with gr.Row():
                    pos_node = gr.Textbox(label="Prompt (positive) node ID",
                                           value=CFG.node_mapping.positive_prompt_node, info="e.g. '6'")
                    pos_input = gr.Textbox(label="...field name on that node",
                                            value=CFG.node_mapping.positive_prompt_input, info="Usually 'text'")
                with gr.Row():
                    neg_node = gr.Textbox(label="What-to-avoid node ID",
                                           value=CFG.node_mapping.negative_prompt_node)
                    neg_input = gr.Textbox(label="...field name on that node",
                                            value=CFG.node_mapping.negative_prompt_input)
                with gr.Row():
                    seed_node = gr.Textbox(label="Seed node ID", value=CFG.node_mapping.seed_node)
                    seed_input = gr.Textbox(label="...field name on that node",
                                             value=CFG.node_mapping.seed_input)

        with gr.Group(elem_classes=["step-card", "step-3"]):
            storyboard_path = gr.Textbox(
                label="Where should locked images be recorded?",
                value=CFG.storyboard_path, info="A spreadsheet file, created automatically.",
            )

        setup_status = gr.Markdown("")
        gr.Button("Save Setup", variant="primary").click(
            setup_save,
            inputs=[comfyui_url, workflow_file, pos_node, pos_input, neg_node, neg_input,
                    seed_node, seed_input, storyboard_path, mock_mode],
            outputs=setup_status,
        )

    with gr.Tab("🛠️ Build"):
        gr.Markdown(
            "Work through the steps top to bottom. Nothing is saved "
            "permanently until you hit **Lock**."
        )

        with gr.Group(elem_classes=["step-card", "step-1"]):
            gr.Markdown("#### Step 1 — Name what you're making")
            with gr.Row():
                entry_id = gr.Textbox(
                    label="Name",
                    placeholder="e.g. CHAR:pig, MASTER:farm_field, or a shot number like 1.1",
                    scale=9,
                )
                id_btn = gr.Button("❓", scale=0, min_width=36, size="sm", elem_classes=["help-btn"])
                entry_type = gr.Dropdown(label="Type", choices=["SHOT", "CHAR", "MASTER", "PROP"],
                                          value="SHOT", info="What kind of thing this is.")
            id_help = gr.Markdown(
                "Use `CHAR:name` for a character (e.g. `CHAR:pig`) or `MASTER:name` "
                "for a recurring backdrop (e.g. `MASTER:farm_field`) — both build a "
                "full reference sheet, panel by panel. Use a shot number (e.g. `1.1`) "
                "for a single scene image. Reusing the same name updates that entry "
                "instead of creating a new one.",
                visible=False, elem_classes=["help-panel"])
            id_state = gr.State(False)
            id_btn.click(_toggle_help, inputs=id_state, outputs=[id_state, id_help])

            beat = gr.Textbox(label="Section / beat (optional)")
            description = gr.Textbox(label="Short description (for your own reference)", lines=2,
                                      placeholder="e.g. 'the pig, front-facing, tweed cap'")

            # --- CHAR / MASTER only: two-stage build (concept, then panels) ---
            with gr.Group(visible=False) as panel_group:
                gr.Markdown(
                    "#### This is a sheet — built in two stages\n"
                    "**Concept (mashup):** explore freely using your reference images as "
                    "inspiration, then lock the one design that's *the* character or backdrop.\n\n"
                    "**Sheet panel:** once a concept is locked, build each pose/expression/"
                    "variant of the fixed template — every panel should stay tight to the "
                    "locked concept's design, not reinterpret it."
                )
                with gr.Row():
                    build_stage = gr.Radio(label="Stage", choices=[CONCEPT_STAGE, PANEL_STAGE],
                                            value=CONCEPT_STAGE, scale=2)
                    concept_thumb = gr.Image(label="Locked concept", interactive=False, scale=1,
                                              height=120)
                panel_key = gr.Dropdown(label="Which panel are you building?", choices=[], visible=False)
                panel_status = gr.Markdown("")

                with gr.Accordion("📎 Reference images (mood board, optional)", open=False):
                    gr.Markdown(
                        "Attach downloaded/inspiration images here before you've "
                        "settled on a design. These are for your own reference — they "
                        "don't get used as generation input directly."
                    )
                    with gr.Row():
                        ref_file = gr.File(label="Image to attach", file_types=["image"])
                        ref_note = gr.Textbox(label="What to borrow from it", scale=2,
                                               placeholder="e.g. 'like this jacket silhouette'")
                    ref_add_btn = gr.Button("Add reference")
                    ref_status = gr.Markdown("")
                    ref_gallery = gr.Gallery(label="Attached references", columns=4, height=200)
                    ref_add_btn.click(add_reference_ui, inputs=[entry_id, ref_file, ref_note],
                                       outputs=[ref_status, ref_gallery])

        with gr.Group(elem_classes=["step-card", "step-2"]):
            gr.Markdown("#### Step 2 — Describe what you want to see")
            prompt_positive = gr.Textbox(
                label="Describe the image", lines=3,
                placeholder="e.g. a dignified pig wearing a tweed cap and waistcoat, hand-painted illustration style",
            )
            prompt_negative = gr.Textbox(
                label="Things to avoid (optional)", lines=2,
                placeholder="e.g. no watercolor, no extra limbs",
            )

        with gr.Group(elem_classes=["step-card", "step-3"]):
            gr.Markdown("#### Step 3 — Generate some options")
            with gr.Row():
                base_seed = gr.Number(label="Seed (leave blank for random)", value=None)
                n_variants = gr.Slider(label="How many versions", minimum=1, maximum=8, step=1, value=4)
            gen_btn = gr.Button("Generate", variant="primary")
            gallery = gr.Gallery(label="Your options — each one is labeled with its seed", columns=4)
            gen_status = gr.Markdown("")
            gen_btn.click(
                generate_click,
                inputs=[entry_id, entry_type, build_stage, panel_key, prompt_positive, prompt_negative,
                        base_seed, n_variants],
                outputs=[gallery, gen_status],
            )

        with gr.Group(elem_classes=["step-card", "step-4"]):
            gr.Markdown(
                "#### Step 4 — Which one looks right?\n"
                "Click an image above to see it larger, then copy its file path and "
                "seed into the two boxes below."
            )
            with gr.Row():
                winner_path = gr.Textbox(label="Winning image's file path")
                winner_seed = gr.Number(label="Winning image's seed")
            note = gr.Textbox(
                label="Why this one? (saved permanently with the record)", lines=2,
                placeholder="e.g. 'first version where the tweed cap read clearly at this angle'",
            )

        with gr.Group(elem_classes=["step-card", "step-5"]):
            gr.Markdown("#### Step 5 — Lock it in")
            lock_btn = gr.Button("🔒 Lock this in", variant="primary")
            lock_status = gr.Markdown("")
            lock_btn.click(
                lock_click,
                inputs=[entry_id, entry_type, build_stage, panel_key, beat, description, prompt_positive,
                        prompt_negative, winner_path, winner_seed, note],
                outputs=lock_status,
            ).then(concept_thumb_refresh, inputs=[entry_id, entry_type], outputs=concept_thumb)

        # --- CHAR / MASTER only: composite sheet rendering ---
        with gr.Group(visible=False, elem_classes=["step-card", "step-6"]) as composite_group:
            gr.Markdown(
                "#### Step 6 — Render the sheet\n"
                "Puts every locked panel together into one composite reference image. "
                "You can render a preview at any point — panels you haven't locked "
                "yet just show as a placeholder."
            )
            render_btn = gr.Button("Render preview")
            composite_image = gr.Image(label="Composite sheet preview", type="filepath")
            render_status = gr.Markdown("")
            render_btn.click(render_composite_preview, inputs=[entry_id, entry_type],
                              outputs=[composite_image, render_status])

            sheet_note = gr.Textbox(label="Note for this sheet version", lines=1,
                                     placeholder="e.g. 'first full pass, 9/13 panels'")
            save_sheet_btn = gr.Button("🔒 Save as this entry's reference sheet", variant="primary")
            save_sheet_status = gr.Markdown("")
            save_sheet_btn.click(save_composite_ui, inputs=[entry_id, entry_type, composite_image, sheet_note],
                                  outputs=save_sheet_status)

        # --- Wiring for entry_type / stage / panel context awareness ---
        entry_type.change(on_entry_type_change, inputs=entry_type,
                           outputs=[panel_group, panel_key, composite_group, build_stage])
        build_stage.change(on_build_stage_change, inputs=build_stage, outputs=panel_key)
        entry_id.change(load_panel_context, inputs=[entry_id, entry_type, build_stage, panel_key],
                         outputs=[prompt_positive, prompt_negative, panel_status])
        build_stage.change(load_panel_context, inputs=[entry_id, entry_type, build_stage, panel_key],
                            outputs=[prompt_positive, prompt_negative, panel_status])
        panel_key.change(load_panel_context, inputs=[entry_id, entry_type, build_stage, panel_key],
                          outputs=[prompt_positive, prompt_negative, panel_status])
        entry_id.change(refresh_references, inputs=entry_id, outputs=ref_gallery)
        entry_id.change(concept_thumb_refresh, inputs=[entry_id, entry_type], outputs=concept_thumb)
        entry_type.change(concept_thumb_refresh, inputs=[entry_id, entry_type], outputs=concept_thumb)

    with gr.Tab("📋 Registry"):
        gr.Markdown(
            "Everything you've locked so far. Locking a new version of "
            "something you've already named updates that entry and adds "
            "your new note underneath the old one — it won't duplicate."
        )
        refresh_btn = gr.Button("Refresh")
        registry_table = gr.Dataframe(
            headers=["Name", "Type", "Section", "Description", "Model", "Seed", "Locked", "Template", "Image path"],
            interactive=False,
        )
        refresh_btn.click(registry_refresh, outputs=registry_table)
        demo.load(registry_refresh, outputs=registry_table)

        gr.Markdown("#### 🖼️ Character & backdrop sheets")
        sheets_refresh_btn = gr.Button("Refresh sheets")
        sheets_gallery = gr.Gallery(label="Rendered composite sheets", columns=3, height=300)
        sheets_refresh_btn.click(sheets_gallery_refresh, outputs=sheets_gallery)
        demo.load(sheets_gallery_refresh, outputs=sheets_gallery)


if __name__ == "__main__":
    demo.launch()
