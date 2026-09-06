"""
app.py — ComfyUI Director Harness, Module 1: Image Build UI.

One shell, tabs for capability (see SUITE.md):
  Welcome  — plain-language explanation of what this tool does and how the
             four tabs fit together. Read this first.
  Setup    — point at a ComfyUI instance + a workflow exported via
             "Save (API Format)", map which node holds the positive prompt,
             negative prompt, and seed. One-time, works with any workflow.
  Build    — pick an entry_id (a shot like "1.1" or an asset like
             "CHAR:pig" / "MASTER:farm_field"), iterate the prompt,
             generate N variants, pick a winner, lock it into the registry.
  Registry — read-only view of everything locked so far.

Mock mode (default ON) generates placeholder images instead of calling
ComfyUI, so the full loop can be tried with zero setup and no GPU.

Every field with room for confusion carries a short inline hint (Gradio's
info=) plus a "❓ More info" toggle button that reveals a longer explanation
without cluttering the form for people who don't need it.
"""

import io
import json
import os
import random

import gradio as gr
from PIL import Image, ImageDraw

import config as cfgmod
import storyboard_store as store
from comfy_client import ComfyClient, ComfyClientError, apply_node_overrides

CFG = cfgmod.load_config()
os.makedirs(CFG.images_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Mock generation (no GPU / ComfyUI required)
# ---------------------------------------------------------------------------

def _mock_image(entry_id: str, prompt: str, seed: int) -> Image.Image:
    random.seed(seed)
    color = tuple(random.randint(40, 200) for _ in range(3))
    img = Image.new("RGB", (512, 288), color)
    draw = ImageDraw.Draw(img)
    text = f"{entry_id}\nseed {seed}\n{prompt[:60]}"
    draw.multiline_text((12, 12), text, fill=(255, 255, 255))
    return img


def _save_image(img: Image.Image, entry_id: str, seed: int) -> str:
    safe_id = str(entry_id).replace(":", "_").replace("/", "_")
    fname = f"{safe_id}_{seed}.png"
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
# Build tab logic
# ---------------------------------------------------------------------------

def build_generate(entry_id, prompt_positive, prompt_negative, base_seed, n_variants):
    if not entry_id:
        return [], "⚠️ Give this a name first (Step 1) — try 'CHAR:pig' or '1.1'."

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
            img = _mock_image(entry_id, prompt_positive, seed)
            path = _save_image(img, entry_id, seed)
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
            path = _save_image(img, entry_id, seed)
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


def build_lock(entry_id, entry_type, beat, description, prompt_positive,
                prompt_negative, winner_path, winner_seed, note):
    if not entry_id:
        return "⚠️ Give this a name first (Step 1)."
    if not winner_path:
        return "⚠️ Generate some variants (Step 3) and tell me which one won (Step 4) before locking."

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
    return f"🔒 Locked '{entry_id}' with seed {winner_seed}. Check the Registry tab to see it."


def registry_refresh():
    rows = store.list_entries(CFG.storyboard_path)
    table = [
        [r.get("entry_id"), r.get("entry_type"), r.get("beat"), r.get("description"),
         r.get("model"), r.get("seed"), r.get("locked"), r.get("image_path")]
        for r in rows
    ]
    return table


# ---------------------------------------------------------------------------
# Per-field "❓ More info" helper
# ---------------------------------------------------------------------------

def _toggle_help(is_visible):
    """Shared click handler for every help button: flip a per-field boolean
    kept in a gr.State, and show/hide that field's detail panel to match."""
    new_val = not bool(is_visible)
    return new_val, gr.update(visible=new_val)


def with_help(component, help_markdown: str):
    """Wrap an already-created component with a '❓ More info' button and a
    collapsible detail panel underneath. Must be called inside the gr.Row
    that lays the component out, immediately after creating it — see usage
    below. Returns the same component unchanged so it can still be wired
    into other event handlers."""
    help_btn = gr.Button("❓", scale=0, min_width=36, size="sm", elem_classes=["help-btn"])
    help_panel = gr.Markdown(help_markdown, visible=False, elem_classes=["help-panel"])
    state = gr.State(False)
    help_btn.click(_toggle_help, inputs=state, outputs=[state, help_panel])
    return component


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

This tool is that memory. It walks you through: **generate a batch of
options → pick your favorite → lock it in with a note about why** — and it
keeps a running, searchable record of every locked image so nothing gets
lost in a folder of "final_v3_REAL_final.png" files.

### The three tabs, in the order you'll actually use them

1. **⚙️ Setup** — tell the tool where your image generator lives. You only
   do this once. *(Not sure yet? Skip it — Mock Mode is on by default and
   lets you try the whole flow with fake placeholder images first.)*
2. **🛠️ Build** — your day-to-day workspace. Name what you're making,
   describe it, generate a few versions, pick your favorite, lock it in.
3. **📋 Registry** — a read-only list of everything you've locked so far,
   like a photo album with notes attached to each picture.

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
  <text x="445" y="160" font-size="12" text-anchor="middle" fill="#888">not happy? tweak the description and generate again — nothing is locked until Step 5</text>
</svg>
"""

WELCOME_MARKDOWN_2 = """
Once something is locked, it shows up permanently in the **Registry** tab —
that's your project's single source of truth going forward.

Look for a **❓ More info** button next to any field you're unsure about —
click it for a longer explanation with examples.

**Ready?** Click the **🛠️ Build** tab above and try it — Mock Mode is on, so
this costs nothing and can't break anything. Come back to **⚙️ Setup** only
once you're ready to connect a real image generator.
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
                mock_mode = with_help(
                    gr.Checkbox(
                        label="Mock Mode (recommended while you're learning the tool)",
                        value=CFG.mock_mode,
                    ),
                    "**When ON:** the Build tab never contacts a real image generator — "
                    "it makes simple colored placeholder images instead, instantly, for "
                    "free. Use this to learn the generate → review → lock flow risk-free.\n\n"
                    "**When OFF:** Build sends real requests to the ComfyUI URL and "
                    "workflow file configured below. Only turn this off once you've "
                    "built and tested a workflow in ComfyUI's own interface first.",
                )

        with gr.Accordion("Real image generator connection (advanced)", open=not CFG.mock_mode):
            gr.Markdown("This section only matters once you turn Mock Mode off.")

            with gr.Group(elem_classes=["step-card", "step-2"]):
                with gr.Row():
                    comfyui_url = with_help(
                        gr.Textbox(
                            label="ComfyUI URL",
                            value=CFG.comfyui_url,
                            info="Where ComfyUI is running.",
                        ),
                        "If ComfyUI is running on this same computer, the default "
                        "`http://127.0.0.1:8188` is almost always correct. If it's running "
                        "on another machine on your network, replace `127.0.0.1` with that "
                        "machine's IP address, e.g. `http://192.168.1.20:8188`.",
                    )

                gr.Markdown(
                    "In ComfyUI, build and test your image workflow, then use "
                    "**Save (API Format)** to export it as a `.json` file, and "
                    "upload it below."
                )
                with gr.Row():
                    workflow_file = with_help(
                        gr.File(label="Workflow file", file_types=[".json"]),
                        "This is **not** the same as ComfyUI's normal 'Save' — that "
                        "produces a file meant only for ComfyUI's own editor. Look for "
                        "**Save (API Format)** specifically in ComfyUI's menu; that "
                        "version is structured so this tool can read and modify it "
                        "automatically for each new prompt and seed.",
                    )

                gr.Markdown(
                    "**Which part of the workflow does what?** Every workflow is laid "
                    "out a little differently, so tell the tool which piece is which:"
                )
                with gr.Row():
                    pos_node = with_help(
                        gr.Textbox(label="Prompt (positive) node ID",
                                   value=CFG.node_mapping.positive_prompt_node,
                                   info="e.g. '6'"),
                        "Open your workflow's `.json` file in a text editor, or hover "
                        "the relevant node in ComfyUI — the node ID is the number shown "
                        "in its corner. This should be the node where your main image "
                        "description text goes in.",
                    )
                    pos_input = gr.Textbox(label="...field name on that node",
                                            value=CFG.node_mapping.positive_prompt_input,
                                            info="Usually 'text' — leave as-is unless you know otherwise.")
                with gr.Row():
                    neg_node = with_help(
                        gr.Textbox(label="What-to-avoid node ID",
                                   value=CFG.node_mapping.negative_prompt_node),
                        "The node where you list things you don't want to see in the "
                        "image. Optional — leave blank if your workflow doesn't use a "
                        "separate negative prompt.",
                    )
                    neg_input = gr.Textbox(label="...field name on that node",
                                            value=CFG.node_mapping.negative_prompt_input)
                with gr.Row():
                    seed_node = with_help(
                        gr.Textbox(label="Seed node ID",
                                   value=CFG.node_mapping.seed_node),
                        "Controls randomness. The same seed plus the same prompt "
                        "reproduces the exact same image later — useful for revisiting "
                        "a specific result.",
                    )
                    seed_input = gr.Textbox(label="...field name on that node",
                                             value=CFG.node_mapping.seed_input)

        with gr.Group(elem_classes=["step-card", "step-3"]):
            with gr.Row():
                storyboard_path = with_help(
                    gr.Textbox(
                        label="Where should locked images be recorded?",
                        value=CFG.storyboard_path,
                        info="A spreadsheet file.",
                    ),
                    "This is the file the Registry tab reads from. It's created "
                    "automatically the first time you lock something — you don't need "
                    "to create it yourself. Use a shared drive path if teammates need "
                    "to see the same registry.",
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
            "permanently until you hit **Lock** at the very end."
        )

        with gr.Group(elem_classes=["step-card", "step-1"]):
            gr.Markdown("#### Step 1 — Name what you're making")
            with gr.Row():
                entry_id = with_help(
                    gr.Textbox(
                        label="Name",
                        placeholder="e.g. CHAR:pig, MASTER:farm_field, or a shot number like 1.1",
                    ),
                    "Use `CHAR:name` for a character (e.g. `CHAR:pig`), `MASTER:name` "
                    "for a recurring backdrop or prop (e.g. `MASTER:farm_field`), or "
                    "just a shot number (e.g. `1.1`) for a single scene. Reusing the "
                    "same name later updates that entry instead of creating a new one.",
                )
                entry_type = gr.Dropdown(
                    label="Type",
                    choices=["SHOT", "CHAR", "MASTER", "PROP"],
                    value="SHOT",
                    info="What kind of thing this is.",
                )
            beat = gr.Textbox(label="Section / beat (optional)", info="Which part of the story this belongs to, if relevant.")
            description = gr.Textbox(label="Short description (for your own reference later)", lines=2,
                                      placeholder="e.g. 'the pig, front-facing, tweed cap'")

        with gr.Group(elem_classes=["step-card", "step-2"]):
            gr.Markdown("#### Step 2 — Describe what you want to see")
            with gr.Row():
                prompt_positive = with_help(
                    gr.Textbox(
                        label="Describe the image", lines=3,
                        placeholder="e.g. a dignified pig wearing a tweed cap and waistcoat, hand-painted illustration style",
                    ),
                    "This is what the generator will try to draw. Be specific — "
                    "style, colors, pose, and framing all help. If you're refining an "
                    "existing character, keep the wording consistent with earlier "
                    "locked versions so the look doesn't drift.",
                )
            with gr.Row():
                prompt_negative = with_help(
                    gr.Textbox(
                        label="Things to avoid (optional)", lines=2,
                        placeholder="e.g. no watercolor, no extra limbs",
                    ),
                    "List anything that keeps showing up in results that you don't "
                    "want — extra limbs, a wrong art style, unwanted objects in the "
                    "background, etc.",
                )

        with gr.Group(elem_classes=["step-card", "step-3"]):
            gr.Markdown("#### Step 3 — Generate some options")
            with gr.Row():
                base_seed = with_help(
                    gr.Number(label="Seed (leave blank for random)", value=None),
                    "Only fill this in if you want a reproducible starting point — "
                    "for example, to nudge a previous winning seed slightly rather "
                    "than starting over randomly. Leave blank most of the time.",
                )
                n_variants = gr.Slider(label="How many versions to generate", minimum=1, maximum=8, step=1, value=4)
            gen_btn = gr.Button("Generate", variant="primary")
            gallery = gr.Gallery(label="Your options — each one is labeled with its seed", columns=4)
            gen_status = gr.Markdown("")
            gen_btn.click(
                build_generate,
                inputs=[entry_id, prompt_positive, prompt_negative, base_seed, n_variants],
                outputs=[gallery, gen_status],
            )

        with gr.Group(elem_classes=["step-card", "step-4"]):
            gr.Markdown(
                "#### Step 4 — Which one looks right?\n"
                "Click an image above to see it larger, then copy its file path and "
                "seed (shown under the image) into the two boxes below."
            )
            with gr.Row():
                winner_path = gr.Textbox(label="Winning image's file path")
                winner_seed = gr.Number(label="Winning image's seed")
            note = with_help(
                gr.Textbox(
                    label="Why this one? (this gets saved with the record, permanently)",
                    lines=2,
                    placeholder="e.g. 'first version where the tweed cap read clearly at this angle'",
                ),
                "This becomes part of the permanent audit trail for this asset. Future "
                "you (or a teammate) will thank you for writing down *why* this one "
                "won, not just that it did.",
            )

        with gr.Group(elem_classes=["step-card", "step-5"]):
            gr.Markdown("#### Step 5 — Lock it in")
            lock_btn = gr.Button("🔒 Lock this into the Registry", variant="primary")
            lock_status = gr.Markdown("")
            lock_btn.click(
                build_lock,
                inputs=[entry_id, entry_type, beat, description, prompt_positive,
                        prompt_negative, winner_path, winner_seed, note],
                outputs=lock_status,
            )

    with gr.Tab("📋 Registry"):
        gr.Markdown(
            "Everything you've locked so far. If you lock a new version of "
            "something you've already named (say, a second pass on "
            "`CHAR:pig`), it updates that same entry and adds your new note "
            "underneath the old one — it won't create a duplicate."
        )
        refresh_btn = gr.Button("Refresh")
        registry_table = gr.Dataframe(
            headers=["Name", "Type", "Section", "Description", "Model", "Seed", "Locked", "Image path"],
            interactive=False,
        )
        refresh_btn.click(registry_refresh, outputs=registry_table)
        demo.load(registry_refresh, outputs=registry_table)


if __name__ == "__main__":
    demo.launch()
