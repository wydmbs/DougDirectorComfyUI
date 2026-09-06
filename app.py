"""
app.py — ComfyUI Director Harness, Module 1: Image Build UI.

One shell, tabs for capability (see SUITE.md):
  Setup    — point at a ComfyUI instance + a workflow exported via
             "Save (API Format)", map which node holds positive/negative
             prompt and seed. One-time, works with any workflow.
  Build    — pick an entry_id (a shot like "1.1" or an asset like
             "CHAR:pig" / "MASTER:farm_field"), iterate the prompt,
             generate N variants, pick a winner, lock it into the registry.
  Registry — read-only view of everything locked so far.

Mock mode (default ON) generates placeholder images instead of calling
ComfyUI, so the full loop can be tested without a live GPU.
"""

import io
import json
import os
import random

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

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
# Setup tab
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
    return f"Saved. Mock mode is {'ON' if CFG.mock_mode else 'OFF'}."


# ---------------------------------------------------------------------------
# Build tab
# ---------------------------------------------------------------------------

def build_generate(entry_id, prompt_positive, prompt_negative, base_seed, n_variants):
    if not entry_id:
        return [], "Enter an entry_id first (e.g. 1.1, CHAR:pig, MASTER:farm_field)."

    base_seed = int(base_seed) if base_seed not in (None, "") else random.randint(0, 2**31 - 1)
    n_variants = max(1, min(int(n_variants), 8))

    gallery = []
    status_lines = []
    client = None
    workflow = None
    if not CFG.mock_mode:
        if not CFG.workflow_json_path or not os.path.exists(CFG.workflow_json_path):
            return [], "No workflow JSON configured — set one in the Setup tab, or turn mock mode on."
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

    status = f"Generated {len(gallery)}/{n_variants} variant(s)." if gallery else "No images generated."
    if status_lines:
        status += " " + " | ".join(status_lines)
    return gallery, status


def build_lock(entry_id, entry_type, beat, description, prompt_positive,
                prompt_negative, winner_path, winner_seed, note):
    if not entry_id:
        return "Enter an entry_id first."
    if not winner_path:
        return "Generate variants and pick a winner path first."

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
    return f"Locked {entry_id} (seed {winner_seed})."


# ---------------------------------------------------------------------------
# Registry tab
# ---------------------------------------------------------------------------

def registry_refresh():
    rows = store.list_entries(CFG.storyboard_path)
    table = [
        [r.get("entry_id"), r.get("entry_type"), r.get("beat"), r.get("description"),
         r.get("model"), r.get("seed"), r.get("locked"), r.get("image_path")]
        for r in rows
    ]
    return table


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ComfyUI Director Harness") as demo:
    gr.Markdown("# ComfyUI Director Harness — Module 1: Image Build")

    with gr.Tab("Setup"):
        gr.Markdown(
            "Point this at a ComfyUI instance and a workflow exported via "
            "**Save (API Format)**. Map which node IDs hold the positive prompt, "
            "negative prompt, and seed. Leave mock mode on to test the loop "
            "without a live ComfyUI/GPU."
        )
        comfyui_url = gr.Textbox(label="ComfyUI URL", value=CFG.comfyui_url)
        workflow_file = gr.File(label="Workflow JSON (API format)", file_types=[".json"])
        with gr.Row():
            pos_node = gr.Textbox(label="Positive prompt node ID", value=CFG.node_mapping.positive_prompt_node)
            pos_input = gr.Textbox(label="…input key", value=CFG.node_mapping.positive_prompt_input)
        with gr.Row():
            neg_node = gr.Textbox(label="Negative prompt node ID", value=CFG.node_mapping.negative_prompt_node)
            neg_input = gr.Textbox(label="…input key", value=CFG.node_mapping.negative_prompt_input)
        with gr.Row():
            seed_node = gr.Textbox(label="Seed node ID", value=CFG.node_mapping.seed_node)
            seed_input = gr.Textbox(label="…input key", value=CFG.node_mapping.seed_input)
        storyboard_path = gr.Textbox(label="Registry .xlsx path", value=CFG.storyboard_path)
        mock_mode = gr.Checkbox(label="Mock mode (no real ComfyUI calls)", value=CFG.mock_mode)
        setup_status = gr.Textbox(label="Status", interactive=False)
        gr.Button("Save Setup").click(
            setup_save,
            inputs=[comfyui_url, workflow_file, pos_node, pos_input, neg_node, neg_input,
                    seed_node, seed_input, storyboard_path, mock_mode],
            outputs=setup_status,
        )

    with gr.Tab("Build"):
        entry_id = gr.Textbox(label="entry_id", placeholder="1.1  /  CHAR:pig  /  MASTER:farm_field")
        entry_type = gr.Dropdown(label="entry_type", choices=["SHOT", "CHAR", "MASTER", "PROP"], value="SHOT")
        beat = gr.Textbox(label="beat")
        description = gr.Textbox(label="description", lines=2)
        prompt_positive = gr.Textbox(label="prompt_positive", lines=3)
        prompt_negative = gr.Textbox(label="prompt_negative_add", lines=2)
        with gr.Row():
            base_seed = gr.Number(label="base seed (blank = random)", value=None)
            n_variants = gr.Slider(label="variants", minimum=1, maximum=8, step=1, value=4)
        gen_btn = gr.Button("Generate variants")
        gallery = gr.Gallery(label="Variants", columns=4)
        gen_status = gr.Textbox(label="Generation status", interactive=False)
        gen_btn.click(
            build_generate,
            inputs=[entry_id, prompt_positive, prompt_negative, base_seed, n_variants],
            outputs=[gallery, gen_status],
        )

        gr.Markdown("Pick the winner (path + seed as shown in the gallery caption), then lock it.")
        winner_path = gr.Textbox(label="Winner image path")
        winner_seed = gr.Number(label="Winner seed")
        note = gr.Textbox(label="Note (what was tried / why this won)", lines=2)
        lock_btn = gr.Button("Lock winner into registry")
        lock_status = gr.Textbox(label="Lock status", interactive=False)
        lock_btn.click(
            build_lock,
            inputs=[entry_id, entry_type, beat, description, prompt_positive,
                    prompt_negative, winner_path, winner_seed, note],
            outputs=lock_status,
        )

    with gr.Tab("Registry"):
        gr.Markdown("Read-only view of everything locked so far.")
        refresh_btn = gr.Button("Refresh")
        registry_table = gr.Dataframe(
            headers=["entry_id", "entry_type", "beat", "description", "model", "seed", "locked", "image_path"],
            interactive=False,
        )
        refresh_btn.click(registry_refresh, outputs=registry_table)
        demo.load(registry_refresh, outputs=registry_table)


if __name__ == "__main__":
    demo.launch()
