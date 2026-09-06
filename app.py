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
import harry_advisor as harry
from comfy_client import ComfyClient, ComfyClientError, apply_node_overrides

CFG = cfgmod.load_config()
os.makedirs(CFG.images_dir, exist_ok=True)

SHEET_TYPES = ("CHAR", "MASTER", "PROP")


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
                seed_node, seed_input, storyboard_path, mock_mode, harry_provider):
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
        harry_provider=harry_provider,
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
# Harry the Advisor
# ---------------------------------------------------------------------------

def harry_provider_status(provider):
    return harry.provider_status(provider)


def harry_save_source(title, source_text, audio_file):
    try:
        metadata = harry.save_source(title, source_text, audio_file.name if audio_file else None)
        return metadata["source_id"], f"Saved source to Harry's project library: {metadata['source_id']}."
    except Exception as error:
        return "", f"⚠️ {error}"


def harry_extract_text(text_file):
    try:
        return harry.extract_text_document(text_file.name if text_file else None), "Document text loaded. Review or edit it before asking Harry."
    except harry.HarryError as error:
        return gr.update(), f"⚠️ {error}"


def harry_transcribe(audio_file):
    try:
        return harry.transcribe_audio(audio_file.name if audio_file else None), "Transcription complete. Review it, then ask Harry for recommendations."
    except harry.HarryError as error:
        return gr.update(), f"⚠️ {error}"


def harry_analyze(provider, title, source_text, source_id, audio_file):
    try:
        plan = harry.analyze(provider, title, source_text, source_id, audio_file.name if audio_file else None)
        return plan, plan["summary"], "\n".join(f"• {question}" for question in plan["questions"]) or "Harry has no essential questions.", harry.plan_to_rows(plan), plan.get("source_path", ""), f"Harry recommended {len(plan['items'])} editable draft artifacts."
    except harry.HarryError as error:
        return {}, "", "", [], f"⚠️ {error}"


def harry_apply_drafts(rows, plan):
    approved = harry.rows_to_plan(rows, plan)
    harry.save_presets(approved)
    return (gr.update(choices=harry.preset_choices(), value=None),
            gr.update(choices=harry.checklist_choices(), value=[]),
            f"Applied {len(approved['items'])} approved drafts to Build. They are presets only; nothing is locked.")


def apply_build_preset(index):
    if index is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    items = harry.load_presets()
    try:
        item = items[int(index)]
    except (IndexError, ValueError, TypeError):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    return (gr.update(value=item.get("kind", "CHAR")), item.get("suggested_id", ""),
            item.get("description", ""), item.get("positive_prompt", ""), item.get("negative_prompt", ""),
            item.get("beat", ""), item.get("reused_from", ""))


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


def lock_click(entry_id, entry_type, build_stage, panel_key, beat, description, reused_from, prompt_positive,
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
            reused_from=reused_from.strip(),
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
            reused_from=reused_from.strip(),
            image_path=winner_path,
            note=note.strip(),
        )
        return (f"🔒 Locked the concept design for '{entry_id}'. Switch to **Sheet panel** below "
                "to start building its panels — each one will start from this design.")

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

BUILD_CONTEXT = {
    "CHAR": ("🦊", "Character", "Build a reusable character design: lock a concept first, then make its 18 consistent pose, expression, costume, and palette panels."),
    "MASTER": ("🏞️", "Backdrop", "Build a recurring location: lock its concept, then define its establishing, day, night, weather, and detail views."),
    "PROP": ("🧰", "Prop", "Build a recurring object: lock its concept, then define its turnaround, surface, detail, scale, and material panels."),
    "SHOT": ("🎬", "Shot", "Build one scene keyframe. Generate options, select the strongest image, and lock it with its prompt and seed."),
}

BUILD_TYPE_CHOICES = [("Character", "CHAR"), ("Backdrop", "MASTER"), ("Prop", "PROP"), ("Shot", "SHOT")]
GENERIC_IDENTITY = {
    "MASTER": ("Backdrop name", "e.g. MASTER:farm_field", "Use a stable backdrop ID for this recurring place."),
    "PROP": ("Prop name", "e.g. PROP:crate", "Use a stable prop ID for this recurring object."),
    "SHOT": ("Shot number", "e.g. 1.1", "Use the storyboard shot number for this one-off keyframe."),
}


def build_context(entry_type):
    icon, title, description = BUILD_CONTEXT.get(entry_type, BUILD_CONTEXT["CHAR"])
    return (
        f'<div class="build-context build-context-{entry_type.lower()}">'
        f'<div class="build-context-icon" aria-hidden="true">{icon}</div>'
        f'<div><div class="build-context-kicker">YOU ARE BUILDING</div>'
        f'<div class="build-context-title">{title}</div>'
        f'<div class="build-context-copy">{description}</div></div></div>'
    )


def on_entry_type_change(entry_type):
    """Set the active workflow, identity form, and banner for the selected build type."""
    generic_update = gr.update()
    if entry_type != "CHAR":
        label, placeholder, info = GENERIC_IDENTITY[entry_type]
        generic_update = gr.update(label=label, placeholder=placeholder, info=info, value="")

    if entry_type in SHEET_TYPES:
        template = composer.get_template(entry_type)
        choices = [panel["key"] for panel in template]
        return (
            gr.update(visible=(entry_type == "CHAR")),
            gr.update(visible=(entry_type != "CHAR")),
            gr.update(choices=choices, value=choices[0] if choices else None),
            gr.update(visible=True),
            gr.update(value=CONCEPT_STAGE),
            build_context(entry_type),
            generic_update,
        )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(choices=[], value=None),
        gr.update(visible=False),
        gr.update(),
        build_context(entry_type),
        generic_update,
    )

def character_choices():
    return [(f"{row.get('display_name') or row['trigger_id']} — {row['trigger_id']}", row["trigger_id"])
            for row in store.list_characters(CFG.storyboard_path)]


def character_trigger(value):
    slug = "_".join((value or "").strip().lower().replace("-", " ").split())
    return f"CHAR:{slug}" if slug else "CHAR:your_character_name"


def character_trigger_preview(value):
    return f'<div class="trigger-preview">Canonical trigger: <code>{character_trigger(value)}</code></div>'


def select_character_ui(trigger_id):
    character = store.get_character(CFG.storyboard_path, trigger_id)
    if not character:
        return "", "", "", "No character selected."
    return (trigger_id, character.get("display_name") or "", character.get("working_note") or "",
            f"Active trigger: `{trigger_id}`. This is the identity used for future LoRA training and prompting.")


def create_character_ui(trigger_slug, display_name, working_note):
    try:
        trigger_id = character_trigger(trigger_slug)
        if trigger_id == "CHAR:your_character_name":
            raise ValueError("Enter a character name to create its trigger.")
        store.create_character(CFG.storyboard_path, trigger_id, display_name, working_note)
        return (gr.update(choices=character_choices(), value=trigger_id), trigger_id, display_name.strip(),
                working_note.strip(), "", "", "", f"Created and selected `{trigger_id}`.")
    except ValueError as error:
        return (gr.update(), "", "", "", gr.update(), gr.update(), gr.update(), f"⚠️ {error}")


def save_character_ui(trigger_id, display_name, working_note):
    try:
        store.update_character(CFG.storyboard_path, trigger_id, display_name, working_note)
        return gr.update(choices=character_choices(), value=trigger_id), "Character details saved."
    except ValueError as error:
        return gr.update(), f"⚠️ {error}"


def rename_character_ui(trigger_id, new_trigger_slug):
    try:
        new_trigger_id = character_trigger(new_trigger_slug)
        if new_trigger_id == "CHAR:your_character_name":
            raise ValueError("Enter a character name for the new trigger.")
        store.rename_character(CFG.storyboard_path, trigger_id, new_trigger_id)
        character = store.get_character(CFG.storyboard_path, new_trigger_id)
        return (gr.update(choices=character_choices(), value=new_trigger_id), new_trigger_id,
                character.get("display_name") or "", character.get("working_note") or "", "",
                f"Renamed character trigger to `{new_trigger_id}`.")
    except ValueError as error:
        return gr.update(), trigger_id, gr.update(), gr.update(), gr.update(), f"⚠️ {error}"


def delete_character_ui(trigger_id, confirmed):
    if not confirmed:
        return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Tick the confirmation box before deleting."
    try:
        store.delete_character(CFG.storyboard_path, trigger_id)
        return gr.update(choices=character_choices(), value=None), "", "", "", "Character records deleted. Image files were kept."
    except ValueError as error:
        return gr.update(), gr.update(), gr.update(), gr.update(), f"⚠️ {error}"


def draft_prompt_brief(target_model, entry_id, display_name, working_note, positive_prompt, negative_prompt):
    identity = display_name.strip() or entry_id.strip() or "the character"
    trigger = entry_id.strip() or "CHAR:your_trigger"
    note = working_note.strip() or "No additional continuity note supplied."
    positive = positive_prompt.strip() or "Describe the character, composition, wardrobe, pose, style, and lighting."
    negative = negative_prompt.strip() or "List anatomy, style, rendering, or continuity failures to avoid."
    return f'''TARGET: {target_model}

Create a production-ready visual prompt for {identity}. Preserve the canonical trigger exactly: {trigger}.

Continuity note: {note}

Positive prompt draft:
{positive}

Negative prompt draft:
{negative}

Return:
1. A refined positive prompt, with {trigger} placed naturally near the beginning.
2. A concise negative prompt.
3. A short continuity checklist for the next panel or shot.
Do not invent a new character identity, wardrobe, or art direction unless requested.'''


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
        [r.get("entry_id"), r.get("entry_type"), r.get("beat"), r.get("reused_from"),
         r.get("description"), r.get("model"), r.get("seed"), r.get("locked"),
         r.get("template"), r.get("composite_image_path") or r.get("image_path")]
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


def beats_table_refresh():
    rows = store.list_beats(CFG.storyboard_path)
    return [[r.get("order"), r.get("beat"), r.get("start_s"), r.get("end_s"),
             r.get("duration_s"), r.get("source"), r.get("notes")] for r in rows]


def import_beats_ui(beats_file):
    if beats_file is None:
        return "⚠️ Choose a beats JSON file first.", beats_table_refresh()
    try:
        with open(beats_file.name, "r", encoding="utf-8") as f:
            beats = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return f"⚠️ Couldn't read that file: {e}", beats_table_refresh()
    source = beats[0].get("source", "imported") if beats and isinstance(beats[0], dict) else "imported"
    store.set_beats(CFG.storyboard_path, beats, source)
    total = sum(b.get("duration_s", 0) for b in beats)
    return f"✅ Imported {len(beats)} beats, total {total:.1f}s.", beats_table_refresh()


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
    primary_hue="orange",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="lg",
).set(
    body_background_fill="#DCE2DF",
    body_text_color="#172534",
    block_background_fill="#F5F1E8",
    block_border_color="#B8C1BE",
    block_label_text_color="#172534",
    input_background_fill="#F5F1E8",
    input_border_color="#61717B",
    input_border_color_focus="#E99322",
    button_primary_background_fill="#E8AE72",
    button_primary_background_fill_hover="#9A5735",
    button_primary_text_color="#172534",
    button_primary_text_color_hover="#F5F1E8",
    button_secondary_background_fill="#F5F1E8",
    button_secondary_background_fill_hover="#E9D2AB",
    button_secondary_border_color="#61717B",
    button_secondary_text_color="#172534",
    block_title_text_weight="600",
)

CUSTOM_CSS = """
:root {
    --horizon-navy: #172534;
    --storm-slate: #273746;
    --weathered-blue-gray: #5C6C77;
    --sea-mist: #DCE2DF;
    --cloud-linen: #F5F1E8;
    --sunlit-sand: #E9D2AB;
    --signal-amber: #E8AE72;
    --horizon-orange: #9A5735;
    --sunbeam-gold: #F6BE45;
    --ember: #B8462C;
    --sea-glass: #3E7B68;
    --pale-gold: #FFE09A;
}

body, .gradio-container {
    background: var(--sea-mist) !important;
    color: var(--horizon-navy);
}
.gradio-container { max-width: 1440px !important; }
.tabs {
    background: var(--storm-slate);
    border-radius: 16px;
    padding: 6px;
    margin-bottom: 18px;
}
.tabs button {
    color: var(--cloud-linen) !important;
    border-radius: 10px !important;
}
.tabs button.selected {
    background: var(--cloud-linen) !important;
    color: var(--horizon-navy) !important;
    box-shadow: inset 0 -4px 0 var(--sunbeam-gold);
}
.step-card {
    background: var(--cloud-linen);
    border: 1px solid #B8C1BE;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-left: 5px solid var(--weathered-blue-gray);
    box-shadow: 0 4px 12px rgb(23 37 52 / 8%);
}
.step-1 { border-left-color: var(--weathered-blue-gray); }
.step-2 { border-left-color: #C98D68; }
.step-3 { border-left-color: var(--signal-amber); }
.step-4 { border-left-color: var(--sunbeam-gold); }
.step-5 { border-left-color: var(--sea-glass); }
.step-6 { border-left-color: var(--horizon-orange); }
.step-card, .step-card .block, .step-card .form, .step-card .wrap,
.step-card .gr-box, .step-card .gr-group, .step-card fieldset {
    background: var(--cloud-linen) !important;
    color: var(--horizon-navy) !important;
}
.step-card h1, .step-card h2, .step-card h3, .step-card h4,
.step-card p, .step-card span, .step-card legend {
    color: var(--horizon-navy) !important;
}
.step-card h4 { margin-top: 0 !important; }
.step-card label {
    color: var(--horizon-navy) !important;
    font-weight: 600;
}
.step-card input, .step-card textarea, .step-card select {
    background: #5A6C80 !important;
    color: #FFFFFF !important;
}
.step-card input::placeholder, .step-card textarea::placeholder {
    color: #F5F1E8 !important;
}
.help-btn { max-width: 40px !important; }
.build-context {
    display: flex;
    gap: 14px;
    align-items: center;
    background: var(--cloud-linen);
    border: 1px solid var(--weathered-blue-gray);
    border-left: 7px solid var(--signal-amber);
    border-radius: 14px;
    box-shadow: 0 4px 12px rgb(23 37 52 / 10%);
    color: var(--horizon-navy);
    margin: 8px 0 18px;
    padding: 14px 18px;
}
.build-context-icon {
    align-items: center;
    background: var(--pale-gold);
    border-radius: 50%;
    display: flex;
    flex: 0 0 52px;
    font-size: 28px;
    height: 52px;
    justify-content: center;
    width: 52px;
}
.build-context-kicker { color: var(--weathered-blue-gray); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; }
.build-context-title { color: var(--horizon-navy); font-size: 1.3rem; font-weight: 700; line-height: 1.25; }
.build-context-copy { color: var(--horizon-navy); line-height: 1.4; margin-top: 2px; }
.build-context-master { border-left-color: var(--sea-glass); }
.build-context-prop { border-left-color: #B9824A; }
.build-context-shot { border-left-color: var(--horizon-orange); }

.stage-guide {
    background: #FFF9EA;
    border: 1px solid var(--weathered-blue-gray);
    border-left: 6px solid var(--signal-amber);
    border-radius: 10px;
    color: var(--horizon-navy);
    line-height: 1.5;
    margin-bottom: 12px;
    padding: 12px 14px;
}
.stage-guide strong { color: var(--horizon-navy); }
.character-manager {
    background: #FFF9EA !important;
    border: 1px solid var(--weathered-blue-gray) !important;
    border-radius: 10px;
    padding: 14px 16px !important;
}
.character-manager .block, .character-manager .form, .character-manager .wrap,
.character-manager .gr-box, .character-manager .gr-group, .character-manager fieldset {
    background: var(--cloud-linen) !important;
    border-color: #B8C1BE !important;
    box-shadow: none !important;
}
.character-manager p, .character-manager strong, .character-manager span,
.character-manager label, .character-manager legend {
    color: var(--horizon-navy) !important;
}
.character-manager > .prose { margin: 0 0 10px !important; }
.character-manager label {
    background: transparent !important;
    border-radius: 0 !important;
    font-weight: 600;
    padding: 0 !important;
}
.character-manager input, .character-manager textarea, .character-manager select {
    background: #5A6C80 !important;
    color: #FFFFFF !important;
}
.character-manager input::placeholder, .character-manager textarea::placeholder {
    color: #F5F1E8 !important;
}
.trigger-preview {
    background: #E9D2AB;
    border-left: 4px solid var(--signal-amber);
    border-radius: 6px;
    color: var(--horizon-navy);
    font-weight: 600;
    margin: 4px 0 10px;
    padding: 8px 10px;
}
.trigger-preview code {
    background: var(--storm-slate);
    border-radius: 4px;
    color: #FFFFFF;
    padding: 2px 5px;
}
.harry-intake {
    background: #FFF9EA;
    border: 1px solid var(--weathered-blue-gray);
    border-left: 6px solid var(--signal-amber);
    border-radius: 10px;
    line-height: 1.5;
    margin: 8px 0 14px;
    padding: 12px 14px;
}
.harry-intake, .harry-intake p, .harry-intake span, .harry-intake strong {
    color: var(--horizon-navy) !important;
    opacity: 1 !important;
}
.harry-source-option {
    background: var(--cloud-linen) !important;
    border: 1px solid #B8C1BE !important;
    border-radius: 10px;
    box-sizing: border-box;
    min-height: 470px;
    padding: 12px !important;
}
.harry-source-option p, .harry-source-option strong {
    color: var(--horizon-navy) !important;
}
.help-panel {
    background: #FFF9EA;
    border: 1px dashed var(--signal-amber);
    border-radius: 10px;
    padding: 8px 14px !important;
    margin-top: -6px;
    margin-bottom: 10px;
    font-size: 0.92em;
}
#welcome-hero {
    background: linear-gradient(135deg, var(--storm-slate) 0%, var(--horizon-navy) 52%, #6C543F 100%);
    color: var(--cloud-linen);
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 8px 20px rgb(23 37 52 / 18%);
}
#welcome-hero h1, #welcome-hero h2, #welcome-hero h3,
#welcome-hero p, #welcome-hero li, #welcome-hero strong { color: var(--cloud-linen) !important; }
button.primary {
    background: var(--signal-amber) !important;
    color: var(--horizon-navy) !important;
    border-color: var(--signal-amber) !important;
}
button.primary:hover {
    background: var(--horizon-orange) !important;
    color: var(--cloud-linen) !important;
    border-color: var(--horizon-orange) !important;
}
input, textarea, select,
input[type="text"], input[type="number"], textarea {
    background: #5A6C80 !important;
    color: #FFFFFF !important;
    border-color: #9CACB8 !important;
}
input::placeholder, textarea::placeholder {
    color: #F5F1E8 !important;
    opacity: 1 !important;
}
.block, .form, .wrap, .gr-box, .gr-group {
    border-color: #9CACB8;
}
input:focus, textarea:focus, select:focus, button:focus-visible {
    outline: 3px solid var(--pale-gold) !important;
    outline-offset: 2px;
}
"""

# ---------------------------------------------------------------------------
# Welcome tab content
# ---------------------------------------------------------------------------

WELCOME_MARKDOWN = """
# Welcome — from story to production plan

ComfyUI Director Harness turns a script, story, poem, or narration into a
clear production workspace. It keeps the creative decisions that matter:
what must be built, how it should look, which prompt and seed produced it,
and what needs to remain consistent later.

**Start with Harry the Advisor** when you have source material. Harry reads
your text or local transcription, recommends characters, backdrops, props,
and draft shots, then gives you an editable approval list. Approved items
become **Build presets** only — nothing is generated or locked automatically.

In **Build**, first choose whether you are making a **Character**, **Backdrop**,
**Prop**, or **Shot**. A Character has a permanent `CHAR:name` trigger for
future prompting and LoRA work. Reusable assets begin with a concept, then
continue into consistent reference-sheet panels. A Shot is a direct
prompt → generate → choose → lock loop.

### The tabs, in the order you will usually use them

1. **Setup** — connect ComfyUI and choose Harry's provider. Claude is the
   default; Azure OpenAI can reuse your local OpenScout configuration.
2. **Harry the Advisor** — attach a text document, paste text, or attach audio
   and transcribe it locally. Review, edit, and approve Harry's recommendations.
3. **Build** — choose the artifact type, load an approved draft if useful,
   generate options, and lock only the decisions you want to keep.
4. **Registry** — inspect everything that is permanently locked, including
   reference sheets and narration beat timing.

### The production flow
"""

FLOW_SVG = """
<svg viewBox="0 0 1120 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:1040px;font-family:sans-serif;">
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#5C6C77"/></marker></defs>
  <g font-size="14" text-anchor="middle" fill="#172534">
    <rect x="10" y="48" width="200" height="78" rx="10" fill="#F5F1E8" stroke="#5C6C77" stroke-width="1.5"/>
    <text x="110" y="78" font-weight="bold">1. Bring your source</text><text x="110" y="101" font-size="12">attach text, paste text,</text><text x="110" y="116" font-size="12">or transcribe audio</text>
    <rect x="235" y="48" width="200" height="78" rx="10" fill="#FFF9EA" stroke="#E8AE72" stroke-width="1.5"/>
    <text x="335" y="78" font-weight="bold">2. Ask Harry</text><text x="335" y="101" font-size="12">review and refine his</text><text x="335" y="116" font-size="12">prep recommendations</text>
    <rect x="460" y="48" width="200" height="78" rx="10" fill="#F5F1E8" stroke="#C98D68" stroke-width="1.5"/>
    <text x="560" y="78" font-weight="bold">3. Approve drafts</text><text x="560" y="101" font-size="12">editable Build presets;</text><text x="560" y="116" font-size="12">nothing is locked yet</text>
    <rect x="685" y="48" width="200" height="78" rx="10" fill="#FFF4D6" stroke="#E8AE72" stroke-width="1.5"/>
    <text x="785" y="78" font-weight="bold">4. Build it</text><text x="785" y="101" font-size="12">choose type, prompt,</text><text x="785" y="116" font-size="12">generate and select</text>
    <rect x="910" y="48" width="200" height="78" rx="10" fill="#F2F7F2" stroke="#3E7B68" stroke-width="1.5"/>
    <text x="1010" y="78" font-weight="bold">5. Lock it</text><text x="1010" y="101" font-size="12">record the winning asset,</text><text x="1010" y="116" font-size="12">prompt, seed, and note</text>
  </g>
  <line x1="210" y1="87" x2="233" y2="87" stroke="#5C6C77" stroke-width="2" marker-end="url(#arrow)"/><line x1="435" y1="87" x2="458" y2="87" stroke="#5C6C77" stroke-width="2" marker-end="url(#arrow)"/><line x1="660" y1="87" x2="683" y2="87" stroke="#5C6C77" stroke-width="2" marker-end="url(#arrow)"/><line x1="885" y1="87" x2="908" y2="87" stroke="#5C6C77" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="560" y="160" font-size="12" text-anchor="middle" fill="#F5F1E8">Harry's recommendations are editable drafts. Only Build's Lock action makes a permanent registry record.</text>
</svg>
"""

WELCOME_MARKDOWN_2 = """
Use the **Harry prep checklist** in Build to track recommended artifacts as you
complete them. The Registry remains the project's permanent source of truth;
drafts are planning aids until you lock a result.

**Ready?** Start in **Harry the Advisor** if you have a story or narration.
Otherwise, open **Build** and create the first artifact directly.
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
            gr.Markdown("#### Harry the Advisor provider")
            harry_provider = gr.Dropdown(label="Advisor model provider", choices=["Claude", "Azure OpenAI", "OpenAI", "Grok", "Ollama"], value=CFG.harry_provider,
                                         info="Claude is the default. Azure OpenAI reuses the local OpenScout Azure configuration and its environment key.")
            harry_provider_note = gr.Markdown(harry.provider_status(CFG.harry_provider))
            harry_provider.change(harry_provider_status, inputs=harry_provider, outputs=harry_provider_note)

        setup_status = gr.Markdown("")
        gr.Button("Save Setup", variant="primary").click(
            setup_save,
            inputs=[comfyui_url, workflow_file, pos_node, pos_input, neg_node, neg_input,
                    seed_node, seed_input, storyboard_path, mock_mode, harry_provider],
            outputs=setup_status,
        )

    with gr.Tab("🛠️ Build"):
        gr.Markdown(
            "Work through the steps top to bottom. Nothing is saved "
            "permanently until you hit **Lock**."
        )
        build_context_banner = gr.HTML(build_context("CHAR"))
        with gr.Group(elem_classes=["step-card", "step-1"]):
            build_preset = gr.Dropdown(label="Start from a Harry-approved draft (optional)", choices=harry.preset_choices(), value=None,
                                       info="This fills Build fields from an approved draft. It does not create or lock an asset.")
            build_checklist = gr.CheckboxGroup(label="Harry prep checklist", choices=harry.checklist_choices(), value=[],
                                                info="Tick artifacts as you complete them. This is a working checklist; locking remains the permanent record.")

        with gr.Group(elem_classes=["step-card", "step-1"]):
            gr.Markdown("#### Step 1 — Choose what you are building")
            entry_type = gr.Dropdown(label="What are you building?", choices=BUILD_TYPE_CHOICES, value="CHAR",
                                     info="Choose this first. The identity, references, and workflow below change for Character, Backdrop, Prop, or Shot.")
            gr.Markdown("#### Step 2 — Name it and add references")
            with gr.Group(elem_classes=["character-manager"]) as character_manager:
                gr.Markdown("**Create a Character**")
                gr.Markdown("The character name below automatically becomes its permanent LoRA and prompt trigger. You do not need to type `CHAR:`.")
                with gr.Row():
                    new_trigger = gr.Textbox(label="Character name for the trigger", placeholder="e.g. wilbur_the_pig")
                    new_display_name = gr.Textbox(label="Display name", placeholder="e.g. Wilbur the Pig")
                trigger_preview = gr.HTML(character_trigger_preview(""))
                new_working_note = gr.Textbox(label="Working continuity note", lines=2,
                                               placeholder="e.g. practical Victorian country gentleman; tweed cap, waistcoat, calm and capable")
                create_character_btn = gr.Button("Create and lock character", variant="primary")
                character_status = gr.Markdown("")
                gr.Markdown("---\n**Choose or manage an existing Character**")
                character_select = gr.Dropdown(label="Existing Character", choices=character_choices(), value=None,
                                               info="Select a character after it has been created.")
                with gr.Row():
                    character_display_name = gr.Textbox(label="Selected display name")
                    character_working_note = gr.Textbox(label="Selected continuity note", lines=2)
                with gr.Row():
                    save_character_btn = gr.Button("Save character details")
                    rename_trigger = gr.Textbox(label="New character name for the trigger", placeholder="e.g. wilbur_the_pig")
                    rename_character_btn = gr.Button("Rename character trigger")
                delete_confirmation = gr.Checkbox(label="I understand deletion removes this Character's registry records, concepts, panels, and references. Image files are kept.")
                delete_character_btn = gr.Button("Delete Character records", variant="stop")
            entry_id = gr.Textbox(visible=False)
            with gr.Group(visible=False) as generic_identity_group:
                generic_entry_id = gr.Textbox(label="Backdrop name", placeholder="e.g. MASTER:farm_field",
                                               info="Use a stable backdrop ID for this recurring place.")
                generic_entry_id.change(lambda value: value, inputs=generic_entry_id, outputs=entry_id)
            with gr.Row():
                beat = gr.Textbox(label="Section / beat (optional)", placeholder="e.g. The Storm, or Beat 3")
                reused_from = gr.Textbox(label="Chained from (optional)", placeholder="e.g. 1.1, CHAR:alistair_pig, or MASTER:ship_deck")
            description = gr.Textbox(label="Working description (optional)", lines=2,
                                      placeholder="For your own reference only, e.g. Alistair facing camera in his everyday tweed")
            with gr.Group(visible=True) as panel_group:
                gr.HTML('<div class="stage-guide"><strong>Choose a stage.</strong> Start with <strong>Concept (mashup)</strong>: use your references and prompts to lock the canonical design. Use <strong>Sheet panel</strong> only after that, to make consistent turnarounds, expressions, and details from the locked concept.</div>')
                with gr.Row():
                    build_stage = gr.Radio(label="Build stage", choices=[CONCEPT_STAGE, PANEL_STAGE], value=CONCEPT_STAGE, scale=2)
                    concept_thumb = gr.Image(label="Locked concept", interactive=False, scale=1, height=120)
                panel_key = gr.Dropdown(label="Sheet panel", choices=[panel["key"] for panel in composer.get_template("CHAR")], value="fullbody_front", visible=False)
                panel_status = gr.Markdown("")
                with gr.Accordion("Reference images — add a mood board", open=True):
                    gr.Markdown("Upload visual references before creating the concept. They are retained with the character record for continuity; this version does not send them into the generator automatically.")
                    with gr.Row():
                        ref_file = gr.File(label="Upload a reference image", file_types=["image"])
                        ref_note = gr.Textbox(label="What should this reference contribute?", placeholder="e.g. jacket silhouette, color palette, attitude")
                    ref_add_btn = gr.Button("Add reference")
                    ref_status = gr.Markdown("")
                    ref_gallery = gr.Gallery(label="Character reference board", columns=4, height=200)

        with gr.Group(elem_classes=["step-card", "step-2"]):
            gr.Markdown("#### Step 3 — Prompt the image or video model")
            prompt_positive = gr.Textbox(label="Positive prompt", lines=3,
                placeholder="Describe subject, wardrobe, pose, composition, style, lighting, and continuity.")
            prompt_negative = gr.Textbox(label="Negative prompt (optional)", lines=2,
                placeholder="Describe failures to avoid: anatomy errors, unwanted style, wrong wardrobe, text, etc.")
            with gr.Accordion("Draft prompt brief — copy into your preferred AI assistant", open=False):
                gr.Markdown("No API call is made. This formats your current character and prompt information into a model-aware brief for you to paste into the AI model of your choice.")
                target_model = gr.Dropdown(label="Intended generation target", choices=["ComfyUI image workflow", "Wan image-to-video", "LTX-Video", "Flux", "Generic image/video model"], value="ComfyUI image workflow")
                draft_prompt_btn = gr.Button("Draft copyable prompt brief")
                prompt_brief = gr.Textbox(label="Copy this into your preferred AI assistant", lines=14, interactive=False)

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
                inputs=[entry_id, entry_type, build_stage, panel_key, beat, description, reused_from, prompt_positive,
                        prompt_negative, winner_path, winner_seed, note],
                outputs=lock_status,
            ).then(concept_thumb_refresh, inputs=[entry_id, entry_type], outputs=concept_thumb)

        # --- CHAR / MASTER only: composite sheet rendering ---
        with gr.Group(visible=True, elem_classes=["step-card", "step-6"]) as composite_group:
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
                           outputs=[character_manager, generic_identity_group, panel_key, panel_group, composite_group, build_stage, build_context_banner, generic_entry_id])
        build_preset.change(apply_build_preset, inputs=build_preset,
                            outputs=[entry_type, entry_id, description, prompt_positive, prompt_negative, beat, reused_from])
        character_select.change(select_character_ui, inputs=character_select,
                                outputs=[entry_id, character_display_name, character_working_note, character_status]).then(
            refresh_references, inputs=entry_id, outputs=ref_gallery).then(
            concept_thumb_refresh, inputs=[entry_id, entry_type], outputs=concept_thumb)
        new_trigger.change(character_trigger_preview, inputs=new_trigger, outputs=trigger_preview)
        create_character_btn.click(create_character_ui, inputs=[new_trigger, new_display_name, new_working_note],
                                   outputs=[character_select, entry_id, character_display_name, character_working_note,
                                            new_trigger, new_display_name, new_working_note, character_status])
        save_character_btn.click(save_character_ui, inputs=[entry_id, character_display_name, character_working_note],
                                 outputs=[character_select, character_status])
        rename_character_btn.click(rename_character_ui, inputs=[entry_id, rename_trigger],
                                   outputs=[character_select, entry_id, character_display_name, character_working_note,
                                            rename_trigger, character_status])
        delete_character_btn.click(delete_character_ui, inputs=[entry_id, delete_confirmation],
                                   outputs=[character_select, entry_id, character_display_name, character_working_note,
                                            character_status])
        ref_add_btn.click(add_reference_ui, inputs=[entry_id, ref_file, ref_note], outputs=[ref_status, ref_gallery])
        draft_prompt_btn.click(draft_prompt_brief,
                               inputs=[target_model, entry_id, character_display_name, character_working_note, prompt_positive, prompt_negative],
                               outputs=prompt_brief)
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

    with gr.Tab("🧭 Harry the Advisor"):
        gr.Markdown("## Harry the Advisor\nStart with the script, story, poem, or narration. Harry proposes the production prep list; you edit and approve it before it becomes Build presets.")
        with gr.Group(elem_classes=["step-card", "step-1"]):
            harry_title = gr.Textbox(label="Project or story title", placeholder="e.g. Pig and Rooster")
            gr.HTML('<div class="harry-intake"><strong>Choose one source route:</strong> <strong>Option A — attach a text document</strong>, <strong>Option B — paste text</strong>, or <strong>Option C — attach narration audio</strong> and transcribe it locally. You may combine them when useful.</div>')
            with gr.Row():
                with gr.Group(elem_classes=["harry-source-option"]):
                    gr.Markdown("**Option A — Attach text**\n\nAttach a `.txt`, `.md`, `.docx`, or `.pdf` script, story, poem, or treatment. Load it into the text area to review before Harry sees it.")
                    harry_text_file = gr.File(label="Text document", file_types=[".txt", ".md", ".docx", ".pdf"])
                    harry_extract_btn = gr.Button("Load attached text")
                with gr.Group(elem_classes=["harry-source-option"]):
                    gr.Markdown("**Option B — Paste text**\n\nPaste a script, story, poem, narration, or treatment directly.")
                    harry_source = gr.Textbox(label="Written source text", lines=14, placeholder="Paste your script, story, poem, or narration here…")
                with gr.Group(elem_classes=["harry-source-option"]):
                    gr.Markdown("**Option C — Attach audio**\n\nAttach narration when no written text is available, then transcribe it locally. Review the transcription before asking Harry.")
                    harry_audio = gr.File(label="Narration audio file", file_types=["audio"])
                    harry_transcribe_btn = gr.Button("Transcribe audio locally")
            harry_save_btn = gr.Button("Save current source to project library")
            harry_source_id = gr.State("")
            harry_source_status = gr.Markdown("")
        with gr.Group(elem_classes=["step-card", "step-2"]):
            gr.Markdown("#### Let Harry recommend the preparation list")
            harry_provider_run = gr.Dropdown(label="Provider", choices=["Claude", "Azure OpenAI", "OpenAI", "Grok", "Ollama"], value=CFG.harry_provider)
            harry_privacy = gr.Markdown(harry.provider_status(CFG.harry_provider))
            harry_provider_run.change(harry_provider_status, inputs=harry_provider_run, outputs=harry_privacy)
            harry_run_btn = gr.Button("Ask Harry for recommendations", variant="primary")
            harry_status = gr.Markdown("")
            harry_summary = gr.Textbox(label="Harry's production reading", lines=4, interactive=False)
            harry_questions = gr.Textbox(label="Only if essential: Harry's clarification questions", lines=3, interactive=False)
        with gr.Group(elem_classes=["step-card", "step-3"]):
            gr.Markdown("#### Review, refine, and approve drafts\nUntick anything you do not want. Every field is editable. Applying creates Build presets only; it does not lock assets.")
            harry_plan = gr.State({})
            harry_table = gr.Dataframe(headers=["Use", "Type", "Name", "Suggested ID", "Description", "Continuity note", "Positive prompt", "Negative prompt", "Beat", "Chained from"],
                                      datatype=["bool", "str", "str", "str", "str", "str", "str", "str", "str", "str"], interactive=True, wrap=True)
            harry_apply_btn = gr.Button("Apply approved drafts to Build", variant="primary")
            harry_apply_status = gr.Markdown("")
        harry_extract_btn.click(harry_extract_text, inputs=harry_text_file, outputs=[harry_source, harry_source_status])
        harry_save_btn.click(harry_save_source, inputs=[harry_title, harry_source, harry_audio], outputs=[harry_source_id, harry_source_status])
        harry_transcribe_btn.click(harry_transcribe, inputs=harry_audio, outputs=[harry_source, harry_source_status])
        harry_run_btn.click(harry_analyze, inputs=[harry_provider_run, harry_title, harry_source, harry_source_id, harry_audio],
                            outputs=[harry_plan, harry_summary, harry_questions, harry_table, harry_source_id, harry_status])
        harry_apply_btn.click(harry_apply_drafts, inputs=[harry_table, harry_plan], outputs=[build_preset, build_checklist, harry_apply_status])

    with gr.Tab("📋 Registry"):
        gr.Markdown(
            "Everything you've locked so far. Locking a new version of "
            "something you've already named updates that entry and adds "
            "your new note underneath the old one — it won't duplicate. The "
            "Chained from column records continuity with an earlier locked shot or asset."
        )
        refresh_btn = gr.Button("Refresh")
        registry_table = gr.Dataframe(
            headers=["Name", "Type", "Section", "Chained from", "Description", "Model", "Seed", "Locked", "Template", "Image path"],
            interactive=False,
        )
        refresh_btn.click(registry_refresh, outputs=registry_table)
        demo.load(registry_refresh, outputs=registry_table)

        gr.Markdown("#### 🖼️ Character & backdrop sheets")
        sheets_refresh_btn = gr.Button("Refresh sheets")
        sheets_gallery = gr.Gallery(label="Rendered composite sheets", columns=3, height=300)
        sheets_refresh_btn.click(sheets_gallery_refresh, outputs=sheets_gallery)
        demo.load(sheets_gallery_refresh, outputs=sheets_gallery)

        gr.Markdown(
            "#### ⏱️ Beat timing\n"
            "Import a beats JSON (either the word-count first-pass estimate, or "
            "real word-level timing from a forced-alignment run against the "
            "narration audio) — this replaces whatever was there before, since "
            "beat timing is recomputed as a whole rather than edited beat by beat."
        )
        with gr.Row():
            beats_file = gr.File(label="Beats JSON", file_types=[".json"])
            beats_import_btn = gr.Button("Import")
        beats_status = gr.Markdown("")
        beats_table = gr.Dataframe(
            headers=["#", "Beat", "Start (s)", "End (s)", "Duration (s)", "Source", "Notes"],
            interactive=False,
        )
        beats_import_btn.click(import_beats_ui, inputs=beats_file, outputs=[beats_status, beats_table])
        demo.load(beats_table_refresh, outputs=beats_table)


if __name__ == "__main__":
    demo.launch()
