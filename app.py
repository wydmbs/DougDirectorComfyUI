"""
app.py — ComfyUI Director Harness, Module 1: Image Build UI.

One shell, tabs for capability (see SUITE.md):
  Welcome  — plain-language explanation of what this tool does.
  Setup    — point at a ComfyUI instance + a workflow exported via
             "Save (API Format)". One-time, works with any workflow.
  Build    — the day-to-day workspace. Two shapes, chosen by Type:
               SHOT   — one image, one prompt. Generate, pick, lock.
               CHARACTER / BACKDROP — a fixed-template composite sheet (front
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
import base64

import gradio as gr
from PIL import Image, ImageDraw

import config as cfgmod
import storyboard_store as store
import sheet_composer as composer
import harry_advisor as harry
import project_manager as projects
import director_engine as engine
import harry_ui
from comfy_client import ComfyClient, ComfyClientError, apply_node_overrides

# ---------------------------------------------------------------------------
# UI graphics — generated via the batch job, resized/optimized locally by
# prepare_ui_assets.py into assets/ui/. Every usage below degrades
# gracefully to the previous emoji/CSS-only look if a file isn't present,
# so a fresh clone without assets/ui/ populated still runs correctly.
# ---------------------------------------------------------------------------

UI_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ui")


def _asset_data_uri(filename: str) -> str:
    """Base64-encodes a PNG from assets/ui/ into an inline data URI. Using
    data URIs (rather than Gradio's static-file serving) sidesteps
    version-dependent static-path behavior entirely -- this works
    identically whether the app is running on Gradio 4.x or 6.x."""
    path = os.path.join(UI_ASSETS_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


UI_IMAGES = {
    "app_icon": _asset_data_uri("app_icon.png"),
    "welcome_hero": _asset_data_uri("welcome_hero_banner.png"),
    "tab_icon_harry": _asset_data_uri("tab_icon_harry.png"),
    "tab_icon_build": _asset_data_uri("tab_icon_build.png"),
    "tab_icon_registry": _asset_data_uri("tab_icon_registry.png"),
    "reward_seal": _asset_data_uri("reward_seal_print_the_take.png"),
    "wrap_banner": _asset_data_uri("celebration_wrap_banner.png"),
    "registry_empty_illustration": _asset_data_uri("registry_empty_state.png"),
}

FAVICON_PATH = os.path.join(UI_ASSETS_DIR, "favicon.ico")

CFG = cfgmod.load_config()
ACTIVE_PROJECT_ID = projects.ensure_active(CFG.active_project_id)


def activate_project(project_id):
    global ACTIVE_PROJECT_ID, CFG
    ACTIVE_PROJECT_ID = projects.ensure_active(project_id)
    workspace = projects.workspace(ACTIVE_PROJECT_ID)
    CFG.active_project_id = ACTIVE_PROJECT_ID
    CFG.storyboard_path = workspace["storyboard"]
    CFG.images_dir = workspace["images"]
    harry.set_library_dir(workspace["library"])
    os.makedirs(CFG.images_dir, exist_ok=True)
    cfgmod.save_config(CFG)


activate_project(ACTIVE_PROJECT_ID)

SHEET_TYPES = ("CHARACTER", "BACKDROP", "PROP")


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


def project_progress_html():
    project = projects.get_project(ACTIVE_PROJECT_ID)
    entries = [row for row in store.list_entries(CFG.storyboard_path) if row.get("locked")]
    count = lambda kind: len([row for row in entries if row.get("entry_type") == kind])
    source_count = len(list(harry.LIBRARY_DIR.glob("*/metadata.json"))) if harry.LIBRARY_DIR.exists() else 0
    plan_count = len(list(harry.LIBRARY_DIR.glob("plan_*.json"))) if harry.LIBRARY_DIR.exists() else 0
    preset_count = len(harry.load_presets())
    beat_count = len(store.list_beats(CFG.storyboard_path))
    rows = [("Source", f"{source_count} revision(s)", source_count), ("Harry plan", f"{plan_count} revision(s)", plan_count), ("Build drafts", f"{preset_count} approved", preset_count), ("Characters", f"{count('CHARACTER')} locked", count('CHARACTER')), ("Backdrops", f"{count('BACKDROP')} locked", count('BACKDROP')), ("Props", f"{count('PROP')} locked", count('PROP')), ("Shots", f"{count('SHOT')} locked", count('SHOT')), ("Narration timing", f"{beat_count} beat(s)", beat_count)]
    items = "".join(f'<li class="{"done" if done else "next"}"><b>{"✓" if done else "○"} {label}</b><span>{detail}</span></li>' for label, detail, done in rows)
    return f'<aside class="project-progress"><div class="progress-kicker">ACTIVE PROJECT</div><h3>{project.get("title", "Untitled Project")}</h3><div class="progress-version">{project.get("version", "v1")}</div><h4>Progress</h4><ul>{items}<li><b>○ Video & QA</b><span>future module</span></li></ul></aside>'


def switch_project_ui(project_id):
    activate_project(project_id)
    project = projects.get_project(ACTIVE_PROJECT_ID)
    return gr.update(value=ACTIVE_PROJECT_ID), f"**{project['title']}** — {project['version']}", project_progress_html()


def create_project_ui(title, version, description):
    if not (title or "").strip():
        return gr.update(), "", gr.update(visible=True), "⚠️ Enter a project title."
    project = projects.create_project(title, version, description)
    activate_project(project["project_id"])
    return gr.update(choices=projects.choices(), value=project["project_id"]), f"**{project['title']}** — {project['version']}", gr.update(visible=False), f"Created and activated {project['title']}."


def harry_provider_status(provider):
    return harry.provider_status(provider)


def harry_save_source(title, source_text, audio_file):
    if not (source_text or "").strip() and not audio_file:
        return "", ("⚠️ There's no text to save yet — the source box is empty. If you attached a document, "
                    "click 'Load attached text' first (or check whether it actually extracted any text).")
    try:
        metadata = harry.save_source(title, source_text, audio_file.name if audio_file else None)
        return metadata["source_id"], f"Saved source to Harry's project library: {metadata['source_id']}."
    except Exception as error:
        return "", f"⚠️ {error}"


def harry_extract_text(text_file):
    try:
        text = harry.extract_text_document(text_file.name if text_file else None)
    except harry.HarryError as error:
        return gr.update(), f"⚠️ {error}"
    if not text.strip():
        return gr.update(), (
            "⚠️ That document loaded, but no readable text came out of it. Common causes: the content sits "
            "inside a text box or a caption/comment (still not read), the file is scanned images with no real "
            "text layer, or it's protected/encrypted. Try Option B (paste the text directly) instead, or export "
            "the document differently."
        )
    return text, "✅ Document text loaded. Review or edit it before asking Harry."


def harry_transcribe(audio_file):
    try:
        return harry.transcribe_audio(audio_file.name if audio_file else None), "Transcription complete. Review it, then ask Harry for recommendations."
    except harry.HarryError as error:
        return gr.update(), f"⚠️ {error}"


def harry_analyze(provider, title, source_text, source_id, audio_file, era=""):
    try:
        plan = harry.analyze(provider, title, source_text, source_id, audio_file.name if audio_file else None, era)
        return plan, plan["summary"], "\n".join(f"• {question}" for question in plan["questions"]) or "Harry has no essential questions.", harry.plan_to_rows(plan), plan.get("source_path", ""), f"Harry recommended {len(plan['items'])} editable draft artifacts."
    except harry.HarryError as error:
        # NOTE: this branch previously returned 5 values against the success
        # path's 6, which would raise inside Gradio's callback dispatch the
        # first time Harry actually failed. Fixed to match arity.
        return {}, "", "", [], "", f"⚠️ {error}"


def harry_analyze_ui(provider, title, source_text, source_id, audio_file, text_file, era=""):
    # Defensive auto-fill: attaching a document or audio file and going
    # straight to "Ask Harry" is the natural expectation -- don't make that
    # a dead end just because the separate Load/Transcribe button wasn't
    # clicked first. Only kicks in when the source box is actually empty.
    prefix_status = ""
    if not (source_text or "").strip():
        if text_file is not None:
            extracted, msg = harry_extract_text(text_file)
            if isinstance(extracted, str) and extracted.strip():
                source_text = extracted
                prefix_status = msg + " "
            else:
                # Extraction ran but genuinely produced nothing usable --
                # surface THAT specific reason (empty document / table-only
                # content / scanned images / etc) rather than falling
                # through to the generic "paste text or transcribe" error,
                # which is misleading once a file actually has been
                # attached and processed.
                return {}, "", "", [], source_id, source_text, reward_card(msg)
        elif audio_file is not None:
            transcribed, msg = harry_transcribe(audio_file)
            if isinstance(transcribed, str) and transcribed.strip():
                source_text = transcribed
                prefix_status = msg + " "
            else:
                return {}, "", "", [], source_id, source_text, reward_card(msg)

    plan, summary, questions, rows, new_source_id, status = harry_analyze(
        provider, title, source_text, source_id, audio_file, era)
    return plan, summary, questions, rows, new_source_id, source_text, reward_card(prefix_status + status)


def harry_apply_drafts(rows, plan):
    approved = harry.rows_to_plan(rows, plan)
    harry.save_presets(approved)
    return (gr.update(choices=harry.preset_choices(), value=None),
            gr.update(choices=harry.checklist_choices(), value=[]),
            f"Approved {len(approved['items'])} call sheet items → sent to Build. They are presets only; nothing is printed yet.")


def harry_apply_drafts_ui(rows, plan):
    preset_update, checklist_update, status = harry_apply_drafts(rows, plan)
    return preset_update, checklist_update, reward_card(status)


def harry_export_call_sheet_ui(title, rows, plan):
    """Exports the call sheet AS CURRENTLY SHOWN in the review table --
    including any manual edits or unticked rows -- not just the raw
    provider output, since the table is the actual source of truth by the
    time someone wants to export and share it."""
    return harry.export_call_sheet(title, rows, plan)


def harry_start_over():
    """Explicit exit path back to a blank Harry tab -- clears the title,
    era, source text, both file attachments, and every downstream result,
    without needing to reload the whole app."""
    return (
        "",              # harry_title
        "",              # harry_era
        "",              # harry_source
        None,            # harry_text_file
        None,            # harry_audio
        "",              # harry_source_id (State)
        {},              # harry_plan (State)
        "",              # harry_summary
        "",              # harry_questions
        [],              # harry_table
        "",              # harry_source_status
        "",              # harry_status
        "",              # harry_apply_status
    )


def apply_build_preset(index):
    if index is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    items = harry.load_presets()
    try:
        item = items[int(index)]
    except (IndexError, ValueError, TypeError):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    return (gr.update(value=item.get("kind", "CHARACTER")), item.get("suggested_id", ""),
            item.get("description", ""), item.get("positive_prompt", ""), item.get("negative_prompt", ""),
            item.get("beat", ""), item.get("reused_from", ""))


# ---------------------------------------------------------------------------
# Core generation (shared by SHOT and per-panel CHARACTER/BACKDROP generation)
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
        return [], "⚠️ Give this a name first (Step 1) — try 'CHARACTER:pig' or '1.1'."
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
# Panel-template / context-aware form logic (CHARACTER / BACKDROP only)
# ---------------------------------------------------------------------------

BUILD_CONTEXT = {
    "CHARACTER": ("🦊", "Character", "Build a reusable character design: lock a concept first, then make its 18 consistent pose, expression, costume, and palette panels."),
    "BACKDROP": ("🏞️", "Backdrop", "Build a recurring location: lock its concept, then define its establishing, day, night, weather, and detail views."),
    "PROP": ("🧰", "Prop", "Build a recurring object: lock its concept, then define its turnaround, surface, detail, scale, and material panels."),
    "SHOT": ("🎬", "Shot", "Build one scene keyframe. Generate options, select the strongest image, and lock it with its prompt and seed."),
}

BUILD_TYPE_CHOICES = [("Character", "CHARACTER"), ("Backdrop", "BACKDROP"), ("Prop", "PROP"), ("Shot", "SHOT")]
GENERIC_IDENTITY = {
    "BACKDROP": ("Backdrop name", "e.g. BACKDROP:farm_field", "Use a stable backdrop ID for this recurring place."),
    "PROP": ("Prop name", "e.g. PROP:crate", "Use a stable prop ID for this recurring object."),
    "SHOT": ("Shot number", "e.g. 1.1", "Use the storyboard shot number for this one-off keyframe."),
}


def build_context(entry_type):
    icon, title, description = BUILD_CONTEXT.get(entry_type, BUILD_CONTEXT["CHARACTER"])
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
    if entry_type != "CHARACTER":
        label, placeholder, info = GENERIC_IDENTITY[entry_type]
        generic_update = gr.update(label=label, placeholder=placeholder, info=info, value="")

    if entry_type in SHEET_TYPES:
        template = composer.get_template(entry_type)
        choices = [panel["key"] for panel in template]
        return (
            gr.update(visible=(entry_type == "CHARACTER")),
            gr.update(visible=(entry_type != "CHARACTER")),
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
    return f"CHARACTER:{slug}" if slug else "CHARACTER:your_character_name"


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
        if trigger_id == "CHARACTER:your_character_name":
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
        if new_trigger_id == "CHARACTER:your_character_name":
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
    trigger = entry_id.strip() or "CHARACTER:your_trigger"
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
        return None, "⚠️ Composite sheets are only for CHARACTER/BACKDROP entries."
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
        return "⚠️ Preview the reel first (the button above)."
    store.set_composite_image(CFG.storyboard_path, entry_id.strip(), entry_type, entry_type,
                               composite_path, (sheet_note or "").strip())
    return f"🎞️ Printed {entry_id}'s reel. Check the Registry tab."


def save_composite_ui_html(entry_id, entry_type, composite_path, sheet_note):
    return reward_card(save_composite_ui(entry_id, entry_type, composite_path, sheet_note))


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


def registry_empty_state():
    rows = store.list_entries(CFG.storyboard_path)
    if rows:
        return gr.update(visible=False)
    illustration = (f'<img src="{UI_IMAGES["registry_empty_illustration"]}" alt="" '
                    f'style="max-width:360px;display:block;margin:0 auto 12px;border-radius:10px;" />'
                    if UI_IMAGES["registry_empty_illustration"] else "")
    return gr.update(visible=True, value=(
        f'<div class="empty-state">{illustration}🎬 Nothing locked yet. Head to the <strong>Build</strong> tab '
        'to create your first character, backdrop, prop, or shot.</div>'
    ))


def sheets_gallery_refresh():
    rows = store.list_entries(CFG.storyboard_path)
    items = []
    for r in rows:
        cp = r.get("composite_image_path")
        if cp and os.path.exists(cp):
            items.append((cp, r.get("entry_id")))
    return items


def sheets_empty_state():
    if sheets_gallery_refresh():
        return gr.update(visible=False)
    return gr.update(visible=True, value=(
        '<div class="empty-state">No reference sheets rendered yet. Lock a Character, Backdrop, '
        'or Prop\'s panels in Build, then render its sheet.</div>'
    ))


def beats_table_refresh():
    rows = store.list_beats(CFG.storyboard_path)
    return [[r.get("order"), r.get("beat"), r.get("start_s"), r.get("end_s"),
             r.get("duration_s"), r.get("source"), r.get("notes")] for r in rows]


def beats_empty_state():
    if store.list_beats(CFG.storyboard_path):
        return gr.update(visible=False)
    return gr.update(visible=True, value=(
        '<div class="empty-state">No beat timing imported yet. Import a beats JSON above, '
        'or run <code>align_narration.py</code> against your narration audio.</div>'
    ))


# ---------------------------------------------------------------------------
# Reward moments — a printed take should feel like something happened.
# One-time entrance animation only, triggered by an actual lock or generate,
# never a looping/decorative animation (see UI rules: motion clarifies a
# state change, it isn't there for decoration).
# ---------------------------------------------------------------------------

def reward_card(text: str) -> str:
    if not text:
        return ""
    kind = "warn" if text.strip().startswith("⚠️") else "win"
    seal_html = ""
    if kind == "win" and UI_IMAGES["reward_seal"]:
        seal_html = f'<img src="{UI_IMAGES["reward_seal"]}" alt="" style="width:28px;height:28px;vertical-align:middle;margin-right:8px;" />'
    return f'<div class="reward-card {kind}">{seal_html}{text}</div>'


def generate_click_ui(entry_id, entry_type, build_stage, panel_key, prompt_positive, prompt_negative,
                       base_seed, n_variants):
    gallery, status = generate_click(entry_id, entry_type, build_stage, panel_key, prompt_positive,
                                      prompt_negative, base_seed, n_variants)
    return gallery, reward_card(status)


def lock_click_ui(entry_id, entry_type, build_stage, panel_key, beat, description, reused_from,
                   prompt_positive, prompt_negative, winner_path, winner_seed, note):
    status = lock_click(entry_id, entry_type, build_stage, panel_key, beat, description, reused_from,
                         prompt_positive, prompt_negative, winner_path, winner_seed, note)
    html = reward_card(status)
    if entry_type in SHEET_TYPES and build_stage == PANEL_STAGE and status.startswith("🔒"):
        template = composer.get_template(entry_type)
        filled = len(store.get_panel_images(CFG.storyboard_path, entry_id))
        if template and filled >= len(template):
            banner_img = (f'<img src="{UI_IMAGES["wrap_banner"]}" alt="" style="width:100%;border-radius:10px;display:block;margin-bottom:10px;" />'
                          if UI_IMAGES["wrap_banner"] else "")
            html += (
                f'<div class="celebration-banner">{banner_img}🎬 <strong>That\'s a wrap!</strong> Every panel for '
                f'{entry_id} is printed — render the sheet below to see the finished reel.</div>'
            )
    return html


# ---------------------------------------------------------------------------
# Harry on set — the assistant director working inside the pipeline.
#
# He is deliberately not a separate tab. The tab strip already is the
# production process, so Harry rides along it: a rail that shows where the
# production stands, and this panel inside Build where work is commissioned
# and watched.
# ---------------------------------------------------------------------------

HARRY_RUNNER = None


def _overview_safe():
    try:
        return engine.project_overview(CFG)
    except Exception:
        return {"entries": 0, "by_type": {}, "sheets": [], "sheets_complete": 0,
                "total_panels": 0, "locked_panels": 0, "panel_progress": 0.0,
                "beats": 0, "characters": 0, "mock_mode": CFG.mock_mode}


def harry_rail_html():
    """The persistent strip. Reads the same overview Harry's tools read, so the
    rail and the assistant can never disagree about the state of the build."""
    overview = _overview_safe()
    snapshot = HARRY_RUNNER.snapshot() if HARRY_RUNNER else {}
    nxt = engine.next_unfinished_panel(CFG)
    return harry_ui.rail(
        overview,
        mood=harry_ui.mood_for(snapshot),
        line=harry_ui.status_line(snapshot, overview),
        next_action=f"{nxt['entry_id']} {nxt['panel_key']}" if nxt else "",
    )


def harry_agent_start(goal, posture, max_steps, provider_name):
    global HARRY_RUNNER
    from harry_agent import AgentRunner, build_registry
    from harry_agent.providers import build_provider

    if not (goal or "").strip():
        return (harry_rail_html(), harry_ui.feed([]),
                reward_card("⚠️ Tell Harry what you want done first."), "")

    if HARRY_RUNNER and HARRY_RUNNER.running:
        return (harry_rail_html(), harry_ui.feed(HARRY_RUNNER.snapshot()["events"]),
                reward_card("⚠️ Harry is already working. Let him finish, or stop him."), "")

    try:
        provider = build_provider(provider_name or CFG.harry_provider, CFG)
    except Exception as error:
        return (harry_rail_html(), harry_ui.feed([]),
                reward_card(f"⚠️ {error}"), "")

    registry = build_registry(CFG)
    HARRY_RUNNER = AgentRunner(CFG, provider, registry, posture=posture,
                               audit_path="harry_agent_audit.sqlite3",
                               max_steps=int(max_steps or CFG.agent.max_steps))
    context = (f"Project: {ACTIVE_PROJECT_ID}. Registry: {CFG.storyboard_path}. "
               f"Mock mode is {'on' if CFG.mock_mode else 'off'}.")
    try:
        HARRY_RUNNER.start(goal, context)
    except RuntimeError as error:
        return harry_rail_html(), harry_ui.feed([]), reward_card(f"⚠️ {error}"), ""
    return (harry_rail_html(), harry_ui.feed(HARRY_RUNNER.snapshot()["events"]),
            reward_card("🎬 Action — Harry is on set."), "")


def harry_agent_poll():
    """Called on a timer while a run is live, to keep the feed moving."""
    if not HARRY_RUNNER:
        return harry_rail_html(), harry_ui.feed([]), "", gr.update(visible=False)
    snapshot = HARRY_RUNNER.snapshot()
    consent = snapshot.get("awaiting_consent")
    return (harry_rail_html(),
            harry_ui.feed(snapshot["events"]),
            harry_ui.consent_card(consent),
            gr.update(visible=bool(consent)))


def harry_agent_stop():
    if HARRY_RUNNER:
        HARRY_RUNNER.cancel()
    return harry_rail_html(), reward_card("🛑 Harry stopped. Nothing half-written was left behind.")


def harry_agent_answer(allow_session):
    from harry_agent import Decision
    if not HARRY_RUNNER:
        return harry_rail_html(), "", gr.update(visible=False)
    HARRY_RUNNER.answer(Decision.ALLOW_SESSION if allow_session else Decision.ALLOW_ONCE)
    return harry_rail_html(), "", gr.update(visible=False)


def harry_agent_deny():
    from harry_agent import Decision
    if HARRY_RUNNER:
        HARRY_RUNNER.answer(Decision.DENY)
    return harry_rail_html(), "", gr.update(visible=False)


HARRY_GOALS = [
    "Finish the next unfinished panel on any sheet",
    "Complete every panel for one character sheet",
    "Check my ComfyUI setup against the current workflow",
    "Review what's locked and tell me what's missing",
]


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
    /* Neutrals — the locked baseline, unchanged */
    --horizon-navy: #172534;
    --storm-slate: #273746;
    --weathered-blue-gray: #5C6C77;
    --sea-mist: #DCE2DF;
    --cloud-linen: #F5F1E8;
    --sunlit-sand: #E9D2AB;

    /* Accent 1 — amber family: everyday action (buttons, "you can act here") */
    --signal-amber: #E8AE72;
    --horizon-orange: #9A5735; /* amber's pressed/hover shade, not a separate accent */

    /* Accent 2 — gold family: reserved for reward/completion moments only */
    --sunbeam-gold: #F6BE45;
    --pale-gold: #FFE09A; /* gold's lighter tint, used for focus rings + reward glows */

    /* Narrow semantic exception: a single hue reserved only for "done/complete"
       checkmark-style indicators, always paired with an icon, never used as a
       general decorative accent. Not counted against the 2-accent budget. */
    --sea-glass: #3E7B68;
}

body, .gradio-container {
    background: var(--sea-mist) !important;
    color: var(--horizon-navy);
}
.gradio-container { max-width: 1440px !important; }
.tabs {
    background: var(--storm-slate);
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 18px;
}
.tabs button {
    color: var(--cloud-linen) !important;
    border-radius: 6px !important;
}
.tabs button.selected {
    background: var(--cloud-linen) !important;
    color: var(--horizon-navy) !important;
    box-shadow: inset 0 -4px 0 var(--sunbeam-gold);
}
.step-card {
    background: var(--cloud-linen);
    border: 1px solid #B8C1BE;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-left: 5px solid var(--signal-amber);
    box-shadow: 0 4px 12px rgb(23 37 52 / 8%);
}
/* Every step card uses the same amber border now — one accent for "this is
   part of the build flow," not six different hues per step. */
.step-card h4 { margin-top: 0 !important; }
.step-card, .step-card .block, .step-card .form, .step-card .wrap,
.step-card .gr-box, .step-card .gr-group, .step-card fieldset {
    background: var(--cloud-linen) !important;
    color: var(--horizon-navy) !important;
}
.step-card h1, .step-card h2, .step-card h3, .step-card h4,
.step-card p, .step-card span, .step-card legend {
    color: var(--horizon-navy) !important;
}
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
    border-radius: 12px;
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
/* All four build-type banners share the same amber accent now, distinguished
   by icon and copy rather than a different border hue each. */

.stage-guide {
    background: #FFF9EA;
    border: 1px solid var(--weathered-blue-gray);
    border-left: 6px solid var(--signal-amber);
    border-radius: 12px;
    color: var(--horizon-navy);
    line-height: 1.5;
    margin-bottom: 12px;
    padding: 12px 14px;
}
.stage-guide strong { color: var(--horizon-navy); }
.character-manager {
    background: #FFF9EA !important;
    border: 1px solid var(--weathered-blue-gray) !important;
    border-radius: 12px;
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
    border-radius: 6px;
    color: #FFFFFF;
    padding: 2px 5px;
}
.harry-intake {
    background: #FFF9EA;
    border: 1px solid var(--weathered-blue-gray);
    border-left: 6px solid var(--signal-amber);
    border-radius: 12px;
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
    border-radius: 12px;
    box-sizing: border-box;
    min-height: 470px;
    padding: 12px !important;
}
.harry-source-option p, .harry-source-option strong {
    color: var(--horizon-navy) !important;
}
.project-toolbar { background: var(--storm-slate); border-radius: 12px; margin: 8px 0 14px; padding: 8px 12px; }
.project-toolbar p { color: var(--cloud-linen) !important; margin: 7px 0 !important; }
.project-progress { background: var(--cloud-linen); border: 1px solid #B8C1BE; border-left: 6px solid var(--signal-amber); border-radius: 12px; color: var(--horizon-navy); padding: 14px; position: fixed; right: 18px; top: 112px; width: 250px; z-index: 10; box-shadow: 0 4px 12px rgb(23 37 52 / 10%); }
.project-progress h3 { color: var(--horizon-navy); font-size: 1.1rem; margin: 3px 0; }.project-progress h4 { border-top: 1px solid #B8C1BE; color: var(--horizon-navy); margin: 12px 0 6px; padding-top: 9px; }.progress-kicker, .progress-version { color: var(--weathered-blue-gray); font-size: .75rem; font-weight: 700; letter-spacing: .06em; }.project-progress ul { list-style: none; margin: 0; padding: 0; }.project-progress li { border-bottom: 1px solid #DCE2DF; display: flex; justify-content: space-between; padding: 7px 0; }.project-progress li span { color: var(--weathered-blue-gray); font-size: .8rem; }.project-progress .done b { color: var(--sea-glass); }.project-progress .next b { color: var(--horizon-navy); }
@media (max-width: 1150px) { .project-progress { position: static; width: auto; margin: 10px 0; } }

.help-panel {
    background: #FFF9EA;
    border: 1px dashed var(--signal-amber);
    border-radius: 12px;
    padding: 8px 14px !important;
    margin-top: -6px;
    margin-bottom: 10px;
    font-size: 0.92em;
}
#welcome-hero {
    background: linear-gradient(135deg, var(--storm-slate) 0%, var(--horizon-navy) 52%, #6C543F 100%);
    color: var(--cloud-linen);
    border-radius: 12px;
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

/* Empty states — Registry should never just look blank */
.empty-state {
    background: #FFF9EA;
    border: 1px dashed var(--weathered-blue-gray);
    border-radius: 12px;
    color: var(--weathered-blue-gray);
    padding: 22px;
    text-align: center;
    font-style: italic;
    margin-bottom: 12px;
}

/* Reward moments — a lock or a wrap should feel like something happened.
   One-time entrance animation only, triggered by a real user action
   (generating or locking), never a looping/decorative animation. */
.reward-card {
    border-radius: 12px; padding: 14px 18px; margin-top: 8px;
    font-weight: 500; animation: reward-pop 0.35s ease-out;
}
.reward-card.win {
    background: #FFF4D6;
    border: 1px solid var(--sunbeam-gold);
    color: var(--horizon-navy);
    box-shadow: 0 2px 10px rgb(246 190 69 / 25%);
}
.reward-card.warn {
    background: #FDEDEA;
    border: 1px solid var(--horizon-orange);
    color: var(--horizon-navy);
}
@keyframes reward-pop {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
.celebration-banner {
    margin-top: 10px; padding: 18px 22px; border-radius: 12px; text-align: center;
    font-size: 1.05em; font-weight: 600;
    background: linear-gradient(135deg, #FFF4D6, #FFE9B8);
    border: 1px solid var(--sunbeam-gold); color: var(--horizon-navy);
    box-shadow: 0 4px 16px rgb(246 190 69 / 30%);
    animation: reward-pop 0.4s ease-out;
}
"""

# ---------------------------------------------------------------------------
# Welcome tab content
# ---------------------------------------------------------------------------

WELCOME_MARKDOWN = """
# 🎬 Welcome to the studio

You're the director on this production. ComfyUI Director Harness turns a
script, story, poem, or narration into a real shoot: what needs building,
how it should look, which take you printed and why, and what has to stay
consistent from scene to scene.

**Start with Harry the Advisor** when you have source material — think of
him as your first assistant director. Harry reads your text or a local
transcription, recommends characters, backdrops, props, and draft shots,
then hands you an editable call sheet. Approved items become **Build
presets** only — nothing is generated or printed automatically. You're
still the one who says "print it."

On set, in **Build**, first choose whether you're building a **Character**,
**Backdrop**, **Prop**, or **Shot**. A Character gets a permanent
`CHARACTER:name` trigger for every future prompt and LoRA run — its screen
credit, in a sense. Reusable assets get a concept locked first, then a full
reference sheet of consistent takes. A Shot is a direct
roll camera → review dailies → print loop.

### The tabs, in the order you'll usually use them

1. **Setup** — connect ComfyUI and choose Harry's provider. Claude is the
   default; Azure OpenAI is configured here too, in its own fields.
2. **Harry the Advisor** — attach a text document, paste text, or attach audio
   and transcribe it locally. Review, edit, and approve Harry's call sheet.
3. **Build** — choose the artifact type, load an approved draft if useful,
   roll camera, and print only the takes worth keeping.
4. **Registry** — your screening room: everything permanently printed,
   including reference sheets and narration beat timing.

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
complete them. The Registry — your screening room — remains the project's
permanent source of truth; drafts are planning aids until you print a result.

**Ready?** Start in **Harry the Advisor** if you have a story or narration.
Otherwise, head to **Build** and call "action" on the first scene yourself.
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ComfyUI Director Harness", theme=THEME, css=CUSTOM_CSS) as demo:
    if UI_IMAGES["app_icon"]:
        gr.HTML(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
            f'<img src="{UI_IMAGES["app_icon"]}" alt="" style="width:40px;height:40px;border-radius:8px;" />'
            f'<h1 style="margin:0;">ComfyUI Director Harness</h1></div>'
        )
    else:
        gr.Markdown("# 🎬 ComfyUI Director Harness")
    with gr.Row(elem_classes=["project-toolbar"]):
        project_selector = gr.Dropdown(label="Active project", choices=projects.choices(), value=ACTIVE_PROJECT_ID, scale=3)
        project_label = gr.Markdown(f"**{projects.get_project(ACTIVE_PROJECT_ID)['title']}** — {projects.get_project(ACTIVE_PROJECT_ID)['version']}", scale=2)
        new_project_btn = gr.Button("New project", scale=0)
    with gr.Group(visible=False, elem_classes=["step-card"]) as new_project_group:
        new_project_title = gr.Textbox(label="Project title", placeholder="e.g. Pig and Rooster")
        with gr.Row():
            new_project_version = gr.Textbox(label="Starting version", value="v1")
            new_project_description = gr.Textbox(label="Description (optional)")
        new_project_confirm = gr.Button("Create project", variant="primary")
        new_project_status = gr.Markdown("")
    progress_panel = gr.HTML(project_progress_html())

    with gr.Tabs() as main_tabs:
        with gr.Tab("👋 Welcome", id="welcome"):
            with gr.Group(elem_id="welcome-hero"):
                if UI_IMAGES["welcome_hero"]:
                    gr.HTML(f'<img src="{UI_IMAGES["welcome_hero"]}" alt="" style="width:100%;border-radius:10px;display:block;margin-bottom:14px;" />')
                gr.Markdown(WELCOME_MARKDOWN)
                gr.HTML(FLOW_SVG)
                gr.Markdown(WELCOME_MARKDOWN_2)

        with gr.Tab("⚙️ Setup", id="setup"):
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
                                             info="Claude is the default. Azure OpenAI uses the endpoint and deployment set below.")
                harry_provider_note = gr.Markdown(harry.provider_status(CFG.harry_provider))
                harry_provider.change(harry_provider_status, inputs=harry_provider, outputs=harry_provider_note)

            setup_status = gr.Markdown("")
            setup_save_event = gr.Button("Save Setup", variant="primary").click(
                setup_save,
                inputs=[comfyui_url, workflow_file, pos_node, pos_input, neg_node, neg_input,
                        seed_node, seed_input, storyboard_path, mock_mode, harry_provider],
                outputs=setup_status,
            )

        with gr.Tab("🧭 Harry the Advisor", id="harry"):
            harry_icon = (f'<img src="{UI_IMAGES["tab_icon_harry"]}" alt="" style="width:32px;height:32px;vertical-align:middle;margin-right:10px;" />'
                         if UI_IMAGES["tab_icon_harry"] else "🎬 ")
            gr.HTML(
                f'<h2 style="display:flex;align-items:center;">{harry_icon}Harry — your assistant director</h2>'
            )
            gr.Markdown(
                "Every production needs a first AD to read the script and build the "
                "call sheet before the director steps on set. That's Harry. Bring "
                "him the script, story, poem, or narration; he proposes the "
                "production prep list; you edit and approve it before it becomes "
                "Build presets."
            )
            with gr.Group(elem_classes=["step-card", "step-1"]):
                harry_title = gr.Textbox(label="Project or story title", placeholder="e.g. Pig and Rooster")
                harry_era = gr.Textbox(
                    label="Era / setting (optional, but strongly recommended)",
                    placeholder="e.g. early 20th century maritime, pre-automobile",
                    info="Grounds every stage's inferences in the right period -- without this, "
                         "'moved from field to ship' could just as easily infer a truck as a horse and cart.",
                )
                gr.HTML('<div class="harry-intake"><strong>Choose one source route:</strong> <strong>Option A — attach a text document</strong>, <strong>Option B — paste text</strong>, or <strong>Option C — attach narration audio</strong> and transcribe it locally. You may combine them when useful.</div>')
                with gr.Row():
                    with gr.Group(elem_classes=["harry-source-option"]):
                        gr.Markdown("**Option A — Attach text**\n\nAttach a `.txt`, `.md`, `.docx`, or `.pdf` script, story, poem, or treatment. Load it into the text area to review before Harry reads it.")
                        harry_text_file = gr.File(label="Text document", file_types=[".txt", ".md", ".docx", ".pdf"])
                        harry_extract_btn = gr.Button("Load attached text")
                    with gr.Group(elem_classes=["harry-source-option"]):
                        gr.Markdown("**Option B — Paste text**\n\nPaste a script, story, poem, narration, or treatment directly.")
                        harry_source = gr.Textbox(label="Written source text", lines=14, placeholder="Paste your script, story, poem, or narration here…")
                    with gr.Group(elem_classes=["harry-source-option"]):
                        gr.Markdown("**Option C — Attach audio**\n\nAttach narration when no written text is available, then transcribe it locally. Review the transcription before Harry reads it.")
                        harry_audio = gr.File(label="Narration audio file", file_types=["audio"])
                        harry_transcribe_btn = gr.Button("Transcribe audio locally")
                harry_save_btn = gr.Button("Save current source to project library")
                harry_source_id = gr.State("")
                harry_source_status = gr.Markdown("")
                harry_reset_btn = gr.Button("↩ Start over (clear this source)")
            with gr.Group(elem_classes=["step-card", "step-2"]):
                gr.Markdown("#### Ask Harry for the call sheet")
                harry_provider_run = gr.Dropdown(label="Provider", choices=["Claude", "Azure OpenAI", "OpenAI", "Grok", "Ollama"], value=CFG.harry_provider)
                harry_privacy = gr.Markdown(harry.provider_status(CFG.harry_provider))
                harry_provider_run.change(harry_provider_status, inputs=harry_provider_run, outputs=harry_privacy)
                setup_save_event.then(lambda: gr.update(value=CFG.harry_provider), outputs=harry_provider_run)
                setup_save_event.then(harry_provider_status, inputs=harry_provider_run, outputs=harry_privacy)
                harry_run_btn = gr.Button("Ask Harry for recommendations", variant="primary")
                harry_status = gr.HTML("")
                harry_summary = gr.Textbox(label="Harry's read on the production", lines=4, interactive=False)
                harry_questions = gr.Textbox(label="Only if essential: Harry's clarification questions", lines=3, interactive=False)
            with gr.Group(elem_classes=["step-card", "step-3"]):
                gr.Markdown("#### Review, refine, and approve the call sheet\nUntick anything you do not want. Every field is editable. Approving creates Build presets only; it does not print any takes.")
                harry_plan = gr.State({})
                harry_table = gr.Dataframe(headers=["Use", "Type", "Name", "Suggested ID", "Description", "Continuity note", "Positive prompt", "Negative prompt", "Beat", "Chained from"],
                                          datatype=["bool", "str", "str", "str", "str", "str", "str", "str", "str", "str"], interactive=True, wrap=True)
                harry_apply_btn = gr.Button("Approve call sheet → send to Build", variant="primary")
                harry_apply_status = gr.HTML("")
                harry_goto_build_btn = gr.Button("🎬 Go to Build →")
                harry_export_btn = gr.DownloadButton("⬇️ Export call sheet (.xlsx)")
            # Attaching a document should load it right away, not require a
            # separate manual click before it's usable -- the button stays too,
            # for re-loading after swapping the attached file.
            harry_text_file.upload(harry_extract_text, inputs=harry_text_file, outputs=[harry_source, harry_source_status])
            harry_extract_btn.click(harry_extract_text, inputs=harry_text_file, outputs=[harry_source, harry_source_status])
            harry_save_btn.click(harry_save_source, inputs=[harry_title, harry_source, harry_audio], outputs=[harry_source_id, harry_source_status])
            harry_transcribe_btn.click(harry_transcribe, inputs=harry_audio, outputs=[harry_source, harry_source_status])
            harry_run_btn.click(harry_analyze_ui, inputs=[harry_provider_run, harry_title, harry_source, harry_source_id, harry_audio, harry_text_file, harry_era],
                                outputs=[harry_plan, harry_summary, harry_questions, harry_table, harry_source_id, harry_source, harry_status])
            harry_goto_build_btn.click(lambda: gr.Tabs(selected="build"), outputs=main_tabs)
            harry_export_btn.click(harry_export_call_sheet_ui, inputs=[harry_title, harry_table, harry_plan], outputs=harry_export_btn)
            harry_reset_btn.click(
                harry_start_over,
                outputs=[harry_title, harry_era, harry_source, harry_text_file, harry_audio, harry_source_id,
                         harry_plan, harry_summary, harry_questions, harry_table,
                         harry_source_status, harry_status, harry_apply_status],
            )

        with gr.Tab("🛠️ Build", id="build"):
            build_icon = (f'<img src="{UI_IMAGES["tab_icon_build"]}" alt="" style="width:32px;height:32px;vertical-align:middle;margin-right:10px;" />'
                         if UI_IMAGES["tab_icon_build"] else "🎥 ")
            gr.HTML(f'<h2 style="display:flex;align-items:center;">{build_icon}On set</h2>')
            gr.Markdown(
                "Every scene starts here. Work the slate top to bottom — nothing "
                "makes it into the final reel until you **print the take**."
            )
            build_rail = gr.HTML(harry_rail_html())

            with gr.Accordion("🎬 On set with Harry — let him work the slate", open=False):
                gr.Markdown(
                    "Harry can run the slate himself: generate, judge the dailies, and print "
                    "the takes that earn it. He reads the registry as he goes, so you can stop "
                    "him at any point and nothing is left half-written."
                )
                with gr.Row():
                    harry_goal = gr.Dropdown(
                        label="What should Harry do?", choices=HARRY_GOALS,
                        value=HARRY_GOALS[0], allow_custom_value=True, scale=3,
                        info="Pick one, or type your own.")
                    harry_posture = gr.Radio(
                        label="How much rope?",
                        choices=[("Check with me each time", "attended"),
                                 ("Generate freely, ask before printing", "supervised"),
                                 ("Work alone", "unattended")],
                        value="supervised", scale=2)
                with gr.Row():
                    harry_max_steps = gr.Slider(label="Step budget", minimum=5, maximum=120,
                                                value=40, step=5, scale=2,
                                                info="A hard ceiling on how long he can work.")
                    harry_agent_provider = gr.Dropdown(
                        label="Model", choices=["Claude", "Azure OpenAI", "OpenAI", "Grok", "Ollama"],
                        value=CFG.harry_provider, scale=2)
                with gr.Row():
                    harry_action_btn = gr.Button("🎬 Action!", variant="primary", scale=2)
                    harry_stop_btn = gr.Button("🛑 Cut", scale=1)
                harry_agent_status = gr.HTML("")
                harry_consent = gr.HTML("")
                with gr.Row(visible=False) as harry_consent_row:
                    harry_allow_once_btn = gr.Button("Allow once", variant="primary")
                    harry_allow_session_btn = gr.Button("Allow for this session")
                    harry_deny_btn = gr.Button("No")
                harry_feed = gr.HTML(harry_ui.feed([]))
                harry_refresh_btn = gr.Button("↻ Refresh Harry's progress", size="sm")

            build_context_banner = gr.HTML(build_context("CHARACTER"))
            with gr.Group(elem_classes=["step-card", "step-1"]):
                build_preset = gr.Dropdown(label="Start from a Harry-approved draft (optional)", choices=harry.preset_choices(), value=None,
                                           info="This fills Build fields from an approved draft. It does not create or lock an asset.")
                build_checklist = gr.CheckboxGroup(label="Harry prep checklist", choices=harry.checklist_choices(), value=[],
                                                    info="Tick artifacts as you complete them. This is a working checklist; locking remains the permanent record.")

            with gr.Group(elem_classes=["step-card", "step-1"]):
                gr.Markdown("#### Step 1 — Choose what you are building")
                entry_type = gr.Dropdown(label="What are you building?", choices=BUILD_TYPE_CHOICES, value="CHARACTER",
                                         info="Choose this first. The identity, references, and workflow below change for Character, Backdrop, Prop, or Shot.")
                gr.Markdown("#### Step 2 — Name it and add references")
                with gr.Group(elem_classes=["character-manager"]) as character_manager:
                    gr.Markdown("**Create a Character**")
                    gr.Markdown("The character name below automatically becomes its permanent LoRA and prompt trigger. You do not need to type `CHARACTER:`.")
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
                    generic_entry_id = gr.Textbox(label="Backdrop name", placeholder="e.g. BACKDROP:farm_field",
                                                   info="Use a stable backdrop ID for this recurring place.")
                    generic_entry_id.change(lambda value: value, inputs=generic_entry_id, outputs=entry_id)
                with gr.Row():
                    beat = gr.Textbox(label="Section / beat (optional)", placeholder="e.g. The Storm, or Beat 3")
                    reused_from = gr.Textbox(label="Chained from (optional)", placeholder="e.g. 1.1, CHARACTER:alistair_pig, or BACKDROP:ship_deck")
                description = gr.Textbox(label="Working description (optional)", lines=2,
                                          placeholder="For your own reference only, e.g. Alistair facing camera in his everyday tweed")
                with gr.Group(visible=True) as panel_group:
                    gr.HTML('<div class="stage-guide"><strong>Choose a stage.</strong> Start with <strong>Concept (mashup)</strong>: use your references and prompts to lock the canonical design. Use <strong>Sheet panel</strong> only after that, to make consistent turnarounds, expressions, and details from the locked concept.</div>')
                    with gr.Row():
                        build_stage = gr.Radio(label="Build stage", choices=[CONCEPT_STAGE, PANEL_STAGE], value=CONCEPT_STAGE, scale=2)
                        concept_thumb = gr.Image(label="Locked concept", interactive=False, scale=1, height=120)
                    panel_key = gr.Dropdown(label="Sheet panel", choices=[panel["key"] for panel in composer.get_template("CHARACTER")], value="fullbody_front", visible=False)
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
                gr.Markdown("#### Step 3 — Roll camera")
                with gr.Row():
                    base_seed = gr.Number(label="Seed (leave blank for random)", value=None)
                    n_variants = gr.Slider(label="How many takes", minimum=1, maximum=8, step=1, value=4)
                gen_btn = gr.Button("🎬 Action!", variant="primary")
                gallery = gr.Gallery(label="The takes — each one labeled with its seed", columns=4)
                gen_status = gr.HTML("")
                gen_btn.click(
                    generate_click_ui,
                    inputs=[entry_id, entry_type, build_stage, panel_key, prompt_positive, prompt_negative,
                            base_seed, n_variants],
                    outputs=[gallery, gen_status],
                )

            with gr.Group(elem_classes=["step-card", "step-4"]):
                gr.Markdown(
                    "#### Step 4 — Review the dailies\n"
                    "Click a take above to see it larger, then copy its file path and "
                    "seed into the two boxes below."
                )
                with gr.Row():
                    winner_path = gr.Textbox(label="Winning take's file path")
                    winner_seed = gr.Number(label="Winning take's seed")
                note = gr.Textbox(
                    label="Why this take? (saved permanently with the record)", lines=2,
                    placeholder="e.g. 'first version where the tweed cap read clearly at this angle'",
                )

            with gr.Group(elem_classes=["step-card", "step-5"]):
                gr.Markdown("#### Step 5 — Print the take")
                lock_btn = gr.Button("🎞️ Print it", variant="primary")
                lock_status = gr.HTML("")
                lock_btn.click(
                    lock_click_ui,
                    inputs=[entry_id, entry_type, build_stage, panel_key, beat, description, reused_from, prompt_positive,
                            prompt_negative, winner_path, winner_seed, note],
                    outputs=lock_status,
                ).then(concept_thumb_refresh, inputs=[entry_id, entry_type], outputs=concept_thumb)

            # --- CHARACTER / BACKDROP only: composite sheet rendering ---
            with gr.Group(visible=True, elem_classes=["step-card", "step-6"]) as composite_group:
                gr.Markdown(
                    "#### Step 6 — Assemble the reel\n"
                    "Puts every printed panel together into one composite reference sheet. "
                    "You can preview it at any point — panels you haven't printed "
                    "yet just show as a placeholder."
                )
                render_btn = gr.Button("Preview the reel")
                composite_image = gr.Image(label="Composite sheet preview", type="filepath")
                render_status = gr.Markdown("")
                render_btn.click(render_composite_preview, inputs=[entry_id, entry_type],
                                  outputs=[composite_image, render_status])

                sheet_note = gr.Textbox(label="Note for this sheet version", lines=1,
                                         placeholder="e.g. 'first full pass, 9/13 panels'")
                save_sheet_btn = gr.Button("🎞️ Print the reel", variant="primary")
                save_sheet_status = gr.HTML("")
                save_sheet_btn.click(save_composite_ui_html, inputs=[entry_id, entry_type, composite_image, sheet_note],
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

        with gr.Tab("📋 Registry", id="registry"):
            registry_icon = (f'<img src="{UI_IMAGES["tab_icon_registry"]}" alt="" style="width:32px;height:32px;vertical-align:middle;margin-right:10px;" />'
                             if UI_IMAGES["tab_icon_registry"] else "🎞️ ")
            gr.HTML(f'<h2 style="display:flex;align-items:center;">{registry_icon}The screening room</h2>')
            gr.Markdown(
                "Every take you've printed. Locking a new version of something "
                "you've already named updates that entry and adds your new note "
                "underneath the old one — it won't duplicate. The Chained from "
                "column records continuity with an earlier locked shot or asset."
            )
            registry_rail = gr.HTML(harry_rail_html())
            refresh_btn = gr.Button("Refresh")
            registry_empty = gr.HTML(visible=False)
            registry_table = gr.Dataframe(
                headers=["Name", "Type", "Section", "Chained from", "Description", "Model", "Seed", "Locked", "Template", "Image path"],
                interactive=False,
            )
            refresh_btn.click(registry_refresh, outputs=registry_table).then(registry_empty_state, outputs=registry_empty)
            demo.load(registry_refresh, outputs=registry_table)
            demo.load(registry_empty_state, outputs=registry_empty)

            gr.Markdown("#### 🖼️ Character, backdrop & prop sheets")
            sheets_refresh_btn = gr.Button("Refresh sheets")
            sheets_empty = gr.HTML(visible=False)
            sheets_gallery = gr.Gallery(label="Rendered composite sheets", columns=3, height=300)
            sheets_refresh_btn.click(sheets_gallery_refresh, outputs=sheets_gallery).then(sheets_empty_state, outputs=sheets_empty)
            demo.load(sheets_gallery_refresh, outputs=sheets_gallery)
            demo.load(sheets_empty_state, outputs=sheets_empty)

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
            beats_empty = gr.HTML(visible=False)
            beats_table = gr.Dataframe(
                headers=["#", "Beat", "Start (s)", "End (s)", "Duration (s)", "Source", "Notes"],
                interactive=False,
            )
            beats_import_btn.click(import_beats_ui, inputs=beats_file, outputs=[beats_status, beats_table]).then(
                beats_empty_state, outputs=beats_empty)
            demo.load(beats_table_refresh, outputs=beats_table)
            demo.load(beats_empty_state, outputs=beats_empty)



    # Harry's "approve" wiring needs Build's build_preset/build_checklist,
    # deferred here since Harry's tab now renders before Build's in the
    # visual order (Setup -> Harry -> Build matches the actual workflow),
    # but both components already exist by this point in the script either way.
    harry_apply_btn.click(harry_apply_drafts_ui, inputs=[harry_table, harry_plan], outputs=[build_preset, build_checklist, harry_apply_status])

    new_project_btn.click(lambda: gr.update(visible=True), outputs=new_project_group)
    new_project_confirm.click(create_project_ui, inputs=[new_project_title, new_project_version, new_project_description], outputs=[project_selector, project_label, new_project_group, new_project_status])
    project_selector.change(switch_project_ui, inputs=project_selector, outputs=[project_selector, project_label, progress_panel])


if __name__ == "__main__":
        # ----------------------------------------------------------------- Harry
    # Wired here, at the end, so every component above already exists.
    harry_action_btn.click(
        harry_agent_start,
        inputs=[harry_goal, harry_posture, harry_max_steps, harry_agent_provider],
        outputs=[build_rail, harry_feed, harry_agent_status, harry_consent],
    )
    harry_stop_btn.click(harry_agent_stop, outputs=[build_rail, harry_agent_status])
    harry_refresh_btn.click(
        harry_agent_poll,
        outputs=[build_rail, harry_feed, harry_consent, harry_consent_row],
    )
    harry_allow_once_btn.click(
        lambda: harry_agent_answer(False),
        outputs=[build_rail, harry_consent, harry_consent_row],
    ).then(harry_agent_poll, outputs=[build_rail, harry_feed, harry_consent, harry_consent_row])
    harry_allow_session_btn.click(
        lambda: harry_agent_answer(True),
        outputs=[build_rail, harry_consent, harry_consent_row],
    ).then(harry_agent_poll, outputs=[build_rail, harry_feed, harry_consent, harry_consent_row])
    harry_deny_btn.click(
        harry_agent_deny,
        outputs=[build_rail, harry_consent, harry_consent_row],
    ).then(harry_agent_poll, outputs=[build_rail, harry_feed, harry_consent, harry_consent_row])

    # A live run refreshes itself; when nothing is running this is a cheap
    # no-op read of the registry, so it doubles as keeping the rail current.
    harry_timer = getattr(gr, "Timer", None)
    if harry_timer is not None:
        _tick = harry_timer(2.0)
        _tick.tick(harry_agent_poll,
                   outputs=[build_rail, harry_feed, harry_consent, harry_consent_row])

    # Locking by hand should move the rail too -- the rail reflects the
    # production, not just Harry's own work.
    lock_btn.click(harry_rail_html, outputs=build_rail)
    demo.load(harry_rail_html, outputs=build_rail)
    demo.load(harry_rail_html, outputs=registry_rail)
    refresh_btn.click(harry_rail_html, outputs=registry_rail)

demo.launch(favicon_path=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None)
