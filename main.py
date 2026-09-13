import asyncio
import html
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import config as cfgmod
import director_engine as engine
import harry_advisor as harry
import model_pipeline
import project_manager as projects
import storyboard_store as store
import specialists
from comfy_client import ComfyClient
from video_client import readiness

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
COMFY_ROOT = Path(os.environ.get("COMFYUI_PORTABLE_ROOT", r"C:\Users\dougl\ComfyUI_windows_portable"))
app = FastAPI(title="ComfyUI Director Harness")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
logs: list[str] = []
call_sheets: dict[str, dict] = {}
last_variants: list[dict] = []


def esc(value) -> str:
    return html.escape(str(value or ""))


def cfg():
    return cfgmod.load_config()


def set_project(project_id: str):
    current = cfg()
    project_id = projects.ensure_active(project_id)
    workspace = projects.workspace(project_id)
    current.active_project_id = project_id
    current.storyboard_path = workspace["storyboard"]
    current.images_dir = workspace["images"]
    harry.set_library_dir(workspace["library"])
    os.makedirs(current.images_dir, exist_ok=True)
    cfgmod.save_config(current)
    return current


def current_cfg():
    current = cfg()
    return set_project(current.active_project_id)



def active_call_sheet_path(project_id: str):
    return projects.workspace(project_id)["library"] / "active_call_sheet.json"


def save_active_call_sheet(project_id: str, record: dict):
    path = active_call_sheet_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_active_call_sheet(project_id: str):
    path = active_call_sheet_path(project_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_files(project_id: str):
    library = projects.workspace(project_id)["library"]
    values = []
    for path in sorted(library.glob("plan_*.json"), key=lambda value: value.stat().st_mtime, reverse=True):
        if "_pre_" in path.name:
            continue
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values.append((path.name, plan.get("title") or path.stem, plan.get("created_at") or ""))
    return values


def next_run_version(record: dict):
    import re
    current = str((record or {}).get("run_version") or "v0")
    match = re.search(r"v(\d+)$", current)
    return f"v{int(match.group(1)) + 1}" if match else "v1"


def run_control_fragment(record: dict):
    current = current_cfg()
    active_file = ""
    title = record.get("plan", {}).get("title") or "Untitled run"
    version = record.get("run_version") or "Saved run"
    options = []
    for filename, run_title, created_at in run_files(current.active_project_id):
        selected = " selected" if filename == active_file else ""
        options.append(f'<option value="{esc(filename)}"{selected}>{esc(run_title)} · {esc(created_at)}</option>')
    picker = "".join(options) or '<option>No archived runs yet</option>'
    return '<section class="run-control"><div><span class="panel-kicker">ACTIVE RUN</span><h2>' + esc(title) + '</h2><small>Project: ' + esc(projects.get_project(current.active_project_id).get("title")) + ' · ' + esc(version) + ' · <b>Saved</b></small></div><div class="run-actions"><form hx-post="/harry/run/select" hx-target="#call-sheet"><select name="run_file">' + picker + '</select><button>Open saved run</button></form><form hx-post="/harry/run/save-confirmed" hx-target="#call-sheet"><button>Save confirmed version</button></form></div></section>'


def clare_handoff_path(project_id: str):
    return projects.workspace(project_id)["library"] / "clare_handoff.json"


def save_clare_handoff(project_id: str, record: dict):
    characters = [item for item in record.get("plan", {}).get("items", []) if item.get("kind") == "CHARACTER"]
    payload = {"plan_id": record.get("plan_id", ""), "run_version": record.get("run_version", ""), "title": record.get("plan", {}).get("title", ""), "characters": characters}
    clare_handoff_path(project_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_clare_handoff(project_id: str):
    path = clare_handoff_path(project_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clare_development_path(project_id: str):
    return projects.workspace(project_id)["library"] / "clare_visual_development.json"


def load_clare_development(project_id: str):
    path = clare_development_path(project_id)
    if not path.exists():
        return {"look_status": "draft", "look_version": "v0", "selected_character": "", "look": {}, "character_notes": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"look_status": "draft", "look_version": "v0", "selected_character": "", "look": {}, "character_notes": {}}


def save_clare_development(project_id: str, state: dict):
    clare_development_path(project_id).write_text(json.dumps(state, indent=2), encoding="utf-8")


def clare_reference_dir(project_id: str):
    path = projects.workspace(project_id)["library"] / "clare_references"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clare_scene_options(handoff):
    record = load_active_call_sheet(current_cfg().active_project_id) or {}
    shots = [item for item in record.get("plan", {}).get("items", []) if item.get("kind") == "SHOT"]
    return {item.get("suggested_id"): item for item in shots}


def clare_workspace_fragment(handoff, dossier=None):
    current = current_cfg()
    state = load_clare_development(current.active_project_id)
    look = state.get("look", {})
    status = state.get("look_status", "draft")
    scenes = clare_scene_options(handoff)
    selected_scenes = state.get("test_scene_ids") or ["2.1c", "4.4"]
    selected_scenes = [scene for scene in selected_scenes if scene in scenes]
    if len(selected_scenes) < 2:
        selected_scenes = list(scenes)[:2]
    references = state.get("references", [])
    scene_options = "".join('<option value="' + esc(scene_id) + '"' + (' selected' if scene_id in selected_scenes else '') + '>' + esc(scene_id + " - " + scenes[scene_id].get("name", "")) + '</option>' for scene_id in scenes)
    ref_rows = "".join('<li><b>' + esc(reference.get("name")) + '</b><span>' + esc(reference.get("role")) + '</span></li>' for reference in references) or '<li>No references yet. You can start with words alone.</li>'
    ready = status == "test-ready"
    locked = status == "locked"
    status_pill = pill("Look Lock " + state.get("look_version", "v1"), "done") if locked else pill("Test board ready" if ready else "Start here", "active")
    stage_one = '<section class="clare-stage"><div class="clare-stage-head"><div><span class="panel-kicker">STAGE 1: DESIGN THE FILM LOOK</span><h2>Tell Clare what you want the movie to feel like.</h2><p>Use ordinary language and images you like. Clare turns that into an original visual recipe; you judge paired real-scene tests.</p></div>' + status_pill + '</div><form hx-post="/clare/look/brief" hx-target="#clare-workspace" class="clare-form"><label>YOUR VISION FOR THE MOVIE<textarea name="vision" rows="4" placeholder="For example: a heartfelt funny old storybook voyage. Safe and warm at the farm, dark and stormy at sea, then glowing and hopeful on the island.">' + esc(look.get("vision", "")) + '</textarea><small>Optional: Clare has already read the storyboard’s farm, storm, open-sea and island progression.</small></label><label>ANY EXTRA GUIDANCE<textarea name="guidance" rows="3" placeholder="Anything you especially want more or less of. You can write: more handmade, less clean, warmer island ending, quieter comedy.">' + esc(look.get("guidance", "")) + '</textarea></label><div class="clare-actions"><button>Save my direction</button><span>Next, add images and choose two storyboard scenes for a whole-film style test.</span></div></form><div class="clare-reference-area"><div><span class="eyebrow">SHOW CLARE IMAGES YOU LIKE</span><h3>Reference evidence, made simple</h3><p>Clare will interpret the image. You only choose what you like about it.</p></div><form hx-post="/clare/look/reference" hx-encoding="multipart/form-data" hx-target="#clare-workspace" class="clare-upload"><input type="file" name="reference_file" accept="image/*" required><select name="role"><option value="Overall feeling">I like this overall</option><option value="Feeling not subject">Use the feeling, not the subject</option><option value="Color">Use the color</option><option value="Drawing and paint">Use the drawing and paint treatment</option><option value="Scene staging">Use the scene staging</option><option value="Avoid">Avoid this aspect</option></select><button>Add reference</button></form><ul class="clare-reference-list">' + ref_rows + '</ul></div></section>'
    test_stage = '<section class="clare-stage"><div class="clare-stage-head"><div><span class="panel-kicker">STAGE 2: PROVE THE LOOK ON THE FILM</span><h2>Test one visual proposition against two real scenes.</h2><p>A direction must work for the whole ask, not just a pretty isolated image.</p></div>' + (pill("Ready", "active") if look.get("vision") or references else pill("Add a thought or image first", "slate")) + '</div><form hx-post="/clare/look/test-plan" hx-target="#clare-workspace" class="clare-form"><div class="clare-fields"><label>SCENE ONE<select name="scene_one">' + scene_options + '</select></label><label>SCENE TWO<select name="scene_two">' + scene_options + '</select></label></div><div class="clare-test-context"><article><b>Recommended first test</b><p>Capture-to-cart proves animals, anonymous workers, crates, horse, cart, dock and comic restraint.</p></article><article><b>Recommended second test</b><p>Storm hull-snap proves ship identity, weather, lighting, story stakes and handmade treatment.</p></article></div><div class="clare-actions"><button>Prepare 5 paired test variants</button><span>Each variant keeps the scenes fixed and changes the whole visual recipe, not one isolated attribute.</span></div></form>'
    if ready or locked:
        names = ["Balanced original hybrid", "More handmade ink", "Richer watercolor release", "Denser story staging", "Quieter storybook clarity"]
        cards = []
        for index, name in enumerate(names, start=1):
            cards.append('<article class="clare-test-card"><span class="eyebrow">VARIANT ' + str(index) + '</span><h3>' + name + '</h3><p>' + esc(scenes[selected_scenes[0]].get("name", "Scene one")) + '</p><p>' + esc(scenes[selected_scenes[1]].get("name", "Scene two")) + '</p><small>Paired ComfyUI test package: same story requirements, original recipe variation.</small></article>')
        test_stage += '<div class="clare-test-board">' + ''.join(cards) + '</div><div class="clare-actions"><button disabled>Generate paired tests in ComfyUI</button><span>Test packages are ready. Generation unlocks when the local render engine is available.</span></div>'
    test_stage += '</section>'
    cast_gate = '<section class="clare-stage gated"><div class="clare-stage-head"><div><span class="panel-kicker">STAGE 3: CHARACTER DEVELOPMENT</span><h2>Choose the first character after the production look is approved.</h2><p>Harry’s detailed cast is waiting. Character variants, costume states and reference sheets come after the cross-scene look lock.</p></div>' + pill("Awaiting style approval", "slate") + '</div><div class="clare-sheet-preview"><span>Pig</span><span>Rooster</span><span>States</span><span>Variants</span><span>Character sheet</span></div></section>'
    dossier_html = clare_fragment(dossier) if dossier else ""
    return '<section class="clare-journey"><div class="run-control"><div><span class="panel-kicker">CLARE VISUAL DEVELOPMENT</span><h2>' + esc(handoff.get("title")) + '</h2><small>Start by proving an original production look on real storyboard scenes. Character selection comes next.</small></div>' + pill("Guided build", "active") + '</div>' + stage_one + test_stage + cast_gate + dossier_html + '</section>'


def coverage_checkpoint_path(project_id: str):
    return projects.workspace(project_id)["library"] / "coverage_checkpoint.json"


def save_coverage_checkpoint(project_id: str, checkpoint: dict):
    path = coverage_checkpoint_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def load_coverage_checkpoint(project_id: str):
    path = coverage_checkpoint_path(project_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_coverage_checkpoint(project_id: str):
    try:
        coverage_checkpoint_path(project_id).unlink()
    except FileNotFoundError:
        pass


def coverage_ready(checkpoint: dict | None):
    return set((checkpoint or {}).get("completed_stages", [])) >= {"cast", "world", "props"}


def event(message: str):
    logs.append(message)
    del logs[:-40]


def pill(text: str, tone: str = "slate") -> str:
    return f'<span class="pill {tone}">{esc(text)}</span>'


def logs_fragment():
    entries = "".join(f'<li>{esc(line)}</li>' for line in reversed(logs)) or "<li>Harry The Helper is standing by.</li>"
    return f'<section class="panel log-panel"><div class="panel-kicker">PRODUCTION LOG</div><ol>{entries}</ol></section>'


def project_fragment(current=None):
    current = current or current_cfg()
    project = projects.get_project(current.active_project_id)
    overview = engine.project_overview(current)
    chapters = [
        ("Brief", bool(list(harry.LIBRARY_DIR.glob("*/metadata.json"))) if harry.LIBRARY_DIR.exists() else False),
        ("Call sheet", bool(list(harry.LIBRARY_DIR.glob("plan_*.json"))) if harry.LIBRARY_DIR.exists() else False),
        ("Look development", overview["entries"] > 0),
        ("Keyframes", False), ("Dailies", False), ("Edit", False),
    ]
    current_chapter = next((name for name, done in chapters if not done), "Edit")
    steps = "".join(pill(("Done · " if done else "Now · " if name == current_chapter else "Later · ") + name,
                         "done" if done else "active" if name == current_chapter else "slate") for name, done in chapters)
    return f'''<section class="panel project-panel">
      <div><div class="panel-kicker">THE DIRECTOR'S QUEST</div><h2>{esc(project.get("title", "Untitled project"))}</h2><p>Chapter: <strong>{esc(current_chapter)}</strong>. Make one clear creative decision, then move the reel forward.</p></div>
      <div class="steps">{steps}</div>
    </section>'''


def status_fragment(current=None):
    current = current or current_cfg()
    try:
        report = readiness(current)
    except Exception as error:
        return f'<div class="notice warn">Render engine check failed: {esc(error)}</div>'
    if report["mock_mode"]:
        return '<div class="notice">Practice mode is on. Builds will make local placeholder dailies.</div>'
    if not report["comfyui_reachable"]:
        return f'<div class="notice warn"><strong>Render engine offline.</strong> ComfyUI is not answering at <code>{esc(report["comfyui_url"])}</code>.</div>'
    ready = sum(1 for item in report["models"].values() if item["ready"])
    return f'<div class="notice good"><strong>Render engine connected.</strong> {esc(report.get("gpu", "GPU"))} · {ready} clip route(s) ready.</div>'


def assets_fragment(current=None):
    current = current or current_cfg()
    rows = engine.list_entries(current)
    if not rows:
        return '<div class="empty">Nothing is printed yet. Begin with Harry The Helper’s call sheet or stage a first look.</div>'
    cards = []
    for row in rows:
        image = row.get("image_path") or ""
        image_html = f'<img src="/file?path={esc(image)}" alt="">' if image and Path(image).exists() else '<div class="asset-placeholder">No frame</div>'
        cards.append(f'''<article class="asset-card">{image_html}<div><span class="eyebrow">{esc(row.get("entry_type"))}</span><h3>{esc(row.get("entry_id"))}</h3><p>{esc(row.get("description"))}</p></div></article>''')
    return '<div class="asset-grid">' + "".join(cards) + "</div>"


def _script_passages(source: str):
    """Split a source into reviewable story passages, never arbitrary paragraphs.

    Explicit headings own a passage. Long unheaded prose is grouped into
    sentence-aware chunks, so a director can point Harry at one story moment
    without selecting a whole act or duplicated adjacent text.
    """
    import re

    lines = [line.strip() for line in (source or "").replace("\r\n", "\n").split("\n")]
    heading = None
    buffer = []
    passages = []

    def flush():
        nonlocal buffer
        body = " ".join(part for part in buffer if part).strip()
        if not body:
            buffer = []
            return
        sentences = re.split(r"(?<=[.!?])\s+", body)
        chunk = []
        for sentence in sentences:
            proposed = " ".join(chunk + [sentence]).strip()
            if chunk and len(proposed) > 700:
                passages.append((heading or _passage_label(" ".join(chunk)), " ".join(chunk)))
                chunk = [sentence]
            else:
                chunk.append(sentence)
        if chunk:
            passages.append((heading or _passage_label(" ".join(chunk)), " ".join(chunk)))
        buffer = []

    for line in lines:
        if not line:
            continue
        is_heading = (
            (line.startswith("[") and line.endswith("]"))
            or (line.startswith("<") and line.endswith(">"))
            or line.upper() == line and len(line) <= 90
            or re.match(r"^(chapter|scene|act|prologue|epilogue|part)\b", line, re.I)
        )
        if is_heading:
            flush()
            heading = line.strip("[]<>").strip().title()
        else:
            buffer.append(line)
    flush()

    unique = []
    seen = set()
    for label, excerpt in passages:
        normalized = " ".join(excerpt.lower().split())
        if normalized and normalized not in seen:
            unique.append((label, excerpt))
            seen.add(normalized)
    return unique


def _passage_label(text: str):
    words = text.split()
    return " ".join(words[:8]).rstrip(".,:;-") or "Script passage"


def script_context_fragment(source: str, semantic_passages=None) -> str:
    passages = semantic_passages or [{"label": label, "excerpt": excerpt} for label, excerpt in _script_passages(source)]
    if not passages:
        return ""
    rows = []
    for number, passage in enumerate(passages, start=1):
        label, excerpt = passage["label"], passage["excerpt"]
        rows.append(
            '<button type="button" class="script-passage" data-script-excerpt="' + esc(excerpt) + '" '
            'data-script-label="' + esc(label) + '"><span>Passage ' + str(number) + ' - ' + esc(label) + '</span>'
            '<p>' + esc(excerpt) + '</p></button>'
        )
    return '<details class="script-context"><summary>Script context - select one or more passages for Harry The Helper</summary><p>Choose the story moments your note refers to. Selected passages become evidence for the revision; use the scroll list to review the whole source.</p><div class="script-passages">' + "".join(rows) + '</div></details>'


def clare_fragment(dossier):
    if not dossier:
        return "<div class='empty'>Clare has not been called yet. Once Harry's call sheet is ready, ask her to protect the cast and costume continuity.</div>"
    cards = []
    for item in dossier.get("dossiers", []):
        states = "".join('<li><b>' + esc(state.get("beat")) + '</b> - ' + esc(state.get("state")) + '</li>' for state in item.get("wardrobe_states", []))
        cards.append('<article class="specialist-card"><span class="eyebrow">' + esc(item.get("character_id")) + '</span><h3>' + esc(item.get("character_name")) + '</h3><h4>Identity lock</h4><p>' + esc(item.get("identity_lock")) + '</p><h4>Base costume / appearance</h4><p>' + esc(item.get("base_wardrobe")) + '</p><h4>Visibility</h4><p>' + esc(item.get("visibility_rules")) + '</p><h4>Watch for drift</h4><p>' + esc(item.get("drift_risks")) + '</p>' + ("<ul>" + states + "</ul>" if states else "") + '</article>')
    return '<section class="specialist-output"><div class="section-heading"><div><div class="panel-kicker">CLARE - CASTING AND COSTUME</div><h2>Character continuity dossiers</h2></div>' + pill('Ready', 'done') + '</div><div class="specialist-grid">' + "".join(cards) + '</div></section>'


def call_sheet_fragment(plan: dict, source: str = "", semantic_passages=None, persist=True, record=None):
    import uuid
    plan = dict(plan or {})
    plan["items"] = sorted(plan.get("items", []), key=harry._proposal_order)
    plan_id = str((record or {}).get("plan_id") or uuid.uuid4())
    if persist:
        record = {"plan_id": plan_id, "plan": plan, "source": source, "semantic_passages": semantic_passages or []}
        call_sheets[plan_id] = record
        save_active_call_sheet(current_cfg().active_project_id, record)
    else:
        record = record or {"plan_id": plan_id, "plan": plan, "source": source, "semantic_passages": semantic_passages or []}
    groups = [("CHARACTER", "Cast"), ("BACKDROP", "World"), ("PROP", "Props"), ("SHOT", "Coverage")]
    sections = []
    for kind, label in groups:
        items = [item for item in plan.get("items", []) if item.get("kind") == kind]
        if not items:
            continue
        cards = []
        for item in items:
            camera = ""
            if item.get("kind") == "SHOT" and item.get("camera_direction"):
                camera = '<div class="camera-note"><b>Camera</b> ' + esc(item["camera_direction"]) + "</div>"
            lighting = ""
            if item.get("kind") == "SHOT" and item.get("lighting_direction"):
                lighting = '<div class="lighting-note"><b>Lighting</b> ' + esc(item["lighting_direction"]) + "</div>"
            scope = '<label class="amend-scope"><input form="amendment-form" type="checkbox" name="selected_ids" value="' + esc(item.get("suggested_id")) + '" data-amend-search="' + esc(" ".join(str(item.get(field) or "") for field in ("kind", "name", "suggested_id", "description", "continuity_note", "beat"))) + '"> Amend this card</label>'
            cards.append('<article class="recommendation"><span class="eyebrow">' + esc(item.get("suggested_id")) + "</span><h3>" + esc(item.get("name")) + "</h3><p>" + esc(item.get("description")) + "</p>" + camera + lighting + "<small>" + esc(item.get("continuity_note")) + "</small>" + scope + "</article>")
        sections.append('<section><div class="section-heading"><h2>' + label + "</h2>" + pill(str(len(items)) + " proposed") + '</div><div class="recommendation-grid">' + "".join(cards) + "</div></section>")
    questions = "".join("<li>" + esc(question) + "</li>" for question in plan.get("questions", []))
    question_html = "<ul class='questions'>" + questions + "</ul>" if questions else ""
    clare_call = f"""<form hx-post="/clare/import" hx-target="#clare-import-result" class="specialist-call">
        <input type="hidden" name="plan_id" value="{plan_id}">
        <div><span class="eyebrow">NEXT WORKSPACE</span><h3>Clare - Casting and Costume</h3>
        <p>Import this run’s cast into the Clare Character & Costume workspace. Dossiers are built there, not during Harry review.</p></div>
        <button>Import cast into Clare workspace</button>
    </form><div id="clare-import-result"></div>"""
    return run_control_fragment(record) + '<section class="call-sheet"><div class="section-heading"><div><div class="panel-kicker">HARRY THE HELPER CALL SHEET</div><h2>' + esc(plan.get("title")) + "</h2></div>" + pill(str(len(plan.get("items", []))) + " pieces", "active") + '</div><p class="summary">' + esc(plan.get("summary")) + "</p>" + question_html + script_context_fragment(source, semantic_passages) + '<form id="amendment-form" hx-post="/harry/feedback" hx-target="#call-sheet" hx-indicator="#feedback-loading, #feedback-progress" class="feedback"><input type="hidden" name="plan_id" value="' + plan_id + '"><input type="hidden" name="context_excerpt" id="context-excerpt" value=""><div class="selected-context" id="selected-context">No script passages selected - choose one or more passages to ground this amendment.</div><div class="selected-context-text" id="selected-context-text" hidden></div><p class="amendment-scope-note">Harry receives only the selected passage, your note, and the cards marked below. Matching cards are suggested automatically; confirm or adjust them before sending.</p><label>DIRECTOR FEEDBACK<textarea name="feedback" rows="3" placeholder="Example: Make the dockyard a recurring backdrop. The selected passage shows crates travelling from the farm to the dock."></textarea></label><div><button>Ask Harry The Helper to amend the proposals</button><span id="feedback-loading" class="htmx-indicator">Harry The Helper is revising the call sheet...</span></div><div id="feedback-progress" class="feedback-progress htmx-indicator"><span></span><div><b>Revision in progress</b><small>Harry The Helper is mapping your note, checking continuity, then applying only the changed proposals.</small></div></div></form>' + "".join(sections) + clare_call + "</section>"


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(INDEX)


@app.get("/file")
async def file(path: str):
    target = Path(path).resolve()
    images = Path(current_cfg().images_dir).resolve()
    if images not in target.parents or not target.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(target)


@app.get("/fragments/dashboard", response_class=HTMLResponse)
async def dashboard():
    current = current_cfg()
    retry = ''
    if coverage_ready(load_coverage_checkpoint(current.active_project_id)):
        retry = '<div class="notice warn"><strong>Coverage checkpoint saved.</strong> Cast, world, and props are ready. <button hx-post="/harry/retry-coverage" hx-target="#call-sheet" hx-indicator="#coverage-retry-loading">Retry Coverage</button><span id="coverage-retry-loading" class="htmx-indicator">Resuming camera and lighting coverage...</span></div>'
    engine_placeholder = '<div id="engine-status" hx-get="/fragments/engine-status" hx-trigger="load" class="notice">Checking render engine separately...</div>'
    return project_fragment(current) + retry + engine_placeholder + assets_fragment(current)


@app.get("/fragments/engine-status", response_class=HTMLResponse)
async def engine_status():
    # The ComfyUI client may wait on a dead/offline host. Keep that check off
    # the dashboard critical path so Harry and Retry Coverage stay usable.
    return await asyncio.to_thread(status_fragment, current_cfg())


@app.get("/fragments/logs", response_class=HTMLResponse)
async def get_logs():
    return logs_fragment()


@app.get("/harry/current", response_class=HTMLResponse)
async def current_harry():
    record = load_active_call_sheet(current_cfg().active_project_id)
    if record is None:
        return "<div class='empty'>Harry The Helper’s creative call sheet will arrive here after you give him a story.</div>"
    return call_sheet_fragment(record["plan"], record.get("source", ""), record.get("semantic_passages", []), persist=False, record=record)


@app.post("/harry/run/select", response_class=HTMLResponse)
async def select_harry_run(run_file: Annotated[str, Form()]):
    current = current_cfg()
    filename = Path(run_file).name
    if filename != run_file or not filename.startswith("plan_") or not filename.endswith(".json"):
        return '<div class="notice warn">That saved run is not available.</div>'
    path = projects.workspace(current.active_project_id)["library"] / filename
    if not path.exists():
        return '<div class="notice warn">That saved run is not available.</div>'
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return '<div class="notice warn">That saved run could not be read.</div>'
    prior = load_active_call_sheet(current.active_project_id) or {}
    record = {"plan_id": plan.get("plan_id") or path.stem, "plan": plan, "source": prior.get("source", ""), "semantic_passages": prior.get("semantic_passages", []), "run_version": prior.get("run_version", "saved-run")}
    save_active_call_sheet(current.active_project_id, record)
    event("Opened saved Harry run: " + plan.get("title", path.stem))
    return call_sheet_fragment(plan, record["source"], record["semantic_passages"], persist=False, record=record)


@app.post("/harry/run/save-confirmed", response_class=HTMLResponse)
async def save_confirmed_harry_run():
    current = current_cfg()
    record = load_active_call_sheet(current.active_project_id)
    if record is None:
        return '<div class="notice warn">There is no active Harry run to save.</div>'
    version = next_run_version(record)
    plan = dict(record["plan"])
    plan["plan_id"] = plan.get("plan_id", "run") + "_" + version
    plan["created_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    saved = {**record, "plan_id": plan["plan_id"], "plan": plan, "run_version": version}
    archive = projects.workspace(current.active_project_id)["library"] / f"plan_{plan['plan_id']}_confirmed.json"
    archive.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    save_active_call_sheet(current.active_project_id, saved)
    event("Saved confirmed Harry version " + version + ".")
    return call_sheet_fragment(plan, saved.get("source", ""), saved.get("semantic_passages", []), persist=False, record=saved)


@app.post("/project/select", response_class=HTMLResponse)
async def select_project(project_id: Annotated[str, Form()]):
    current = set_project(project_id)
    event(f"Project switched to {projects.get_project(project_id).get('title')}")
    return project_fragment(current) + status_fragment(current) + assets_fragment(current)


@app.post("/project/new", response_class=HTMLResponse)
async def new_project(title: Annotated[str, Form()], version: Annotated[str, Form()] = "v1"):
    project = projects.create_project(title, version)
    current = set_project(project["project_id"])
    event(f"Created project: {project['title']}")
    return project_fragment(current) + status_fragment(current) + assets_fragment(current)


@app.post("/engine/start", response_class=HTMLResponse)
async def start_engine():
    current = current_cfg()
    if current.comfyui_url.rstrip("/") not in {"http://127.0.0.1:8188", "http://localhost:8188"}:
        return '<div class="notice warn">This project points at a remote render engine. Start that machine there.</div>'
    if ComfyClient(current.comfyui_url).ping():
        return '<div class="notice good">The local render engine is already awake.</div>'
    python = COMFY_ROOT / "python_embeded" / "python.exe"
    main = COMFY_ROOT / "ComfyUI" / "main.py"
    if not python.exists() or not main.exists():
        return '<div class="notice warn">The local ComfyUI portable installation was not found.</div>'
    env = os.environ | {"TRANSFORMERS_NO_FLASH_ATTENTION": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0"}
    log = open(COMFY_ROOT / "comfy_boot.log", "ab")
    subprocess.Popen([str(python), "-s", "ComfyUI/main.py", "--windows-standalone-build", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch", "--cuda-malloc", "--fast", "--use-pytorch-cross-attention", "--disable-xformers", "--preview-method", "auto"], cwd=COMFY_ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    event("Started the local ComfyUI render engine.")
    return '<div class="notice">ComfyUI is warming up. Check the engine again in a minute.</div>'


@app.post("/harry/extract", response_class=HTMLResponse)
async def extract_source(source_file: UploadFile | None = File(default=None)):
    if source_file is None or not source_file.filename:
        return '<textarea id="story-source" name="source" rows="13" placeholder="Paste the story, script, or narration here…" class="w-full rounded-xl border border-slate-700 bg-slate-950 p-4 text-slate-100"></textarea>'
    suffix = Path(source_file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".docx", ".pdf"}:
        return '<div class="notice warn">Harry The Helper can read .txt, .md, .docx, or .pdf source files.</div>'
    upload_dir = ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    uploaded_path = upload_dir / f"preview_{random.randint(100000, 999999)}{suffix}"
    uploaded_path.write_bytes(await source_file.read())
    try:
        source = harry.extract_text_document(str(uploaded_path))
    except harry.HarryError as error:
        return f'<div class="notice warn"><strong>Amendment not applied.</strong> {esc(error)} The active call sheet remains unchanged. Review the Production Log and the project diagnostics folder for details.</div>'
    if not source.strip():
        return '<div class="notice warn">That file opened, but no readable source text was found.</div>'
    event(f"Harry The Helper loaded {source_file.filename} for review.")
    return HTMLResponse(f'<textarea id="story-source" name="source" rows="13" class="w-full rounded-xl border border-violet-400 bg-slate-950 p-4 text-slate-100">{esc(source)}</textarea>', headers={"HX-Trigger": "source-extracted"})


@app.post("/harry/retry-coverage", response_class=HTMLResponse)
async def retry_coverage():
    current = current_cfg()
    checkpoint = load_coverage_checkpoint(current.active_project_id)
    if not checkpoint:
        return "<div class='notice warn'>No paused coverage checkpoint is available. Run Harry The Helper's brief first.</div>"
    if not coverage_ready(checkpoint):
        return "<div class='notice warn'>This checkpoint cannot resume coverage because cast, world, and props were not all completed.</div>"
    event("Retry Coverage requested. Reusing completed cast, world, and props; only coverage will call the provider.")
    def progress(message):
        event(message)
    try:
        plan = await asyncio.to_thread(harry.resume_coverage, current.harry_provider, checkpoint, progress)
    except harry.HarryRateLimitError as error:
        event(f"Coverage remains paused: provider rate limit reached. {error}")
        return f'<div class="notice warn"><strong>Coverage is still rate limited.</strong> {esc(error)} The checkpoint remains saved; retry only coverage after the stated wait.</div>'
    except harry.HarryError as error:
        event(f"Coverage retry failed: {error}")
        return f'<div class="notice warn">Coverage retry failed: {esc(error)} The checkpoint remains saved.</div>'
    clear_coverage_checkpoint(current.active_project_id)
    semantic_passages = await asyncio.to_thread(harry.label_script_passages, current.harry_provider, _script_passages(checkpoint["source_text"]))
    plan["script_passages"] = semantic_passages
    event(f"Ready to Review: Coverage resumed with {len([i for i in plan['items'] if i.get('kind') == 'SHOT'])} camera-and-lighting shots.")
    return call_sheet_fragment(plan, checkpoint["source_text"], semantic_passages)


@app.post("/harry/analyze", response_class=HTMLResponse)
async def analyze_story(title: Annotated[str, Form()], source: Annotated[str, Form()] = "", era: Annotated[str, Form()] = "", source_file: UploadFile | None = File(default=None)):
    uploaded_path = None
    if source_file and source_file.filename:
        suffix = Path(source_file.filename).suffix.lower()
        if suffix not in {".txt", ".md", ".docx", ".pdf"}:
            return '<div class="notice warn">Harry The Helper can read .txt, .md, .docx, or .pdf source files.</div>'
        upload_dir = ROOT / "uploads"
        upload_dir.mkdir(exist_ok=True)
        uploaded_path = upload_dir / f"source_{random.randint(100000, 999999)}{suffix}"
        uploaded_path.write_bytes(await source_file.read())
        try:
            extracted = harry.extract_text_document(str(uploaded_path)).strip()
            source = source.strip() if extracted and extracted in source else (source + "\n\n" + extracted).strip()
        except harry.HarryError as error:
            return f'<div class="notice warn">{esc(error)}</div>'
    if not source.strip():
        return '<div class="notice warn">Give Harry The Helper a script, story, treatment, narration, or source file first.</div>'
    current = current_cfg()
    harry.set_library_dir(projects.workspace(current.active_project_id)["library"])
    logs.clear()
    event("Harry The Helper opened the script. Starting four production passes: cast, world, props, then camera and lighting coverage.")
    def progress(message):
        event(message)
    checkpoint = {"title": title, "era": era, "source_text": source, "source_path": "", "characters": [], "backdrops": [], "props": [], "completed_stages": []}
    try:
        for stage in harry.analyze_staged(current.harry_provider, title, source, "", str(uploaded_path) if uploaded_path else None, era, progress):
            if stage["stage"] == "cast":
                checkpoint["characters"] = stage["items"]
                checkpoint["source_path"] = stage.get("source_path", "")
                checkpoint["completed_stages"].append("cast")
                save_coverage_checkpoint(current.active_project_id, checkpoint)
            elif stage["stage"] == "world":
                checkpoint["backdrops"] = stage["items"]
                checkpoint["completed_stages"].append("world")
                save_coverage_checkpoint(current.active_project_id, checkpoint)
            elif stage["stage"] == "props":
                checkpoint["props"] = stage["items"]
                checkpoint["completed_stages"].append("props")
                save_coverage_checkpoint(current.active_project_id, checkpoint)
            elif stage["stage"] == "complete":
                plan = stage["plan"]
                clear_coverage_checkpoint(current.active_project_id)
                break
        else:
            raise harry.HarryError("Harry The Helper stopped before producing a call sheet.")
    except harry.HarryRateLimitError as error:
        event(f"Harry The Helper paused: provider rate limit reached. {error}")
        saved = save_coverage_checkpoint(current.active_project_id, checkpoint)
        if coverage_ready(checkpoint):
            return f'<div class="notice warn"><strong>Coverage paused, not lost.</strong> {esc(error)} Cast, world, and props are checkpointed. Wait for the provider window, then use <b>Retry Coverage</b> to make only the camera-and-lighting request.</div>'
        completed = ", ".join(checkpoint["completed_stages"]) or "no"
        return f'<div class="notice warn"><strong>Brief paused.</strong> {esc(error)} Completed stages: {esc(completed)}. Coverage retry is unavailable until cast, world, and props all complete.</div>'
    except harry.HarryError as error:
        event(f"Harry The Helper could not finish the brief: {error}")
        return f'<div class="notice warn"><strong>Harry The Helper could not complete the call sheet.</strong> {esc(error)} The Production Log records the pass that failed; a local diagnostic was saved for review.</div>'
    passages = _script_passages(source)
    semantic_passages = await asyncio.to_thread(harry.label_script_passages, current.harry_provider, passages)
    plan["script_passages"] = semantic_passages
    event(f"Ready to Review: Harry The Helper prepared {len(plan['items'])} proposals and labeled {len(semantic_passages)} script moments. Next step: review Harry The Helper's call sheet.")
    return call_sheet_fragment(plan, source, semantic_passages)

@app.post("/clare/import", response_class=HTMLResponse)
async def import_cast_to_clare(plan_id: Annotated[str, Form()]):
    record = call_sheets.get(plan_id) or load_active_call_sheet(current_cfg().active_project_id)
    if record is None:
        return '<div class="notice warn">No active Harry run is available to import.</div>'
    handoff = save_clare_handoff(current_cfg().active_project_id, record)
    event(f"Imported {len(handoff['characters'])} character record(s) into the Clare Character & Costume workspace.")
    return '<div class="notice good">Cast imported into the Clare Character &amp; Costume workspace. Open that tab to continue.</div>'


@app.post("/clare/look/brief", response_class=HTMLResponse)
async def save_clare_look_brief(vision: Annotated[str, Form()] = "", guidance: Annotated[str, Form()] = ""):
    current = current_cfg()
    state = load_clare_development(current.active_project_id)
    state.setdefault("look", {}).update({"vision": vision.strip(), "guidance": guidance.strip()})
    save_clare_development(current.active_project_id, state)
    event("Clare saved the director's plain-language look direction.")
    return await current_clare()


@app.post("/clare/look/reference", response_class=HTMLResponse)
async def add_clare_reference(reference_file: UploadFile = File(), role: Annotated[str, Form()] = "Overall feeling"):
    current = current_cfg()
    suffix = Path(reference_file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return '<div class="notice warn">Add a JPG, PNG or WEBP image reference.</div>'
    destination = clare_reference_dir(current.active_project_id) / (str(random.randint(100000, 999999)) + suffix)
    destination.write_bytes(await reference_file.read())
    state = load_clare_development(current.active_project_id)
    state.setdefault("references", []).append({"name": reference_file.filename, "path": str(destination), "role": role})
    save_clare_development(current.active_project_id, state)
    event("Clare added visual reference: " + reference_file.filename)
    return await current_clare()


@app.post("/clare/look/test-plan", response_class=HTMLResponse)
async def prepare_clare_test_plan(scene_one: Annotated[str, Form()], scene_two: Annotated[str, Form()]):
    current = current_cfg()
    scenes = clare_scene_options({})
    if scene_one not in scenes or scene_two not in scenes or scene_one == scene_two:
        return '<div class="notice warn">Choose two different storyboard scenes for the paired test.</div>'
    state = load_clare_development(current.active_project_id)
    state["test_scene_ids"] = [scene_one, scene_two]
    state["look_status"] = "test-ready"
    save_clare_development(current.active_project_id, state)
    event("Clare prepared five paired whole-film style test variants.")
    return await current_clare()


@app.post("/specialists/clare", response_class=HTMLResponse)
async def run_clare(plan_id: Annotated[str, Form()]):
    record = call_sheets.get(plan_id) or load_active_call_sheet(current_cfg().active_project_id)
    if record is None:
        return '<div class="notice warn">Clare needs an active call sheet. Ask Harry The Helper for a brief first.</div>'
    current = current_cfg()
    event("Clare began reviewing character identity, costume, and framing constraints.")
    try:
        dossier = await asyncio.to_thread(specialists.clare_character_costume, current.harry_provider, record["plan"], record.get("source", ""), record["plan"].get("era", ""))
        specialists.save_dossier(projects.workspace(current.active_project_id)["library"], dossier)
    except specialists.SpecialistError as error:
        event(f"Clare could not complete the character dossiers: {error}")
        return f'<div class="notice warn">{esc(error)}</div>'
    event("Clare is ready: character identity and costume dossiers are available for review.")
    return clare_fragment(dossier)


@app.get("/specialists/clare/current", response_class=HTMLResponse)
async def current_clare():
    current = current_cfg()
    handoff = load_clare_handoff(current.active_project_id)
    if handoff:
        dossier = specialists.load_dossier(projects.workspace(current.active_project_id)["library"])
        return clare_workspace_fragment(handoff, dossier)
    record = load_active_call_sheet(current.active_project_id)
    if record:
        return '<div class="empty">Import the cast from Harry The Helper to begin the Clare Character & Costume workspace.</div>'
    return '<div class="empty">Clare joins after Harry The Helper has prepared a call sheet.</div>'


@app.post("/harry/feedback", response_class=HTMLResponse)
async def director_feedback(plan_id: Annotated[str, Form()], feedback: Annotated[str, Form()], context_excerpt: Annotated[str, Form()] = "", selected_ids: Annotated[list[str], Form()] = []):
    record = call_sheets.get(plan_id)
    if record is None:
        current_project = current_cfg().active_project_id
        saved = load_active_call_sheet(current_project)
        if saved:
            record = saved
            event("Recovered the active call sheet from the project library for amendment.")
        else:
            return '<div class="notice warn">No active call sheet was found for this project. Create one with Harry The Helper first.</div>'
    plan = record["plan"]
    source = record.get("source", "")
    semantic_passages = record.get("semantic_passages", plan.get("script_passages", []))
    current = current_cfg()
    event("Director Feedback received." + (" Selected script context attached." if context_excerpt.strip() else ""))
    event("Amendment stage 1/3: Harry The Helper is mapping the note to active proposals.")
    def progress(message):
        event(message)
    try:
        revised = await asyncio.to_thread(harry.revise_plan, current.harry_provider, plan, feedback, context_excerpt, selected_ids, progress)
    except harry.HarryError as error:
        event(f"Harry The Helper could not revise the proposals: {error}")
        return f'<div class="notice warn">{esc(error)}</div>'
    event(f"Harry The Helper applied a focused amendment: {len(revised['items'])} proposals now in play.")
    revised["script_passages"] = semantic_passages
    return call_sheet_fragment(revised, source, semantic_passages)


@app.post("/generate", response_class=HTMLResponse)
async def generate(entry_id: Annotated[str, Form()], entry_type: Annotated[str, Form()], prompt: Annotated[str, Form()], variants: Annotated[int, Form()] = 4):
    current = current_cfg()
    if not entry_id.strip() or not prompt.strip():
        return '<div class="notice warn">Name the asset and describe the frame before rolling camera.</div>'
    event(f"Rolling camera for {entry_id}.")
    try:
        result = await asyncio.to_thread(engine.generate, current, entry_id, prompt, "", None, variants)
    except engine.EngineError as error:
        event(f"Generation failed: {error}")
        return f'<div class="notice warn">{esc(error)}</div>'
    global last_variants
    last_variants = [{"path": item.path, "seed": item.seed} for item in result.variants]
    event(f"Dailies returned: {len(last_variants)} variant(s).")
    cards = "".join(f'<label class="daily"><input type="radio" name="winner" value="{esc(v["path"])}|{v["seed"]}" {"checked" if i == 0 else ""}><img src="/file?path={esc(v["path"])}" alt="variant"><span>Take {i + 1} · seed {v["seed"]}</span></label>' for i, v in enumerate(last_variants))
    warning = " ".join(esc(item) for item in result.warnings)
    return f'<div class="dailies">{cards}</div>{f"<div class=notice>{warning}</div>" if warning else ""}'


@app.post("/lock", response_class=HTMLResponse)
async def lock_take(entry_id: Annotated[str, Form()], entry_type: Annotated[str, Form()], description: Annotated[str, Form()] = "", prompt: Annotated[str, Form()] = "", winner: Annotated[str, Form()] = ""):
    if "|" not in winner:
        return '<div class="notice warn">Choose a take before printing it.</div>'
    path, seed = winner.rsplit("|", 1)
    try:
        message = engine.lock_concept(current_cfg(), entry_id, entry_type, "", description, prompt, "", seed, path)
    except engine.EngineError as error:
        return f'<div class="notice warn">{esc(error)}</div>'
    event(message)
    return f'<div class="notice good">{esc(message)} It is now in the production book.</div>'


@app.post("/setup/save", response_class=HTMLResponse)
async def save_setup(comfyui_url: Annotated[str, Form()], mock_mode: Annotated[str | None, Form()] = None, provider: Annotated[str, Form()] = "Azure OpenAI", azure_endpoint: Annotated[str, Form()] = "", azure_deployment: Annotated[str, Form()] = "", azure_api_version: Annotated[str, Form()] = "2025-04-01-preview"):
    current = current_cfg()
    current.comfyui_url = comfyui_url.strip() or current.comfyui_url
    current.mock_mode = mock_mode is not None
    current.harry_provider = provider
    current.azure.endpoint = azure_endpoint.strip()
    current.azure.deployment = azure_deployment.strip()
    current.azure.api_version = azure_api_version.strip() or current.azure.api_version
    cfgmod.save_config(current)
    event("Production settings saved.")
    return status_fragment(current)
