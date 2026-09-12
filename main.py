import asyncio
import html
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


def call_sheet_fragment(plan: dict, source: str = "", semantic_passages=None):
    import uuid
    plan_id = str(uuid.uuid4())
    call_sheets[plan_id] = {"plan": plan, "source": source, "semantic_passages": semantic_passages or []}
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
            cards.append('<article class="recommendation"><span class="eyebrow">' + esc(item.get("suggested_id")) + "</span><h3>" + esc(item.get("name")) + "</h3><p>" + esc(item.get("description")) + "</p>" + camera + "<small>" + esc(item.get("continuity_note")) + "</small></article>")
        sections.append('<section><div class="section-heading"><h2>' + label + "</h2>" + pill(str(len(items)) + " proposed") + '</div><div class="recommendation-grid">' + "".join(cards) + "</div></section>")
    questions = "".join("<li>" + esc(question) + "</li>" for question in plan.get("questions", []))
    question_html = "<ul class='questions'>" + questions + "</ul>" if questions else ""
    return '<section class="call-sheet"><div class="section-heading"><div><div class="panel-kicker">HARRY THE HELPER CALL SHEET</div><h2>' + esc(plan.get("title")) + "</h2></div>" + pill(str(len(plan.get("items", []))) + " pieces", "active") + '</div><p class="summary">' + esc(plan.get("summary")) + "</p>" + question_html + script_context_fragment(source, semantic_passages) + '<form hx-post="/harry/feedback" hx-target="#call-sheet" hx-indicator="#feedback-loading" class="feedback"><input type="hidden" name="plan_id" value="' + plan_id + '"><input type="hidden" name="context_excerpt" id="context-excerpt" value=""><div class="selected-context" id="selected-context">No script passages selected - feedback will apply to the whole call sheet.</div><div class="selected-context-text" id="selected-context-text" hidden></div><label>DIRECTOR FEEDBACK<textarea name="feedback" rows="3" placeholder="Example: Make the dockyard a recurring backdrop. The selected passage shows crates travelling from the farm to the dock."></textarea></label><div><button>Ask Harry The Helper to amend the proposals</button><span id="feedback-loading" class="htmx-indicator">Harry The Helper is revising the call sheet...</span></div></form>' + "".join(sections) + "</section>"


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
    return project_fragment(current) + status_fragment(current) + assets_fragment(current)


@app.get("/fragments/logs", response_class=HTMLResponse)
async def get_logs():
    return logs_fragment()


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
        return f'<div class="notice warn">{esc(error)}</div>'
    if not source.strip():
        return '<div class="notice warn">That file opened, but no readable source text was found.</div>'
    event(f"Harry The Helper loaded {source_file.filename} for review.")
    return f'<textarea id="story-source" name="source" rows="13" class="w-full rounded-xl border border-violet-400 bg-slate-950 p-4 text-slate-100">{esc(source)}</textarea>'


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
            source = (source + "\n\n" + harry.extract_text_document(str(uploaded_path))).strip()
        except harry.HarryError as error:
            return f'<div class="notice warn">{esc(error)}</div>'
    if not source.strip():
        return '<div class="notice warn">Give Harry The Helper a script, story, treatment, narration, or source file first.</div>'
    current = current_cfg()
    harry.set_library_dir(projects.workspace(current.active_project_id)["library"])
    event("Harry The Helper opened the script and began scouting the production.")
    try:
        plan = await asyncio.to_thread(harry.analyze, current.harry_provider, title, source, "", str(uploaded_path) if uploaded_path else None, era)
    except harry.HarryError as error:
        event(f"Harry The Helper could not finish the brief: {error}")
        return f'<div class="notice warn">{esc(error)}</div>'
    passages = _script_passages(source)
    semantic_passages = await asyncio.to_thread(harry.label_script_passages, current.harry_provider, passages)
    plan["script_passages"] = semantic_passages
    event(f"Ready to Review: Harry The Helper prepared {len(plan['items'])} proposals and labeled {len(semantic_passages)} script moments.")
    return call_sheet_fragment(plan, source, semantic_passages)

@app.post("/harry/feedback", response_class=HTMLResponse)
async def director_feedback(plan_id: Annotated[str, Form()], feedback: Annotated[str, Form()], context_excerpt: Annotated[str, Form()] = ""):
    record = call_sheets.get(plan_id)
    if record is None:
        return '<div class="notice warn">That call sheet is no longer active. Ask Harry The Helper for a fresh brief first.</div>'
    plan = record["plan"]
    source = record.get("source", "")
    semantic_passages = record.get("semantic_passages", plan.get("script_passages", []))
    current = current_cfg()
    event("The director gave Harry The Helper a call-sheet note." + (" Script context attached." if context_excerpt.strip() else ""))
    try:
        revised = await asyncio.to_thread(harry.revise_plan, current.harry_provider, plan, feedback, context_excerpt)
    except harry.HarryError as error:
        event(f"Harry The Helper could not revise the proposals: {error}")
        return f'<div class="notice warn">{esc(error)}</div>'
    learned = await asyncio.to_thread(harry.learn_from_feedback, current.harry_provider, revised, feedback, context_excerpt)
    event(f"Harry The Helper amended the call sheet: {len(revised['items'])} proposals now in play.")
    if learned:
        event("Director Playbook updated: " + learned)
    revised["script_passages"] = semantic_passages
    result = call_sheet_fragment(revised, source, semantic_passages)
    if learned:
        result = '<div class="notice good"><strong>Director Playbook updated.</strong> Harry The Helper will apply this next time: ' + esc(learned) + '</div>' + result
    return result


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
