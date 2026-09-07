"""Harry the Advisor: source analysis, provider calls, and editable Build drafts."""

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
LIBRARY_DIR = PROJECT_ROOT / "harry_library"


def set_library_dir(path):
    global LIBRARY_DIR
    LIBRARY_DIR = Path(path)
OPENSCOUT_CONFIG = Path.home() / "OpenScout" / "openscout" / "config.yaml"

HARRY_PERSONA = """You are Harry the Advisor, a thoughtful pre-production advisor for AI-assisted film.
Read the supplied script, story, poem, or narration transcript."""

ERA_NOTE = """
IF AN ERA/SETTING IS SPECIFIED, TREAT IT AS A HARD CONSTRAINT:
Ground every visual detail, object, and technology in that era. Do not introduce anachronistic
equipment (e.g. a motor vehicle in a pre-automobile setting, electric lighting in a candlelit era).
When inferring something the text doesn't name outright, the era is exactly what should decide
which real-world version of it is correct -- if animals are "moved from field to ship" in an era
before motor transport, the era tells you that's a horse and cart, not a truck."""

ITEM_SHAPE_NOTE = """Each item must have this exact shape:
{
  "kind": "%s",
  "name": "human readable name",
  "suggested_id": "%s",
  "description": "what it is and why it matters",
  "continuity_note": "visual or narrative continuity guidance",
  "positive_prompt": "initial target-model-aware visual prompt",
  "negative_prompt": "things to avoid",
  "beat": "the story section this is introduced or most associated with",
  "reused_from": "optional prior item id"
}"""

CHARACTER_SYSTEM_PROMPT = f"""{HARRY_PERSONA}

TASK: identify ONLY the characters in the material -- do not identify backdrops, props, or shots in this pass; those come in separate passes. Do not invent named people without textual evidence.
{ERA_NOTE}

Return JSON only, with this exact shape:
{{"items": [ {ITEM_SHAPE_NOTE % ('CHARACTER', 'CHARACTER:snake_case')} ]}}"""

BACKDROP_SYSTEM_PROMPT = f"""{HARRY_PERSONA}

TASK: identify ONLY recurring backdrops/locations in the material (kind "BACKDROP") -- do not identify characters, props, or shots in this pass; those are handled separately. You will be told which characters were already identified in an earlier pass, for context only.
{ERA_NOTE}

A location the story clearly requires but never directly describes (e.g. a farmhouse implied by "farrow & field") MAY be suggested as a tentative item, but mark its continuity_note as "inferred, not explicit in source -- confirm before locking" so the director knows it's a suggestion, not something read directly off the page.

Return JSON only, with this exact shape:
{{"items": [ {ITEM_SHAPE_NOTE % ('BACKDROP', 'BACKDROP:snake_case')} ]}}"""

PROP_SYSTEM_PROMPT = f"""{HARRY_PERSONA}

TASK: identify ONLY recurring key objects/props in the material (kind "PROP") -- do not identify characters, backdrops, or shots in this pass; those are handled separately. You will be told which characters and backdrops were already identified in earlier passes, for context only.
{ERA_NOTE}

RECURRING PHYSICAL OBJECTS ARE PROPS, EVEN WHEN DESCRIBED POETICALLY:
A recurring object central to multiple beats (a vehicle, a vessel, a key object the characters interact with repeatedly) warrants its own PROP entry even if the text never uses a single plain noun for it consistently -- reading "sail ship," "cargo hold," "the hull," and "a sleeping hulk" across a few lines is the same object described four ways, not four separate ideas. Do not require one literal name to appear before recommending something the narrative obviously depends on.
Separately, when the text implies a real-world object or mechanism the story logically requires but never actually states (e.g. characters are "moved from field to ship" with no vehicle named), you MAY suggest it as a tentative item, but mark its continuity_note as "inferred, not explicit in source -- confirm before locking" so the director knows it's a suggestion, not something read directly off the page.

Return JSON only, with this exact shape:
{{"items": [ {ITEM_SHAPE_NOTE % ('PROP', 'PROP:snake_case')} ]}}"""

SHOT_SYSTEM_PROMPT = f"""{HARRY_PERSONA}

TASK: break the material into beats/scenes and produce shot drafts (kind "SHOT") -- one pass, focused entirely on scene coverage, since this is the task most likely to get shortchanged when asked alongside everything else. You will be told which characters, backdrops, and props were already identified in earlier passes -- reference them by name in shot descriptions/continuity notes for continuity, don't redefine them.
{ERA_NOTE}

SCENE/BEAT COVERAGE IS MANDATORY, NOT OPTIONAL:
If the source text contains explicit section markers (bracketed headers like "< The Ship >", chapter titles, scene headings, or similarly clear structural breaks), treat each one as a beat and include at least one SHOT item for every single marked section -- do not silently skip a marked section because it seems minor or because you are trying to keep the list short. A missing beat is a more serious omission than a slightly longer list. If the source has no explicit markers, infer a reasonable beat structure yourself (opening, rising action, climax, resolution, etc.) and still cover it with SHOT items.

Return JSON only, with this exact shape:
{{
  "summary": "short production reading of the whole piece",
  "questions": ["only essential clarification questions; empty when the material is clear"],
  "items": [ {ITEM_SHAPE_NOTE % ('SHOT', 'beat.shot, e.g. 1.1')} ]
}}"""


class HarryError(Exception):
    pass


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _slug(value):
    return "_".join(re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).split())


def _load_openscout_config():
    if not OPENSCOUT_CONFIG.exists():
        raise HarryError(f"OpenScout config was not found at {OPENSCOUT_CONFIG}.")
    with OPENSCOUT_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def provider_status(provider):
    provider = provider or "Claude"
    if provider == "Azure OpenAI":
        try:
            azure = _load_openscout_config().get("azure", {})
            key = os.environ.get(azure.get("api_key_env", "AZURE_FOUNDRY_API_KEY"))
            return "Azure OpenAI is ready through OpenScout." if azure.get("endpoint") and azure.get("deployment") and key else "Azure OpenAI settings or its environment key are unavailable to this app process."
        except HarryError as error:
            return str(error)
    env_map = {"Claude": "ANTHROPIC_API_KEY", "OpenAI": "OPENAI_API_KEY", "Grok": "XAI_API_KEY"}
    if provider == "Ollama":
        return "Ollama runs locally at its configured address; no material leaves this machine."
    return f"{provider} is ready." if os.environ.get(env_map[provider]) else f"Set {env_map[provider]} in the app process environment to use {provider}."


def _request(url, headers, payload, timeout=120):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:600]
        raise HarryError(f"Provider returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise HarryError(f"Could not reach the provider: {error.reason}") from error


def _chat(provider, system_prompt, user_prompt):
    if provider == "Azure OpenAI":
        azure = _load_openscout_config().get("azure", {})
        key = os.environ.get(azure.get("api_key_env", "AZURE_FOUNDRY_API_KEY"))
        if not key:
            raise HarryError("Azure API key is not available to this app process. Restart the app after setting the OpenScout Azure key environment variable.")
        endpoint = azure.get("endpoint", "").rstrip("/")
        url = f"{endpoint}/openai/v1/chat/completions"
        response = _request(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, {
            "model": azure.get("deployment"), "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_completion_tokens": 5000,
        })
        return response["choices"][0]["message"]["content"]
    if provider == "Claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise HarryError("ANTHROPIC_API_KEY is not available to this app process.")
        response = _request("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, {
            "model": os.environ.get("HARRY_CLAUDE_MODEL", "claude-sonnet-4-5"), "max_tokens": 5000,
            "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}],
        })
        return "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")
    if provider == "OpenAI":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HarryError("OPENAI_API_KEY is not available to this app process.")
        response = _request("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, {
            "model": os.environ.get("HARRY_OPENAI_MODEL", "gpt-4.1"), "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "response_format": {"type": "json_object"},
        })
        return response["choices"][0]["message"]["content"]
    if provider == "Grok":
        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise HarryError("XAI_API_KEY is not available to this app process.")
        response = _request("https://api.x.ai/v1/chat/completions", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, {
            "model": os.environ.get("HARRY_GROK_MODEL", "grok-3"), "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "response_format": {"type": "json_object"},
        })
        return response["choices"][0]["message"]["content"]
    response = _request(os.environ.get("HARRY_OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat", {"Content-Type": "application/json"}, {
        "model": os.environ.get("HARRY_OLLAMA_MODEL", "qwen2.5:7b"), "stream": False,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "format": "json",
    })
    return response["message"]["content"]


def _parse_json_block(raw):
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        return json.loads(fenced)
    except json.JSONDecodeError as error:
        raise HarryError("Harry returned a response that was not valid JSON. Try again or choose another provider.") from error


def _clean_items(raw_items):
    clean = []
    for item in raw_items or []:
        if not isinstance(item, dict) or item.get("kind") not in {"CHARACTER", "BACKDROP", "PROP", "SHOT"}:
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        kind = item["kind"]
        prefix = {"CHARACTER": "CHARACTER", "BACKDROP": "BACKDROP", "PROP": "PROP"}.get(kind)
        suggested = str(item.get("suggested_id") or "").strip()
        if prefix and not suggested.startswith(prefix + ":"):
            suggested = f"{prefix}:{_slug(name)}"
        if kind == "SHOT" and not suggested:
            suggested = _slug(name) or "draft_shot"
        item["suggested_id"] = suggested
        clean.append({key: str(item.get(key) or "") for key in ("kind", "name", "suggested_id", "description", "continuity_note", "positive_prompt", "negative_prompt", "beat", "reused_from")})
    return clean


def _parse_plan(raw):
    plan = _parse_json_block(raw)
    if not isinstance(plan, dict) or not isinstance(plan.get("items"), list):
        raise HarryError("Harry's response did not contain a usable recommendation list.")
    plan["items"] = _clean_items(plan["items"])
    plan["summary"] = str(plan.get("summary") or "")
    plan["questions"] = [str(question) for question in plan.get("questions", []) if str(question).strip()]
    return plan


def _parse_items(raw):
    """Same cleaning as _parse_plan, but for a stage call that returns only
    {"items": [...]} with no summary/questions expected."""
    data = _parse_json_block(raw)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise HarryError("Harry's response did not contain a usable item list for this pass.")
    return _clean_items(data["items"])


def save_source(title, text, uploaded_path=None):
    LIBRARY_DIR.mkdir(exist_ok=True)
    source_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(title) or 'source'}"
    folder = LIBRARY_DIR / source_id
    folder.mkdir()
    text_path = folder / "source.txt"
    text_path.write_text(text or "", encoding="utf-8")
    if uploaded_path:
        shutil.copy2(uploaded_path, folder / Path(uploaded_path).name)
    metadata = {"source_id": source_id, "title": title or source_id, "created_at": _timestamp(), "text_path": str(text_path), "audio_path": str(folder / Path(uploaded_path).name) if uploaded_path else ""}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def extract_text_document(file_path):
    if not file_path:
        raise HarryError("Attach a .txt, .md, .docx, or .pdf document first.")
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rtf"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        try:
            from docx import Document
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as error:
            raise HarryError("Reading .docx needs python-docx. Install it with: pip install python-docx") from error
        doc = Document(path)
        parts = []
        # Walk the document body in actual document order, handling both
        # paragraphs AND tables. document.paragraphs alone only sees
        # top-level body paragraphs -- text laid out in a table (a common
        # choice for scripts/treatments, since it keeps character names
        # and dialogue aligned) is otherwise silently invisible and
        # extraction returns an empty string with no error at all.
        for child in doc.element.body.iterchildren():
            if child.tag == qn("w:p"):
                para = Paragraph(child, doc)
                if para.text.strip():
                    parts.append(para.text)
            elif child.tag == qn("w:tbl"):
                table = Table(child, doc)
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
        return "\n".join(parts).strip()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise HarryError("Reading PDFs needs pypdf. Install it with: pip install pypdf") from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    raise HarryError("Supported text documents are .txt, .md, .docx, and .pdf.")


def transcribe_audio(audio_path):
    if not audio_path:
        raise HarryError("Upload audio before asking for transcription.")
    try:
        import whisper
    except ImportError as error:
        raise HarryError("Local transcription needs openai-whisper. Install it with: pip install openai-whisper") from error
    model = whisper.load_model(os.environ.get("HARRY_WHISPER_MODEL", "medium.en"))
    result = model.transcribe(audio_path, language="en")
    return result.get("text", "").strip()


def _context_line(label, items):
    if not items:
        return f"{label}: none identified yet."
    names = ", ".join(f"{item['name']} ({item['suggested_id']})" for item in items)
    return f"{label}: {names}"


def _run_stage(provider, system_prompt, user_prompt, stage_name, warnings):
    """Runs one focused pass. A failure here doesn't abort the whole
    analysis -- it's logged as a warning (surfaced to the director via the
    questions list) and that stage simply contributes no items, so a
    transient failure in, say, the props pass doesn't also cost the
    characters and shots that already succeeded."""
    try:
        raw = _chat(provider, system_prompt, user_prompt)
        return _parse_items(raw)
    except HarryError as error:
        warnings.append(f"⚠️ {stage_name} pass failed: {error}")
        return []


def analyze(provider, title, source_text, source_path="", uploaded_path=None, era=""):
    if not source_text.strip():
        raise HarryError("Paste script/story text, or transcribe the uploaded audio first.")
    if not source_path:
        source_path = save_source(title, source_text, uploaded_path).get("source_id", "")
    notice = "The source stays local." if provider == "Ollama" else f"The source text will be sent to {provider} for this analysis."
    era_line = f"ERA / SETTING: {era}\n" if (era or "").strip() else ""
    base = f"PROJECT TITLE: {title or 'Untitled project'}\n{era_line}\nSOURCE:\n{source_text}\n\n{notice}\n"

    # Four separate, focused passes instead of one call trying to do
    # everything at once. A single call asking for characters, backdrops,
    # props, AND a full beat-by-beat shot breakdown in one JSON response
    # spreads the model's attention thin across very different tasks --
    # scene coverage in particular tends to get shortchanged since it's
    # both the largest sub-task and the last thing reasoned about. Each
    # stage below gets its own prompt, its own token budget, and knows
    # what earlier stages already found so it can reference them for
    # continuity without redefining them.
    warnings = []
    character_items = _run_stage(provider, CHARACTER_SYSTEM_PROMPT, base + "Return the JSON now.", "Characters", warnings)
    backdrop_items = _run_stage(
        provider, BACKDROP_SYSTEM_PROMPT,
        base + _context_line("Already-identified characters", character_items) + "\nReturn the JSON now.",
        "Backdrops", warnings,
    )
    prop_items = _run_stage(
        provider, PROP_SYSTEM_PROMPT,
        base + _context_line("Already-identified characters", character_items) + "\n"
        + _context_line("Already-identified backdrops", backdrop_items) + "\nReturn the JSON now.",
        "Props", warnings,
    )

    shot_user_prompt = (
        base + _context_line("Characters", character_items) + "\n"
        + _context_line("Backdrops", backdrop_items) + "\n"
        + _context_line("Props", prop_items) + "\nReturn the JSON now."
    )
    try:
        shot_plan = _parse_plan(_chat(provider, SHOT_SYSTEM_PROMPT, shot_user_prompt))
        shot_items = shot_plan["items"]
        summary = shot_plan["summary"]
        questions = shot_plan["questions"]
    except HarryError as error:
        warnings.append(f"⚠️ Shots pass failed: {error}")
        shot_items, summary, questions = [], "", []

    all_items = character_items + backdrop_items + prop_items + shot_items
    if not all_items:
        raise HarryError("Harry could not produce any recommendations. " + " ".join(warnings))

    if not summary:
        summary = (f"Identified {len(character_items)} character(s), {len(backdrop_items)} backdrop(s), "
                   f"{len(prop_items)} prop(s), and {len(shot_items)} shot(s) across separate focused passes.")

    plan = {"summary": summary, "questions": questions + warnings, "items": all_items}
    plan.update({"plan_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "title": title or "Untitled project", "provider": provider, "era": era or "", "created_at": _timestamp(), "source_path": source_path})
    LIBRARY_DIR.mkdir(exist_ok=True)
    (LIBRARY_DIR / f"plan_{plan['plan_id']}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def plan_to_rows(plan):
    return [[True, item["kind"], item["name"], item["suggested_id"], item["description"], item["continuity_note"], item["positive_prompt"], item["negative_prompt"], item["beat"], item["reused_from"]] for item in plan.get("items", [])]


CALL_SHEET_COLUMNS = ["Use", "Type", "Name", "Suggested ID", "Description", "Continuity note",
                      "Positive prompt", "Negative prompt", "Beat", "Chained from"]


def export_call_sheet(title, rows, plan):
    """Writes the current call sheet (as it stands in the review table --
    including any manual edits/unticks, not just the raw provider output)
    to a real .xlsx file for offline review or sharing. Two sheets:
    Call Sheet (the rows themselves) and Summary (Harry's reading + any
    clarification questions), so the export is self-contained -- someone
    reviewing it doesn't need the app open to understand the context."""
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter

    LIBRARY_DIR.mkdir(exist_ok=True)
    export_dir = LIBRARY_DIR / "exports"
    export_dir.mkdir(exist_ok=True)

    wb = Workbook()
    sheet = wb.active
    sheet.title = "Call Sheet"
    sheet.append(CALL_SHEET_COLUMNS)
    for row in (rows or []):
        # rows may come from the Gradio Dataframe as-is; pad/trim defensively
        # in case of manual edits that changed column count somehow.
        padded = list(row) + [""] * (len(CALL_SHEET_COLUMNS) - len(row))
        sheet.append(padded[:len(CALL_SHEET_COLUMNS)])
    for i, _ in enumerate(CALL_SHEET_COLUMNS, start=1):
        sheet.column_dimensions[get_column_letter(i)].width = 24

    summary_sheet = wb.create_sheet("Summary")
    summary_sheet.append(["Title", title or "Untitled project"])
    summary_sheet.append(["Exported", _timestamp()])
    summary_sheet.append(["Provider", (plan or {}).get("provider", "")])
    summary_sheet.append(["Era / setting", (plan or {}).get("era", "")])
    summary_sheet.append([])
    summary_sheet.append(["Harry's reading"])
    summary_sheet.append([(plan or {}).get("summary", "")])
    summary_sheet.append([])
    summary_sheet.append(["Clarification questions"])
    for question in (plan or {}).get("questions", []):
        summary_sheet.append([question])
    summary_sheet.column_dimensions["A"].width = 90

    safe_title = _slug(title) or "call_sheet"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_title}_call_sheet.xlsx"
    out_path = export_dir / filename
    wb.save(out_path)
    return str(out_path)


def rows_to_plan(rows, template=None):
    items = []
    for row in rows or []:
        if not row or len(row) < 10 or not row[0]:
            continue
        items.append(dict(zip(("selected", "kind", "name", "suggested_id", "description", "continuity_note", "positive_prompt", "negative_prompt", "beat", "reused_from"), row)))
    return {"summary": (template or {}).get("summary", ""), "questions": (template or {}).get("questions", []), "items": items}


def preset_path():
    return LIBRARY_DIR / "approved_build_presets.json"


def save_presets(plan):
    LIBRARY_DIR.mkdir(exist_ok=True)
    preset_path().write_text(json.dumps(plan, indent=2), encoding="utf-8")


def load_presets():
    if not preset_path().exists():
        return []
    return json.loads(preset_path().read_text(encoding="utf-8")).get("items", [])


def preset_choices():
    return [(f"{item.get('kind')}: {item.get('name')}", index) for index, item in enumerate(load_presets())]


def checklist_choices():
    return [f"{item.get('kind')}: {item.get('name')}" for item in load_presets()]
