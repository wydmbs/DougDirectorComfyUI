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
OPENSCOUT_CONFIG = Path.home() / "OpenScout" / "openscout" / "config.yaml"

SYSTEM_PROMPT = """You are Harry the Advisor, a thoughtful pre-production advisor for AI-assisted film.
Read the supplied script, story, poem, or narration transcript. Recommend only the reusable preparation artifacts that will genuinely help production: characters, recurring props, recurring backdrops, and key shot/beat drafts. Prefer a concise, useful list over exhaustive fragmentation.

Return JSON only, with this exact shape:
{
  "summary": "short production reading",
  "questions": ["only essential clarification questions; empty when the material is clear"],
  "items": [
    {
      "kind": "CHAR|MASTER|PROP|SHOT",
      "name": "human readable name",
      "suggested_id": "CHAR:snake_case|MASTER:snake_case|PROP:snake_case|beat.shot",
      "description": "what it is and why it matters",
      "continuity_note": "visual or narrative continuity guidance",
      "positive_prompt": "initial target-model-aware visual prompt",
      "negative_prompt": "things to avoid",
      "beat": "optional story section",
      "reused_from": "optional prior item id"
    }
  ]
}
Do not invent named people or key objects without textual evidence. Questions are exceptional: ask only when a decision materially changes the prep plan."""


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


def _chat(provider, user_prompt):
    if provider == "Azure OpenAI":
        azure = _load_openscout_config().get("azure", {})
        key = os.environ.get(azure.get("api_key_env", "AZURE_FOUNDRY_API_KEY"))
        if not key:
            raise HarryError("Azure API key is not available to this app process. Restart the app after setting the OpenScout Azure key environment variable.")
        endpoint = azure.get("endpoint", "").rstrip("/")
        url = f"{endpoint}/openai/v1/chat/completions"
        response = _request(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, {
            "model": azure.get("deployment"), "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], "max_completion_tokens": 5000,
        })
        return response["choices"][0]["message"]["content"]
    if provider == "Claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise HarryError("ANTHROPIC_API_KEY is not available to this app process.")
        response = _request("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, {
            "model": os.environ.get("HARRY_CLAUDE_MODEL", "claude-sonnet-4-5"), "max_tokens": 5000,
            "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": user_prompt}],
        })
        return "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")
    if provider == "OpenAI":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HarryError("OPENAI_API_KEY is not available to this app process.")
        response = _request("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, {
            "model": os.environ.get("HARRY_OPENAI_MODEL", "gpt-4.1"), "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], "response_format": {"type": "json_object"},
        })
        return response["choices"][0]["message"]["content"]
    if provider == "Grok":
        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise HarryError("XAI_API_KEY is not available to this app process.")
        response = _request("https://api.x.ai/v1/chat/completions", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, {
            "model": os.environ.get("HARRY_GROK_MODEL", "grok-3"), "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], "response_format": {"type": "json_object"},
        })
        return response["choices"][0]["message"]["content"]
    response = _request(os.environ.get("HARRY_OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat", {"Content-Type": "application/json"}, {
        "model": os.environ.get("HARRY_OLLAMA_MODEL", "qwen2.5:7b"), "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], "format": "json",
    })
    return response["message"]["content"]


def _parse_plan(raw):
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        plan = json.loads(fenced)
    except json.JSONDecodeError as error:
        raise HarryError("Harry returned a response that was not valid JSON. Try again or choose another provider.") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("items"), list):
        raise HarryError("Harry's response did not contain a usable recommendation list.")
    clean = []
    for item in plan["items"]:
        if not isinstance(item, dict) or item.get("kind") not in {"CHAR", "MASTER", "PROP", "SHOT"}:
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        kind = item["kind"]
        prefix = {"CHAR": "CHAR", "MASTER": "MASTER", "PROP": "PROP"}.get(kind)
        suggested = str(item.get("suggested_id") or "").strip()
        if prefix and not suggested.startswith(prefix + ":"):
            suggested = f"{prefix}:{_slug(name)}"
        if kind == "SHOT" and not suggested:
            suggested = _slug(name) or "draft_shot"
        item["suggested_id"] = suggested
        clean.append({key: str(item.get(key) or "") for key in ("kind", "name", "suggested_id", "description", "continuity_note", "positive_prompt", "negative_prompt", "beat", "reused_from")})
    plan["items"] = clean
    plan["summary"] = str(plan.get("summary") or "")
    plan["questions"] = [str(question) for question in plan.get("questions", []) if str(question).strip()]
    return plan


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


def analyze(provider, title, source_text, source_path="", uploaded_path=None):
    if not source_text.strip():
        raise HarryError("Paste script/story text, or transcribe the uploaded audio first.")
    if not source_path:
        source_path = save_source(title, source_text, uploaded_path).get("source_id", "")
    notice = "The source stays local." if provider == "Ollama" else f"The source text will be sent to {provider} for this analysis."
    user_prompt = f"PROJECT TITLE: {title or 'Untitled project'}\n\nSOURCE:\n{source_text}\n\n{notice}\nReturn the JSON recommendation now."
    plan = _parse_plan(_chat(provider, user_prompt))
    plan.update({"plan_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "title": title or "Untitled project", "provider": provider, "created_at": _timestamp(), "source_path": source_path})
    LIBRARY_DIR.mkdir(exist_ok=True)
    (LIBRARY_DIR / f"plan_{plan['plan_id']}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def plan_to_rows(plan):
    return [[True, item["kind"], item["name"], item["suggested_id"], item["description"], item["continuity_note"], item["positive_prompt"], item["negative_prompt"], item["beat"], item["reused_from"]] for item in plan.get("items", [])]


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
