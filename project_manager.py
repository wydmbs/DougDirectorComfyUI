import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECTS_DIR = Path(__file__).resolve().parent / "projects"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _slug(value):
    return "_".join(re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).split()) or "untitled_project"


def _metadata(project_id):
    return PROJECTS_DIR / project_id / "project.json"


def _read(project_id):
    path = _metadata(project_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def list_projects():
    if not PROJECTS_DIR.exists():
        return []
    values = [_read(path.name) for path in PROJECTS_DIR.iterdir() if _metadata(path.name).exists()]
    return sorted(values, key=lambda value: value.get("updated_at", ""), reverse=True)


def create_project(title, version="v1", description=""):
    PROJECTS_DIR.mkdir(exist_ok=True)
    base = _slug(title)
    project_id = base
    index = 2
    while (PROJECTS_DIR / project_id).exists():
        project_id = f"{base}_{index}"
        index += 1
    root = PROJECTS_DIR / project_id
    (root / "images").mkdir(parents=True)
    (root / "harry_library").mkdir()
    data = {"project_id": project_id, "title": title.strip() or "Untitled Project", "version": version.strip() or "v1", "description": description.strip(), "created_at": _now(), "updated_at": _now()}
    _metadata(project_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def ensure_active(project_id=""):
    if project_id and _metadata(project_id).exists():
        return project_id
    existing = list_projects()
    return existing[0]["project_id"] if existing else create_project("Untitled Project")["project_id"]


def get_project(project_id):
    return _read(project_id)


def choices():
    return [(f"{item['title']} — {item.get('version') or 'v1'}", item["project_id"]) for item in list_projects()]


def workspace(project_id):
    root = PROJECTS_DIR / project_id
    return {"storyboard": str(root / "storyboard.xlsx"), "images": str(root / "images"), "library": root / "harry_library"}
