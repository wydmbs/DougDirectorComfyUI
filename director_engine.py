"""
director_engine.py -- the headless core of the harness.

Everything here runs without Gradio, without module-level mutable state, and
without touching the UI. The Build tab calls it; the assistant director calls
the same functions through its tools. One implementation, two callers, so the
agent can never drift from what the buttons do.

Config arrives as an argument rather than being read from a module global. That
matters once more than one thing is running: a background build must not change
behaviour because someone switched project in the UI mid-run.
"""

import io
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image, ImageDraw

import sheet_composer as composer
import storyboard_store as store
from comfy_client import ComfyClient, ComfyClientError, apply_node_overrides

SHEET_TYPES = ("CHARACTER", "BACKDROP", "PROP")
ENTRY_PREFIXES = ("CHARACTER:", "BACKDROP:", "PROP:")
CONCEPT_STAGE = "Concept (mashup)"
PANEL_STAGE = "Sheet panel"


class EngineError(Exception):
    """Something the caller asked for cannot be done, with a reason worth showing."""


@dataclass
class Variant:
    """One generated candidate image."""
    image_path: str
    seed: int
    label: str = ""

    def as_gallery_item(self):
        return (self.image_path, f"seed {self.seed}")


@dataclass
class GenerationResult:
    variants: list
    warnings: list

    @property
    def ok(self) -> bool:
        return bool(self.variants)

    def as_gallery(self):
        return [v.as_gallery_item() for v in self.variants]


# ---------------------------------------------------------------- image output


def _mock_image(label: str, prompt: str, seed: int) -> Image.Image:
    """A deterministic placeholder so the whole flow is exercisable with no GPU."""
    rng = random.Random(seed)
    img = Image.new("RGB", (512, 512), (rng.randint(30, 90), rng.randint(40, 100), rng.randint(60, 120)))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 492, 492], outline=(240, 240, 240), width=3)
    draw.text((36, 40), f"{label}", fill=(255, 255, 255))
    draw.text((36, 70), f"seed {seed}", fill=(230, 230, 230))
    draw.text((36, 100), (prompt or "")[:300], fill=(215, 215, 215))
    return img


def _save_image(img: Image.Image, images_dir: str, label: str, seed: int) -> str:
    os.makedirs(images_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (label or "frame"))
    path = os.path.join(images_dir, f"{safe}__{seed}.png")
    img.save(path)
    return path


# ------------------------------------------------------------------ generation


def generation_label(entry_id: str, entry_type: str, build_stage: str, panel_key: str) -> str:
    """The filename stem for a generation, matching the Build tab's scheme."""
    if entry_type not in SHEET_TYPES:
        return entry_id
    if build_stage == CONCEPT_STAGE:
        return f"{entry_id}__concept"
    return f"{entry_id}__{panel_key}"


def generate(cfg, label: str, prompt_positive: str, prompt_negative: str = "",
             base_seed=None, n_variants: int = 4,
             reference_image_path: str = "", scene_image_path: str = "",
             on_progress: Optional[Callable[[str], None]] = None) -> GenerationResult:
    """Render n variants, either mocked or through a live ComfyUI.

    reference_image_path is the character turnaround, which drives IP-Adapter so
    the face survives into this composition. scene_image_path is the environment
    concept, which anchors the background through img2img. Either can be passed
    while conditioning is off; that isn't an error, but it is reported as a
    warning so a run never silently pretends a reference was honoured.
    """
    base_seed = int(base_seed) if base_seed not in (None, "") else random.randint(0, 2**31 - 1)
    n_variants = max(1, min(int(n_variants), 8))

    variants, warnings = [], []
    client = workflow = None

    if not cfg.mock_mode:
        if not cfg.workflow_json_path or not os.path.exists(cfg.workflow_json_path):
            raise EngineError(
                "No workflow file is set up yet. Go to Setup and either upload one or turn Mock Mode back on."
            )
        with open(cfg.workflow_json_path, "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
        client = ComfyClient(cfg.comfyui_url)

    for index in range(n_variants):
        seed = base_seed + index
        if on_progress:
            on_progress(f"variant {index + 1}/{n_variants} (seed {seed})")

        if cfg.mock_mode:
            path = _save_image(_mock_image(label, prompt_positive, seed), cfg.images_dir, label, seed)
            variants.append(Variant(path, seed, label))
            continue

        try:
            work = apply_node_overrides(workflow, cfg.node_mapping, prompt_positive, prompt_negative, seed)
            if reference_image_path or scene_image_path:
                from reference_conditioning import apply_reference
                work, note = apply_reference(work, cfg, reference_image_path, scene_image_path)
                if note and note not in warnings:
                    warnings.append(note)
            prompt_id = client.queue_prompt(work)
            history = client.wait_for_completion(prompt_id)
            refs = client.extract_image_refs(history)
            if not refs:
                warnings.append(f"seed {seed}: no images returned")
                continue
            filename, subfolder, folder_type = refs[0]
            img = Image.open(io.BytesIO(client.fetch_image_bytes(filename, subfolder, folder_type)))
            variants.append(Variant(_save_image(img, cfg.images_dir, label, seed), seed, label))
        except ComfyClientError as error:
            warnings.append(f"seed {seed}: {error}")

    return GenerationResult(variants, warnings)


# -------------------------------------------------------------------- registry


def list_entries(cfg) -> list:
    return store.list_entries(cfg.storyboard_path)


def get_entry(cfg, entry_id: str) -> dict:
    return store.get_entry(cfg.storyboard_path, entry_id)


def lock_concept(cfg, entry_id: str, entry_type: str, beat: str, description: str,
                 prompt_positive: str, prompt_negative: str, seed, image_path: str,
                 reused_from: str = "", note: str = "") -> str:
    if not entry_id:
        raise EngineError("An entry needs a name before it can be locked.")
    if not image_path:
        raise EngineError("Pick a winning variant before locking.")
    store.lock_entry(
        path=cfg.storyboard_path, entry_id=entry_id, entry_type=entry_type, beat=beat,
        description=description, prompt_positive=prompt_positive,
        prompt_negative_add=prompt_negative,
        model=os.path.basename(cfg.workflow_json_path or "") or ("mock" if cfg.mock_mode else ""),
        seed=seed, reused_from=reused_from, image_path=image_path, note=note,
    )
    return f"Locked {entry_id}."


def lock_panel(cfg, entry_id: str, panel_key: str, prompt_positive: str,
               prompt_negative: str, seed, image_path: str, note: str = "") -> str:
    if not entry_id or not panel_key:
        raise EngineError("Locking a panel needs both an entry name and a panel key.")
    if not image_path:
        raise EngineError("Pick a winning variant before locking.")
    store.lock_panel(
        path=cfg.storyboard_path, entry_id=entry_id, panel_key=panel_key,
        prompt_positive=prompt_positive, prompt_negative_add=prompt_negative,
        seed=seed, image_path=image_path, note=note,
    )
    return f"Locked {entry_id} panel '{panel_key}'."


# ----------------------------------------------------------------- sheet state


def sheet_status(cfg, entry_id: str, entry_type: str) -> dict:
    """What's done and what's outstanding for one sheet-shaped asset."""
    if entry_type not in SHEET_TYPES:
        raise EngineError(f"'{entry_type}' does not use a panel sheet.")
    template = composer.get_template(entry_type)
    filled = store.get_panel_images(cfg.storyboard_path, entry_id)
    done = [p["key"] for p in template if filled.get(p["key"])]
    missing = [p["key"] for p in template if not filled.get(p["key"])]
    concept = store.get_entry(cfg.storyboard_path, entry_id) or {}
    return {
        "entry_id": entry_id,
        "entry_type": entry_type,
        "total_panels": len(template),
        "locked_panels": done,
        "missing_panels": missing,
        "complete": not missing,
        "concept_locked": bool(concept.get("image_path")),
        "concept_image_path": concept.get("image_path", ""),
        "labels": {p["key"]: p["label"] for p in template},
    }


def render_sheet(cfg, entry_id: str, entry_type: str) -> str:
    if entry_type not in SHEET_TYPES:
        raise EngineError("Composite sheets are only for CHARACTER, BACKDROP and PROP entries.")
    panel_images = store.get_panel_images(cfg.storyboard_path, entry_id)
    concept = store.get_entry(cfg.storyboard_path, entry_id) or {}
    safe_id = entry_id.strip().replace(":", "_")
    out_dir = os.path.join(cfg.images_dir, "sheets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_id}_sheet.png")
    composer.render_sheet(entry_id, entry_type, panel_images, out_path,
                          concept_image_path=concept.get("image_path") or None)
    return out_path


def save_sheet(cfg, entry_id: str, entry_type: str, composite_path: str, note: str = "") -> str:
    store.set_composite_image(cfg.storyboard_path, entry_id.strip(), entry_type,
                              entry_type, composite_path, note)
    return f"Printed the reel for {entry_id}."


# ------------------------------------------------------------------- overview


def project_overview(cfg) -> dict:
    """One cheap call that answers 'where is this project up to'.

    This is the assistant director's orientation step and the data behind the
    progress rail, so both always agree.
    """
    rows = store.list_entries(cfg.storyboard_path)
    by_type = {}
    for row in rows:
        by_type.setdefault(row.get("entry_type") or "?", []).append(row)

    sheets = []
    for row in rows:
        etype = row.get("entry_type")
        if etype in SHEET_TYPES:
            try:
                sheets.append(sheet_status(cfg, row.get("entry_id"), etype))
            except EngineError:
                continue

    total_panels = sum(s["total_panels"] for s in sheets)
    locked_panels = sum(len(s["locked_panels"]) for s in sheets)
    return {
        "entries": len(rows),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "sheets": sheets,
        "sheets_complete": sum(1 for s in sheets if s["complete"]),
        "total_panels": total_panels,
        "locked_panels": locked_panels,
        "panel_progress": (locked_panels / total_panels) if total_panels else 0.0,
        "beats": len(store.list_beats(cfg.storyboard_path)),
        "characters": len(store.list_characters(cfg.storyboard_path)),
        "mock_mode": bool(cfg.mock_mode),
    }


def next_unfinished_panel(cfg) -> Optional[dict]:
    """The next thing worth working on, or None when every sheet is complete.

    Resumability lives here: the agent asks the registry what remains rather
    than tracking its own progress, so a crashed or cancelled run picks up
    exactly where it stopped with no bookkeeping to reconcile.
    """
    for row in store.list_entries(cfg.storyboard_path):
        etype = row.get("entry_type")
        if etype not in SHEET_TYPES:
            continue
        status = sheet_status(cfg, row.get("entry_id"), etype)
        if status["missing_panels"]:
            key = status["missing_panels"][0]
            return {
                "entry_id": status["entry_id"],
                "entry_type": etype,
                "panel_key": key,
                "panel_label": status["labels"].get(key, key),
                "remaining": len(status["missing_panels"]),
                "concept_image_path": status["concept_image_path"],
            }
    return None
