"""
director_tools.py -- what the assistant director can actually do.

Every tool here calls director_engine, which is the same code path the Build
tab's buttons use. There is deliberately no separate "agent version" of
generating or locking, because two implementations of the same action drift
apart and then the agent quietly does something the UI never would.

Safety semantics, per the permission policy:
  reading            -- not mutating, always allowed
  generating         -- mutating but routine: costs GPU time, changes nothing
  locking            -- mutating and routine: the ordinary business of building
  installing         -- always_confirm: runs commands on the director's PC
"""

import json
import os

import director_engine as engine
import sheet_composer as composer
import storyboard_store as store

from .registry import Tool


def build_tools(cfg, generate_fn=None, critique_fn=None) -> list:
    """Bind the tool set to one config.

    generate_fn and critique_fn are injectable so tests can run the whole loop
    without a GPU or a vision model.
    """
    generate = generate_fn or engine.generate

    # ------------------------------------------------------------- reading

    def project_overview() -> str:
        overview = engine.project_overview(cfg)
        lines = [
            f"Entries: {overview['entries']} ({', '.join(f'{k}:{v}' for k, v in overview['by_type'].items()) or 'none yet'})",
            f"Panels locked: {overview['locked_panels']}/{overview['total_panels']}"
            f" ({overview['panel_progress']*100:.0f}%)",
            f"Sheets complete: {overview['sheets_complete']}/{len(overview['sheets'])}",
            f"Beats: {overview['beats']}   Characters: {overview['characters']}",
            f"Mock mode: {'on (no real generation)' if overview['mock_mode'] else 'off (live ComfyUI)'}",
        ]
        pending = engine.next_unfinished_panel(cfg)
        if pending:
            lines.append(
                f"Next unfinished: {pending['entry_id']} panel '{pending['panel_key']}'"
                f" ({pending['panel_label']}), {pending['remaining']} left on that sheet")
        else:
            lines.append("Next unfinished: nothing — every sheet is complete.")
        return "\n".join(lines)

    def list_entries(entry_type: str = "") -> str:
        rows = engine.list_entries(cfg)
        if entry_type:
            rows = [r for r in rows if (r.get("entry_type") or "").upper() == entry_type.upper()]
        if not rows:
            return "No entries yet."
        return json.dumps([
            {"entry_id": r.get("entry_id"), "entry_type": r.get("entry_type"),
             "beat": r.get("beat"), "locked": bool(r.get("locked")),
             "description": (r.get("description") or "")[:200],
             "reused_from": r.get("reused_from") or ""}
            for r in rows], indent=2)

    def get_entry(entry_id: str) -> str:
        row = engine.get_entry(cfg, entry_id)
        if not row:
            return f"No entry called '{entry_id}'."
        return json.dumps(row, indent=2, default=str)

    def sheet_status(entry_id: str, entry_type: str) -> str:
        status = engine.sheet_status(cfg, entry_id, entry_type)
        return json.dumps({
            "entry_id": status["entry_id"],
            "complete": status["complete"],
            "concept_locked": status["concept_locked"],
            "locked_panels": status["locked_panels"],
            "missing_panels": [
                {"key": k, "label": status["labels"].get(k, k)} for k in status["missing_panels"]],
        }, indent=2)

    def list_panel_template(entry_type: str) -> str:
        try:
            template = composer.get_template(entry_type)
        except Exception as error:  # noqa: BLE001
            return f"No panel template for '{entry_type}': {error}"
        return json.dumps([{"key": p["key"], "label": p["label"]} for p in template], indent=2)

    def list_beats() -> str:
        beats = store.list_beats(cfg.storyboard_path)
        return json.dumps(beats, indent=2, default=str) if beats else "No beats imported yet."

    def list_characters() -> str:
        rows = store.list_characters(cfg.storyboard_path)
        return json.dumps(rows, indent=2, default=str) if rows else "No characters registered yet."

    # ---------------------------------------------------------- generating

    def generate_variants(entry_id: str, prompt_positive: str, prompt_negative: str = "",
                          entry_type: str = "", panel_key: str = "", n_variants: int = 0,
                          use_reference: bool = True, base_seed: int = 0) -> str:
        """Render candidates for a concept or a sheet panel."""
        entry_type = entry_type or (engine.get_entry(cfg, entry_id) or {}).get("entry_type", "") or "SHOT"
        stage = engine.PANEL_STAGE if panel_key else engine.CONCEPT_STAGE
        label = engine.generation_label(entry_id, entry_type, stage, panel_key)

        reference = ""
        if use_reference and panel_key:
            concept = engine.get_entry(cfg, entry_id) or {}
            reference = concept.get("image_path") or ""

        result = generate(
            cfg, label, prompt_positive, prompt_negative,
            base_seed=base_seed or None,
            n_variants=n_variants or cfg.agent.variants_per_generation,
            reference_image_path=reference,
        )
        if not result.ok:
            return "No variants came back. " + " | ".join(result.warnings)
        payload = {
            "label": label,
            "variants": [{"index": i, "image_path": v.image_path, "seed": v.seed}
                         for i, v in enumerate(result.variants)],
        }
        if result.warnings:
            payload["warnings"] = result.warnings
        return json.dumps(payload, indent=2)

    def critique_variants(entry_id: str, image_paths: list, criteria: str = "") -> str:
        """Judge candidates against the locked concept and continuity note."""
        if critique_fn is None:
            return ("No critic is configured, so pick on the evidence you have and say why. "
                    "Variant image paths: " + ", ".join(image_paths or []))
        return critique_fn(cfg, entry_id, image_paths, criteria)

    # ------------------------------------------------------------- locking

    def lock_concept(entry_id: str, entry_type: str, image_path: str, seed: int,
                     prompt_positive: str, prompt_negative: str = "", beat: str = "",
                     description: str = "", reused_from: str = "", note: str = "") -> str:
        return engine.lock_concept(
            cfg, entry_id, entry_type, beat, description, prompt_positive,
            prompt_negative, seed, image_path, reused_from,
            note=f"[HARRY] {note}" if note else "[HARRY] locked by the assistant director")

    def lock_panel(entry_id: str, panel_key: str, image_path: str, seed: int,
                   prompt_positive: str, prompt_negative: str = "", note: str = "") -> str:
        return engine.lock_panel(
            cfg, entry_id, panel_key, prompt_positive, prompt_negative, seed, image_path,
            note=f"[HARRY] {note}" if note else "[HARRY] locked by the assistant director")

    def render_sheet(entry_id: str, entry_type: str) -> str:
        path = engine.render_sheet(cfg, entry_id, entry_type)
        status = engine.sheet_status(cfg, entry_id, entry_type)
        return (f"Rendered {entry_id}: {len(status['locked_panels'])}/{status['total_panels']}"
                f" panels filled. Sheet at {path}")

    def print_sheet(entry_id: str, entry_type: str, note: str = "") -> str:
        path = engine.render_sheet(cfg, entry_id, entry_type)
        return engine.save_sheet(cfg, entry_id, entry_type, path,
                                 note=f"[HARRY] {note}" if note else "[HARRY] reel printed")

    def register_character(trigger_id: str, display_name: str, working_note: str = "") -> str:
        store.create_character(cfg.storyboard_path, trigger_id, display_name, working_note)
        return f"Registered {trigger_id} ({display_name})."

    # --------------------------------------------------------------- schema

    def obj(**properties):
        return {"type": "object", "properties": properties,
                "required": [k for k, v in properties.items() if v.pop("_required", False)]}

    def s(desc, required=False):
        return {"type": "string", "description": desc, "_required": required}

    def i(desc, required=False):
        return {"type": "integer", "description": desc, "_required": required}

    def b(desc):
        return {"type": "boolean", "description": desc}

    return [
        Tool("project_overview",
             "Where the project stands: entry counts, panel progress, and the next unfinished panel. Call this first.",
             obj(), project_overview),
        Tool("list_entries",
             "List registry entries, optionally filtered to one entry_type (CHARACTER, BACKDROP, PROP, SHOT).",
             obj(entry_type=s("Optional filter")), list_entries),
        Tool("get_entry",
             "Full registry row for one entry_id, including its prompts, seed, continuity notes and lock state.",
             obj(entry_id=s("e.g. CHARACTER:pig", True)), get_entry),
        Tool("sheet_status",
             "Which panels of a sheet are locked and which are still missing.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 entry_type=s("CHARACTER, BACKDROP or PROP", True)), sheet_status),
        Tool("list_panel_template",
             "The full panel list for a sheet type, with each panel's key and label.",
             obj(entry_type=s("CHARACTER, BACKDROP or PROP", True)), list_panel_template),
        Tool("list_beats", "The narration beats and their timings.", obj(), list_beats),
        Tool("list_characters", "Registered characters and their LoRA trigger ids.", obj(), list_characters),

        Tool("generate_variants",
             "Render candidate images for a concept or a sheet panel. Returns each variant's path and seed. "
             "Give panel_key when building a sheet panel; leave it out for the concept.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 prompt_positive=s("The full positive prompt", True),
                 prompt_negative=s("Things to avoid"),
                 entry_type=s("CHARACTER, BACKDROP, PROP or SHOT"),
                 panel_key=s("Panel key when building a sheet panel"),
                 n_variants=i("How many candidates (1-8)"),
                 use_reference=b("Condition on the locked concept image when available"),
                 base_seed=i("Fix the starting seed for reproducibility")),
             generate_variants, mutating=True, routine=True, timeout_s=1800),
        Tool("critique_variants",
             "Judge candidate images against the locked concept and continuity note, and recommend one.",
             obj(entry_id=s("The entry being judged", True),
                 image_paths={"type": "array", "items": {"type": "string"},
                              "description": "Variant image paths", "_required": True},
                 criteria=s("What matters most for this asset")),
             critique_variants, timeout_s=600),

        Tool("lock_concept",
             "Lock the winning concept image for an entry. This is a real decision recorded in the registry.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 entry_type=s("CHARACTER, BACKDROP, PROP or SHOT", True),
                 image_path=s("Winning variant's path", True),
                 seed=i("Winning variant's seed", True),
                 prompt_positive=s("Prompt that produced it", True),
                 prompt_negative=s("Negative prompt used"),
                 beat=s("Story beat this belongs to"),
                 description=s("What it is and why it matters"),
                 reused_from=s("Prior entry_id this was chained from"),
                 note=s("Why this one won")),
             lock_concept, mutating=True, routine=True),
        Tool("lock_panel",
             "Lock the winning image for one panel of a sheet.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 panel_key=s("Which panel", True),
                 image_path=s("Winning variant's path", True),
                 seed=i("Winning variant's seed", True),
                 prompt_positive=s("Prompt that produced it", True),
                 prompt_negative=s("Negative prompt used"),
                 note=s("Why this one won")),
             lock_panel, mutating=True, routine=True),
        Tool("render_sheet",
             "Render the composite reference sheet for an entry and report how full it is.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 entry_type=s("CHARACTER, BACKDROP or PROP", True)),
             render_sheet, mutating=True, routine=True),
        Tool("print_sheet",
             "Render and record the composite sheet against the entry — the 'print it' step.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 entry_type=s("CHARACTER, BACKDROP or PROP", True),
                 note=s("Any note to record")),
             print_sheet, mutating=True, routine=True),
        Tool("register_character",
             "Register a character and its CHARACTER:name LoRA trigger id.",
             obj(trigger_id=s("Must start with CHARACTER:", True),
                 display_name=s("Human readable name", True),
                 working_note=s("Working note")),
             register_character, mutating=True, routine=True),
    ]
