"""
storyboard_store.py — shared Excel data backbone for the Director Harness.

Assets holds one row per shot or reusable asset. Panels holds independently
locked sheet panels. References holds mood-board images. Beats holds
narration timing rows. Re-locking an asset or panel updates its row and
appends to its decision log.
"""

import os
from datetime import datetime, timezone

from openpyxl import Workbook, load_workbook

ASSETS_SHEET = "Assets"
ASSETS_COLUMNS = [
    "entry_id", "entry_type", "beat", "description",
    "prompt_positive", "prompt_negative_add",
    "model", "seed", "reused_from", "locked", "image_path", "notes",
    "template", "composite_image_path",
]

PANELS_SHEET = "Panels"
PANELS_COLUMNS = [
    "entry_id", "panel_key", "prompt_positive", "prompt_negative_add",
    "seed", "image_path", "locked", "notes",
]

REFERENCES_SHEET = "References"
REFERENCES_COLUMNS = ["entry_id", "image_path", "note", "added_at"]

BEATS_SHEET = "Beats"
BEATS_COLUMNS = ["order", "beat", "start_s", "end_s", "duration_s", "source", "notes"]

CHARACTERS_SHEET = "Characters"
CHARACTERS_COLUMNS = ["trigger_id", "display_name", "working_note", "created_at", "updated_at"]


def _open_or_create_workbook(path: str) -> Workbook:
    if os.path.exists(path):
        return load_workbook(path)
    workbook = Workbook()
    workbook.remove(workbook.active)
    return workbook


def _ensure_sheet(workbook: Workbook, sheet_name: str, columns: list):
    if sheet_name not in workbook.sheetnames:
        worksheet = workbook.create_sheet(sheet_name)
        worksheet.append(columns)
        return worksheet

    worksheet = workbook[sheet_name]
    existing = [cell.value for cell in worksheet[1]] if worksheet.max_row >= 1 else []
    for column in columns:
        if column not in existing:
            worksheet.cell(1, worksheet.max_column + 1 if worksheet.max_row >= 1 else 1, column)
            existing.append(column)
    return worksheet


def _col_index(worksheet, name: str) -> int:
    for index, cell in enumerate(worksheet[1], start=1):
        if cell.value == name:
            return index
    raise KeyError(f"Column '{name}' not found in '{worksheet.title}' sheet header row.")


def _find_row(worksheet, id_col: int, key: str, id_col2: int = None, key2: str = None):
    for row in range(2, worksheet.max_row + 1):
        value = worksheet.cell(row, id_col).value
        if value is None or str(value) != str(key):
            continue
        if id_col2 is not None:
            value2 = worksheet.cell(row, id_col2).value
            if value2 is None or str(value2) != str(key2):
                continue
        return row
    return None


def _read_sheet_rows(path: str, sheet_name: str, columns: list) -> list:
    if not os.path.exists(path):
        return []
    workbook = load_workbook(path)
    if sheet_name not in workbook.sheetnames:
        return []
    worksheet = workbook[sheet_name]
    header = [cell.value for cell in worksheet[1]] if worksheet.max_row >= 1 else []
    rows = []
    for row in range(2, worksheet.max_row + 1):
        values = [worksheet.cell(row, column).value for column in range(1, len(header) + 1)]
        if all(value is None for value in values):
            continue
        record = dict(zip(header, values))
        rows.append({column: record.get(column) for column in columns})
    return rows


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def list_entries(path: str) -> list:
    return _read_sheet_rows(path, ASSETS_SHEET, ASSETS_COLUMNS)


def get_entry(path: str, entry_id: str) -> dict:
    for row in list_entries(path):
        if str(row.get("entry_id")) == str(entry_id):
            return row
    return {}


def lock_entry(path: str, entry_id: str, entry_type: str, beat: str, description: str,
               prompt_positive: str, prompt_negative_add: str, model: str, seed,
               reused_from: str, image_path: str, note: str) -> None:
    workbook = _open_or_create_workbook(path)
    worksheet = _ensure_sheet(workbook, ASSETS_SHEET, ASSETS_COLUMNS)
    id_col = _col_index(worksheet, "entry_id")
    row = _find_row(worksheet, id_col, entry_id)
    note_line = f"[{_timestamp()}] {note}" if note else f"[{_timestamp()}] locked"

    if row is None:
        row = worksheet.max_row + 1
        worksheet.cell(row, id_col, entry_id)
        existing_notes = ""
    else:
        existing_notes = worksheet.cell(row, _col_index(worksheet, "notes")).value or ""

    values = {
        "entry_type": entry_type,
        "beat": beat,
        "description": description,
        "prompt_positive": prompt_positive,
        "prompt_negative_add": prompt_negative_add,
        "model": model,
        "seed": seed,
        "reused_from": reused_from,
        "locked": True,
        "image_path": image_path,
        "notes": (existing_notes + "\n" + note_line).strip() if existing_notes else note_line,
    }
    for column, value in values.items():
        worksheet.cell(row, _col_index(worksheet, column), value)
    workbook.save(path)


def set_composite_image(path: str, entry_id: str, entry_type: str, template: str,
                        composite_image_path: str, note: str = "") -> None:
    workbook = _open_or_create_workbook(path)
    worksheet = _ensure_sheet(workbook, ASSETS_SHEET, ASSETS_COLUMNS)
    id_col = _col_index(worksheet, "entry_id")
    row = _find_row(worksheet, id_col, entry_id)
    note_line = f"[{_timestamp()}] {note}" if note else f"[{_timestamp()}] composite sheet rendered"

    if row is None:
        row = worksheet.max_row + 1
        worksheet.cell(row, id_col, entry_id)
        existing_notes = ""
    else:
        existing_notes = worksheet.cell(row, _col_index(worksheet, "notes")).value or ""

    values = {
        "entry_type": entry_type,
        "locked": True,
        "template": template,
        "composite_image_path": composite_image_path,
        "notes": (existing_notes + "\n" + note_line).strip() if existing_notes else note_line,
    }
    for column, value in values.items():
        worksheet.cell(row, _col_index(worksheet, column), value)
    workbook.save(path)


def list_panels(path: str, entry_id: str) -> list:
    return [row for row in _read_sheet_rows(path, PANELS_SHEET, PANELS_COLUMNS)
            if str(row.get("entry_id")) == str(entry_id)]


def get_panel_images(path: str, entry_id: str) -> dict:
    return {row["panel_key"]: row["image_path"] for row in list_panels(path, entry_id)
            if row.get("image_path")}


def lock_panel(path: str, entry_id: str, panel_key: str, prompt_positive: str,
               prompt_negative_add: str, seed, image_path: str, note: str) -> None:
    workbook = _open_or_create_workbook(path)
    worksheet = _ensure_sheet(workbook, PANELS_SHEET, PANELS_COLUMNS)
    id_col = _col_index(worksheet, "entry_id")
    panel_col = _col_index(worksheet, "panel_key")
    row = _find_row(worksheet, id_col, entry_id, panel_col, panel_key)
    note_line = f"[{_timestamp()}] {note}" if note else f"[{_timestamp()}] locked"

    if row is None:
        row = worksheet.max_row + 1
        worksheet.cell(row, id_col, entry_id)
        worksheet.cell(row, panel_col, panel_key)
        existing_notes = ""
    else:
        existing_notes = worksheet.cell(row, _col_index(worksheet, "notes")).value or ""

    values = {
        "prompt_positive": prompt_positive,
        "prompt_negative_add": prompt_negative_add,
        "seed": seed,
        "image_path": image_path,
        "locked": True,
        "notes": (existing_notes + "\n" + note_line).strip() if existing_notes else note_line,
    }
    for column, value in values.items():
        worksheet.cell(row, _col_index(worksheet, column), value)
    workbook.save(path)


def list_references(path: str, entry_id: str) -> list:
    return [row for row in _read_sheet_rows(path, REFERENCES_SHEET, REFERENCES_COLUMNS)
            if str(row.get("entry_id")) == str(entry_id)]


def add_reference(path: str, entry_id: str, image_path: str, note: str) -> None:
    workbook = _open_or_create_workbook(path)
    worksheet = _ensure_sheet(workbook, REFERENCES_SHEET, REFERENCES_COLUMNS)
    row = worksheet.max_row + 1
    worksheet.cell(row, _col_index(worksheet, "entry_id"), entry_id)
    worksheet.cell(row, _col_index(worksheet, "image_path"), image_path)
    worksheet.cell(row, _col_index(worksheet, "note"), note)
    worksheet.cell(row, _col_index(worksheet, "added_at"), _timestamp())
    workbook.save(path)



def list_characters(path: str) -> list:
    return _read_sheet_rows(path, CHARACTERS_SHEET, CHARACTERS_COLUMNS)


def get_character(path: str, trigger_id: str) -> dict:
    normalized = (trigger_id or "").strip()
    return next((row for row in list_characters(path) if row.get("trigger_id") == normalized), {})


def _require_character_trigger(trigger_id: str) -> str:
    normalized = (trigger_id or "").strip()
    if not normalized.startswith("CHAR:") or len(normalized) <= len("CHAR:"):
        raise ValueError("Character trigger must start with CHAR:, for example CHAR:alistair_pig.")
    return normalized


def create_character(path: str, trigger_id: str, display_name: str, working_note: str) -> None:
    trigger_id = _require_character_trigger(trigger_id)
    if get_character(path, trigger_id):
        raise ValueError(f"{trigger_id} already exists.")
    workbook = _open_or_create_workbook(path)
    worksheet = _ensure_sheet(workbook, CHARACTERS_SHEET, CHARACTERS_COLUMNS)
    row = worksheet.max_row + 1
    now = _timestamp()
    values = {
        "trigger_id": trigger_id,
        "display_name": (display_name or "").strip(),
        "working_note": (working_note or "").strip(),
        "created_at": now,
        "updated_at": now,
    }
    for column, value in values.items():
        worksheet.cell(row, _col_index(worksheet, column), value)
    workbook.save(path)


def update_character(path: str, trigger_id: str, display_name: str, working_note: str) -> None:
    trigger_id = _require_character_trigger(trigger_id)
    workbook = _open_or_create_workbook(path)
    worksheet = _ensure_sheet(workbook, CHARACTERS_SHEET, CHARACTERS_COLUMNS)
    row = _find_row(worksheet, _col_index(worksheet, "trigger_id"), trigger_id)
    if row is None:
        raise ValueError(f"{trigger_id} does not exist.")
    worksheet.cell(row, _col_index(worksheet, "display_name"), (display_name or "").strip())
    worksheet.cell(row, _col_index(worksheet, "working_note"), (working_note or "").strip())
    worksheet.cell(row, _col_index(worksheet, "updated_at"), _timestamp())
    workbook.save(path)


def rename_character(path: str, old_trigger_id: str, new_trigger_id: str) -> None:
    old_trigger_id = _require_character_trigger(old_trigger_id)
    new_trigger_id = _require_character_trigger(new_trigger_id)
    if old_trigger_id == new_trigger_id:
        return
    workbook = _open_or_create_workbook(path)
    characters = _ensure_sheet(workbook, CHARACTERS_SHEET, CHARACTERS_COLUMNS)
    row = _find_row(characters, _col_index(characters, "trigger_id"), old_trigger_id)
    if row is None:
        raise ValueError(f"{old_trigger_id} does not exist.")
    if _find_row(characters, _col_index(characters, "trigger_id"), new_trigger_id) is not None:
        raise ValueError(f"{new_trigger_id} already exists.")
    characters.cell(row, _col_index(characters, "trigger_id"), new_trigger_id)
    characters.cell(row, _col_index(characters, "updated_at"), _timestamp())
    for sheet_name, columns in ((ASSETS_SHEET, ASSETS_COLUMNS), (PANELS_SHEET, PANELS_COLUMNS), (REFERENCES_SHEET, REFERENCES_COLUMNS)):
        if sheet_name not in workbook.sheetnames:
            continue
        worksheet = _ensure_sheet(workbook, sheet_name, columns)
        id_col = _col_index(worksheet, "entry_id")
        for item_row in range(2, worksheet.max_row + 1):
            if worksheet.cell(item_row, id_col).value == old_trigger_id:
                worksheet.cell(item_row, id_col, new_trigger_id)
    workbook.save(path)


def delete_character(path: str, trigger_id: str) -> None:
    trigger_id = _require_character_trigger(trigger_id)
    workbook = _open_or_create_workbook(path)
    for sheet_name, columns in ((CHARACTERS_SHEET, CHARACTERS_COLUMNS), (ASSETS_SHEET, ASSETS_COLUMNS), (PANELS_SHEET, PANELS_COLUMNS), (REFERENCES_SHEET, REFERENCES_COLUMNS)):
        if sheet_name not in workbook.sheetnames:
            continue
        worksheet = _ensure_sheet(workbook, sheet_name, columns)
        key_col = _col_index(worksheet, "trigger_id" if sheet_name == CHARACTERS_SHEET else "entry_id")
        for row in range(worksheet.max_row, 1, -1):
            if worksheet.cell(row, key_col).value == trigger_id:
                worksheet.delete_rows(row, 1)
    workbook.save(path)


def list_beats(path: str) -> list:
    rows = _read_sheet_rows(path, BEATS_SHEET, BEATS_COLUMNS)
    return sorted(rows, key=lambda row: row.get("order") if row.get("order") is not None else 0)


def _rounded_or_blank(value):
    return round(value, 2) if value is not None else None


def set_beats(path: str, beats: list, source: str) -> None:
    """Replace timing rows while preserving unresolved values as blank cells."""
    workbook = _open_or_create_workbook(path)
    if BEATS_SHEET in workbook.sheetnames:
        workbook.remove(workbook[BEATS_SHEET])
    worksheet = _ensure_sheet(workbook, BEATS_SHEET, BEATS_COLUMNS)
    for index, beat in enumerate(beats, start=1):
        row = index + 1
        worksheet.cell(row, _col_index(worksheet, "order"), index)
        worksheet.cell(row, _col_index(worksheet, "beat"), beat.get("beat"))
        worksheet.cell(row, _col_index(worksheet, "start_s"), _rounded_or_blank(beat.get("start_s")))
        worksheet.cell(row, _col_index(worksheet, "end_s"), _rounded_or_blank(beat.get("end_s")))
        worksheet.cell(row, _col_index(worksheet, "duration_s"), _rounded_or_blank(beat.get("duration_s")))
        worksheet.cell(row, _col_index(worksheet, "source"), beat.get("source") or source)
        worksheet.cell(row, _col_index(worksheet, "notes"), beat.get("notes", ""))
    workbook.save(path)
