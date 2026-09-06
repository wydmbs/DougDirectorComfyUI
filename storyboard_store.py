"""
storyboard_store.py — the shared data backbone (see SUITE.md).

Three sheets in one .xlsx workbook:

  Assets — one row per entry_id (a shot, or a CHAR/MASTER/PROP). Schema
    generalized from Pig_and_Rooster_Storyboard_v14.xlsx's Shot List tab.
    For CHAR/MASTER entries, prompt_positive/seed/image_path describe the
    ASSEMBLED COMPOSITE SHEET, not any one panel — the individual panel
    generations live in the Panels sheet below.

  Panels — one row per (entry_id, panel_key). A CHAR or MASTER entry is
    built from several independently-generated panels (front view, angry
    expression, night variant, etc.) against a FIXED template per
    entry_type (see sheet_composer.py) — fixed on purpose, so downstream
    motion/animation work can rely on a panel always sitting in the same
    position across every character. A plain SHOT entry has no panels; it
    locks directly into Assets.

  References — mood-board / inspiration images attached to an entry_id
    before a design is locked. These aren't generations and carry no
    prompt of their own — just an image and a note on what to borrow from
    it. Purely input, not part of the final registry record.

Locking again on the same entry_id (or entry_id+panel_key) UPDATES that
row in place and appends a timestamped line to notes, rather than
duplicating — this is the inline versioned decision log the old shot list
kept by hand.

entry_id convention:
    shot_id       e.g. "1.1", "4.3b"
    CHAR:<name>   e.g. "CHAR:pig"
    MASTER:<name> e.g. "MASTER:farm_field"
    PROP:<name>   e.g. "PROP:crate"
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


# ---------------------------------------------------------------------------
# Generic sheet plumbing (shared by Assets / Panels / References)
# ---------------------------------------------------------------------------

def _open_or_create_workbook(path: str) -> Workbook:
    if os.path.exists(path):
        return load_workbook(path)
    wb = Workbook()
    wb.remove(wb.active)
    return wb


def _ensure_sheet(wb: Workbook, sheet_name: str, columns: list):
    """Create the sheet with a header row if missing. If it exists but is
    missing columns added in a later version of this schema, append those
    columns to the header rather than failing — keeps older workbooks
    forward-compatible without a manual migration step."""
    if sheet_name not in wb.sheetnames:
        ws = wb.create_sheet(sheet_name)
        ws.append(columns)
        return ws
    ws = wb[sheet_name]
    existing = [c.value for c in ws[1]] if ws.max_row >= 1 else []
    for col in columns:
        if col not in existing:
            ws.cell(1, ws.max_column + 1 if ws.max_row >= 1 else 1, col)
            existing.append(col)
    return ws


def _col_index(ws, name: str) -> int:
    for idx, cell in enumerate(ws[1], start=1):
        if cell.value == name:
            return idx
    raise KeyError(f"Column '{name}' not found in '{ws.title}' sheet header row.")


def _find_row(ws, id_col: int, key: str, id_col2: int = None, key2: str = None):
    for row in range(2, ws.max_row + 1):
        val = ws.cell(row, id_col).value
        if val is None or str(val) != str(key):
            continue
        if id_col2 is not None:
            val2 = ws.cell(row, id_col2).value
            if val2 is None or str(val2) != str(key2):
                continue
        return row
    return None


def _read_sheet_rows(path: str, sheet_name: str, columns: list) -> list:
    if not os.path.exists(path):
        return []
    wb = load_workbook(path)
    if sheet_name not in wb.sheetnames:
        return []
    ws = wb[sheet_name]
    header = [c.value for c in ws[1]] if ws.max_row >= 1 else []
    rows = []
    for r in range(2, ws.max_row + 1):
        row_vals = [ws.cell(r, c).value for c in range(1, len(header) + 1)]
        if all(v is None for v in row_vals):
            continue
        record = dict(zip(header, row_vals))
        rows.append({col: record.get(col) for col in columns})
    return rows


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Assets sheet
# ---------------------------------------------------------------------------

def list_entries(path: str) -> list:
    return _read_sheet_rows(path, ASSETS_SHEET, ASSETS_COLUMNS)


def get_entry(path: str, entry_id: str) -> dict:
    for row in list_entries(path):
        if str(row.get("entry_id")) == str(entry_id):
            return row
    return {}


def lock_entry(
    path: str,
    entry_id: str,
    entry_type: str,
    beat: str,
    description: str,
    prompt_positive: str,
    prompt_negative_add: str,
    model: str,
    seed,
    reused_from: str,
    image_path: str,
    note: str,
) -> None:
    """Used directly for plain SHOT entries (one image, one prompt). CHAR /
    MASTER entries instead accumulate panels via lock_panel() and reach
    Assets only through set_composite_image()."""
    wb = _open_or_create_workbook(path)
    ws = _ensure_sheet(wb, ASSETS_SHEET, ASSETS_COLUMNS)
    id_col = _col_index(ws, "entry_id")
    row = _find_row(ws, id_col, entry_id)

    note_line = f"[{_timestamp()}] {note}" if note else f"[{_timestamp()}] locked"

    if row is None:
        row = ws.max_row + 1
        ws.cell(row, id_col, entry_id)
        existing_notes = ""
    else:
        existing_notes = ws.cell(row, _col_index(ws, "notes")).value or ""

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
    for col_name, val in values.items():
        ws.cell(row, _col_index(ws, col_name), val)

    wb.save(path)


def set_composite_image(path: str, entry_id: str, entry_type: str, template: str,
                          composite_image_path: str, note: str = "") -> None:
    """Save/update the assembled composite sheet for a CHAR/MASTER entry.
    Distinct from lock_entry: this doesn't touch prompt/seed fields (those
    live per-panel), it just records the rendered sheet as this entry's
    reference image."""
    wb = _open_or_create_workbook(path)
    ws = _ensure_sheet(wb, ASSETS_SHEET, ASSETS_COLUMNS)
    id_col = _col_index(ws, "entry_id")
    row = _find_row(ws, id_col, entry_id)

    note_line = f"[{_timestamp()}] {note}" if note else f"[{_timestamp()}] composite sheet rendered"

    if row is None:
        row = ws.max_row + 1
        ws.cell(row, id_col, entry_id)
        existing_notes = ""
    else:
        existing_notes = ws.cell(row, _col_index(ws, "notes")).value or ""

    values = {
        "entry_type": entry_type,
        "locked": True,
        "template": template,
        "composite_image_path": composite_image_path,
        "notes": (existing_notes + "\n" + note_line).strip() if existing_notes else note_line,
    }
    for col_name, val in values.items():
        ws.cell(row, _col_index(ws, col_name), val)

    wb.save(path)


# ---------------------------------------------------------------------------
# Panels sheet (per-panel generations for CHAR / MASTER sheets)
# ---------------------------------------------------------------------------

def list_panels(path: str, entry_id: str) -> list:
    return [r for r in _read_sheet_rows(path, PANELS_SHEET, PANELS_COLUMNS)
            if str(r.get("entry_id")) == str(entry_id)]


def get_panel_images(path: str, entry_id: str) -> dict:
    """panel_key -> image_path, for feeding straight into sheet_composer."""
    return {r["panel_key"]: r["image_path"] for r in list_panels(path, entry_id) if r.get("image_path")}


def lock_panel(
    path: str,
    entry_id: str,
    panel_key: str,
    prompt_positive: str,
    prompt_negative_add: str,
    seed,
    image_path: str,
    note: str,
) -> None:
    """Locking the same (entry_id, panel_key) again updates that panel's
    row in place and appends to its note log, same pattern as lock_entry."""
    wb = _open_or_create_workbook(path)
    ws = _ensure_sheet(wb, PANELS_SHEET, PANELS_COLUMNS)
    id_col = _col_index(ws, "entry_id")
    panel_col = _col_index(ws, "panel_key")
    row = _find_row(ws, id_col, entry_id, panel_col, panel_key)

    note_line = f"[{_timestamp()}] {note}" if note else f"[{_timestamp()}] locked"

    if row is None:
        row = ws.max_row + 1
        ws.cell(row, id_col, entry_id)
        ws.cell(row, panel_col, panel_key)
        existing_notes = ""
    else:
        existing_notes = ws.cell(row, _col_index(ws, "notes")).value or ""

    values = {
        "prompt_positive": prompt_positive,
        "prompt_negative_add": prompt_negative_add,
        "seed": seed,
        "image_path": image_path,
        "locked": True,
        "notes": (existing_notes + "\n" + note_line).strip() if existing_notes else note_line,
    }
    for col_name, val in values.items():
        ws.cell(row, _col_index(ws, col_name), val)

    wb.save(path)


# ---------------------------------------------------------------------------
# References sheet (mood-board / inspiration images)
# ---------------------------------------------------------------------------

def list_references(path: str, entry_id: str) -> list:
    return [r for r in _read_sheet_rows(path, REFERENCES_SHEET, REFERENCES_COLUMNS)
            if str(r.get("entry_id")) == str(entry_id)]


def add_reference(path: str, entry_id: str, image_path: str, note: str) -> None:
    """Always appends — a mood board can hold several images for the same
    entry_id, there's no single 'current' reference to overwrite."""
    wb = _open_or_create_workbook(path)
    ws = _ensure_sheet(wb, REFERENCES_SHEET, REFERENCES_COLUMNS)
    row = ws.max_row + 1
    ws.cell(row, _col_index(ws, "entry_id"), entry_id)
    ws.cell(row, _col_index(ws, "image_path"), image_path)
    ws.cell(row, _col_index(ws, "note"), note)
    ws.cell(row, _col_index(ws, "added_at"), _timestamp())
    wb.save(path)
