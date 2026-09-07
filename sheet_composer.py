"""
sheet_composer.py — renders a fixed-template composite reference sheet
(character sheet or master-plate sheet) from individually generated panel
images.

The template per entry_type is FIXED by design: every CHARACTER sheet has the
same panels in the same grid position, and every BACKDROP sheet likewise.
That consistency is what lets downstream motion/animation work assume,
e.g., "the front view is always at the same normalized position" rather
than re-discovering layout per character.

A sheet can be rendered at any point during production, even with most
panels still missing — unfinished panels render as a labeled placeholder
so you can see progress, not just a finished-or-nothing result.
"""

import os
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Fixed panel templates
# ---------------------------------------------------------------------------

PANEL_TEMPLATES = {
    "CHARACTER": [
        # Full-body turnaround
        {"key": "fullbody_front", "label": "Full body — front"},
        {"key": "fullbody_back", "label": "Full body — back"},
        {"key": "fullbody_left", "label": "Full body — left"},
        {"key": "fullbody_right", "label": "Full body — right"},
        # Upper-third (bust) turnaround
        {"key": "bust_front", "label": "Upper third — front"},
        {"key": "bust_back", "label": "Upper third — back"},
        {"key": "bust_left", "label": "Upper third — left"},
        {"key": "bust_right", "label": "Upper third — right"},
        # Forward-facing emotion shots
        {"key": "expression_neutral", "label": "Neutral"},
        {"key": "expression_happy", "label": "Happy"},
        {"key": "expression_angry", "label": "Angry"},
        {"key": "expression_sad", "label": "Sad"},
        {"key": "expression_surprised", "label": "Surprised"},
        {"key": "expression_determined", "label": "Determined"},
        # Supporting reference
        {"key": "attitude_pose", "label": "Attitude pose"},
        {"key": "costume_closeup", "label": "Costume detail"},
        {"key": "signature_prop", "label": "Signature prop"},
        {"key": "color_palette", "label": "Color palette"},
    ],
    "PROP": [
        {"key": "turnaround_front", "label": "Front"},
        {"key": "turnaround_back", "label": "Back"},
        {"key": "turnaround_left", "label": "Left"},
        {"key": "turnaround_right", "label": "Right"},
        {"key": "interior", "label": "Interior (if any)"},
        {"key": "on_it", "label": "On it / surface (if any)"},
        {"key": "detail_closeup", "label": "Detail / texture"},
        {"key": "in_context", "label": "In context (scale)"},
        {"key": "material_palette", "label": "Material palette"},
    ],
    "BACKDROP": [
        {"key": "establishing_wide", "label": "Establishing (wide)"},
        {"key": "day", "label": "Day"},
        {"key": "night", "label": "Night"},
        {"key": "weather_variant", "label": "Weather variant"},
        {"key": "detail_closeup", "label": "Detail close-up"},
    ],
}

GRID_COLUMNS = {"CHARACTER": 4, "BACKDROP": 3, "PROP": 4}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

CELL_W, CELL_H = 220, 220
PADDING = 16
LABEL_H = 28
HEADER_H = 70

BG = (250, 249, 246)
CELL_BG = (255, 255, 255)
CELL_BORDER = (214, 209, 198)
ACCENT = (18, 116, 99)
PLACEHOLDER_BG = (233, 231, 224)
PLACEHOLDER_TEXT = (150, 147, 138)
TEXT_COLOR = (45, 44, 40)
HEADER_TEXT = (255, 255, 255)

FONT_DIR = "/usr/share/fonts/truetype/google-fonts"


def _font(size, weight="Regular"):
    path = os.path.join(FONT_DIR, f"Poppins-{weight}.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def get_template(entry_type: str):
    return PANEL_TEMPLATES.get((entry_type or "").upper(), PANEL_TEMPLATES["CHARACTER"])


def render_sheet(entry_id: str, entry_type: str, panel_images: dict, out_path: str,
                  concept_image_path: str = None) -> str:
    """panel_images: dict of panel_key -> image path. Missing or unreadable
    entries render as a soft placeholder rather than failing, so an
    in-progress sheet can still be previewed.

    concept_image_path: the locked Phase-1 "mashup" design this sheet's
    panels are meant to extend. Drawn as a small thumbnail in the header so
    the sheet visibly documents what the panels were built to match."""
    template = get_template(entry_type)
    cols = GRID_COLUMNS.get((entry_type or "").upper(), 4)
    rows = (len(template) + cols - 1) // cols

    width = PADDING + cols * (CELL_W + PADDING)
    height = HEADER_H + PADDING + rows * (CELL_H + LABEL_H + PADDING)

    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = _font(26, "Bold")
    subtitle_font = _font(13, "Regular")
    label_font = _font(15, "Medium")
    placeholder_font = _font(13, "Regular")

    draw.rectangle([0, 0, width, HEADER_H], fill=ACCENT)
    draw.text((PADDING, 14), entry_id, font=title_font, fill=HEADER_TEXT)
    filled = sum(1 for p in template if panel_images.get(p["key"]) and os.path.exists(panel_images[p["key"]]))
    draw.text((PADDING, 44), f"{filled} / {len(template)} panels", font=subtitle_font, fill=(220, 240, 235))

    if concept_image_path and os.path.exists(concept_image_path):
        try:
            thumb_size = HEADER_H - 16
            concept_thumb = Image.open(concept_image_path).convert("RGB")
            concept_thumb.thumbnail((thumb_size, thumb_size))
            tx = width - PADDING - thumb_size
            ty = (HEADER_H - concept_thumb.height) // 2
            canvas.paste(concept_thumb, (tx + (thumb_size - concept_thumb.width), ty))
            draw.text((tx - 78, HEADER_H / 2 - 8), "concept:", font=subtitle_font, fill=(220, 240, 235))
        except OSError:
            pass

    for i, panel in enumerate(template):
        col = i % cols
        row = i // cols
        x = PADDING + col * (CELL_W + PADDING)
        y = HEADER_H + PADDING + row * (CELL_H + LABEL_H + PADDING)

        draw.rectangle([x, y, x + CELL_W, y + CELL_H], fill=CELL_BG, outline=CELL_BORDER, width=2)

        img_path = panel_images.get(panel["key"])
        drew_image = False
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((CELL_W - 12, CELL_H - 12))
                ix = x + (CELL_W - img.width) // 2
                iy = y + (CELL_H - img.height) // 2
                canvas.paste(img, (ix, iy))
                drew_image = True
            except OSError:
                drew_image = False

        if not drew_image:
            draw.rectangle([x + 6, y + 6, x + CELL_W - 6, y + CELL_H - 6], fill=PLACEHOLDER_BG)
            msg = "not yet generated" if not img_path else "couldn't load"
            tw = draw.textlength(msg, font=placeholder_font)
            draw.text((x + (CELL_W - tw) / 2, y + CELL_H / 2 - 8), msg, font=placeholder_font, fill=PLACEHOLDER_TEXT)

        draw.text((x + 2, y + CELL_H + 6), panel["label"], font=label_font, fill=TEXT_COLOR)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)
    return out_path
