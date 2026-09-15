"""Deterministic Clare conformance checks for approved Harry shot drafts."""

import json
from datetime import datetime, timezone

import storyboard_store as store

THELWELL_STYLE_BLOCK = "Thelwell style illustration, vintage 1960s British children's book, pen and ink with watercolour wash, loose hand-drawn sketchy linework, slightly wobbly confident lines, watercolour texture fills with visible brushstrokes, muted warm desaturated palette, rough textured paper, ink bleed at line edges, uneven watercolour wash, hand painted feel."

IDENTITY_LOCKS = {
    "Pig": {
        "trigger_phrase": "pig farmer character",
        "lora_file": "pig_lora_final.safetensors",
        "lora_strength": [0.85, 0.85],
        "base_wardrobe": "tweed cap, cream linen shirt, leather braces, tweed waistcoat, cord trousers, bare trotters",
    },
    "Bertie": {
        "trigger_phrase": "bertie show cockerel character",
        "lora_file": "rooster_lora_p3.safetensors",
        "lora_strength": [0.85, 0.85],
        "base_wardrobe": "approved locked Bertie wardrobe and plumage descriptor cluster",
    },
}

VALID_STATUSES = {"pending", "approved", "flagged", "rejected"}
VALID_FRAMING = {"close-up", "medium", "full-body", "silhouette"}
VALID_VISIBILITY = {"waist-down", "full-figure", "face-obscured", "n/a"}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def cache_identity_locks(storyboard_path):
    for name, lock in IDENTITY_LOCKS.items():
        trigger_id = f"CHARACTER:{name.lower()}"
        current = store.get_character(storyboard_path, trigger_id)
        if current:
            store.set_character_identity_lock(storyboard_path, trigger_id, lock)
        else:
            store.create_character(storyboard_path, trigger_id, name, "Clare Tier 1 identity lock")
            store.set_character_identity_lock(storyboard_path, trigger_id, lock)


def _shot_text(shot):
    return " ".join(str(shot.get(key) or "") for key in ("name", "description", "continuity_note", "camera_direction", "positive_prompt")).lower()


def _framing(shot):
    explicit = str(shot.get("framing") or "").strip().lower()
    if explicit in VALID_FRAMING:
        return explicit
    text = _shot_text(shot)
    if any(term in text for term in ("close-up", "close up", "close two-shot", "tight side-angle")):
        return "close-up"
    if "silhouette" in text:
        return "silhouette"
    if any(term in text for term in ("full-body", "full body", "wide shot", "wide angle")):
        return "full-body"
    return "medium"


def _has_human(shot):
    return bool(shot.get("human_visibility_tier")) or any(term in _shot_text(shot) for term in ("farm hand", "farm worker", "sailor", "cart driver", "human"))


def _cast_names(shot):
    text = _shot_text(shot)
    names = []
    if "pig" in text:
        names.append("Pig")
    if any(term in text for term in ("rooster", "bertie", "cockerel")):
        names.append("Bertie")
    return names


def build_character_blocks(shot):
    text = _shot_text(shot)
    framing = _framing(shot)
    human_tier = "n/a"
    if _has_human(shot):
        human_tier = "face-obscured" if "hammock" in text else "waist-down" if any(term in text for term in ("crate", "capture", "load")) else "full-figure"
    blocks = []
    for name in _cast_names(shot):
        lock = IDENTITY_LOCKS[name]
        storm = any(term in text for term in ("storm", "ship pitches", "shipwreck", "wet", "afloat"))
        variant = "storm-soaked and weathered transformation of locked base" if storm else "none"
        block = {
            "name": name,
            "trigger_phrase": lock["trigger_phrase"],
            "lora_file": lock["lora_file"],
            "lora_strength": lock["lora_strength"],
            "style_block_required": True,
            "wardrobe": {"base": lock["base_wardrobe"], "variant": variant},
            "pose_state": "flopped" if name == "Bertie" and storm else "upright" if name == "Bertie" else "n/a",
            "framing": framing,
            "out_of_frame_check": "pass",
            "human_visibility_tier": human_tier,
            "children_appropriate": "pass",
        }
        blocks.append(block)
    return blocks


def merge_prompt(shot, character_blocks):
    character_text = "; ".join(
        f"{block['trigger_phrase']}, {block['wardrobe']['base']}, wardrobe state: {block['wardrobe']['variant']}"
        for block in character_blocks
    )
    existing = str(shot.get("positive_prompt") or "").strip()
    parts = [THELWELL_STYLE_BLOCK, character_text, existing]
    return ". ".join(part for part in parts if part)


def _validate_block(block, prompt, shot):
    reasons, flags = [], []
    name = block.get("name")
    lock = IDENTITY_LOCKS.get(name)
    if not lock:
        return [f"Unknown Clare character '{name}'."], flags
    if block.get("trigger_phrase") != lock["trigger_phrase"]:
        reasons.append("Trigger phrase missing or malformed.")
    if block.get("lora_file") != lock["lora_file"]:
        reasons.append("LoRA file is not the locked production file.")
    if block.get("lora_strength") != lock["lora_strength"]:
        flags.append("LoRA strength differs from the default and needs a documented exception.")
    if block.get("style_block_required") and THELWELL_STYLE_BLOCK not in prompt:
        reasons.append("Full Thelwell style block is missing.")
    framing = block.get("framing")
    if framing not in VALID_FRAMING:
        reasons.append("Framing is invalid or absent.")
    if framing == "close-up" and block.get("out_of_frame_check") != "pass":
        reasons.append("Close-up describes or fails to exclude out-of-frame body parts.")
    tier = block.get("human_visibility_tier")
    if tier not in VALID_VISIBILITY:
        reasons.append("Human visibility tier is invalid.")
    if _has_human(shot) and tier == "n/a":
        reasons.append("Human visibility tier is required when a human is present.")
    if not _has_human(shot) and tier != "n/a":
        flags.append("Human visibility tier is set although no human is in the shot.")
    if _has_human(shot) and any(term in _shot_text(shot) for term in ("crate", "capture", "load")) and tier != "waist-down":
        reasons.append("Crate-loading/capture human visibility must be waist-down.")
    if block.get("children_appropriate") != "pass":
        reasons.append("Children-appropriate check failed.")
    variant = str((block.get("wardrobe") or {}).get("variant") or "none").lower()
    if any(term in variant for term in ("redesign", "new costume", "entirely new", "replacement outfit")):
        reasons.append("Wardrobe variant is a redesign, not a transformation of the locked base.")
    if name == "Bertie" and block.get("pose_state") not in {"upright", "flopped"}:
        flags.append("Bertie comb/plumage state must be upright or flopped; Harry proposes it and Clare validates it.")
    return reasons, flags


def validate_shot(shot, character_blocks, prompt=None):
    prompt = prompt if prompt is not None else str(shot.get("positive_prompt") or shot.get("prompt") or "")
    results = []
    all_reasons, all_flags = [], []
    for original in character_blocks:
        block = dict(original)
        reasons, flags = _validate_block(block, prompt, shot)
        block["status"] = "rejected" if reasons else "flagged" if flags else "approved"
        block["notes"] = "; ".join(reasons + flags)
        results.append(block)
        all_reasons.extend(reasons)
        all_flags.extend(flags)
    status = "rejected" if all_reasons else "flagged" if all_flags else "approved"
    return {"status": status, "character_blocks": results, "notes": "; ".join(all_reasons + all_flags), "checked_at": _now(), "prompt_positive": prompt}


def validate_and_store(storyboard_path, shot, character_blocks=None):
    character_blocks = character_blocks if character_blocks is not None else build_character_blocks(shot)
    prompt = merge_prompt(shot, character_blocks)
    result = validate_shot(shot, character_blocks, prompt)
    store.set_clare_result(storyboard_path, shot["suggested_id"], result["status"], result["character_blocks"], result["notes"], result["checked_at"], result["prompt_positive"])
    return result


def is_downstream_eligible(entry):
    return entry.get("clare_status") in {"approved", "flagged"}


def load_result(entry):
    raw = entry.get("clare_character_json") or "[]"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
