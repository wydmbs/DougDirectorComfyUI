"""Specialist personas for the Director Harness.

Harry establishes story intent. Specialists turn that intent into production
constraints with structured, inspectable outputs.
"""

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import harry_advisor as harry


class SpecialistError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _json(raw):
    try:
        return harry._parse_json_block(raw)
    except harry.HarryError as error:
        raise SpecialistError(str(error)) from error


def _validate_clare(data, characters):
    dossiers = data.get("dossiers")
    if not isinstance(dossiers, list):
        raise SpecialistError("Clare returned no character dossiers.")
    expected = {item["suggested_id"] for item in characters}
    received = {str(item.get("character_id") or "") for item in dossiers if isinstance(item, dict)}
    missing = expected - received
    if missing:
        raise SpecialistError("Clare did not cover every proposed character: " + ", ".join(sorted(missing)))
    required = ("character_id", "identity_lock", "base_wardrobe", "visibility_rules", "drift_risks")
    for dossier in dossiers:
        if not isinstance(dossier, dict) or any(not str(dossier.get(key) or "").strip() for key in required):
            raise SpecialistError("Clare returned an incomplete dossier. Identity, wardrobe, visibility, and drift risks are all required.")
    return data


def clare_character_costume(provider, plan, source_text="", era=""):
    """Create a reusable casting and costume dossier from Harry's call sheet."""
    characters = [item for item in (plan or {}).get("items", []) if item.get("kind") == "CHARACTER"]
    shots = [item for item in (plan or {}).get("items", []) if item.get("kind") == "SHOT"]
    if not characters:
        raise SpecialistError("Clare needs at least one CHARACTER proposal before she can cast the production.")

    system = """You are Clare, the casting and costume designer for an AI film production.
You are warmly exacting and protective of what is on-model. You turn a director's broad character ideas into repeatable screen identity without over-designing the story.

NON-NEGOTIABLES:
1. A physical identity lock cannot change between shots unless the story explicitly calls for a recognizable transformation.
2. Wardrobe and surface state evolve from a base; damage, dirt, wetness, and fatigue are states, never new designs.
3. A costume note must honor the stated era and never introduce anachronistic materials or technology.
4. Visibility guidance must respect the shot's framing and never prescribe body detail the camera cannot see.
5. Character distinctions must remain readable in silhouette, material, and one or two signature details.
6. Do not invent named protagonists, major wardrobe changes, or central roles without support in the source/call sheet.
7. Flag likely generative drift before it reaches a render.

Return JSON only:
{
  "persona": "Clare",
  "dossiers": [{
    "character_id": "CHARACTER:name",
    "character_name": "...",
    "identity_lock": "physical/silhouette/color/feature constraints that never drift",
    "base_wardrobe": "period-appropriate base costume or natural appearance treatment",
    "wardrobe_states": [{"beat":"...", "state":"...", "change_reason":"..."}],
    "visibility_rules": "what framing may reveal or must avoid",
    "drift_risks": "specific likely model failures and prevention",
    "hard_rejects": ["...", "..."]
  }],
  "shot_notes": [{
    "shot_id":"...",
    "cast":"character IDs visible or absent",
    "wardrobe_state":"the relevant Clare state",
    "visibility_note":"framing-specific constraint"
  }]
}"""
    payload = {
        "era": era,
        "source_excerpt": (source_text or "")[:6000],
        "characters": characters,
        "coverage": shots,
    }
    try:
        output = _json(harry._chat(provider, system, json.dumps(payload, indent=2)))
    except harry.HarryError as error:
        raise SpecialistError(str(error)) from error
    output = _validate_clare(output, characters)
    output["created_at"] = _now()
    output["status"] = "ready"
    return output


def clare_character_brief(provider, character, director_note, references):
    system = """You are Clare, a casting and costume director helping a filmmaker develop one AI-film character. Be practical, imaginative and exacting. Build from the supplied story role and director's own words; do not replace them with generic film-design language. Treat reference images as evidence about the requested aspect, never as subjects to copy.

Return JSON only with: identity_direction, wardrobe_direction, sheet_plan, drift_risks, questions (a short list of specific questions that would improve the next character pass)."""
    payload = {"character": character, "director_note": director_note, "references": [{"name": item.get("name"), "role": item.get("role")} for item in references]}
    images = [{"path": item.get("path"), "role": item.get("role"), "media_type": mimetypes.guess_type(item.get("path") or "")[0] or "image/jpeg"} for item in references]
    try:
        brief = _json(harry._chat_with_images(provider, system, json.dumps(payload, indent=2), images))
    except harry.HarryError as error:
        raise SpecialistError(str(error)) from error
    required = ("identity_direction", "wardrobe_direction", "sheet_plan", "drift_risks")
    if not isinstance(brief, dict) or any(not str(brief.get(key) or "").strip() for key in required):
        raise SpecialistError("Clare returned an incomplete character brief. Please try again.")
    questions = brief.get("questions", [])
    brief["questions"] = [str(question) for question in questions] if isinstance(questions, list) else []
    brief["created_at"] = _now()
    return brief


def save_dossier(library_dir, dossier):
    root = Path(library_dir) / "specialists"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "clare_character_costume.json"
    path.write_text(json.dumps(dossier, indent=2), encoding="utf-8")
    return path


def load_dossier(library_dir):
    path = Path(library_dir) / "specialists" / "clare_character_costume.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clare_consolidate_character_trials(provider, brief, selected):
    system = """You are Clare, a rigorous casting and costume director. Inspect four selected character reference images as one proposed identity set. Judge against the supplied character brief, not generic visual appeal. Verify that each selected image meets its trial purpose and that all four describe one consistent character. Hard-fail wrong species/anatomy or stance, unwanted extra central character, missing required costume/identity features, generic cartoon/CGI finish, or incompatible identity between selections. Return JSON only: {"verdict":"approved" or "revise", "summary":"...", "trial_reviews":[{"trial":"...","verdict":"pass" or "revise","note":"specific evidence and correction"}], "lock_note":"what is locked if approved"}."""
    images = [{"path": item.get("path"), "role": item.get("trial"), "media_type": mimetypes.guess_type(item.get("path") or "")[0] or "image/jpeg"} for item in selected]
    try:
        result = _json(harry._chat_with_images(provider, system, json.dumps({"brief": brief, "selected_trials": [{"trial": item.get("trial")} for item in selected]}, indent=2), images))
    except harry.HarryError as error:
        raise SpecialistError(str(error)) from error
    if not isinstance(result, dict) or result.get("verdict") not in {"approved", "revise"} or not isinstance(result.get("trial_reviews"), list):
        raise SpecialistError("Clare returned an incomplete consolidation review. Please try again.")
    result["created_at"] = _now()
    result["locked"] = False
    return result


def clare_realign_character_brief(provider, brief, consolidation, director_response):
    system = """You are Clare, a casting and costume director. A director has responded to your trial-consolidation review. Determine exactly which director points can be accepted, which corrections must remain to protect the character, and revise the written character brief accordingly. Do not generate images. Return JSON only with: accepted (list of director points accepted), changed (list of corrections retained or amended), confirmation (plain-language confirmation of the proposed realignment), rerender_instruction (one precise instruction for the next render batch), revised_brief (complete replacement object containing identity_direction, wardrobe_direction, sheet_plan, drift_risks, questions). The revised brief must preserve non-negotiable anatomy, identity, costume and style requirements unless the director explicitly changes them."""
    payload = {"current_brief": brief, "consolidation_review": consolidation, "director_response": director_response}
    try:
        result = _json(harry._chat(provider, system, json.dumps(payload, indent=2)))
    except harry.HarryError as error:
        raise SpecialistError(str(error)) from error
    required = ("accepted", "changed", "confirmation", "rerender_instruction", "revised_brief")
    if not isinstance(result, dict) or any(key not in result for key in required):
        raise SpecialistError("Clare returned an incomplete written realignment. Please try again.")
    revised = result.get("revised_brief")
    brief_keys = ("identity_direction", "wardrobe_direction", "sheet_plan", "drift_risks")
    if not isinstance(revised, dict) or any(not str(revised.get(key) or "").strip() for key in brief_keys):
        raise SpecialistError("Clare’s realigned character brief is incomplete. Please try again.")
    revised["questions"] = [str(question) for question in revised.get("questions", [])] if isinstance(revised.get("questions", []), list) else []
    revised["created_at"] = _now()
    result["accepted"] = [str(item) for item in result.get("accepted", [])]
    result["changed"] = [str(item) for item in result.get("changed", [])]
    result["director_response"] = director_response
    result["created_at"] = _now()
    return result
