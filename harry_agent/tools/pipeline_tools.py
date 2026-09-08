"""
pipeline_tools.py -- the assistant director's view of the three-stage pipeline.

Harry needs to know more than "generate an image". He needs to know which stage
an asset is at, which model that stage uses, and -- for a shot -- which of the
three video models the shot actually calls for. These tools give him that,
grounded in the same registry everything else reads.

The routing tool is deliberately advisory: it returns a recommendation and its
reasoning rather than silently picking, because a wrong route either wastes
cloud credits or produces a melted face, and the director should be able to see
the call being made.
"""

import json
import os

import director_engine as engine
import model_pipeline as pipeline
import storyboard_store as store

from .registry import Tool


def build_pipeline_tools(cfg, video_fn=None) -> list:
    """video_fn is injectable so tests never call a paid API."""

    def pipeline_map() -> str:
        return pipeline.pipeline_summary()

    def asset_stage(entry_id: str) -> str:
        """Where one asset is up to, and what happens next."""
        entry = engine.get_entry(cfg, entry_id)
        if not entry:
            return (f"No entry called '{entry_id}'. It starts life as a ChatGPT draft: "
                    f"{pipeline.stage_guidance(pipeline.STAGE_DRAFT)}")
        stage = pipeline.next_stage(entry)
        payload = {
            "entry_id": entry_id,
            "entry_type": entry.get("entry_type"),
            "stage": stage,
            "stage_label": pipeline.STAGE_LABELS.get(stage, stage),
            "what_to_do": pipeline.stage_guidance(stage),
            "has_draft": bool(entry.get("image_path")),
            "has_keyframe": bool(entry.get("keyframe_path")),
            "has_clip": bool(entry.get("clip_path")),
            "has_turnaround_sheet": bool(entry.get("composite_image_path")),
            "shot_type": entry.get("shot_type") or "",
            "video_model": entry.get("video_model") or "",
        }
        return json.dumps(payload, indent=2, default=str)

    def attach_draft(entry_id: str, entry_type: str, image_path: str,
                     description: str = "", beat: str = "", note: str = "") -> str:
        """Register a ChatGPT concept as the draft everything else anchors to.

        Drafting happens outside this app -- ChatGPT has no API here and no seed
        control -- so the draft arrives as a file. This is how it enters the
        pipeline.
        """
        if not os.path.exists(image_path):
            return f"No file at '{image_path}'. Save the ChatGPT image locally first."
        return engine.lock_concept(
            cfg, entry_id, entry_type, beat, description,
            prompt_positive="", prompt_negative="", seed=0, image_path=image_path,
            note=f"[HARRY] draft attached from ChatGPT: {note}" if note else
                 "[HARRY] draft attached from ChatGPT")

    def route_shot(entry_id: str, description: str = "", motion_prompt: str = "",
                   override: str = "") -> str:
        """Recommend the clip model for a shot, and explain the choice."""
        if not description:
            entry = engine.get_entry(cfg, entry_id) or {}
            description = entry.get("description") or ""
            motion_prompt = motion_prompt or entry.get("motion_prompt") or ""

        route = pipeline.route_shot(description, motion_prompt, override)
        spec = route.model
        return json.dumps({
            "entry_id": entry_id,
            "shot_type": route.shot_type,
            "model_key": route.model_key,
            "model_name": spec.name if spec else "still frame only",
            "why": route.reason,
            "confidence": route.confidence,
            "cost": spec.cost if spec else "",
            "max_seconds": spec.max_seconds if spec else 0,
            "watch_out_for": spec.limits[:2] if spec else [],
        }, indent=2)

    def plan_shot(entry_id: str, shot_type: str, video_model: str,
                  motion_prompt: str = "", note: str = "") -> str:
        """Record how a shot is intended to be animated, before spending on it."""
        if shot_type not in pipeline.SHOT_TYPES:
            return f"'{shot_type}' isn't a shot type. Use one of: {', '.join(pipeline.SHOT_TYPES)}"
        if video_model and video_model not in pipeline.MODELS:
            return f"'{video_model}' isn't a known model. Use one of: {', '.join(pipeline.MODELS)}"
        store.set_shot_plan(cfg.storyboard_path, entry_id, shot_type, video_model,
                            motion_prompt, note=f"[HARRY] {note}" if note else "")
        spec = pipeline.MODELS.get(video_model)
        return (f"Planned {entry_id} as a {shot_type} shot on "
                f"{spec.name if spec else video_model}.")

    def stage_keyframe(entry_id: str, prompt_positive: str, prompt_negative: str = "",
                       character_entry_id: str = "", scene_entry_id: str = "",
                       n_variants: int = 0, base_seed: int = 0) -> str:
        """Render a shot keyframe in FLUX, conditioned on the locked references.

        The character's turnaround sheet drives IP-Adapter and the scene concept
        anchors the background -- which is the whole reason a shot generated
        this way looks like it belongs to the same film as the last one.
        """
        turnaround = scene = ""
        if character_entry_id:
            character = engine.get_entry(cfg, character_entry_id) or {}
            turnaround = character.get("composite_image_path") or character.get("image_path") or ""
        if scene_entry_id:
            backdrop = engine.get_entry(cfg, scene_entry_id) or {}
            scene = backdrop.get("image_path") or ""

        result = engine.generate(
            cfg, f"{entry_id}__keyframe", prompt_positive, prompt_negative,
            base_seed=base_seed or None,
            n_variants=n_variants or cfg.agent.variants_per_generation,
            reference_image_path=turnaround, scene_image_path=scene,
        )
        if not result.ok:
            return "No keyframes came back. " + " | ".join(result.warnings)

        payload = {
            "entry_id": entry_id,
            "identity_from": turnaround or "(none — face is unconstrained)",
            "background_from": scene or "(none — background is unconstrained)",
            "variants": [{"index": i, "image_path": v.image_path, "seed": v.seed}
                         for i, v in enumerate(result.variants)],
        }
        if result.warnings:
            payload["warnings"] = result.warnings
        return json.dumps(payload, indent=2)

    def lock_keyframe(entry_id: str, image_path: str, seed: int = 0,
                      prompt_positive: str = "", note: str = "") -> str:
        if not os.path.exists(image_path):
            return f"No file at '{image_path}'."
        store.set_keyframe(cfg.storyboard_path, entry_id, image_path, prompt_positive,
                           seed, note=f"[HARRY] {note}" if note else "keyframe staged")
        return f"Staged the keyframe for {entry_id}."

    def animate_shot(entry_id: str, motion_prompt: str, video_model: str = "",
                     character_entry_id: str = "", seconds: float = 0.0) -> str:
        """Turn a staged keyframe into a clip using the routed model."""
        from video_client import VideoError, generate_clip

        entry = engine.get_entry(cfg, entry_id)
        if not entry:
            return f"No entry called '{entry_id}'."
        keyframe = entry.get("keyframe_path") or entry.get("image_path")
        if not keyframe:
            return (f"{entry_id} has no keyframe yet. Stage one in FLUX first — "
                    f"a clip is grown from a keyframe, not from a prompt alone.")

        if not video_model:
            video_model = entry.get("video_model") or ""
        if not video_model:
            route = pipeline.route_shot(entry.get("description") or "", motion_prompt)
            video_model = route.model_key
        if not video_model:
            return "This shot routed to 'still' — it doesn't need animating."

        turnaround = ""
        if character_entry_id:
            character = engine.get_entry(cfg, character_entry_id) or {}
            turnaround = character.get("composite_image_path") or character.get("image_path") or ""

        try:
            result = generate_clip(cfg, entry_id, keyframe, motion_prompt,
                                   model_key=video_model, turnaround_path=turnaround,
                                   seconds=seconds)
        except VideoError as error:
            return f"Could not produce the clip: {error}"

        store.set_clip(cfg.storyboard_path, entry_id, result.clip_path, motion_prompt,
                       result.model_key, entry.get("shot_type") or "",
                       note=f"[HARRY] clip via {result.model_key}")
        payload = {"entry_id": entry_id, "clip_path": result.clip_path,
                   "model": result.model_key, "seconds": result.seconds}
        if result.warnings:
            payload["warnings"] = result.warnings
        return json.dumps(payload, indent=2)

    def check_reference_setup() -> str:
        """Can the configured workflow actually hold a character's identity?"""
        from reference_conditioning import describe_mode, inspect_workflow

        report = {"mode": describe_mode(cfg)}
        path = cfg.workflow_json_path
        if not path or not os.path.exists(path):
            report["workflow"] = "none configured"
            report["verdict"] = ("No workflow yet, so nothing can be conditioned. "
                                 "Build one in ComfyUI and export it as API format.")
            return json.dumps(report, indent=2)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                workflow = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            report["workflow"] = f"unreadable: {error}"
            return json.dumps(report, indent=2)

        found = inspect_workflow(workflow)
        report.update(found)
        report["workflow"] = os.path.basename(path)
        if found["can_lock_identity"]:
            report["verdict"] = "Ready — the turnaround will hold the character's face."
        else:
            report["verdict"] = ("This workflow cannot lock identity. Add an "
                                 "IPAdapterAdvanced node fed by a LoadImage, or characters "
                                 "will drift between shots.")
        return json.dumps(report, indent=2)

    def obj(**properties):
        return {"type": "object", "properties": properties,
                "required": [k for k, v in properties.items() if v.pop("_required", False)]}

    def s(desc, required=False):
        return {"type": "string", "description": desc, "_required": required}

    def i(desc, required=False):
        return {"type": "integer", "description": desc, "_required": required}

    def n(desc):
        return {"type": "number", "description": desc}

    return [
        Tool("pipeline_map",
             "The three production stages, which model serves each, and how shot types route to video models.",
             obj(), pipeline_map),
        Tool("asset_stage",
             "Where one asset is in the pipeline (draft, keyframe, clip) and what to do next.",
             obj(entry_id=s("e.g. CHARACTER:pig or 1.1", True)), asset_stage),
        Tool("check_reference_setup",
             "Whether the configured FLUX workflow can actually lock a character's identity via IP-Adapter.",
             obj(), check_reference_setup),
        Tool("attach_draft",
             "Register a ChatGPT/DALL-E concept image as an asset's draft — the reference everything downstream anchors to.",
             obj(entry_id=s("e.g. CHARACTER:pig", True),
                 entry_type=s("CHARACTER, BACKDROP, PROP or SHOT", True),
                 image_path=s("Local path to the saved ChatGPT image", True),
                 description=s("What it is and why it matters"),
                 beat=s("Story beat"),
                 note=s("Any note to record")),
             attach_draft, mutating=True, routine=True),
        Tool("route_shot",
             "Recommend which video model a shot needs — Minimax H3 for performance, LTX-2.5 for camera moves, "
             "Runway Gen-4 for physics — and explain why.",
             obj(entry_id=s("The shot", True),
                 description=s("What happens in the shot"),
                 motion_prompt=s("The intended motion"),
                 override=s("Force a shot type or model")),
             route_shot),
        Tool("plan_shot",
             "Record how a shot will be animated before spending time or credits on it.",
             obj(entry_id=s("The shot", True),
                 shot_type=s("dialogue, camera, action or still", True),
                 video_model=s("Model key from route_shot", True),
                 motion_prompt=s("The intended motion"),
                 note=s("Why this route")),
             plan_shot, mutating=True, routine=True),
        Tool("stage_keyframe",
             "Render shot keyframes in FLUX, with a character's turnaround driving IP-Adapter and a "
             "backdrop anchoring the scene. This is what keeps shots looking like one film.",
             obj(entry_id=s("The shot", True),
                 prompt_positive=s("The cinematic prompt for this shot", True),
                 prompt_negative=s("Things to avoid"),
                 character_entry_id=s("Character whose turnaround locks the face"),
                 scene_entry_id=s("Backdrop whose concept anchors the background"),
                 n_variants=i("How many candidates (1-8)"),
                 base_seed=i("Fix the starting seed")),
             stage_keyframe, mutating=True, routine=True, timeout_s=1800),
        Tool("lock_keyframe",
             "Lock the winning keyframe for a shot. Clips are grown from this frame.",
             obj(entry_id=s("The shot", True),
                 image_path=s("Winning keyframe path", True),
                 seed=i("Its seed"),
                 prompt_positive=s("Prompt that produced it"),
                 note=s("Why this one won")),
             lock_keyframe, mutating=True, routine=True),
        Tool("animate_shot",
             "Turn a locked keyframe into a clip with the routed video model. Cloud models cost credits, "
             "so route the shot first and say what you expect.",
             obj(entry_id=s("The shot", True),
                 motion_prompt=s("What moves, and how", True),
                 video_model=s("Override the routed model"),
                 character_entry_id=s("Character whose turnaround goes to Minimax REF2VA"),
                 seconds=n("Clip length; capped per model")),
             animate_shot, mutating=True, routine=True, timeout_s=2400),
    ]
