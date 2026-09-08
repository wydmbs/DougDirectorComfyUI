"""
model_pipeline.py -- the declarative record of which model does what.

The production runs in three stages, and each stage has a model chosen for a
reason. Writing that down in one place means the harness, the assistant
director, and the person reading the code six months from now all work from the
same map -- rather than the routing rules living scattered through prompt text
and half-remembered decisions.

    DRAFT      ChatGPT / DALL-E 3     concept sheets, turnarounds, scene art
       |
       v
    KEYFRAME   FLUX.1 Dev (local)     IP-Adapter injects the turnaround so the
       |                              character survives into a real composition
       v
    CLIP       one of three routes, chosen by what the shot actually needs

The clip stage is the interesting one. There is no single best video model, so
the shot type picks:

    dialogue / emotion   Minimax H3    REF2VA takes the keyframe AND the
                                       turnaround, so a head turn doesn't melt
    camera / sequence    LTX-2.5       local, fast, native multi-shot cuts
    physics / action     Runway Gen-4  cloud, for destruction and fluids that
                                       open-weight models still fumble

Nothing here calls an API. This is the map; the clients live elsewhere.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------- stages

STAGE_DRAFT = "draft"
STAGE_KEYFRAME = "keyframe"
STAGE_CLIP = "clip"
STAGE_DONE = "done"
STAGES = (STAGE_DRAFT, STAGE_KEYFRAME, STAGE_CLIP, STAGE_DONE)

STAGE_LABELS = {
    STAGE_DRAFT: "Drafting",
    STAGE_KEYFRAME: "Keyframe staging",
    STAGE_CLIP: "Animating",
    STAGE_DONE: "In the can",
}

# ------------------------------------------------------------- shot types

SHOT_DIALOGUE = "dialogue"
SHOT_CAMERA = "camera"
SHOT_ACTION = "action"
SHOT_STILL = "still"
SHOT_TYPES = (SHOT_DIALOGUE, SHOT_CAMERA, SHOT_ACTION, SHOT_STILL)


@dataclass
class ModelSpec:
    key: str
    name: str
    stage: str
    where: str                     # local | cloud | manual
    cost: str
    vram: str = ""
    max_seconds: float = 0.0
    strengths: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    node_classes: List[str] = field(default_factory=list)
    notes: str = ""

    def describe(self) -> str:
        bits = [f"{self.name} ({self.where}, {self.cost}"]
        if self.vram:
            bits.append(f", {self.vram}")
        bits.append(")")
        return "".join(bits)


# ------------------------------------------------------------- the models

DRAFT_CHATGPT = ModelSpec(
    key="chatgpt_dalle",
    name="ChatGPT / DALL-E 3",
    stage=STAGE_DRAFT,
    where="manual",
    cost="subscription",
    strengths=["fast concept turnarounds", "consistent flat studio lighting",
               "locks palette and art direction early"],
    limits=["not driven from this app -- images are produced outside and attached",
            "no seed control, so a redraw is never identical"],
    notes=("Prompt for 'character concept sheet, multiple angles, front, side, back "
           "profile, flat studio lighting' on a solid neutral background. Scenes get "
           "a wide establishing shot. These drafts are what everything downstream "
           "is anchored to, so they are worth getting right before moving on."),
)

KEYFRAME_FLUX = ModelSpec(
    key="flux1_dev",
    name="FLUX.1 Dev (FP8)",
    stage=STAGE_KEYFRAME,
    where="local",
    cost="free",
    vram="~12-16GB",
    strengths=["high-fidelity stills", "IP-Adapter holds facial features",
               "img2img anchors a scene's look"],
    limits=["still frames only", "needs IP-Adapter-Plus nodes installed"],
    node_classes=["IPAdapterAdvanced", "IPAdapterModelLoader", "LoadImage",
                  "CLIPVisionLoader"],
    notes=("The turnaround sheet goes into IP-Adapter-Plus to lock the face; the "
           "scene concept anchors the background. Output is the reference keyframe "
           "every clip is grown from."),
)

CLIP_MINIMAX = ModelSpec(
    key="minimax_h3",
    name="Minimax H3 (REF2VA)",
    stage=STAGE_CLIP,
    where="cloud",
    cost="api credits",
    max_seconds=15.0,
    strengths=["multi-image reference alignment", "faces survive a head turn",
               "longest usable take at 15s", "dialogue and subtle emotion"],
    limits=["slow locally", "cloud credits per clip"],
    notes=("Fed both the FLUX keyframe and the original turnaround sheet through "
           "REF2VA. Having the side and back profiles on hand is exactly why the "
           "face doesn't distort when the character turns."),
)

CLIP_LTX = ModelSpec(
    key="ltx_2_5",
    name="LTX-2.5",
    stage=STAGE_CLIP,
    where="local",
    cost="free",
    vram="~14GB",
    max_seconds=10.0,
    strengths=["native multi-shot sequencing", "camera pans and tracking",
               "fast, and free because it's local"],
    limits=["weaker on complex physics", "less reliable for close-up faces"],
    node_classes=["LTXVImgToVideo", "LTXVConditioning"],
    notes="The default workhorse: free, fast, and good at moving the camera.",
)

CLIP_RUNWAY = ModelSpec(
    key="runway_gen4_turbo",
    name="Runway Gen-4 Turbo",
    stage=STAGE_CLIP,
    where="cloud",
    cost="api credits",
    max_seconds=5.0,
    strengths=["fluid and rigid-body physics", "explosions, shattering, water",
               "cloth in wind"],
    limits=["short bursts only", "burns credits fastest of the three"],
    notes=("Reserved for the shots local models fumble. Use it for 4-5 second "
           "high-impact bursts, not general coverage."),
)

MODELS = {m.key: m for m in (DRAFT_CHATGPT, KEYFRAME_FLUX, CLIP_MINIMAX, CLIP_LTX, CLIP_RUNWAY)}

# Which clip model serves which kind of shot.
SHOT_ROUTING = {
    SHOT_DIALOGUE: CLIP_MINIMAX.key,
    SHOT_CAMERA: CLIP_LTX.key,
    SHOT_ACTION: CLIP_RUNWAY.key,
    SHOT_STILL: "",
}

# Words in a shot description that suggest what it really is. Deliberately a
# hint, not a verdict -- route_shot() explains itself and the director can
# always override, because a wrong route wastes either credits or a take.
_DIALOGUE_HINTS = (
    "close up", "close-up", "closeup", "speaking", "says", "talking", "dialogue",
    "expression", "reaction", "turns to", "turning", "looks at", "smiles",
    "frowns", "eyes", "face", "portrait", "whisper", "shout",
)
_ACTION_HINTS = (
    "explosion", "explodes", "shatter", "smash", "splash", "water", "wave",
    "fire", "flame", "smoke", "storm", "wind", "debris", "crash", "collapse",
    "spray", "foam", "rain", "lightning", "destruction", "fluttering",
)
_CAMERA_HINTS = (
    "pan", "pans", "tracking", "dolly", "crane", "zoom", "push in", "pull back",
    "wide", "establishing", "cuts to", "sequence", "follows", "sweep", "aerial",
)


@dataclass
class Route:
    shot_type: str
    model_key: str
    reason: str
    confidence: str = "medium"

    @property
    def model(self) -> Optional[ModelSpec]:
        return MODELS.get(self.model_key)

    def describe(self) -> str:
        model = self.model
        name = model.name if model else "no video model"
        return f"{self.shot_type} -> {name} ({self.reason})"


def route_shot(description: str, motion_prompt: str = "", override: str = "") -> Route:
    """Pick the clip model for a shot, and say why.

    Priority order matters: physics beats everything, because Runway is the only
    one of the three that handles it and a wrong call there is a wasted take.
    Dialogue beats camera, because a melting face ruins a shot in a way a
    slightly duller camera move does not.
    """
    if override:
        if override in SHOT_ROUTING:
            return Route(override, SHOT_ROUTING[override], "chosen by the director", "certain")
        if override in MODELS:
            spec = MODELS[override]
            return Route(SHOT_CAMERA, override, "model named by the director", "certain")

    text = f"{description or ''} {motion_prompt or ''}".lower()
    if not text.strip():
        return Route(SHOT_CAMERA, SHOT_ROUTING[SHOT_CAMERA],
                     "nothing described yet, so defaulting to the local workhorse", "low")

    action_hits = [h for h in _ACTION_HINTS if h in text]
    if action_hits:
        return Route(SHOT_ACTION, SHOT_ROUTING[SHOT_ACTION],
                     f"physics in the description ({', '.join(action_hits[:3])})", "high")

    dialogue_hits = [h for h in _DIALOGUE_HINTS if h in text]
    if dialogue_hits:
        return Route(SHOT_DIALOGUE, SHOT_ROUTING[SHOT_DIALOGUE],
                     f"performance in the description ({', '.join(dialogue_hits[:3])})", "high")

    camera_hits = [h for h in _CAMERA_HINTS if h in text]
    if camera_hits:
        return Route(SHOT_CAMERA, SHOT_ROUTING[SHOT_CAMERA],
                     f"camera move in the description ({', '.join(camera_hits[:3])})", "high")

    return Route(SHOT_CAMERA, SHOT_ROUTING[SHOT_CAMERA],
                 "nothing specific, so the free local model", "low")


# ------------------------------------------------------------ stage logic


def next_stage(entry: dict) -> str:
    """Where this entry is up to, read from what it actually has.

    Same principle as the rest of the harness: derive state from the registry
    rather than storing a status that can fall out of step with reality.
    """
    if not entry:
        return STAGE_DRAFT
    if not entry.get("image_path"):
        return STAGE_DRAFT
    entry_type = (entry.get("entry_type") or "").upper()
    if entry_type == "SHOT":
        if not entry.get("keyframe_path"):
            return STAGE_KEYFRAME
        if not entry.get("clip_path"):
            return STAGE_CLIP
        return STAGE_DONE
    # Sheet-shaped assets finish at their composite; clips are made from shots.
    if not entry.get("composite_image_path"):
        return STAGE_KEYFRAME
    return STAGE_DONE


def stage_guidance(stage: str) -> str:
    """What the director should be doing at this stage, in one line."""
    return {
        STAGE_DRAFT: ("Draft it in ChatGPT first -- a concept sheet on a neutral "
                      "background, multiple angles, flat studio lighting -- then attach it here."),
        STAGE_KEYFRAME: ("Stage the keyframe in FLUX.1 Dev, with the turnaround "
                         "driving IP-Adapter so the face survives the new composition."),
        STAGE_CLIP: ("Animate it. The shot type picks the model: performance goes to "
                     "Minimax H3, camera moves to LTX-2.5, physics to Runway."),
        STAGE_DONE: "Finished -- draft, keyframe and clip are all in place.",
    }.get(stage, "")


def pipeline_summary() -> str:
    """The whole map, for a person or for Harry's context."""
    lines = ["PRODUCTION PIPELINE", ""]
    for stage in (STAGE_DRAFT, STAGE_KEYFRAME, STAGE_CLIP):
        specs = [m for m in MODELS.values() if m.stage == stage]
        lines.append(f"{STAGE_LABELS[stage].upper()}")
        for spec in specs:
            lines.append(f"  {spec.describe()}")
            if spec.max_seconds:
                lines.append(f"    up to {spec.max_seconds:g}s per clip")
            lines.append(f"    good at: {', '.join(spec.strengths[:3])}")
            if spec.limits:
                lines.append(f"    watch:   {spec.limits[0]}")
        lines.append("")
    lines.append("SHOT ROUTING")
    for shot_type, key in SHOT_ROUTING.items():
        model = MODELS.get(key)
        lines.append(f"  {shot_type:9s} -> {model.name if model else 'still frame only'}")
    return "\n".join(lines)
