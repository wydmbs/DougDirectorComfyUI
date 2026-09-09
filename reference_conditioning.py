"""
reference_conditioning.py -- injecting a locked draft into a FLUX generation.

This is the mechanism that makes continuity real rather than aspirational.
Without it, "the same pig, angrier" is just the words "a pig, angry" with a
different seed, and every character drifts. With it, the ChatGPT turnaround
sheet drives IP-Adapter-Plus and the face survives into a new composition.

Three ways in, because they do different jobs:

  Kontext      the character's identity, exactly. The reference is VAE-encoded
               into a full latent grid and rides along in the sequence the
               model is denoising, so it attends to real pixels.
  IP-Adapter   the character's identity, approximately. The reference goes
               through a vision encoder and arrives as a handful of embedding
               tokens -- a summary of the picture, not the picture. Costume
               detail, button count and comb shape do not survive that
               bottleneck at any weight.
  img2img      the scene's look. Fed the environment concept, at low denoise
               so the layout and palette hold.

Kontext and IP-Adapter are not two settings of one thing; they are different
mechanisms, which is why turning the adapter weight up never closes the gap.
Kontext is an edit model, so it wants an instruction ("turn him to face left")
rather than a fresh description of the scene -- re-describing the character
makes it redraw instead of transform.

The workflow has to actually contain the nodes. When it doesn't, the caller
gets a message naming exactly what to add -- never a silent no-op that looks
like the reference was honoured when it wasn't.
"""

import os

MODE_OFF = "off"
MODE_IPADAPTER = "ipadapter"
MODE_KONTEXT = "kontext"
MODE_IMG2IMG = "img2img"
MODE_BOTH = "both"
SUPPORTED_MODES = (MODE_OFF, MODE_KONTEXT, MODE_IPADAPTER, MODE_IMG2IMG, MODE_BOTH)

# The modes that carry a character's identity, whichever mechanism they use.
IDENTITY_MODES = (MODE_KONTEXT, MODE_IPADAPTER, MODE_BOTH)

# Node classes that accept a reference, most specific first.
#
# FLUX and SD/SDXL use entirely different IP-Adapter implementations, and the
# distinction is easy to miss: a machine can have the SDXL nodes installed and
# still be unable to condition a FLUX render at all. The FLUX ones are listed
# first so they win on a machine that has both, which is common.
IPADAPTER_FLUX_CLASSES = ("ApplyIPAdapterFluxAdvanced", "ApplyIPAdapterFlux",
                          "IPAdapterFluxAdvanced", "IPAdapterFlux",
                          "ApplyFluxIPAdapter", "XlabsSampler")
IPADAPTER_SD_CLASSES = ("IPAdapterAdvanced", "IPAdapterApply", "IPAdapter",
                        "IPAdapterPlus", "IPAdapterFaceID")
IPADAPTER_CLASSES = IPADAPTER_FLUX_CLASSES + IPADAPTER_SD_CLASSES
# Kontext conditioning lives in ComfyUI core, not a custom node. ReferenceLatent
# is the node that does the actual work -- it attaches the encoded reference to
# the conditioning so its tokens travel with the ones being generated.
KONTEXT_CLASSES = ("ReferenceLatent",)
KONTEXT_SCALE_CLASSES = ("FluxKontextImageScale",)
LOAD_IMAGE_CLASSES = ("LoadImage", "LoadImageFromPath", "ETN_LoadImageBase64")
LATENT_CLASSES = ("VAEEncode",)


class ReferenceError(Exception):
    pass


def _find(workflow: dict, class_types) -> tuple:
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") in class_types:
            return node_id, node
    return None, None


def _find_all(workflow: dict, class_types) -> list:
    return [(nid, n) for nid, n in workflow.items()
            if isinstance(n, dict) and n.get("class_type") in class_types]


def _mode(cfg) -> str:
    return getattr(getattr(cfg, "agent", None), "reference_conditioning", MODE_OFF) or MODE_OFF


def apply_reference(workflow: dict, cfg, reference_image_path: str,
                    scene_image_path: str = "", weight: float = 0.0,
                    upload=None):
    """Return (workflow, warning). warning is "" when everything asked for landed.

    `upload` turns a local path into something ComfyUI can actually open, and is
    required whenever ComfyUI is not on this machine: a LoadImage node given an
    absolute local path from a different computer silently loads nothing, and the
    generation then looks successful while ignoring the reference entirely.
    Callers pass ComfyClient.upload_image. When it is omitted the local path is
    used, which is only correct for a same-machine ComfyUI.

    The workflow dict is already a deep copy by the time it reaches here
    (apply_node_overrides copies), so editing in place is safe.
    """
    mode = _mode(cfg)

    if mode == MODE_OFF:
        if reference_image_path or scene_image_path:
            return workflow, ("reference conditioning is off, so this was generated from the "
                              "text prompt only -- turn it on in Setup to hold continuity")
        return workflow, ""

    if not reference_image_path and not scene_image_path:
        return workflow, ""

    notes = []

    if reference_image_path and mode in IDENTITY_MODES:
        if not os.path.exists(reference_image_path):
            notes.append(f"reference image not found, identity not locked: {reference_image_path}")
        elif mode == MODE_KONTEXT:
            note = _apply_kontext(workflow, reference_image_path, upload)
            if note:
                notes.append(note)
        else:
            note = _apply_ipadapter(workflow, reference_image_path,
                                    weight or _default_weight(cfg), upload)
            if note:
                notes.append(note)

    if scene_image_path and mode in (MODE_IMG2IMG, MODE_BOTH):
        if not os.path.exists(scene_image_path):
            notes.append(f"scene concept not found, background not anchored: {scene_image_path}")
        else:
            note = _apply_img2img(workflow, scene_image_path, upload)
            if note:
                notes.append(note)

    return workflow, " | ".join(notes)


def _default_weight(cfg) -> float:
    return float(getattr(getattr(cfg, "agent", None), "ipadapter_weight", 1.0) or 1.0)


def _resolve(image_path: str, upload) -> str:
    """What this workflow should put in a LoadImage node's `image` field.

    With an uploader, that's the name ComfyUI returned after receiving the file.
    Without one, it's the local path, which only works when ComfyUI is running on
    this same machine.
    """
    if upload is None:
        return os.path.abspath(image_path)
    return upload(image_path)


def _trace_to_loader(workflow: dict, node_id: str, input_key: str, depth: int = 0):
    """Follow an input link back until an actual image loader is reached.

    An IP-Adapter is rarely wired straight to a LoadImage -- there is usually a
    resize, a mask, or a CLIPVision preprocessor in between. Writing a filename
    into that intermediate node sets a field it doesn't have, and ComfyUI then
    rejects the whole prompt. Following the chain finds the node that actually
    reads a file.
    """
    if depth > 8 or not node_id:
        return None, None
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return None, None
    if node.get("class_type") in LOAD_IMAGE_CLASSES:
        return node_id, node

    link = (node.get("inputs") or {}).get(input_key)
    if not (isinstance(link, list) and link):
        # Try any upstream image-ish input rather than giving up immediately.
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and value and key in ("image", "images", "pixels", "source"):
                return _trace_to_loader(workflow, value[0], "image", depth + 1)
        return None, None
    return _trace_to_loader(workflow, link[0], "image", depth + 1)


def _apply_ipadapter(workflow: dict, image_path: str, weight: float, upload=None) -> str:
    """Point the IP-Adapter's image input at the turnaround sheet."""
    adapter_id, adapter = _find(workflow, IPADAPTER_CLASSES)
    if adapter_id is None:
        return ("the workflow has no IP-Adapter node, so the character's identity was not "
                "locked -- for FLUX add ApplyIPAdapterFlux (ComfyUI-IPAdapter-Flux), for "
                "SD/SDXL add IPAdapterAdvanced (ComfyUI_IPAdapter_plus), wired between the "
                "model loader and the sampler")

    # Weight is named differently across the two families; set whichever the
    # node actually exposes rather than inventing a field it will reject.
    inputs = adapter.setdefault("inputs", {})
    for key in ("weight", "ip_adapter_scale", "strength"):
        if key in inputs or key == "weight":
            inputs[key] = float(weight)
            break

    try:
        reference = _resolve(image_path, upload)
    except Exception as error:  # noqa: BLE001 - reported, never silently skipped
        return f"could not send the turnaround to ComfyUI, identity not locked: {error}"

    # Follow the adapter's image input back to whatever actually loads a file,
    # rather than assuming it is wired directly to a LoadImage.
    loader_id, loader = _trace_to_loader(workflow, adapter_id, "image")
    if loader_id is not None:
        loader.setdefault("inputs", {})["image"] = reference
        return ""

    loader_id, loader = _find(workflow, LOAD_IMAGE_CLASSES)
    if loader_id is None:
        return ("the IP-Adapter node has no LoadImage feeding it, so the turnaround was not "
                "applied -- add a LoadImage node and connect it to the adapter's image input")
    loader.setdefault("inputs", {})["image"] = reference
    adapter.setdefault("inputs", {})["image"] = [loader_id, 0]
    return ""


def _apply_kontext(workflow: dict, image_path: str, upload=None) -> str:
    """Point the Kontext reference chain at the locked design.

    The chain is LoadImage -> FluxKontextImageScale -> VAEEncode ->
    ReferenceLatent, so the filename belongs at the far end of it. Starting from
    ReferenceLatent and walking back means the graph can be shaped differently
    -- an extra crop, a stitch, no scaler at all -- and this still finds the
    node that reads a file.
    """
    reference_id, _ = _find(workflow, KONTEXT_CLASSES)
    if reference_id is None:
        return ("the workflow has no ReferenceLatent node, so the character's identity was "
                "not locked -- wire LoadImage -> FluxKontextImageScale -> VAEEncode -> "
                "ReferenceLatent onto the positive conditioning, and drive the sampler with "
                "a FLUX Kontext model (plain FLUX.1 dev cannot do this)")

    try:
        reference = _resolve(image_path, upload)
    except Exception as error:  # noqa: BLE001 - reported, never silently skipped
        return f"could not send the reference to ComfyUI, identity not locked: {error}"

    loader_id, loader = _trace_to_loader(workflow, reference_id, "latent")
    if loader_id is not None:
        loader.setdefault("inputs", {})["image"] = reference
        return _note_kontext_scale(workflow)

    loader_id, loader = _find(workflow, LOAD_IMAGE_CLASSES)
    if loader_id is None:
        return ("the ReferenceLatent node has no LoadImage feeding it, so the reference was "
                "not applied -- add a LoadImage and encode it into the reference latent")
    loader.setdefault("inputs", {})["image"] = reference
    return _note_kontext_scale(workflow)


def _note_kontext_scale(workflow: dict) -> str:
    """Kontext was trained on a fixed set of resolutions.

    Hand it anything else and quality falls off in a way that reads as the model
    being bad rather than the image being the wrong shape, so an absent scaler is
    worth saying out loud even though the render will still complete.
    """
    scale_id, _ = _find(workflow, KONTEXT_SCALE_CLASSES)
    if scale_id is None:
        return ("no FluxKontextImageScale node, so the reference goes in at whatever size it "
                "happens to be -- add one between the LoadImage and the VAEEncode, or expect "
                "softer results that look like the model underperforming")
    return ""


def _apply_img2img(workflow: dict, image_path: str, upload=None) -> str:
    """Anchor the background by encoding the scene concept as the start latent."""
    encode_id, encode = _find(workflow, LATENT_CLASSES)
    if encode_id is None:
        return ("the workflow has no VAEEncode node, so the scene concept did not anchor the "
                "background -- add a LoadImage into a VAEEncode and feed the sampler's "
                "latent_image from it")

    try:
        reference = _resolve(image_path, upload)
    except Exception as error:  # noqa: BLE001 - reported, never silently skipped
        return f"could not send the scene concept to ComfyUI, background not anchored: {error}"

    loader_id, loader = _trace_to_loader(workflow, encode_id, "pixels")
    if loader_id is not None:
        loader.setdefault("inputs", {})["image"] = reference
        return _note_denoise(workflow)

    loaders = _find_all(workflow, LOAD_IMAGE_CLASSES)
    if not loaders:
        return ("no LoadImage node is available for the scene concept -- add one feeding "
                "the VAEEncode")
    # Take the last loader so we don't steal the one IP-Adapter is using.
    loader_id, loader = loaders[-1]
    loader.setdefault("inputs", {})["image"] = reference
    encode.setdefault("inputs", {})["pixels"] = [loader_id, 0]
    return _note_denoise(workflow)


def _note_denoise(workflow: dict) -> str:
    """img2img only holds a scene when denoise is well below 1.

    At denoise 1.0 the sampler discards the encoded latent entirely, so the
    scene reference is present in the graph and absent from the result -- which
    looks exactly like the reference not working.
    """
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        denoise = (node.get("inputs") or {}).get("denoise")
        if isinstance(denoise, (int, float)) and denoise >= 0.95:
            return (f"the sampler's denoise is {denoise:g}, which discards the scene concept -- "
                    f"drop it to about 0.5-0.7 for the background to actually hold")
    return ""


def inspect_workflow(workflow: dict) -> dict:
    """What this workflow can and can't do, before anything is generated."""
    flux_id, _ = _find(workflow, IPADAPTER_FLUX_CLASSES)
    sd_id, _ = _find(workflow, IPADAPTER_SD_CLASSES)
    adapter_id = flux_id or sd_id
    kontext_id, _ = _find(workflow, KONTEXT_CLASSES)
    scale_id, _ = _find(workflow, KONTEXT_SCALE_CLASSES)
    encode_id, _ = _find(workflow, LATENT_CLASSES)
    loaders = _find_all(workflow, LOAD_IMAGE_CLASSES)
    # A Kontext graph encodes its reference through a VAEEncode too, so the
    # presence of one no longer means the graph is doing img2img scene
    # anchoring. Saying otherwise makes the doctor warn about a denoise of 1.0
    # that is entirely correct for Kontext.
    kontext = kontext_id is not None
    return {
        "has_ipadapter": adapter_id is not None,
        "ipadapter_family": "flux" if flux_id else ("sd" if sd_id else ""),
        "has_kontext": kontext,
        "has_kontext_scale": scale_id is not None,
        "identity_mechanism": ("kontext" if kontext
                               else ("ipadapter" if adapter_id is not None else "")),
        "has_img2img": encode_id is not None and not kontext,
        "load_image_nodes": len(loaders),
        "can_lock_identity": (kontext or adapter_id is not None) and bool(loaders),
        "can_anchor_scene": encode_id is not None and bool(loaders) and not kontext,
    }


def describe_mode(cfg) -> str:
    return {
        MODE_OFF: "Off — prompt text only. Continuity rests on wording alone.",
        MODE_KONTEXT: ("Kontext — the reference is encoded into the sequence being generated, "
                       "so the character survives intact."),
        MODE_IPADAPTER: "IP-Adapter — the reference guides identity, approximately.",
        MODE_IMG2IMG: "img2img — the scene concept anchors the background.",
        MODE_BOTH: "IP-Adapter + img2img — identity from the reference, look from the scene.",
    }.get(_mode(cfg), f"Unknown mode '{_mode(cfg)}'.")


# Words that describe a thing Kontext would otherwise have inherited. Naming one
# doesn't preserve it -- it hands the model licence to regenerate it.
INHERITED_TERMS = (
    "wearing", "waistcoat", "vest", "jacket", "shirt", "collar", "buttons",
    "plumage", "feathers", "fur", "comb", "wattle", "beak", "eyes",
    "crimson", "scarlet", "rust-red", "cream", "brass", "amber",
    "lighting", "lit by", "side-lighting", "backlit", "rim light",
)
# An instruction starts with a verb aimed at the picture. A prompt that doesn't
# is a description, and Kontext answers a description by drawing it fresh.
INSTRUCTION_VERBS = (
    "turn", "move", "rotate", "tilt", "raise", "lower", "open", "close",
    "make", "change", "replace", "remove", "add", "put", "place", "swap",
    "zoom", "pan", "crop", "reframe", "show", "have", "give", "point",
    "look", "face", "lean", "step", "walk", "sit", "stand", "hold",
)
# Kontext will reframe all day and barely rotate at all. Swept on this
# project's own reference at guidance 2.5, 4.0 and 6.0, across four ways of
# asking -- an order to the subject, a camera move, the finished pose stated as
# fact, and a deliberately excessive "full side profile" -- twelve renders came
# back at essentially the same angle. Guidance changed the contrast and the
# background tone and left the geometry alone.
#
# That is the model working as designed: it preserves layout, which is the same
# property that keeps the costume. A single front-on reference contains no side
# of the character to rotate towards, so there is nothing to preserve *into*.
# Chaining a turnaround that already holds that angle helps a little; staging
# the angle in the reference itself is what actually works.
POSE_TERMS = (
    "three-quarter", "three quarter", "profile", "from the side", "from behind",
    "side view", "back view", "rear view", "turn him", "turn her", "turn them",
    "facing left", "facing right", "facing away", "over the shoulder",
)
# Reframing is the half Kontext is good at, so a prompt doing both should not be
# warned off wholesale.
FRAMING_TERMS = (
    "close-up", "close up", "wide shot", "zoom", "crop", "reframe",
    "head and shoulders", "full body", "medium shot",
)


def review_instruction(prompt: str, cfg) -> str:
    """What's wrong with this prompt, for the conditioning mode in use.

    Only Kontext is opinionated here, and for a reason that is easy to get
    backwards: it inherits everything the prompt doesn't mention. So the way to
    keep a costume is to say nothing about it, and describing the character
    "to be safe" is precisely what makes it drift. Measured on this project's
    own reference -- the same shot, same seed, same reference image -- an
    instruction that also asked for particular lighting and a particular room
    came back with the waistcoat turned into a lapelled jacket. The instruction
    alone kept the waistcoat, its three brass buttons and the studio light.

    Returns "" when the prompt is shaped the way the model wants.
    """
    if _mode(cfg) != MODE_KONTEXT or not (prompt or "").strip():
        return ""

    text = prompt.strip()
    lowered = text.lower()
    notes = []

    first = lowered.lstrip("\"'([").split(" ", 1)[0].strip(".,:;")
    if first not in INSTRUCTION_VERBS:
        notes.append(
            "this reads as a description rather than an instruction, and Kontext answers a "
            "description by drawing it fresh -- start with what should change "
            "(\"Turn him to face left\") instead of restating the scene")

    named = sorted({term for term in INHERITED_TERMS if term in lowered})
    if named:
        notes.append(
            f"mentions {', '.join(named[:5])} -- Kontext inherits anything the prompt leaves "
            "out, so naming an attribute lets it be regenerated rather than kept")

    pose = sorted({term for term in POSE_TERMS if term in lowered})
    if pose and not any(term in lowered for term in FRAMING_TERMS):
        notes.append(
            f"asks for a change of angle ({pose[0]}) -- Kontext reframes well but barely "
            "rotates, and pushing guidance does not help; supply a reference that already "
            "holds that angle, or chain the turnaround sheet alongside it")

    return " | ".join(notes)
