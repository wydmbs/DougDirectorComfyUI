"""
reference_conditioning.py -- injecting a locked draft into a FLUX generation.

This is the mechanism that makes continuity real rather than aspirational.
Without it, "the same pig, angrier" is just the words "a pig, angry" with a
different seed, and every character drifts. With it, the ChatGPT turnaround
sheet drives IP-Adapter-Plus and the face survives into a new composition.

Two ways in, because they do different jobs:

  IP-Adapter   the character's identity. Fed the turnaround sheet.
  img2img      the scene's look. Fed the environment concept, at low denoise
               so the layout and palette hold.

The workflow has to actually contain the nodes. When it doesn't, the caller
gets a message naming exactly what to add -- never a silent no-op that looks
like the reference was honoured when it wasn't.
"""

import os

MODE_OFF = "off"
MODE_IPADAPTER = "ipadapter"
MODE_IMG2IMG = "img2img"
MODE_BOTH = "both"
SUPPORTED_MODES = (MODE_OFF, MODE_IPADAPTER, MODE_IMG2IMG, MODE_BOTH)

# Node classes that accept a reference, most specific first.
IPADAPTER_CLASSES = ("IPAdapterAdvanced", "IPAdapterApply", "IPAdapter",
                     "IPAdapterPlus", "IPAdapterFaceID")
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

    if reference_image_path and mode in (MODE_IPADAPTER, MODE_BOTH):
        if not os.path.exists(reference_image_path):
            notes.append(f"turnaround sheet not found, identity not locked: {reference_image_path}")
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
    return float(getattr(getattr(cfg, "agent", None), "ipadapter_weight", 0.8) or 0.8)


def _resolve(image_path: str, upload) -> str:
    """What this workflow should put in a LoadImage node's `image` field.

    With an uploader, that's the name ComfyUI returned after receiving the file.
    Without one, it's the local path, which only works when ComfyUI is running on
    this same machine.
    """
    if upload is None:
        return os.path.abspath(image_path)
    return upload(image_path)


def _apply_ipadapter(workflow: dict, image_path: str, weight: float, upload=None) -> str:
    """Point the IP-Adapter's image input at the turnaround sheet."""
    adapter_id, adapter = _find(workflow, IPADAPTER_CLASSES)
    if adapter_id is None:
        return ("the workflow has no IP-Adapter node, so the character's identity was not "
                "locked -- add IPAdapterAdvanced (ComfyUI_IPAdapter_plus) and wire it "
                "between the model loader and the sampler")

    adapter.setdefault("inputs", {})["weight"] = float(weight)

    # The adapter takes its image from a LoadImage node by reference. Find the
    # one it is actually wired to, so a workflow with several loaders doesn't
    # get the wrong one overwritten.
    link = (adapter.get("inputs") or {}).get("image")
    target_id = link[0] if isinstance(link, list) and link else None

    try:
        reference = _resolve(image_path, upload)
    except Exception as error:  # noqa: BLE001 - reported, never silently skipped
        return f"could not send the turnaround to ComfyUI, identity not locked: {error}"

    if target_id and isinstance(workflow.get(target_id), dict):
        workflow[target_id].setdefault("inputs", {})["image"] = reference
        return ""

    loader_id, loader = _find(workflow, LOAD_IMAGE_CLASSES)
    if loader_id is None:
        return ("the IP-Adapter node has no LoadImage feeding it, so the turnaround was not "
                "applied -- add a LoadImage node and connect it to the adapter's image input")
    loader.setdefault("inputs", {})["image"] = reference
    adapter.setdefault("inputs", {})["image"] = [loader_id, 0]
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

    link = (encode.get("inputs") or {}).get("pixels")
    target_id = link[0] if isinstance(link, list) and link else None
    if target_id and isinstance(workflow.get(target_id), dict):
        workflow[target_id].setdefault("inputs", {})["image"] = reference
        return ""

    loaders = _find_all(workflow, LOAD_IMAGE_CLASSES)
    if not loaders:
        return ("no LoadImage node is available for the scene concept -- add one feeding "
                "the VAEEncode")
    # Take the last loader so we don't steal the one IP-Adapter is using.
    loader_id, loader = loaders[-1]
    loader.setdefault("inputs", {})["image"] = reference
    encode.setdefault("inputs", {})["pixels"] = [loader_id, 0]
    return ""


def inspect_workflow(workflow: dict) -> dict:
    """What this workflow can and can't do, before anything is generated."""
    adapter_id, _ = _find(workflow, IPADAPTER_CLASSES)
    encode_id, _ = _find(workflow, LATENT_CLASSES)
    loaders = _find_all(workflow, LOAD_IMAGE_CLASSES)
    return {
        "has_ipadapter": adapter_id is not None,
        "has_img2img": encode_id is not None,
        "load_image_nodes": len(loaders),
        "can_lock_identity": adapter_id is not None and bool(loaders),
        "can_anchor_scene": encode_id is not None and bool(loaders),
    }


def describe_mode(cfg) -> str:
    return {
        MODE_OFF: "Off — prompt text only. Continuity rests on wording alone.",
        MODE_IPADAPTER: "IP-Adapter — the turnaround sheet locks the character's identity.",
        MODE_IMG2IMG: "img2img — the scene concept anchors the background.",
        MODE_BOTH: "IP-Adapter + img2img — identity from the turnaround, look from the scene.",
    }.get(_mode(cfg), f"Unknown mode '{_mode(cfg)}'.")
