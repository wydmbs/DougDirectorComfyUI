"""
reference_conditioning.py -- feeding a locked image back in as a reference.

This is the seam for the still-open Flux 2 vs Nano Banana Pro decision. The two
models take a reference image in different ways, so rather than guess, the
harness defines the contract and provides an adapter per mode. Choosing later
means changing one config value, not reworking generation.

Modes:
  off          -- text prompt only, today's behaviour. A reference passed in is
                  reported back as an honest warning rather than silently dropped.
  flux2        -- inject the image into a LoadImage node the workflow already has
  nano_banana  -- attach the image to an API-model node's image input

Both live adapters need a workflow that actually contains the target node. When
it doesn't, the caller gets a clear message naming what to add, instead of a
silent no-op that looks like the reference worked.
"""

import base64
import os

MODE_OFF = "off"
MODE_FLUX2 = "flux2"
MODE_NANO_BANANA = "nano_banana"
SUPPORTED_MODES = (MODE_OFF, MODE_FLUX2, MODE_NANO_BANANA)


class ReferenceError(Exception):
    pass


def _find_node_of_class(workflow: dict, class_type: str):
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node_id, node
    return None, None


def _encode(image_path: str) -> str:
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def apply_reference(workflow: dict, cfg, reference_image_path: str):
    """Return (workflow, warning). Warning is "" when the reference was applied.

    The workflow is mutated on a copy by the caller (apply_node_overrides has
    already deep-copied), so it is safe to edit in place here.
    """
    mode = getattr(getattr(cfg, "agent", None), "reference_conditioning", MODE_OFF) or MODE_OFF

    if not reference_image_path:
        return workflow, ""
    if not os.path.exists(reference_image_path):
        return workflow, f"reference image not found, generated without it: {reference_image_path}"

    if mode == MODE_OFF:
        return workflow, (
            "reference conditioning is off, so this was generated from the text prompt only "
            "(set it in Setup once the image model is chosen)"
        )

    if mode == MODE_FLUX2:
        node_id, node = _find_node_of_class(workflow, "LoadImage")
        if node_id is None:
            return workflow, (
                "reference conditioning is set to flux2 but the workflow has no LoadImage node -- "
                "add one and point it at the reference input"
            )
        node.setdefault("inputs", {})["image"] = os.path.abspath(reference_image_path)
        return workflow, ""

    if mode == MODE_NANO_BANANA:
        node_id, node = _find_node_of_class(workflow, "ImageAPIModel")
        if node_id is None:
            return workflow, (
                "reference conditioning is set to nano_banana but the workflow has no ImageAPIModel "
                "node -- add the API model node that accepts a reference image"
            )
        node.setdefault("inputs", {})["reference_image"] = _encode(reference_image_path)
        return workflow, ""

    return workflow, f"unknown reference conditioning mode '{mode}', generated without a reference"


def describe_mode(cfg) -> str:
    mode = getattr(getattr(cfg, "agent", None), "reference_conditioning", MODE_OFF) or MODE_OFF
    return {
        MODE_OFF: "Off — generations are driven by prompt text only. Continuity rests on wording.",
        MODE_FLUX2: "Flux 2 — the locked concept is injected through the workflow's LoadImage node.",
        MODE_NANO_BANANA: "Nano Banana Pro — the locked concept is attached to the API model node.",
    }.get(mode, f"Unknown mode '{mode}'.")
