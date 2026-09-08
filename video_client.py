"""
video_client.py -- turning a staged keyframe into a clip, through ComfyUI.

Everything after the ChatGPT draft runs on the GPU machine. LTX-2.5 is local
weights; Minimax H3 and Runway Gen-4 are reached through ComfyUI's API nodes.
That means all three are the same thing from here: a workflow, submitted to the
same ComfyUI, with the keyframe uploaded first.

This is a deliberate simplification over talking to each vendor's REST API
directly. One connection to configure, credentials live in ComfyUI where they
belong rather than in this app's config, and the render happens where the GPU
is. The cost is that each model needs its own exported API-format workflow --
which is work the director has to do once, in ComfyUI, and can then reuse.

Each workflow is patched the same way:
  * the keyframe goes into the LoadImage node
  * the turnaround, if the model takes one, goes into a second LoadImage
  * the motion prompt goes into the text encode
  * the duration goes into whatever node exposes one
"""

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import model_pipeline as pipeline
from comfy_client import ComfyClient, ComfyClientError

DEFAULT_TIMEOUT = 2400

# Node classes that carry each kind of input, in preference order.
IMAGE_LOADERS = ("LoadImage", "LoadImageFromPath")
TEXT_NODES = ("CLIPTextEncode", "MinimaxTextPrompt", "RunwayTextPrompt", "String",
              "PrimitiveString", "LTXVTextPrompt")
DURATION_KEYS = ("duration", "seconds", "length", "num_frames", "frames", "video_length")


class VideoError(Exception):
    """A clip could not be produced, with a reason worth showing."""


@dataclass
class VideoResult:
    clip_path: str
    model_key: str
    seconds: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.clip_path)


def workflow_path_for(cfg, model_key: str) -> str:
    """Where the exported API-format workflow for this model lives."""
    return {
        pipeline.CLIP_MINIMAX.key: cfg.video.minimax_workflow_path,
        pipeline.CLIP_LTX.key: cfg.video.ltx_workflow_path,
        pipeline.CLIP_RUNWAY.key: cfg.video.runway_workflow_path,
    }.get(model_key, "")


def _clip_path(cfg, entry_id: str, model_key: str, extension: str = "mp4") -> str:
    folder = os.path.join(cfg.images_dir, "clips")
    os.makedirs(folder, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in entry_id)
    return os.path.join(folder, f"{safe}__{model_key}.{extension}")


def _find_all(workflow: dict, class_types) -> list:
    return [(nid, n) for nid, n in workflow.items()
            if isinstance(n, dict) and n.get("class_type") in class_types]


def _patch_workflow(workflow: dict, keyframe_ref: str, turnaround_ref: str,
                    motion_prompt: str, seconds: float) -> list:
    """Put this shot's inputs into the workflow. Returns warnings."""
    warnings = []

    loaders = _find_all(workflow, IMAGE_LOADERS)
    if not loaders:
        warnings.append(
            "this workflow has no LoadImage node, so the keyframe was not applied -- "
            "the clip will not be based on your staged frame")
    else:
        loaders[0][1].setdefault("inputs", {})["image"] = keyframe_ref
        if turnaround_ref:
            if len(loaders) > 1:
                loaders[1][1].setdefault("inputs", {})["image"] = turnaround_ref
            else:
                warnings.append(
                    "this workflow has only one LoadImage, so the turnaround sheet was not "
                    "passed -- add a second image input for REF2VA or a large head turn may drift")

    text_nodes = _find_all(workflow, TEXT_NODES)
    if motion_prompt:
        if not text_nodes:
            warnings.append("no text node found, so the motion prompt was not applied")
        for _, node in text_nodes:
            inputs = node.setdefault("inputs", {})
            for key in ("text", "prompt", "value", "string"):
                if isinstance(inputs.get(key), str):
                    inputs[key] = motion_prompt
                    break

    if seconds:
        applied = False
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            for key in DURATION_KEYS:
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = int(seconds) if key in ("num_frames", "frames") else seconds
                    applied = True
                    break
        if not applied:
            warnings.append(
                f"no duration field found, so the clip length is whatever the workflow "
                f"is set to rather than {seconds:g}s")

    return warnings


def generate_mock(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                  model_key: str, **kwargs) -> VideoResult:
    """Write a placeholder so the pipeline is walkable with no GPU and no credits.

    Deliberately not a real video -- it exists to prove the registry wiring, and
    says so in its own contents.
    """
    path = _clip_path(cfg, entry_id, model_key, "txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"MOCK CLIP\nentry: {entry_id}\nmodel: {model_key}\n"
            f"keyframe: {keyframe_path}\nmotion: {motion_prompt}\n")
    return VideoResult(path, model_key, 0.0,
                       ["mock mode: this is a placeholder file, not a real clip"])


def generate_clip(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                  model_key: str = "", turnaround_path: str = "", seconds: float = 0.0,
                  on_progress: Optional[Callable[[str], None]] = None) -> VideoResult:
    """Produce a clip with whichever model the shot calls for.

    All three routes go through the same ComfyUI on the GPU machine; only the
    workflow differs.
    """
    if not model_key:
        raise VideoError("No video model was chosen for this shot.")
    spec = pipeline.MODELS.get(model_key)
    if spec is None:
        raise VideoError(f"'{model_key}' isn't a known video model.")

    if getattr(cfg, "mock_mode", False):
        return generate_mock(cfg, entry_id, keyframe_path, motion_prompt, model_key)

    if not os.path.exists(keyframe_path):
        raise VideoError(f"Keyframe not found: {keyframe_path}")

    path = workflow_path_for(cfg, model_key)
    if not path or not os.path.exists(path):
        raise VideoError(
            f"No workflow is configured for {spec.name}. Build one in ComfyUI on the GPU "
            f"machine, export it with Save (API Format), and point Setup at it.")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise VideoError(f"Could not read the {spec.name} workflow: {error}") from error

    if "nodes" in workflow:
        raise VideoError(
            f"The {spec.name} workflow looks like a UI export, not API format. In ComfyUI "
            f"enable Dev mode options and use 'Save (API Format)'.")

    warnings = []
    if not seconds:
        seconds = min(5.0, spec.max_seconds) if spec.max_seconds else 5.0
    if spec.max_seconds and seconds > spec.max_seconds:
        warnings.append(f"clipped to {spec.name}'s {spec.max_seconds:g}s ceiling")
        seconds = spec.max_seconds

    client = ComfyClient(cfg.comfyui_url)
    if not client.ping():
        raise VideoError(
            f"ComfyUI isn't answering at {cfg.comfyui_url}. Start it on the GPU machine, "
            f"or check the URL in Setup.")

    try:
        keyframe_ref = client.upload_image(keyframe_path)
        turnaround_ref = ""
        if turnaround_path and os.path.exists(turnaround_path):
            turnaround_ref = client.upload_image(turnaround_path)
        elif model_key == pipeline.CLIP_MINIMAX.key:
            warnings.append(
                "no turnaround sheet was passed to REF2VA, so the face has only the keyframe "
                "to work from -- a large head turn may drift")

        work = json.loads(json.dumps(workflow))
        warnings.extend(_patch_workflow(work, keyframe_ref, turnaround_ref,
                                        motion_prompt, seconds))

        if on_progress:
            on_progress(f"{spec.name} rendering on the GPU machine")
        prompt_id = client.queue_prompt(work)
        history = client.wait_for_completion(
            prompt_id, timeout_s=DEFAULT_TIMEOUT,
            on_progress=(lambda s: on_progress(f"{spec.name} rendering ({s}s)")) if on_progress else None)
    except ComfyClientError as error:
        raise VideoError(str(error)) from error

    refs = client.extract_video_refs(history)
    if not refs:
        refs = client.extract_image_refs(history)
        if refs:
            warnings.append(
                "the workflow returned an image rather than a video -- check its output node")
    if not refs:
        raise VideoError(
            f"{spec.name} finished but produced no output. Check the workflow has a "
            f"video output node (VHS_VideoCombine or similar).")

    filename, subfolder, folder_type = refs[0]
    extension = os.path.splitext(filename)[1].lstrip(".") or "mp4"
    destination = _clip_path(cfg, entry_id, model_key, extension)
    try:
        data = client.fetch_image_bytes(filename, subfolder, folder_type)
    except (ComfyClientError, OSError) as error:
        raise VideoError(f"Could not download the finished clip: {error}") from error
    with open(destination, "wb") as handle:
        handle.write(data)

    return VideoResult(destination, model_key, seconds, warnings)


def readiness(cfg) -> dict:
    """Which parts of the clip stage are actually ready to run.

    Worth checking before planning a night of rendering, rather than finding out
    when the first shot fails.
    """
    client = ComfyClient(cfg.comfyui_url)
    reachable = client.ping()
    report = {
        "comfyui_url": cfg.comfyui_url,
        "comfyui_reachable": reachable,
        "mock_mode": bool(cfg.mock_mode),
        "models": {},
    }
    if reachable:
        try:
            stats = client.system_stats()
            devices = stats.get("devices") or []
            if devices:
                device = devices[0]
                report["gpu"] = device.get("name", "unknown")
                total = device.get("vram_total")
                free = device.get("vram_free")
                if isinstance(total, (int, float)):
                    report["vram_total_gb"] = round(total / (1024 ** 3), 1)
                if isinstance(free, (int, float)):
                    report["vram_free_gb"] = round(free / (1024 ** 3), 1)
        except ComfyClientError:
            pass

    available_classes = set()
    if reachable:
        try:
            available_classes = set(client.object_info().keys())
        except ComfyClientError:
            pass

    for key in (pipeline.CLIP_MINIMAX.key, pipeline.CLIP_LTX.key, pipeline.CLIP_RUNWAY.key):
        spec = pipeline.MODELS[key]
        path = workflow_path_for(cfg, key)
        entry = {
            "name": spec.name,
            "workflow": os.path.basename(path) if path else "",
            "workflow_present": bool(path and os.path.exists(path)),
        }
        if entry["workflow_present"] and available_classes:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    work = json.load(handle)
                needed = {n.get("class_type") for n in work.values()
                          if isinstance(n, dict) and n.get("class_type")}
                entry["missing_nodes"] = sorted(needed - available_classes)
            except (OSError, json.JSONDecodeError):
                entry["missing_nodes"] = ["(workflow unreadable)"]
        entry["ready"] = entry["workflow_present"] and not entry.get("missing_nodes")
        report["models"][key] = entry

    return report
