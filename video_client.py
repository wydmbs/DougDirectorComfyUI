"""
video_client.py -- turning a staged keyframe into a moving clip.

Three backends, one interface, because the shot decides the model:

  Minimax H3     performance. Takes the keyframe AND the turnaround sheet
                 through REF2VA, which is precisely why a head turn holds
                 together instead of melting.
  LTX-2.5        camera. Runs locally through ComfyUI, so it's free and fast,
                 and it can cut between shots natively.
  Runway Gen-4   physics. Cloud, short bursts, for the destruction and fluids
                 the local models still fumble.

LTX goes through the existing ComfyUI client, since it is just another
workflow. The two cloud models are HTTP APIs and are implemented as submit +
poll, which is what both actually offer.

Every backend returns the same VideoResult, so the caller -- and Harry -- never
has to care which one ran.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

import model_pipeline as pipeline

POLL_SECONDS = 5.0
DEFAULT_TIMEOUT = 1800


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


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:600]
        raise VideoError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise VideoError(f"Could not reach the service: {error.reason}") from error


def _get_json(url: str, headers: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:600]
        raise VideoError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise VideoError(f"Could not reach the service: {error.reason}") from error


def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=600) as response, open(dest, "wb") as handle:
            handle.write(response.read())
    except (urllib.error.URLError, OSError) as error:
        raise VideoError(f"Could not download the finished clip: {error}") from error
    return dest


def _clip_path(cfg, entry_id: str, model_key: str, extension: str = "mp4") -> str:
    folder = os.path.join(cfg.images_dir, "clips")
    os.makedirs(folder, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in entry_id)
    return os.path.join(folder, f"{safe}__{model_key}.{extension}")


# ------------------------------------------------------------------- mock


def generate_mock(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                  model_key: str, **kwargs) -> VideoResult:
    """Write a placeholder file so the whole pipeline is exercisable with no GPU
    and no credits. Deliberately not a real video -- it exists to prove the
    registry wiring, and says so."""
    path = _clip_path(cfg, entry_id, model_key, "txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"MOCK CLIP\nentry: {entry_id}\nmodel: {model_key}\n"
            f"keyframe: {keyframe_path}\nmotion: {motion_prompt}\n")
    return VideoResult(path, model_key, 0.0,
                       ["mock mode: this is a placeholder file, not a real clip"])


# --------------------------------------------------------------- Minimax H3


def generate_minimax(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                     turnaround_path: str = "", seconds: float = 10.0,
                     on_progress: Optional[Callable[[str], None]] = None,
                     **kwargs) -> VideoResult:
    """REF2VA: the keyframe for layout, the turnaround for identity.

    Passing both is the entire reason this model is chosen for performance
    shots -- with the side and back profiles available it can rotate a head
    without inventing a face.
    """
    key = os.environ.get(cfg.video.minimax_api_key_env, "")
    if not key:
        raise VideoError(
            f"No Minimax key. Set {cfg.video.minimax_api_key_env} and restart the app.")
    if not os.path.exists(keyframe_path):
        raise VideoError(f"Keyframe not found: {keyframe_path}")

    references = [{"type": "keyframe", "image": os.path.abspath(keyframe_path)}]
    warnings = []
    if turnaround_path and os.path.exists(turnaround_path):
        references.append({"type": "character_reference",
                           "image": os.path.abspath(turnaround_path)})
    else:
        warnings.append(
            "no turnaround sheet was passed, so the face has only the keyframe to work "
            "from -- a large head turn may drift")

    cap = pipeline.CLIP_MINIMAX.max_seconds
    if seconds > cap:
        warnings.append(f"clipped to Minimax's {cap:g}s ceiling")
        seconds = cap

    base = cfg.video.minimax_url.rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    submitted = _post_json(f"{base}/video_generation", headers, {
        "model": cfg.video.minimax_model,
        "prompt": motion_prompt,
        "duration": seconds,
        "references": references,
    })
    task_id = submitted.get("task_id") or submitted.get("id")
    if not task_id:
        raise VideoError(f"Minimax did not return a task id: {str(submitted)[:300]}")

    deadline = time.time() + DEFAULT_TIMEOUT
    while time.time() < deadline:
        if on_progress:
            on_progress(f"Minimax H3 rendering ({int(deadline - time.time())}s left)")
        time.sleep(POLL_SECONDS)
        status = _get_json(f"{base}/query/video_generation?task_id={task_id}", headers)
        state = (status.get("status") or status.get("state") or "").lower()
        if state in ("success", "succeeded", "finished"):
            url = (status.get("file_url") or status.get("video_url")
                   or (status.get("result") or {}).get("video_url"))
            if not url:
                raise VideoError("Minimax reported success but returned no video URL.")
            path = _download(url, _clip_path(cfg, entry_id, pipeline.CLIP_MINIMAX.key))
            return VideoResult(path, pipeline.CLIP_MINIMAX.key, seconds, warnings)
        if state in ("failed", "error"):
            raise VideoError(f"Minimax failed: {status.get('message') or str(status)[:300]}")

    raise VideoError(f"Minimax was still rendering after {DEFAULT_TIMEOUT}s.")


# ------------------------------------------------------------------- LTX-2.5


def generate_ltx(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                 seconds: float = 5.0,
                 on_progress: Optional[Callable[[str], None]] = None,
                 **kwargs) -> VideoResult:
    """LTX runs as an ordinary ComfyUI workflow, so it reuses the same client."""
    from comfy_client import ComfyClient, ComfyClientError

    workflow_path = cfg.video.ltx_workflow_path
    if not workflow_path or not os.path.exists(workflow_path):
        raise VideoError(
            "No LTX workflow is configured. Build one in ComfyUI, export it with "
            "Save (API Format), and point Setup at it.")
    if not os.path.exists(keyframe_path):
        raise VideoError(f"Keyframe not found: {keyframe_path}")

    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)

    warnings = []
    patched = False
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        klass = node.get("class_type", "")
        if klass in ("LoadImage", "LoadImageFromPath"):
            node.setdefault("inputs", {})["image"] = os.path.abspath(keyframe_path)
            patched = True
        elif klass == "CLIPTextEncode" and motion_prompt:
            inputs = node.setdefault("inputs", {})
            if isinstance(inputs.get("text"), str):
                inputs["text"] = motion_prompt
    if not patched:
        warnings.append("the LTX workflow has no LoadImage node, so the keyframe was not applied")

    client = ComfyClient(cfg.comfyui_url)
    try:
        if on_progress:
            on_progress("LTX-2.5 rendering locally")
        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_completion(prompt_id, timeout_s=DEFAULT_TIMEOUT)
    except ComfyClientError as error:
        raise VideoError(str(error)) from error

    for node_output in (history.get("outputs") or {}).values():
        for item in (node_output.get("gifs") or node_output.get("videos") or []):
            filename = item.get("filename")
            if not filename:
                continue
            data = client.fetch_image_bytes(filename, item.get("subfolder", ""),
                                            item.get("type", "output"))
            extension = os.path.splitext(filename)[1].lstrip(".") or "mp4"
            path = _clip_path(cfg, entry_id, pipeline.CLIP_LTX.key, extension)
            with open(path, "wb") as handle:
                handle.write(data)
            return VideoResult(path, pipeline.CLIP_LTX.key, seconds, warnings)

    raise VideoError("LTX finished but produced no video output node.")


# -------------------------------------------------------------- Runway Gen-4


def generate_runway(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                    seconds: float = 5.0,
                    on_progress: Optional[Callable[[str], None]] = None,
                    **kwargs) -> VideoResult:
    key = os.environ.get(cfg.video.runway_api_key_env, "")
    if not key:
        raise VideoError(
            f"No Runway key. Set {cfg.video.runway_api_key_env} and restart the app.")
    if not os.path.exists(keyframe_path):
        raise VideoError(f"Keyframe not found: {keyframe_path}")

    import base64

    warnings = []
    cap = pipeline.CLIP_RUNWAY.max_seconds
    if seconds > cap:
        warnings.append(f"clipped to Runway's {cap:g}s ceiling -- it is built for short bursts")
        seconds = cap

    with open(keyframe_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")

    base = cfg.video.runway_url.rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Runway-Version": cfg.video.runway_version}
    submitted = _post_json(f"{base}/image_to_video", headers, {
        "model": cfg.video.runway_model,
        "promptImage": f"data:image/png;base64,{encoded}",
        "promptText": motion_prompt,
        "duration": int(seconds),
    })
    task_id = submitted.get("id")
    if not task_id:
        raise VideoError(f"Runway did not return a task id: {str(submitted)[:300]}")

    deadline = time.time() + DEFAULT_TIMEOUT
    while time.time() < deadline:
        if on_progress:
            on_progress("Runway Gen-4 rendering")
        time.sleep(POLL_SECONDS)
        status = _get_json(f"{base}/tasks/{task_id}", headers)
        state = (status.get("status") or "").upper()
        if state == "SUCCEEDED":
            output = status.get("output") or []
            url = output[0] if isinstance(output, list) and output else None
            if not url:
                raise VideoError("Runway reported success but returned no output URL.")
            path = _download(url, _clip_path(cfg, entry_id, pipeline.CLIP_RUNWAY.key))
            return VideoResult(path, pipeline.CLIP_RUNWAY.key, seconds, warnings)
        if state in ("FAILED", "CANCELLED"):
            raise VideoError(f"Runway failed: {status.get('failure') or state}")

    raise VideoError(f"Runway was still rendering after {DEFAULT_TIMEOUT}s.")


# ------------------------------------------------------------------ router

BACKENDS = {
    pipeline.CLIP_MINIMAX.key: generate_minimax,
    pipeline.CLIP_LTX.key: generate_ltx,
    pipeline.CLIP_RUNWAY.key: generate_runway,
}


def generate_clip(cfg, entry_id: str, keyframe_path: str, motion_prompt: str,
                  model_key: str = "", turnaround_path: str = "", seconds: float = 0.0,
                  on_progress: Optional[Callable[[str], None]] = None) -> VideoResult:
    """Produce a clip with whichever model the shot calls for."""
    if not model_key:
        raise VideoError("No video model was chosen for this shot.")

    if getattr(cfg, "mock_mode", False):
        return generate_mock(cfg, entry_id, keyframe_path, motion_prompt, model_key)

    backend = BACKENDS.get(model_key)
    if backend is None:
        raise VideoError(f"No backend for '{model_key}'. Known: {', '.join(BACKENDS)}")

    spec = pipeline.MODELS.get(model_key)
    if not seconds:
        seconds = min(5.0, spec.max_seconds) if spec and spec.max_seconds else 5.0

    return backend(cfg, entry_id=entry_id, keyframe_path=keyframe_path,
                   motion_prompt=motion_prompt, turnaround_path=turnaround_path,
                   seconds=seconds, on_progress=on_progress)
