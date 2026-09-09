"""critic_worker.py -- the half of the visual critic that needs a GPU stack.

This file exists because of an interpreter split, not because the code wanted
to be in two pieces. The app runs on Python 3.14 in .venv; torch, transformers
and mediapipe do not build for 3.14 and will not for a while. The vision stack
that does exist on this machine is ComfyUI's embedded Python, already carrying
CUDA torch and the DINOv2 weights the calibration downloaded.

So the critic is split at the interpreter boundary: visual_critic.py runs in
the app and decides things, this runs under the embedded Python and measures
things, and JSON on stdout is the seam. That is a subprocess launch and a model
load per critique -- a few seconds, once per generation round, against a render
that took a minute. The alternative was pinning the whole app to an older
Python to satisfy the critic, which is a large tail wagging a small dog.

Invoked as:  python_embeded\python.exe critic_worker.py  < payload.json
Payload:     {"anchor": path, "paths": [path, ...], "metrics": ["identity", ...]}
Prints:      {"ok": true, "scores": {...}, "errors": {...}}  -- and nothing else.

Metric choices and thresholds are argued in visual_critic.py; they were settled
by critic_calibrate.py against the labelled set, not by reputation.
"""

from __future__ import annotations

import json
import os
import sys

# The hub's xet transport resets mid-transfer on this network; the plain HTTP
# path works. Must be set before anything imports huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HAND_TASK = (r"C:\Users\dougl\ComfyUI_windows_portable\ComfyUI\custom_nodes"
             r"\comfyui_controlnet_aux\src\custom_controlnet_aux"
             r"\mesh_graphormer\hand_landmarker.task")


def _pil(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def score_identity(anchor_path, paths):
    """Cosine similarity to the reference in DINOv2 space.

    DINOv2 is trained for instance-level identity, which is the question being
    asked -- is this the same character -- rather than the question CLIP
    answers, which is whether two images share a subject and a style.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    def embed(path):
        inputs = processor(images=_pil(path), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        vec = out.last_hidden_state[:, 0]          # CLS token
        return vec / vec.norm(dim=-1, keepdim=True)

    anchor = embed(anchor_path)
    return [float((embed(p) @ anchor.T).item()) for p in paths]


def score_palette(anchor_path, paths):
    """Negative L1 distance between 8x8x8 colour histograms."""
    import numpy as np

    def hist(path):
        arr = np.asarray(_pil(path).resize((256, 256)), dtype=np.float32) / 255.0
        h, _ = np.histogramdd(arr.reshape(-1, 3), bins=(8, 8, 8),
                              range=((0, 1), (0, 1), (0, 1)))
        h = h.ravel()
        return h / (h.sum() + 1e-8)

    anchor = hist(anchor_path)
    # Negated so that, as with identity, higher means more similar.
    return [float(-np.abs(hist(p) - anchor).sum()) for p in paths]


def count_hands(_anchor_path, paths):
    """How many five-fingered human hands MediaPipe finds. Advisory only."""
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.environ.get("HAND_LANDMARKER_TASK", HAND_TASK)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"hand_landmarker.task not found at {model_path}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.3,
    )
    counts = []
    with vision.HandLandmarker.create_from_options(options) as detector:
        for path in paths:
            arr = np.asarray(_pil(path), dtype=np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            counts.append(float(len(detector.detect(image).hand_landmarks or [])))
    return counts


METRICS = {
    "identity": score_identity,
    "palette": score_palette,
    "hands": count_hands,
}


def main():
    # Everything the libraries print -- HF warnings, TensorFlow delegate notices,
    # tqdm weight bars -- goes to stderr, because stdout carries the JSON and a
    # single stray progress bar on it would make the result unparseable.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        payload = json.loads(sys.stdin.read())
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"bad payload: {error}"}), file=real_stdout)
        return 1

    anchor = payload.get("anchor")
    paths = payload.get("paths") or []
    wanted = payload.get("metrics") or list(METRICS)

    scores, errors = {}, {}
    for name in wanted:
        fn = METRICS.get(name)
        if fn is None:
            errors[name] = f"unknown metric: {name}"
            continue
        try:
            scores[name] = fn(anchor, paths)
        except Exception as error:  # noqa: BLE001
            errors[name] = f"{type(error).__name__}: {error}"

    print(json.dumps({"ok": True, "scores": scores, "errors": errors}), file=real_stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
