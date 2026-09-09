"""critic_calibrate.py -- does a local metric actually separate good from bad?

The point of this file is to refuse to take a metric's reputation on trust.

CLIP similarity is widely used as a "same character?" score, and it has a
well-known failure: the embedding is dominated by global style and palette, so
two *different* characters drawn in the same style score high. DINOv2 is the
usual answer for instance-level identity. MediaPipe detects human hands, which
is the specific anatomy defect this production keeps hitting. Any of those
claims might hold here or might not, and the only way to find out is to run them
against renders whose correct answer is already known.

That labelled set exists, because earlier probes produced it as a by-product:

    verify_v1_control_*   BAD   the v1 assets. Bare pink human hands -- the
                                original defect, and a visibly different design.
    glove_v2_+_tips_*     OLD   the best the previous assets ever managed. Same
                                character concept, earlier version. This is the
                                hard case, and the one that tells CLIP-style
                                similarity apart from real identity matching.
    glove_v3_*            GOOD  the adopted canon, verified by eye across three
                                shots and three conditioning configurations.
    grip_* / gripshot_*   HAND  nine renders that were asked for no hand at all
                                and produced a five-fingered hand every time.

A metric earns its place by ranking GOOD above BAD against the v3 reference.
AUC is reported because it is rank-based and needs no threshold chosen in
advance; a margin is reported alongside it because an AUC of 1.0 with the two
classes almost touching is a different proposition from one with daylight
between them, and only the second survives contact with a new character.

Nothing here writes to the app or its config. It prints a table and exits.

    python critic_calibrate.py                    # all available metrics
    python critic_calibrate.py --metrics hands    # just the anatomy gate
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# The hub's xet transport resets mid-transfer on this network; the plain HTTP
# path works. Set before anything imports huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

OUTPUT_DIR = r"C:\Users\dougl\ComfyUI_windows_portable\ComfyUI\output"
ANCHOR = "smoke_out/rooster_ref_v3.png"

# MediaPipe 0.10.35 removed mp.solutions; the Tasks API needs an explicit
# model file. ComfyUI ships one with the controlnet_aux nodes, so nothing
# new has to be downloaded.
HAND_TASK = (r"C:\Users\dougl\ComfyUI_windows_portable\ComfyUI\custom_nodes"
             r"\comfyui_controlnet_aux\src\custom_controlnet_aux"
             r"\mesh_graphormer\hand_landmarker.task")

# (glob, label, what the label means)
SETS = [
    ("verify_v1_control_*.png", "BAD",  "v1 design, bare human hands"),
    ("glove_v2_+_tips_*.png",   "OLD",  "v2 design, best of the old assets"),
    ("glove_v3_*.png",          "GOOD", "v3 canon, verified by eye"),
    ("grip_*.png",              "HAND", "asked for no hand, produced one"),
    ("gripshot_*.png",          "HAND", "asked for no hand, produced one"),
]


def load_sets(output_dir):
    found = {}
    for pattern, label, meaning in SETS:
        paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if not paths:
            continue
        found.setdefault(label, {"paths": [], "meaning": meaning})
        found[label]["paths"].extend(paths)
    return found


def auc(pos, neg):
    """Probability a random positive outranks a random negative.

    Rank-based, so it needs no threshold. 1.0 is perfect separation, 0.5 is
    a coin flip, below 0.5 means the metric is backwards.
    """
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------- embeddings

def _pil(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def metric_clip(anchor_path, paths):
    """Cosine similarity to the anchor in CLIP image space."""
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()

    def embed(path):
        tensor = preprocess(_pil(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(tensor)
        return vec / vec.norm(dim=-1, keepdim=True)

    anchor = embed(anchor_path)
    return [float((embed(p) @ anchor.T).item()) for p in paths]


def metric_dinov2(anchor_path, paths):
    """Cosine similarity in DINOv2 space -- built for instance identity."""
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


def metric_palette(anchor_path, paths):
    """Negative distance between colour histograms.

    The cheap baseline, included deliberately. If a 3D histogram separates the
    classes as well as a 2GB vision transformer, that is worth knowing before
    anything heavier gets wired into the build loop.
    """
    import numpy as np

    def hist(path):
        arr = np.asarray(_pil(path).resize((256, 256)), dtype=np.float32) / 255.0
        h, _ = np.histogramdd(arr.reshape(-1, 3), bins=(8, 8, 8),
                              range=((0, 1), (0, 1), (0, 1)))
        h = h.ravel()
        return h / (h.sum() + 1e-8)

    anchor = hist(anchor_path)
    # Negated so that, like the others, higher is more similar.
    return [float(-np.abs(hist(p) - anchor).sum()) for p in paths]


def metric_hands(_anchor_path, paths):
    """How confidently a five-fingered human hand is visible.

    Not a similarity score and not compared against the anchor: this is the
    anatomy gate. The production's recurring defect is the model substituting a
    human hand for a limb the design never defined, and unlike identity that is
    an objective, countable thing.

    Returned as a count so that higher means *more* hand. For the identity table
    it is therefore expected to be backwards -- an AUC well below 0.5 here is
    the metric working, not failing.
    """
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.environ.get("HAND_LANDMARKER_TASK", HAND_TASK)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"hand_landmarker.task not found at {model_path}; set HAND_LANDMARKER_TASK")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.3,
    )
    scores = []
    with vision.HandLandmarker.create_from_options(options) as detector:
        for path in paths:
            arr = np.asarray(_pil(path), dtype=np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            result = detector.detect(image)
            scores.append(float(len(result.hand_landmarks or [])))
    return scores


METRICS = {
    "clip": metric_clip,
    "dinov2": metric_dinov2,
    "palette": metric_palette,
    "hands": metric_hands,
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--anchor", default=ANCHOR)
    parser.add_argument("--metrics", default="clip,dinov2,palette,hands")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    if not os.path.exists(args.anchor):
        print(f"No anchor at {args.anchor}")
        return 1

    found = load_sets(args.output_dir)
    if not found:
        print(f"No labelled renders under {args.output_dir}")
        return 1

    print("Labelled set")
    print("=" * 68)
    for label, data in found.items():
        print(f"  {label:<5} {len(data['paths']):>3} renders -- {data['meaning']}")
    print(f"\nAnchor: {args.anchor}\n")

    results = {}
    for name in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        fn = METRICS.get(name)
        if fn is None:
            print(f"  unknown metric: {name}")
            continue
        print(f"--- {name} ---")
        started = time.time()
        try:
            scores = {label: fn(args.anchor, data["paths"])
                      for label, data in found.items()}
        except Exception as error:  # noqa: BLE001
            print(f"  unavailable: {type(error).__name__}: {error}\n")
            continue
        elapsed = time.time() - started
        total = sum(len(v) for v in scores.values())
        results[name] = scores
        for label in scores:
            values = scores[label]
            print(f"  {label:<5} n={len(values):<3} "
                  f"min={min(values):+.4f}  mean={sum(values)/len(values):+.4f}  "
                  f"max={max(values):+.4f}")
        print(f"  ({elapsed:.1f}s for {total} images, "
              f"{elapsed/max(total,1)*1000:.0f}ms each)\n")

    # ------------------------------------------------------------- the verdict
    print("=" * 68)
    print("Separation -- can the metric rank GOOD above the rest?")
    print("=" * 68)
    print(f"{'metric':<10} {'GOOD vs BAD':>13} {'GOOD vs OLD':>13} {'GOOD vs HAND':>13}")
    for name, scores in results.items():
        row = f"{name:<10}"
        for other in ("BAD", "OLD", "HAND"):
            if "GOOD" in scores and other in scores:
                row += f"{auc(scores['GOOD'], scores[other]):>13.3f}"
            else:
                row += f"{'--':>13}"
        print(row)

    print("\n1.000 = perfect, 0.500 = coin flip, below 0.500 = backwards.")
    print("For 'hands', backwards is correct: it measures hand presence,")
    print("not similarity, so GOOD should score lower than the grip set.")

    # The honest question is not just whether the ordering is right but whether
    # a threshold could survive a new character, which needs daylight.
    print("\nMargin (min GOOD - max BAD; positive means a clean threshold exists)")
    for name, scores in results.items():
        if "GOOD" in scores and "BAD" in scores:
            margin = min(scores["GOOD"]) - max(scores["BAD"])
            verdict = "separable" if margin > 0 else "OVERLAPS"
            print(f"  {name:<10} {margin:+.4f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
