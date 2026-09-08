"""
tune_identity.py — find the IP-Adapter weight that actually holds a character.

"It's a rooster, but slightly different" is the whole problem in one sentence.
The adapter is clearly doing something, and just as clearly not doing enough --
but the fix is a number, and nobody can guess a number by staring at one image.

So render the same shot across a range of weights, change nothing else, and
look at them side by side. Weight 0.0 is included deliberately as a control:
it shows what the prompt alone produces, which is the only way to tell how much
of the resemblance is the adapter and how much was always going to happen
because you asked for a rooster in a waistcoat either way.

The seed is held fixed across the sweep for the same reason. Vary two things
and you learn nothing about either.

    python tune_identity.py --reference sheet.png \\
        --prompt "three-quarter view, warm side-lighting, dark kitchen" \\
        --weights 0,0.5,0.7,0.85,1.0

Writes one labelled contact sheet. Look at it, pick the lowest weight that
still holds the face -- going higher than that starts dragging the reference's
pose and lighting into every shot, which is its own kind of drift.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux_workflow_builder import fetch_object_info, resolve_models, build_workflow  # noqa: E402


def _post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get_json(url: str, path: str):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=60) as r:
        return json.load(r)


def _fetch_image(url: str, meta: dict) -> bytes:
    """Pull a render straight out of the server rather than off the disk.

    Saves the caller from knowing where ComfyUI keeps its output folder, which
    differs between a portable install and a source checkout.
    """
    q = urllib.parse.urlencode({
        "filename": meta["filename"],
        "subfolder": meta.get("subfolder", ""),
        "type": meta.get("type", "output")})
    with urllib.request.urlopen(url.rstrip("/") + "/view?" + q, timeout=120) as r:
        return r.read()


def render(url: str, workflow: dict, label: str, timeout: int = 900) -> dict:
    resp = _post(url, "/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError(f"{label}: server rejected the graph -> {resp}")
    started = time.time()
    while time.time() - started < timeout:
        hist = _get_json(url, f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {}) or {}
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"{label}: {json.dumps(status.get('messages', status))[:600]}")
            for out in (entry.get("outputs") or {}).values():
                for img in out.get("images") or []:
                    print(f"  weight {label}: {time.time() - started:.0f}s")
                    return img
            raise RuntimeError(f"{label}: finished with no image")
        time.sleep(2)
    raise RuntimeError(f"{label}: timed out")


def contact_sheet(tiles: list, out_path: str, cell: int = 512) -> str:
    """One image to look at instead of eight to click through."""
    from PIL import Image, ImageDraw

    band = 40
    cols = min(len(tiles), 4)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + band)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (label, data) in enumerate(tiles):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((cell, cell))
        x = (i % cols) * cell
        y = (i // cols) * (cell + band)
        sheet.paste(img, (x + (cell - img.width) // 2, y + band))
        draw.rectangle([x, y, x + cell, y + band], fill="black")
        draw.text((x + 10, y + 12), label, fill="white")

    sheet.save(out_path)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--reference", required=True,
                   help="Filename already in ComfyUI's input folder.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--weights", default="0,0.5,0.7,0.85,1.0")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", default="identity_sweep.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    weights = [float(w) for w in a.weights.split(",") if w.strip()]
    info = fetch_object_info(a.url)
    models = resolve_models(info)

    print(f"Sweeping {len(weights)} weights against {a.reference}")
    print(f"  seed {a.seed} held fixed -- only the adapter weight changes\n")

    tiles = []
    for w in weights:
        wf, mapping = build_workflow(
            models, reference_image=a.reference, width=a.size, height=a.size,
            steps=a.steps, guidance=3.5, weight=w, seed=a.seed)
        wf[mapping["positive_prompt_node"]]["inputs"]["text"] = a.prompt
        wf["11"]["inputs"]["filename_prefix"] = f"sweep_{str(w).replace('.', '_')}"

        if w == 0.0:
            # A weight of zero still runs the adapter and can still tint the
            # result. Removing it outright is the honest control: this is the
            # prompt with no reference at all.
            upstream = wf["4"]["inputs"]["model"]
            wf["9"]["inputs"]["model"] = upstream
            for nid in ("2", "3", "4"):
                wf.pop(nid, None)
            label = "0.00  (no adapter - control)"
        else:
            label = f"weight {w:.2f}"

        meta = render(a.url, wf, f"{w:.2f}")
        tiles.append((label, _fetch_image(a.url, meta)))

    path = contact_sheet(tiles, a.out)
    print(f"\nWrote {path}")
    print("Pick the lowest weight that still looks like the same bird. Higher")
    print("than that and the reference's pose and lighting start coming along")
    print("with the face, which limits what you can stage in later shots.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        raise SystemExit(1)
