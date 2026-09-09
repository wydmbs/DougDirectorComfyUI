"""
compare_conditioning.py — IP-Adapter against Kontext, on the same reference.

Two mechanisms, one reference image, one thing asked of them: turn this
character and light him differently. Anything else held constant, so the sheet
that comes out is about the mechanisms and nothing else.

The comparison is deliberately unflattering to neither. IP-Adapter is given the
weight it does best at, and Kontext is prompted the way it wants to be prompted
-- as an instruction, since it is an edit model and re-describing the character
makes it redraw rather than transform.

    python compare_conditioning.py --reference rooster_ref_v1.png
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
from kontext_workflow import resolve_kontext_models, build_kontext_workflow  # noqa: E402


def _post(url, path, payload):
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(url, path):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=60) as r:
        return json.load(r)


def _img(url, meta):
    q = urllib.parse.urlencode({"filename": meta["filename"],
                                "subfolder": meta.get("subfolder", ""),
                                "type": meta.get("type", "output")})
    with urllib.request.urlopen(url.rstrip("/") + "/view?" + q, timeout=120) as r:
        return r.read()


def render(url, wf, label, timeout=900):
    pid = _post(url, "/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())}).get("prompt_id")
    if not pid:
        raise RuntimeError(f"{label}: rejected")
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = _get(url, f"/history/{pid}")
        if pid in hist:
            st = hist[pid].get("status", {}) or {}
            if st.get("status_str") == "error":
                raise RuntimeError(f"{label}: {json.dumps(st.get('messages', st))[:700]}")
            for out in (hist[pid].get("outputs") or {}).values():
                for im in out.get("images") or []:
                    print(f"  {label}: {time.time() - t0:.0f}s")
                    return im
            raise RuntimeError(f"{label}: no image")
        time.sleep(2)
    raise RuntimeError(f"{label}: timeout")


def sheet(tiles, out_path, cell=560):
    from PIL import Image, ImageDraw
    band = 44
    cols = min(len(tiles), 3)
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * cell, rows * (cell + band)), "white")
    d = ImageDraw.Draw(img)
    for i, (label, data) in enumerate(tiles):
        t = Image.open(io.BytesIO(data)).convert("RGB")
        t.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + band)
        img.paste(t, (x + (cell - t.width) // 2, y + band))
        d.rectangle([x, y, x + cell, y + band], fill="black")
        d.text((x + 10, y + 14), label, fill="white")
    img.save(out_path)
    return out_path


DESC = ("A stout barnyard rooster with deep rust-red plumage and darker bronze "
        "wings, a tall floppy scarlet comb, bright amber eyes, a short stubby "
        "yellow-orange beak, wearing a fitted crimson waistcoat with exactly "
        "three brass buttons over a plain cream collarless shirt")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--reference", required=True)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="conditioning_compare.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    info = fetch_object_info(a.url)
    tiles = []

    # --- IP-Adapter, at the weights it does best at ----------------------
    ipa = resolve_models(info)
    for w in (1.0, 1.3):
        wf, m = build_workflow(ipa, reference_image=a.reference, width=1024,
                               height=1024, steps=a.steps, guidance=3.5,
                               weight=w, seed=a.seed)
        wf[m["positive_prompt_node"]]["inputs"]["text"] = (
            f"{DESC}. Three-quarter view, head turned to his left, dramatic "
            "warm side-lighting, dark rustic kitchen background.")
        wf["11"]["inputs"]["filename_prefix"] = f"cmp_ipa_{w}"
        tiles.append((f"IP-Adapter  weight {w}", _img(a.url, render(a.url, wf, f"ipa {w}"))))

    # --- Kontext, prompted as the edit model it is -----------------------
    kx = resolve_kontext_models(info)
    instructions = [
        ("Kontext  turn + relight",
         "Turn him to a three-quarter view with his head turned to his left. "
         "Light him with dramatic warm side-lighting and place him in a dark "
         "rustic kitchen. Keep the character exactly the same."),
        ("Kontext  close-up",
         "Move the camera to an extreme close-up of his head and shoulders, "
         "squinting. Dramatic warm side-lighting. Keep the character exactly "
         "the same."),
    ]
    for label, instr in instructions:
        wf, m = build_kontext_workflow(kx, reference_image=a.reference,
                                       steps=a.steps, guidance=2.5, seed=a.seed)
        wf[m["positive_prompt_node"]]["inputs"]["text"] = instr
        wf["13"]["inputs"]["filename_prefix"] = "cmp_kontext"
        tiles.append((label, _img(a.url, render(a.url, wf, label))))

    path = sheet(tiles, a.out)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        raise SystemExit(1)
