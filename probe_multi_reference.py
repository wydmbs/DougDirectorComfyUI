"""
probe_multi_reference.py — can Kontext hold a character *and* a place?

A single reference is the easy case. The one that decides whether an assistant
director can stage a scene is two: this character, in this room, both of them
recognisable afterwards.

There are two ways to hand Kontext a second reference and they behave
differently enough to be worth seeing side by side:

    chain    each image encoded separately, ReferenceLatents chained. Both
             references keep their full resolution.
    stitch   the images joined into one picture first, so the model reads them
             as a single frame and each gets a share of the tokens.

The backdrop is generated here rather than supplied, so the run is
self-contained: a plain Kontext-free render, locked to a fixed seed, gives a
room that can be compared against afterwards. The point is not whether the room
is beautiful; it is whether the same room comes back.

    python probe_multi_reference.py --reference rooster_ref_v1.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_conditioning import _img, render, sheet  # noqa: E402
from flux_workflow_builder import fetch_object_info, resolve_models, build_workflow  # noqa: E402
from kontext_workflow import resolve_kontext_models, build_kontext_workflow  # noqa: E402

BACKDROP = ("A warm rustic farmhouse kitchen interior, worn oak table in the "
            "foreground, copper pans on a rail, a deep stone sink under a "
            "small window, late afternoon light, empty of people, wide shot")

# An instruction, not a description -- and it names neither the bird nor the
# room, because both are already in the conditioning and anything named is
# regenerated rather than carried across.
PLACE_HIM = "Put him in this kitchen, standing at the table, seen from the front."


def upload(url, path):
    """ComfyUI needs the bytes; a path from another machine means nothing to it."""
    import mimetypes
    import uuid as _uuid
    import urllib.request

    name = os.path.basename(path)
    boundary = f"----probe{_uuid.uuid4().hex}"
    with open(path, "rb") as handle:
        payload = handle.read()
    ctype = mimetypes.guess_type(name)[0] or "image/png"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
        + payload
        + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\""
          f"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        url.rstrip("/") + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    import json as _json
    with urllib.request.urlopen(req, timeout=120) as response:
        return _json.load(response)["name"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--reference", required=True)
    p.add_argument("--backdrop", default="", help="Skip generating one.")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="multi_reference.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    info = fetch_object_info(a.url)
    if "ImageStitch" not in info:
        print("No ImageStitch node -- only the chain route can be tested.", file=sys.stderr)

    tiles = []

    backdrop_name = a.backdrop
    if not backdrop_name:
        flux = resolve_models(info)
        wf, m = build_workflow(flux, reference_image=a.reference, width=1024,
                               height=1024, steps=a.steps, guidance=3.5,
                               weight=0.0, seed=a.seed)
        wf[m["positive_prompt_node"]]["inputs"]["text"] = BACKDROP
        wf["11"]["inputs"]["filename_prefix"] = "probe_backdrop"
        meta = render(a.url, wf, "backdrop")
        backdrop_bytes = _img(a.url, meta)
        tiles.append(("Backdrop reference", backdrop_bytes))
        # It lands in output/; the loaders read input/, so send it back up.
        local = os.path.join(os.path.dirname(os.path.abspath(a.out)), "probe_backdrop.png")
        with open(local, "wb") as handle:
            handle.write(backdrop_bytes)
        backdrop_name = upload(a.url, local)
        print(f"  backdrop available to LoadImage as {backdrop_name}")

    routes = [("chain", "chain"), ("stitch", "stitch")] if "ImageStitch" in info \
        else [("chain", "chain")]
    kx = resolve_kontext_models(info)
    for label, strategy in routes:
        wf, m = build_kontext_workflow(
            kx, reference_image=a.reference, steps=a.steps, guidance=2.5,
            seed=a.seed, extra_references=[backdrop_name], multi=strategy)
        wf[m["positive_prompt_node"]]["inputs"]["text"] = PLACE_HIM
        wf["13"]["inputs"]["filename_prefix"] = f"probe_multi_{strategy}"
        tiles.append((f"Character + backdrop  /  {label}",
                      _img(a.url, render(a.url, wf, label))))

    print(f"\nWrote {sheet(tiles, a.out)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
