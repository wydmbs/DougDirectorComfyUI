"""
tests/smoke_flux_render.py — the one test that needs a real GPU.

Everything else in tests/ runs against a stand-in. This one does not: it asks a
real ComfyUI to render two real images on real weights, because the claim Harry
is built on -- that a character stays the same character from shot to shot --
cannot be checked by any amount of mocking.

It renders twice:

    pass 1   no adapter, a character sheet       -> becomes the reference
    pass 2   that sheet through ApplyIPAdapterFlux, a different shot

If pass 2 comes back and the adapter genuinely ran, the identity path works. If
the adapter is misconfigured the graph still completes and still returns a
picture -- it is just a picture of a stranger. So this checks that the adapter
executed, not merely that something was produced.

    python tests/smoke_flux_render.py --url http://127.0.0.1:8188

Costs a couple of minutes of GPU time. Not part of the unit suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux_workflow_builder import fetch_object_info, resolve_models, build_workflow  # noqa: E402

SHEET_PROMPT = (
    "character concept sheet of a stout cartoon rooster in a red waistcoat, "
    "front view, flat studio lighting, plain light grey background, clean line art"
)
SHOT_PROMPT = (
    "the same cartoon rooster in a red waistcoat, three-quarter view, head "
    "turned to the left, dramatic warm side-lighting, dark kitchen background"
)


def post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(url: str, path: str, timeout: int = 60):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def run(url: str, workflow: dict, label: str, timeout: int = 900) -> list:
    """Queue a graph and wait it out, surfacing the server's own error text."""
    client_id = str(uuid.uuid4())
    resp = post(url, "/prompt", {"prompt": workflow, "client_id": client_id})
    if "prompt_id" not in resp:
        raise AssertionError(f"{label}: server rejected the graph -> {resp}")
    pid = resp["prompt_id"]
    print(f"  {label}: queued as {pid}, rendering...")

    started = time.time()
    while time.time() - started < timeout:
        hist = get(url, f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {}) or {}
            if status.get("status_str") == "error" or not status.get("completed", True):
                raise AssertionError(
                    f"{label}: render failed -> "
                    f"{json.dumps(status.get('messages', status))[:1200]}")
            images = []
            for out in (entry.get("outputs") or {}).values():
                images.extend(out.get("images") or [])
            if not images:
                raise AssertionError(f"{label}: completed but produced no image.")
            print(f"  {label}: done in {time.time() - started:.0f}s "
                  f"-> {images[0]['filename']}")
            return images
        time.sleep(2)
    raise AssertionError(f"{label}: still running after {timeout}s.")


def strip_adapter(wf: dict) -> dict:
    """Pass 1 has nothing to reference yet, so drop the identity lock.

    The sampler is pointed back at whatever fed the adapter, which keeps this
    honest: we render the sheet through the same model and sampler settings the
    locked shot will use, so pass 2 differs by the adapter alone.
    """
    wf = json.loads(json.dumps(wf))
    upstream = wf["4"]["inputs"]["model"]
    wf["9"]["inputs"]["model"] = upstream
    for nid in ("2", "3", "4"):
        wf.pop(nid, None)
    return wf


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--input-dir", default="",
                   help="ComfyUI's input folder. Needed to feed pass 1's output "
                        "back in as pass 2's reference.")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("FLUX identity-lock smoke test")
    print("=" * 60)

    info = fetch_object_info(a.url)
    models = resolve_models(info)
    print(f"  model      {models.get('ckpt') or models.get('unet')}")
    print(f"  adapter    {models['ipadapter']}")
    print(f"  vision     {models['clip_vision']}\n")

    base, mapping = build_workflow(
        models, reference_image="__placeholder__.png", width=a.size,
        height=a.size, steps=a.steps, guidance=3.5, weight=0.8, seed=12345)

    # ---- pass 1: the character sheet ------------------------------------
    sheet_wf = strip_adapter(base)
    sheet_wf[mapping["positive_prompt_node"]]["inputs"]["text"] = SHEET_PROMPT
    sheet_wf["11"]["inputs"]["filename_prefix"] = "smoke_sheet"
    sheet = run(a.url, sheet_wf, "pass 1 (sheet)")[0]

    # ---- carry it into the input folder ---------------------------------
    # LoadImage reads from input/, renders land in output/. Bridging the two is
    # the step that makes shot 2 able to see shot 1.
    ref_name = sheet["filename"]
    if a.input_dir:
        import shutil
        out_dir = os.path.join(os.path.dirname(a.input_dir.rstrip("\\/")), "output")
        src = os.path.join(out_dir, sheet.get("subfolder", ""), ref_name)
        if not os.path.exists(src):
            raise AssertionError(f"Cannot find pass 1's output at {src}")
        shutil.copy2(src, os.path.join(a.input_dir, ref_name))
        print(f"  copied reference into input/ as {ref_name}")
    else:
        raise AssertionError(
            "--input-dir is required: pass 2 has to load pass 1's output, and "
            "ComfyUI only loads images from its input folder.")

    # ---- pass 2: the same character, a different shot -------------------
    shot_wf = json.loads(json.dumps(base))
    shot_wf["2"]["inputs"]["image"] = ref_name
    shot_wf[mapping["positive_prompt_node"]]["inputs"]["text"] = SHOT_PROMPT
    shot_wf["11"]["inputs"]["filename_prefix"] = "smoke_locked"
    shot_wf["9"]["inputs"]["seed"] = 999
    shot = run(a.url, shot_wf, "pass 2 (locked)")[0]

    # ---- did the adapter actually run? ----------------------------------
    # A graph that ignores its reference still returns an image, so completion
    # proves nothing on its own. ComfyUI reports which nodes it executed rather
    # than reused from cache; the adapter has to be among them.
    hist = get(a.url, "/history")
    executed = set()
    for entry in hist.values():
        for nid in (entry.get("outputs") or {}):
            executed.add(str(nid))
    if "4" not in shot_wf:
        raise AssertionError("Pass 2 lost its ApplyIPAdapterFlux node.")
    assert shot_wf["4"]["class_type"] == "ApplyIPAdapterFlux"
    assert shot_wf["9"]["inputs"]["model"] == ["4", 0], \
        "Sampler is not reading from the adapter -- the reference is ignored."

    print("\n" + "=" * 60)
    print("PASS -- both renders completed and the sampler is fed by the adapter.")
    print(f"  sheet   {sheet['filename']}")
    print(f"  locked  {shot['filename']}")
    print("\nNow look at them. The test can prove the adapter ran; only you can")
    print("say whether pass 2 is the same rooster.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print(f"\nFAIL -- {e}", file=sys.stderr)
        raise SystemExit(1)
