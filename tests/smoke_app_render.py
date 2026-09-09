"""tests/smoke_app_render.py -- the app itself renders, not a probe script.

Everything proven against the live GPU so far was proven by scripts that build a
graph and post it themselves. None of it touched `director_engine.generate()`,
which is the function the Build buttons and Harry's tools both call. So the
pipeline was demonstrated and the *application* was not, which is exactly the
gap BUILD_LIST item 5 describes: until this passes, everything downstream is
theory.

`tests/smoke_flux_render.py` is the neighbouring test and a different claim. It
asks whether FLUX plus an adapter can hold a face. This one asks whether the app
can drive a render at all -- config, workflow file, node mapping, upload, queue,
poll, fetch, save -- with no probe script standing in for any of it.

Three things are checked, because only the first is obvious:

    reachable   the seam is configured: not in mock mode, a workflow on disk,
                and a node mapping that resolves against it. A misconfigured app
                fails here with a sentence rather than a stack trace.

    honoured    the reference actually arrives. This is the check that matters.
                A LoadImage pointed at a filename ComfyUI cannot see loads
                nothing, the graph still completes, and a picture of a stranger
                comes back looking like a success. So the reference is uploaded
                through the app's own client and the graph is inspected to
                confirm the name landed on the loader feeding ReferenceLatent.

    real        an image came back from the GPU rather than the mock. The mock
                is a deterministic 512x512 card, so a render at Kontext's own
                resolution cannot be confused for one.

Costs about a minute of GPU time. Not part of the unit suite -- it needs a live
ComfyUI, which is the whole point of it.

    python tests/smoke_app_render.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import director_engine  # noqa: E402
from comfy_client import ComfyClient  # noqa: E402
from config import load_config  # noqa: E402
from reference_conditioning import apply_reference  # noqa: E402

MOCK_SIZE = (512, 512)

SHOT = "Put him at a wooden table, seen from the front."


def check(condition, ok_message: str, fail_message: str) -> None:
    if not condition:
        raise AssertionError(fail_message)
    print(f"  [ ok ] {ok_message}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="smoke_out/rooster_ref_v3.png",
                   help="A character reference on local disk, as a user would attach.")
    p.add_argument("--label", default="smoke_app")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("App render smoke test")
    print("=" * 60)

    cfg = load_config()

    # ---- reachable ------------------------------------------------------
    check(not cfg.mock_mode,
          "mock mode is off",
          "Mock mode is on, so this would prove nothing. Turn it off in Setup.")
    check(cfg.workflow_json_path and os.path.exists(cfg.workflow_json_path),
          f"workflow on disk: {os.path.basename(cfg.workflow_json_path or '')}",
          "No workflow configured. Run: python kontext_workflow.py --write")
    check(os.path.exists(a.reference),
          f"reference on disk: {a.reference}",
          f"No reference image at {a.reference}")

    client = ComfyClient(cfg.comfyui_url)
    check(client.ping(),
          f"ComfyUI answering at {cfg.comfyui_url}",
          f"No ComfyUI at {cfg.comfyui_url}. Start it, or check the firewall.")

    with open(cfg.workflow_json_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    check(cfg.node_mapping.positive_prompt_node in workflow,
          f"prompt node '{cfg.node_mapping.positive_prompt_node}' resolves",
          "The node mapping points at a node the workflow does not contain.")

    # ---- honoured -------------------------------------------------------
    # Done explicitly rather than trusting the render, because the failure this
    # guards against is silent: the graph completes either way.
    conditioned, note = apply_reference(
        json.loads(json.dumps(workflow)), cfg, a.reference, "",
        upload=client.upload_image)

    uploaded = client.upload_image(a.reference)
    loaders = [n for n in conditioned.values()
               if n.get("class_type") == "LoadImage"
               and n.get("inputs", {}).get("image") == uploaded]
    check(loaders,
          f"reference reached the loader as '{uploaded}'",
          "The reference never landed on a LoadImage node, so the render would "
          f"have quietly ignored it. Conditioning note: {note or '(none)'}")

    # ---- real -----------------------------------------------------------
    print("\n  rendering through director_engine.generate()...")
    result = director_engine.generate(
        cfg, a.label, SHOT, "", base_seed=4242, n_variants=1,
        reference_image_path=a.reference,
        on_progress=lambda message: print(f"    {message}"))

    blocking = [w for w in result.warnings if "ignored" in w.lower() or "no images" in w.lower()]
    check(not blocking,
          "no warnings that would mean a silent miss",
          f"Generation reported: {blocking}")
    check(result.variants,
          f"{len(result.variants)} variant returned",
          f"No variants came back. Warnings: {result.warnings}")

    path = result.variants[0].image_path
    check(os.path.exists(path), f"saved to {path}", f"Variant path missing: {path}")

    with Image.open(path) as image:
        size = image.size
    check(size != MOCK_SIZE,
          f"rendered at {size[0]}x{size[1]} -- a real render, not the mock card",
          "The image is the mock's 512x512 placeholder, so no GPU work happened.")

    if result.warnings:
        print("\n  notes (not failures):")
        for warning in result.warnings:
            print(f"    - {warning}")

    print("\n" + "=" * 60)
    print("PASS -- the app rendered on the GPU with the reference honoured.")
    print(f"  {path}")
    print("\nThe test proves the reference arrived. Only you can say whether the")
    print("picture is the right character.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"\nFAIL -- {error}", file=sys.stderr)
        raise SystemExit(1)
