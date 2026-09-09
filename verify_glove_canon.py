"""verify_glove_canon.py -- is the glove actually canon, or just usually there?

Making a decision is not the same as making it hold. The v2 lesson was that an
asset which looks right on its own sheet can stop working the moment it
conditions a shot, so the redraw and the conditioning have to be separated
again here.

Canon makes a stronger claim than "the hands look better", and it is worth
stating the claim precisely, because that is what decides whether this passes:

    one limb end     the same gloved hand in every shot. A feathered wing tip
                     at rest and a glove under load is still two answers for
                     one limb, which is the continuity bug this exists to kill.
    same glove       colour, cuff and material stable across shots, not
                     re-improvised per render.
    no bare hand     no pink human hand anywhere, which was the original defect.

Three shots, chosen so the limb end has to do three different things:

    at a table   the shot that started this. Limb end at rest.
    holding      the shot that produced the glove. Limb end under load.
    pointing     unseen by any asset, so it tests whether the design generalises
                 rather than whether these three plates memorised three poses.

Four ways of supplying the character, so a win is attributable:

    v2 + tips    the best the old assets ever managed. Present because the
                 comparison has to be against the real previous state.
    v3 alone     the new reference on its own. If this passes, the glove rides
                 in the identity and no shot needs to chain anything.
    v3 + plate   the reference with the gloved-hand study chained.
    v3 + both    reference, turnaround and plate, as a real shot would run.

The interesting outcome is v3 alone. The whole appeal of adopting the glove was
that the model already draws it unprompted and consistently -- if that is true,
canon is free, and every shot from here costs one reference instead of three.

    python verify_glove_canon.py --url http://192.168.1.206:8188
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_conditioning import _img, render, sheet  # noqa: E402
from flux_workflow_builder import fetch_object_info  # noqa: E402
from kontext_workflow import resolve_kontext_models, build_kontext_workflow  # noqa: E402

SHOTS = [
    ("at a table", "Put him at a wooden table, seen from the front."),
    ("holding", "Have him hold a wooden spoon."),
    # Deliberately outside what any plate shows.
    ("pointing", "Have him point off to one side."),
]

CONFIGS = [
    ("v2 + tips", "v2", ["tips"]),
    ("v3 alone", "v3", []),
    ("v3 + plate", "v3", ["plate"]),
    ("v3 + both", "v3", ["sheet", "plate"]),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--v2", default="rooster_ref_v2.png")
    p.add_argument("--tips", default="rooster_wingtips_v2.png")
    p.add_argument("--v3", default="rooster_ref_v3.png")
    p.add_argument("--sheet", default="rooster_sheet_v3.png")
    p.add_argument("--plate", default="rooster_plate_v3.png")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="smoke_out/18_glove_verify.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))
    images = {"v2": a.v2, "tips": a.tips, "v3": a.v3,
              "sheet": a.sheet, "plate": a.plate}

    tiles = []
    for shot, instruction in SHOTS:
        for name, ref_key, extra_keys in CONFIGS:
            wf, m = build_kontext_workflow(
                models, reference_image=images[ref_key], steps=a.steps,
                guidance=2.5, seed=a.seed,
                extra_references=[images[k] for k in extra_keys], multi="chain")
            wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
            wf["13"]["inputs"]["filename_prefix"] = "glove_" + name.replace(" ", "_")
            label = f"{shot}  /  {name}"
            tiles.append((label, _img(a.url, render(a.url, wf, label))))

    print(f"\nWrote {sheet(tiles, a.out, columns=len(CONFIGS))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
