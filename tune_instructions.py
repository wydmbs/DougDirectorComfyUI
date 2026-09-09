"""
tune_instructions.py — how much of a Kontext prompt should be an instruction?

Kontext inherits everything you don't mention. That sounds like a convenience
and is actually the whole discipline: naming the wardrobe or the lighting is how
they drift, because a named attribute gets regenerated rather than carried over.

The first comparison this project ran asked for "dramatic warm side-lighting" and
"a dark rustic kitchen" alongside the pose change, and got back a character in a
different jacket under different light. That looked like Kontext being loose. It
was the prompt asking for it.

So this renders the same two shots at three levels of instruction discipline,
with the reference, the seed, the steps and the guidance all held fixed:

    verbose   the pose change plus lighting and setting, as originally written
    surgical  the pose change and nothing else
    guarded   the pose change, plus an explicit hands-off clause

If surgical holds the waistcoat and the light while verbose doesn't, the fix is
in how the app words instructions, not in the model or its settings.

    python tune_instructions.py --reference rooster_ref_v1.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_conditioning import _img, render, sheet  # noqa: E402
from flux_workflow_builder import fetch_object_info  # noqa: E402
from kontext_workflow import resolve_kontext_models, build_kontext_workflow  # noqa: E402

TURN = "Turn him to a three-quarter view with his head turned to his left."
CLOSE = "Move the camera to an extreme close-up of his head and shoulders."

# The clause is deliberately about categories rather than specifics. Listing
# "crimson waistcoat, three brass buttons" would name the very attributes we
# want inherited, which is the mistake this script exists to measure.
GUARD = (" Keep his clothing, colours, markings and the lighting exactly as they "
         "are, and change nothing else.")
EMBELLISH = (" Light him with dramatic warm side-lighting and place him in a dark "
             "rustic kitchen.")

STYLES = [
    ("verbose", lambda base: base + EMBELLISH + " Keep the character the same."),
    ("surgical", lambda base: base),
    ("guarded", lambda base: base + GUARD),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--reference", required=True)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="instruction_compare.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))
    tiles = []
    for shot_name, base in (("turn", TURN), ("close-up", CLOSE)):
        for style_name, shape in STYLES:
            wf, m = build_kontext_workflow(models, reference_image=a.reference,
                                           steps=a.steps, guidance=2.5, seed=a.seed)
            wf[m["positive_prompt_node"]]["inputs"]["text"] = shape(base)
            wf["13"]["inputs"]["filename_prefix"] = f"instr_{shot_name}_{style_name}"
            label = f"{shot_name}  /  {style_name}"
            tiles.append((label, _img(a.url, render(a.url, wf, label))))

    print(f"\nWrote {sheet(tiles, a.out)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
