"""
verify_wing_design.py -- does the redrawn sheet actually fix the hands?

The v2 turnaround gives the character something the v1 sheet never had: a
defined limb end. The wings fold down each side and the primaries taper to a
point at hip height, and there is no arm silhouette and no hand anywhere on it.

That is the design fixed on paper. It is not yet the problem fixed, because the
failure was never on the sheet -- it was in shots where the limb end has to do
something and the model had nothing to copy. So this puts the new asset against
the exact shot that produced the human hands, and against a harder one.

    at a table   the original failure, unchanged wording
    holding      the limb end has to grip, which is where a folded wing is
                 genuinely awkward and a hand is the easy way out

Four ways of supplying the character, so the answer separates "the redraw fixed
it" from "chaining the sheet fixed it":

    v1 control   the old reference, which is known to fail -- present so the
                 comparison is against the real failure and not a memory of it
    v2 alone     the new front view on its own
    v2 + sheet   the new front view with the full turnaround chained
    v2 + tips    the new front view with the wing-tip plate chained
    v2 + both    everything, which is what a real shot would use

If v2 alone is clean, the design was the whole problem and one reference is
enough. If it needs the tips plate, then any shot where the limb end does
something has to chain it, and the registry has to learn the difference between
a reference and a study before that can happen automatically. Both are
actionable; guessing between them is not.

    python verify_wing_design.py
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
]

# The turnaround shows the folded wing at rest from four angles. The wing-tip
# plate shows it doing things -- resting on a table, cradling a spoon, raised
# mid-gesture -- which is the part every failing shot actually needed. They are
# offered separately because they answer different questions, and together
# because that is how a real shot would be conditioned.
CONFIGS = [
    # Known to fail. Present so the comparison is against the real failure
    # rather than a memory of it.
    ("v1 control", "v1", []),
    ("v2 alone", "v2", []),
    ("v2 + sheet", "v2", ["sheet"]),
    ("v2 + tips", "v2", ["tips"]),
    ("v2 + both", "v2", ["sheet", "tips"]),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--v1", default="rooster_ref_v1.png")
    p.add_argument("--v2", default="rooster_ref_v2.png")
    p.add_argument("--sheet", default="rooster_sheet_v2.png")
    p.add_argument("--tips", default="rooster_wingtips_v2.png")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="wing_verify.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))
    images = {"v1": a.v1, "v2": a.v2, "sheet": a.sheet, "tips": a.tips}

    tiles = []
    for shot, instruction in SHOTS:
        for name, ref_key, extra_keys in CONFIGS:
            wf, m = build_kontext_workflow(
                models, reference_image=images[ref_key], steps=a.steps,
                guidance=2.5, seed=a.seed,
                extra_references=[images[k] for k in extra_keys], multi="chain")
            wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
            wf["13"]["inputs"]["filename_prefix"] = "verify_" + name.replace(" ", "_")
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
