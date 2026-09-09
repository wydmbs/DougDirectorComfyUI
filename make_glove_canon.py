"""make_glove_canon.py -- settle the limb end, once, as an asset.

Nine renders under four instructions all converged on the same thing: a dark
five-fingered glove. `probe_grip.py` established that no amount of conditioning
talks the model out of it, because a grip is a structure the design never had
and the model fills that hole the same way every time.

So the design stops fighting it. The glove becomes canon -- which is also what
animation has done for eighty years, and for the same reason.

The distinction that makes this cheap is the one probe_grip proved the hard way:

    a grip    is a STRUCTURE. Kontext cannot add one. Four studies, zero wins.
    a glove   is an ATTRIBUTE -- a costume element on a limb that already
              exists. That is Kontext's best operation, and it is the same
              repair path that produced the v2 wing tips.

So no ChatGPT redraw is needed. That matters beyond convenience: a redraw
invalidates every reference downstream of the current sheet, and a repair does
not.

What is being pinned down is consistency, not appearance. The model already
draws this glove unprompted; what it does not do is draw the *same* glove
twice, and it disagrees with itself about whether a resting limb is a glove or
a bare feathered wing tip. Two answers for one limb is the continuity bug in
miniature. Canon means one answer, everywhere:

    GLOVE   dark brown leather, turned-back cuff at the wrist, worn where the
            wing feathers end. Present in every shot, at rest and under load.

Finger count is deliberately left unspecified. The model consistently draws
five; insisting on the classic four would be fighting it again, for a detail no
viewer checks.

Three assets, because they answer different questions -- the same split the v2
set uses:

    ref     identity. One clean front view, gloves visible.
    sheet   the turnaround, so the gloves survive a change of angle.
    plate   the gloves doing things -- resting, gripping, gesturing -- which is
            what every failing shot actually needed.

    python make_glove_canon.py --url http://192.168.1.206:8188
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_conditioning import _img, render, sheet  # noqa: E402
from flux_workflow_builder import fetch_object_info  # noqa: E402
from kontext_workflow import resolve_kontext_models, build_kontext_workflow  # noqa: E402
from probe_multi_reference import upload  # noqa: E402

# The one sentence the whole design now rests on. It is repeated verbatim into
# every asset rather than paraphrased per shot, because paraphrase is how the
# glove drifted in the first place.
GLOVE = ("dark brown leather gloves with a turned-back cuff at the wrist, worn "
         "where his wing feathers end")

# Each asset is an edit of the matching v2 source, so the identity that v2
# already establishes is carried rather than re-derived.
ASSETS = [
    (
        "ref",
        "rooster_ref_v2.png",
        [],
        f"Give him {GLOVE}. Keep everything else exactly as it is.",
    ),
    (
        "sheet",
        "rooster_sheet_v2.png",
        [],
        f"Give him {GLOVE} in every view on this turnaround. Keep every pose, "
        f"angle and layout exactly as it is.",
    ),
    (
        # Built on the new reference rather than on v2, so the plate cannot
        # disagree with the reference about what the glove looks like.
        "plate",
        None,
        [],
        f"Show a study sheet of his gloved hands in four views: resting flat on "
        f"a wooden table, gripping a wooden spoon, open with the fingers spread, "
        f"and raised mid-gesture. He wears {GLOVE}. Plain grey background.",
    ),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--outdir", default="smoke_out")
    p.add_argument("--contact", default="smoke_out/17_glove_canon.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))

    tiles = []
    produced: dict = {}
    for name, source, extra, instruction in ASSETS:
        # The plate hangs off the freshly made reference; everything else edits
        # its v2 counterpart.
        source = source or produced["ref"]
        wf, m = build_kontext_workflow(
            models, reference_image=source, steps=a.steps, guidance=a.guidance,
            seed=a.seed, extra_references=extra, multi="chain")
        wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
        wf["13"]["inputs"]["filename_prefix"] = f"canon_{name}"

        data = _img(a.url, render(a.url, wf, f"canon / {name}"))

        local = os.path.join(a.outdir, f"rooster_{name}_v3.png")
        with open(local, "wb") as handle:
            handle.write(data)
        # Re-uploaded so later shots can load it by name: renders land in
        # output/ and LoadImage reads input/.
        produced[name] = upload(a.url, local)
        tiles.append((f"v3 {name}", data))
        print(f"    {name:<6} {local}  ->  {produced[name]}")

    print(f"\nWrote {sheet(tiles, a.contact, columns=len(ASSETS))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
