"""probe_grip.py -- can any reference fix a grip, or only dress one up?

The wing-tip study fixed the limb end at rest: at a table the character now
puts a folded wing on the wood instead of a hand. Under load it fixed nothing.
Asked to hold a spoon, the same conditioning produces a five-fingered hand in a
dark glove -- the study's colour and material transferred, its anatomy did not.

That is a worse failure than the one it replaced. A pink human hand is caught by
anyone glancing at the frame; a gloved one reads as a costume choice and ships.

The open question is not which study to chain. It is whether a study can carry
anatomy at all, or only appearance:

    if a study fixes grip   studies are load-bearing for action shots, and the
                            registry has to tell a reference from a study so a
                            shot can chain the right one
    if none of them do      conditioning cannot supply an affordance the design
                            never defined, and the answer belongs in the design
                            -- no plumbing worth building either way

So each candidate is rendered twice, the same way the wing repair was checked:

    row 1   the study itself -- does it show a wing actually gripping?
    row 2   that study chained into the shot that produced the glove -- did the
            grip survive being used?

A candidate only counts if both rows are right. A study that looks correct and
collapses the moment it conditions a shot has fixed nothing, and row 1 alone
would have scored it a success.

The control column carries v2 + tips with no new study: the known glove. The
comparison is against the real failure rather than a memory of it.

    python probe_grip.py --url http://192.168.1.206:8188
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
from probe_wing_hands import fetch_input  # noqa: E402

# The shot that produced the glove. Unchanged wording, so the comparison is
# against the actual failure.
HOLDING = "Have him hold a wooden spoon."

# Each candidate has to answer the question the design left open: what does the
# limb do when it grips. They differ in the mechanism they name, because "no
# hands" is a constraint and the model needs an instruction -- told only what to
# avoid, it substitutes the nearest thing it knows, which is how the glove got
# there.
#
# Ordered from naming the part, to naming the motion, to naming the contact.
STUDIES = [
    # Control: no new study. The known glove.
    ("tips only", ""),
    # Names the acting part, leaving the motion open.
    ("wingtip grip", "Show his folded wing gripping a wooden rod with the long "
                     "primary feathers themselves, no fingers and no hand."),
    # Names the motion. Curling is something feathers can plausibly do, so this
    # asks for a deformation rather than a new appendage.
    ("feather curl", "Show the long primary feathers at his wingtip curling "
                     "around a wooden rod to hold it, no fingers and no hand."),
    # Names the contact and removes the need to encircle anything -- the
    # easiest grip to draw without inventing a thumb.
    ("cradle", "Show a wooden rod held against the underside of his folded wing, "
               "cradled between wing and body, no fingers and no hand."),
    # Two surfaces pressing. Names an opposition, which is the part a hand
    # supplies and a wing does not have.
    ("pinch", "Show a wooden rod pinched between two layers of feathers at his "
              "wingtip, pressed together to hold it, no fingers and no hand."),
]


def study_reference(a, models, name, instruction):
    """Render one grip study and hand back a name ComfyUI can load.

    The render lands in output/ and LoadImage reads input/, so the bytes are
    fetched and re-uploaded rather than referenced by path.
    """
    if not instruction:
        return None, fetch_input(a.url, a.tips)

    wf, m = build_kontext_workflow(models, reference_image=a.ref, steps=a.steps,
                                   guidance=a.guidance, seed=a.seed,
                                   extra_references=[a.tips], multi="chain")
    wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
    wf["13"]["inputs"]["filename_prefix"] = f"grip_{name.replace(' ', '_')}"
    data = _img(a.url, render(a.url, wf, f"study / {name}"))

    local = f"grip_{name.replace(' ', '_')}.png"
    with open(local, "wb") as handle:
        handle.write(data)
    return upload(a.url, local), data


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--ref", default="rooster_ref_v2.png")
    p.add_argument("--tips", default="rooster_wingtips_v2.png")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--out", default="smoke_out/16_grip.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))

    studies, shots = [], []
    for name, instruction in STUDIES:
        study_name, data = study_reference(a, models, name, instruction)
        suffix = "tips plate as-is" if not instruction else "grip study"
        studies.append((f"{name}  /  {suffix}", data))

        # The shot always keeps the tips plate -- the study is being tested as
        # an addition to the best known conditioning, not as a replacement for
        # it. Otherwise a win could just be the tips plate being dropped.
        extra = [a.tips] + ([study_name] if study_name else [])
        wf, m = build_kontext_workflow(models, reference_image=a.ref, steps=a.steps,
                                       guidance=2.5, seed=a.seed,
                                       extra_references=extra, multi="chain")
        wf[m["positive_prompt_node"]]["inputs"]["text"] = HOLDING
        wf["13"]["inputs"]["filename_prefix"] = f"gripshot_{name.replace(' ', '_')}"
        label = f"{name}  /  holding"
        shots.append((label, _img(a.url, render(a.url, wf, label))))

    print(f"\nWrote {sheet(studies + shots, a.out, columns=len(STUDIES))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
