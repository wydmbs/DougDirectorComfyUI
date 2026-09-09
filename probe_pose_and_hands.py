"""
probe_pose_and_hands.py -- the two things the Kontext sheets got wrong.

Sheet 7 and sheet 8 held identity beautifully and still had two faults worth
chasing, because both would bite in real shot work:

  Pose      "Turn him to a three-quarter view" barely rotated him. Framing
            changes land; asking the body to move does not.
  Hands     His wings came back as human hands whenever a table was involved.

They look like separate problems and may share a cause: Kontext resists moving
away from the reference, so both the pose it was given and the anatomy it was
given tend to survive -- except the anatomy didn't, which is the interesting
part.

POSE. Two hypotheses, crossed so we can tell them apart.

  guidance   FluxGuidance is how hard the instruction is pressed. The default
             2.5 is tuned for edits that should be gentle. A pose change is not
             gentle. Swept at 2.5 / 4.0 / 6.0.
  phrasing   "Turn him" is an imperative aimed at the subject, and a model that
             predicts pixels has no subject to obey. Three alternatives: move
             the camera instead, state the finished pose as fact, or ask for a
             change too large to split the difference on.

If rotation tracks guidance, it is a settings fix. If it tracks phrasing, it is
a prompt-authoring fix and belongs in review_instruction(). If neither moves it,
pose is not a Kontext job and shots need staging another way.

HANDS. The instruction that produced them was "Put him in this kitchen,
standing at the table, seen from the front." Two suspects in that sentence and
one gap in the graph.

  posture    "standing at the table" is a human pose. The nearest thing in the
             model's prior to a figure standing at a table has hands on it.
  silence    The project rule is "don't name what you want kept" -- but that
             rule is about attributes that are already right. Wings that came
             out as hands are already wrong, and the only way to fix something
             the reference failed to carry is to name it.
  negative   The Kontext graph runs cfg 1.0 with ConditioningZeroOut, so there
             is no negative prompt at all. Nothing can be pushed away. This
             tests a real negative at cfg 2.5 before deciding whether the
             builder should offer one -- it doubles sampling cost, so it needs
             to earn its place.

    python probe_pose_and_hands.py --reference rooster_ref_v1.png
    python probe_pose_and_hands.py --reference rooster_ref_v1.png --only hands
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_conditioning import _img, render, sheet  # noqa: E402
from flux_workflow_builder import fetch_object_info  # noqa: E402
from kontext_workflow import resolve_kontext_models, build_kontext_workflow  # noqa: E402

# Same target pose four ways. Only the grammar changes -- the requested end
# state is a three-quarter view in every case except `profile`, which asks for
# more than we want on purpose: if even that does not move him, the ones asking
# for less never will.
POSE_STYLES = [
    ("subject", "Turn him to a three-quarter view with his head turned to his left."),
    ("camera", "Move the camera around to his left so he is seen in three-quarter view."),
    ("state", "He is standing in three-quarter view, his body angled away to his left."),
    ("profile", "Show him in full side profile, facing left."),
]

GUIDANCES = [2.5, 4.0, 6.0]

# The instruction from probe_multi_reference.py that produced hands, and the
# variants that each remove one suspect from it.
HANDS_BASE = "Put him at a wooden table, seen from the front."

HANDS_STYLES = [
    ("baseline", HANDS_BASE, None),
    # Drops the table framing entirely. If the hands go, the posture cue was
    # doing it and the fix is to stop writing human staging directions.
    ("no-table", "Put him in a warm rustic kitchen.", None),
    # Names the wings. Breaks the don't-name-it rule deliberately, because the
    # rule protects what is already correct and this is not.
    ("name-wings", HANDS_BASE + " He has feathered wings, not arms or hands.", None),
    # Names them and gives them something to do, so there is a correct answer
    # in frame rather than only a prohibition.
    ("wings-busy", "Put him at a wooden table with his feathered wings folded at "
                   "his sides, seen from the front.", None),
    # The graph currently cannot say no to anything. This is what it would cost.
    ("negative", HANDS_BASE, "human hands, fingers, human arms, human skin"),
]

# Both failures survived every prompt and every setting, which points at the
# input rather than the instruction. The single reference is one front-on full
# body: it shows no side view to rotate towards, and its wings are feathered
# masses that simply end -- the design has no answer for what is at the end of
# an arm. Asked for a side profile the model has nothing to copy; asked for a
# character at a table it has nothing to put on the table, and reaches for its
# strongest prior in both cases.
#
# The turnaround sheet has front, three-quarter, side and back views. Chaining
# it as a second reference is the one lever not yet pulled, and it is the lever
# the multi-reference work exists to provide.
TURNAROUND_SHOTS = [
    ("side profile", "Show him in full side profile, facing left."),
    ("at a table", HANDS_BASE),
]


def with_negative(wf, text: str, cfg: float = 2.5):
    """Give the sampler something to push away from.

    The stock Kontext graph zeroes the negative and runs cfg 1.0, which means
    classifier-free guidance is switched off and a negative prompt would be
    ignored even if one were wired in. Both have to change together.
    """
    wf["10"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": text, "clip": ["2", 0]},
        "_meta": {"title": "Negative"},
    }
    wf["11"]["inputs"]["cfg"] = cfg
    return wf


def pose_sheet(a, models) -> str:
    tiles = []
    for name, instruction in POSE_STYLES:
        for guidance in GUIDANCES:
            wf, m = build_kontext_workflow(models, reference_image=a.reference,
                                           steps=a.steps, guidance=guidance,
                                           seed=a.seed)
            wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
            wf["13"]["inputs"]["filename_prefix"] = f"pose_{name}_{guidance}"
            label = f"{name}  /  guidance {guidance}"
            tiles.append((label, _img(a.url, render(a.url, wf, label))))
    return sheet(tiles, a.pose_out, columns=len(GUIDANCES))


def hands_sheet(a, models) -> str:
    tiles = []
    for name, instruction, negative in HANDS_STYLES:
        wf, m = build_kontext_workflow(models, reference_image=a.reference,
                                       steps=a.steps, guidance=2.5, seed=a.seed)
        wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
        if negative:
            with_negative(wf, negative)
        wf["13"]["inputs"]["filename_prefix"] = f"hands_{name}"
        label = f"{name}" + ("  /  cfg 2.5" if negative else "")
        tiles.append((label, _img(a.url, render(a.url, wf, label))))
    return sheet(tiles, a.hands_out, columns=len(HANDS_STYLES))


def turnaround_sheet(a, models) -> str:
    """Each shot rendered from the one reference, then from two.

    Paired left and right so the only difference in each pair is whether the
    turnaround was chained on. Anything that changes is attributable to it.
    """
    tiles = []
    for shot, instruction in TURNAROUND_SHOTS:
        for label, extra in (("1 reference", []), ("+ turnaround", [a.turnaround])):
            wf, m = build_kontext_workflow(models, reference_image=a.reference,
                                           steps=a.steps, guidance=2.5, seed=a.seed,
                                           extra_references=extra, multi="chain")
            wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
            wf["13"]["inputs"]["filename_prefix"] = f"turn_{shot.replace(' ', '_')}"
            name = f"{shot}  /  {label}"
            tiles.append((name, _img(a.url, render(a.url, wf, name))))
    return sheet(tiles, a.turnaround_out, columns=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--reference", required=True)
    p.add_argument("--turnaround", default="rooster_sheet_v1.png")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--only", choices=["pose", "hands", "turnaround", "both", "all"],
                   default="all")
    p.add_argument("--pose-out", default="pose_compare.png")
    p.add_argument("--hands-out", default="hands_compare.png")
    p.add_argument("--turnaround-out", default="turnaround_compare.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))

    if a.only in ("pose", "both", "all"):
        print(f"\nWrote {pose_sheet(a, models)}")
    if a.only in ("hands", "both", "all"):
        print(f"\nWrote {hands_sheet(a, models)}")
    if a.only in ("turnaround", "all"):
        print(f"\nWrote {turnaround_sheet(a, models)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
