"""
probe_wing_hands.py -- give the design an answer, then check it holds.

The hands were never a prompt fault. The reference and the turnaround both show
wings as feathered masses that simply stop, so a shot needing him at a table
has no defined limb end to draw and the model supplies the one it knows. Doug
has settled the design question: wings that fold.

That settles what to draw. It does not settle where the fix belongs, and there
are two candidates worth separating, because they cost very different things.

  Repair    Edit the existing reference to define folded wing tips. This is a
            local attribute change on an image Kontext already holds, which is
            the operation it is best at. If it works the design is fixed in one
            render and the whole ChatGPT stage is skipped.
  Redraw    Go back to ChatGPT for a new turnaround built around folded wings
            from the start. Authoritative, slower, manual, and it invalidates
            every reference downstream of the current sheet.

Repair is only worth having if the repaired reference then survives being used.
An image that looks right and stops working the moment it conditions a shot has
fixed nothing. So each variant is rendered twice:

    row 1   the repair itself -- did the wings become wings?
    row 2   that repaired image conditioning the table shot that produced the
            human hands in the first place -- did it hold?

A variant only counts if both rows are right. The `original` column carries the
unrepaired reference through both rows as the control, so the comparison is
against the actual failure rather than against memory of it.

    python probe_wing_hands.py --reference rooster_ref_v1.png
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

# The shot that failed. Unchanged, so the comparison is honest.
TABLE = "Put him at a wooden table, seen from the front."

# Four ways to say "wings that fold", ordered from least to most surgical.
# Each names the wings on purpose: the don't-name-it rule protects attributes
# that are already correct, and this one is the thing being corrected.
REPAIRS = [
    ("original", ""),
    # States the anatomy without dictating how it is drawn.
    ("define", "Give him clearly defined folded wings with long layered primary "
               "feathers at the tips."),
    # Says what they replace, since the failure is a substitution.
    ("instead-of-arms", "Replace the ends of his arms with folded wings that taper "
                        "into long primary feathers."),
    # Positions them, so there is a visible tip in frame rather than a mass that
    # trails off where a hand would be.
    ("fold-forward", "Fold his wings neatly against his sides so the long primary "
                     "feather tips show at the front."),
]


def fetch_input(url, name):
    """Read an image back out of ComfyUI's input folder.

    The control column needs the untouched reference as a tile. It was never
    rendered, so there is no history entry to pull it from -- but ComfyUI will
    serve anything in input/ through the same view endpoint.
    """
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({"filename": name, "type": "input"})
    with urllib.request.urlopen(f"{url.rstrip('/')}/view?{query}", timeout=60) as response:
        return response.read()


def repaired_reference(a, models, name, instruction):
    """Run the repair and hand back a filename ComfyUI can load.

    The output has to go back in as an input, so it is fetched and re-uploaded
    rather than referenced by path -- the render lives in output/ and LoadImage
    reads input/.
    """
    if not instruction:
        return a.reference, fetch_input(a.url, a.reference)

    wf, m = build_kontext_workflow(models, reference_image=a.reference,
                                   steps=a.steps, guidance=a.guidance, seed=a.seed)
    wf[m["positive_prompt_node"]]["inputs"]["text"] = instruction
    wf["13"]["inputs"]["filename_prefix"] = f"wing_{name}"
    data = _img(a.url, render(a.url, wf, f"repair / {name}"))

    local = f"wing_{name}.png"
    with open(local, "wb") as handle:
        handle.write(data)
    return upload(a.url, local), data


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--reference", required=True)
    p.add_argument("--turnaround", default="rooster_sheet_v1.png")
    p.add_argument("--chain-turnaround", action="store_true",
                   help="Also chain the turnaround on the table shot.")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--out", default="wing_hands.png")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    models = resolve_kontext_models(fetch_object_info(a.url))

    repairs, table = [], []
    extra = [a.turnaround] if a.chain_turnaround else []

    for name, instruction in REPAIRS:
        ref_name, data = repaired_reference(a, models, name, instruction)
        suffix = "reference as-is" if not instruction else "repaired"
        repairs.append((f"{name}  /  {suffix}", data))

        wf, m = build_kontext_workflow(models, reference_image=ref_name,
                                       steps=a.steps, guidance=2.5, seed=a.seed,
                                       extra_references=extra, multi="chain")
        wf[m["positive_prompt_node"]]["inputs"]["text"] = TABLE
        wf["13"]["inputs"]["filename_prefix"] = f"wingtable_{name}"
        label = f"{name}  /  at a table"
        table.append((label, _img(a.url, render(a.url, wf, label))))

    print(f"\nWrote {sheet(repairs + table, a.out, columns=len(REPAIRS))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"\nFailed: {error}", file=sys.stderr)
        raise SystemExit(1)
