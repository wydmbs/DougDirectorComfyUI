"""compose_salvage.py -- rebuild a comparison sheet from already-rendered files.

A run that dies partway still leaves its renders on the GPU box. When the shot
that matters completed before the crash, re-rendering it is wasted GPU time --
the answer is already sitting on disk. This composes those files into the same
labelled grid the live run would have produced, so a salvaged pass is read the
same way as a clean one.

    python compose_salvage.py --dir smoke_out/salvage --out smoke_out/15a.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_conditioning import sheet  # noqa: E402

# Order matters: the grid reads left to right as "how much reference did it
# get", so the control sits first and the fully-conditioned case sits last.
TILES = [
    ("v1 control", "verify_v1_control_00007_.png"),
    ("v2 alone", "verify_v2_alone_00007_.png"),
    ("v2 + sheet", "verify_v2_+_sheet_00007_.png"),
    ("v2 + tips", "verify_v2_+_tips_00005_.png"),
    ("v2 + both", "verify_v2_+_both_00005_.png"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="smoke_out/salvage")
    p.add_argument("--shot", default="at a table")
    p.add_argument("--out", default="smoke_out/15a_wing_verify_table.png")
    a = p.parse_args(argv)

    tiles = []
    for name, filename in TILES:
        path = os.path.join(a.dir, filename)
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            return 1
        with open(path, "rb") as handle:
            tiles.append((f"{a.shot}  /  {name}", handle.read()))

    print(f"Wrote {sheet(tiles, a.out, columns=len(TILES))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
