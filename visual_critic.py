"""visual_critic.py -- the local critic that actually measures the renders.

Harry could not see. `critique_variants` had an injection point and nothing
plugged into it, so `build_registry(CFG)` passed no critic and every lock he
made was stamped "image not seen". This is what plugs in, and it runs entirely
on this machine: no API key, no network call, no per-image cost, and the same
answer every time it is asked.

What goes in the gate was decided by critic_calibrate.py against the labelled
set (10 BAD v1 renders, 3 OLD v2, 9 GOOD v3 canon, 9 HAND grip failures)
rather than by reputation, and the calibration disagreed with the reputation
twice:

    dinov2   AUC 1.000 vs BAD, margin +0.0864     -> the gate
    palette  AUC 1.000 vs BAD, margin +0.0221     -> advisory, margin too thin
    clip     AUC 0.667 vs BAD, 0.481 vs OLD       -> dropped, worse than chance
    hands    fires on GOOD too, misses the grip   -> advisory, with the caveat

Two of those deserve their reasoning written down, because the numbers alone
invite the wrong conclusion later.

CLIP is the metric everyone reaches for and it was the worst performer here.
0.481 against the OLD set is below chance: on the one comparison that actually
matters for identity, telling v2 from v3, it was actively misleading. That is
the known CLIP failure mode -- the embedding is dominated by shared style and
palette -- and this production is precisely the case that triggers it, because
every render is the same character in the same style. It is not in the gate,
and it should not be added back without new calibration evidence.

The hands metric measures the wrong thing, in an instructive way. The BAD set
scores a flat 2.0, which looks like a perfect defect detector until you notice
GOOD averages 1.556 and also reaches 2.0 -- the canon design has gloved hands
and MediaPipe counts them. Worse, the HAND set, nine renders that were asked
for no hand and produced one anyway, averages 0.111: the detector does not see
the offending hands at all. It is a bare-human-hand detector pointed at a
stylised subject. So it is reported and never gates. Reporting is still worth
the 99ms, because a 2.0 on a design that has no hands is a real signal even
though the converse is worthless.

Palette earns a place as a cheap cross-check and nothing more. A +0.0221
margin, on a scale where the classes span 1.2, is a threshold that separates
these 31 images and would not survive a new character.

The honesty rule this file inherits from the locking tools: it reports what it
measured. It measured identity against a reference -- not composition, not
expression, not whether the shot is any good. Callers must not round that up
to "reviewed", which is why the lock note it earns says "measured, not seen".

Structure note: the metrics do not live here. See critic_worker.py -- the app
runs on Python 3.14, where torch and mediapipe do not exist, so the measuring
happens under ComfyUI's embedded Python and comes back as JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "critic_worker.py")

# ComfyUI's embedded Python is the interpreter on this machine that has CUDA
# torch, transformers and mediapipe. Overridable because the path is a fact
# about this box, not about the code.
EMBEDDED_PYTHON = os.environ.get(
    "CRITIC_PYTHON",
    r"C:\Users\dougl\ComfyUI_windows_portable\python_embeded\python.exe")

# Generous, because a cold run pays for a model load, and a critique that times
# out is indistinguishable to Harry from a critique that failed -- which is
# fine, both make him say he could not check.
TIMEOUT_S = int(os.environ.get("CRITIC_TIMEOUT_S", "600"))

# Midpoint of the calibrated gap: GOOD floor 0.7127, BAD ceiling 0.6263. Placed
# at the midpoint rather than on the GOOD floor because that floor is a single
# render, and a threshold sitting on your worst accepted sample has no room in
# it for the next character.
DINOV2_PASS = 0.67
# Below this, identity is gone rather than drifting: the BAD set clustered at
# 0.60-0.63, which is "not the same character" territory.
DINOV2_FAIL = 0.64


# ------------------------------------------------------------------ measuring

def measure(anchor: str, paths: list, metrics: list = None) -> dict:
    """Run the metrics in the interpreter that owns the models.

    Returns the worker's parsed result, or raises RuntimeError with something a
    human can act on. Never raises the subprocess's own exceptions upward --
    the caller's job is to tell Harry he could not check, not to crash.
    """
    if not os.path.exists(EMBEDDED_PYTHON):
        raise RuntimeError(
            f"no vision interpreter at {EMBEDDED_PYTHON}; set CRITIC_PYTHON to a "
            "Python that has torch, transformers and mediapipe")
    if not os.path.exists(WORKER):
        raise RuntimeError(f"critic_worker.py missing from {HERE}")

    payload = json.dumps({
        "anchor": anchor,
        "paths": paths,
        "metrics": metrics or ["identity", "palette", "hands"],
    })

    try:
        completed = subprocess.run(
            [EMBEDDED_PYTHON, WORKER],
            input=payload, capture_output=True, text=True,
            timeout=TIMEOUT_S, cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"the critic did not finish within {TIMEOUT_S}s")

    stdout = (completed.stdout or "").strip()
    if not stdout:
        tail = (completed.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("the critic returned nothing. " + " / ".join(tail))

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError("the critic returned unparseable output: " + stdout[:300])

    if not result.get("ok"):
        raise RuntimeError(result.get("error", "the critic reported failure"))
    return result


# ------------------------------------------------------------------ anchor

def find_anchor(cfg, entry_id: str) -> tuple:
    """The image this entry's renders are supposed to look like.

    Preference order matters. An explicit reference row is a deliberate act by
    the director. The locked concept image is merely the last thing accepted,
    which is usually right and occasionally is the render that started the
    drift -- measuring against it would then certify the drift as correct.
    """
    try:
        import storyboard_store as store
        for row in store.list_references(cfg.storyboard_path, entry_id):
            candidate = str(row.get("image_path") or "").strip()
            if candidate and os.path.exists(candidate):
                return candidate, "reference sheet"
    except Exception:  # noqa: BLE001 -- a missing sheet is not a critic failure
        pass

    try:
        import director_engine as engine
        entry = engine.get_entry(cfg, entry_id) or {}
        candidate = str(entry.get("image_path") or "").strip()
        if candidate and os.path.exists(candidate):
            return candidate, "locked concept"
    except Exception:  # noqa: BLE001
        pass

    return None, ""


# ------------------------------------------------------------------ critique

def critique(cfg, entry_id: str, image_paths: list, criteria: str = "") -> str:
    """Measure candidates against the entry's reference and report.

    Wired in as build_registry(critique_fn=...). The tool contract is a string
    back to Harry, so everything below is phrased for a reader who is about to
    act on it -- including the branches where the honest answer is that this
    cannot be judged and he should not pretend otherwise.
    """
    paths = [p for p in (image_paths or []) if p]
    if not paths:
        return "No image paths were given, so there is nothing to measure."

    missing = [p for p in paths if not os.path.exists(p)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return ("None of those files exist on disk: " + ", ".join(missing) +
                ". Nothing was measured, so nothing is known about them.")

    anchor, anchor_source = find_anchor(cfg, entry_id)
    if not anchor:
        return (f"No reference image is on file for {entry_id}, so identity cannot be "
                "measured -- there is nothing to compare against. Add a reference or "
                "lock a concept first. Until then do not claim one variant is more "
                "on-model than another. Candidates: " + ", ".join(paths))

    try:
        result = measure(anchor, paths)
    except RuntimeError as error:
        return (f"The visual critic could not run ({error}), so these images have NOT "
                "been checked. Do not treat any of them as verified, and say so if you "
                "lock one anyway. Candidates: " + ", ".join(paths))

    scores = result.get("scores", {})
    errors = result.get("errors", {})
    identity = scores.get("identity")
    if not identity:
        return (f"The identity metric failed ({errors.get('identity', 'no reason given')}), "
                "and it is the only one that gates. These images have NOT been checked. "
                "Candidates: " + ", ".join(paths))

    palette = scores.get("palette") or [None] * len(paths)
    hands = scores.get("hands") or [None] * len(paths)

    rows = []
    for i, path in enumerate(paths):
        value = identity[i]
        if value >= DINOV2_PASS:
            verdict = "on-model"
        elif value < DINOV2_FAIL:
            verdict = "OFF-MODEL"
        else:
            verdict = "borderline"
        rows.append({"path": path, "identity": value, "palette": palette[i],
                     "hands": hands[i], "verdict": verdict})

    ranked = sorted(rows, key=lambda r: r["identity"], reverse=True)

    lines = [f"Measured {len(paths)} candidate(s) for {entry_id} against the "
             f"{anchor_source}: {os.path.basename(anchor)}"]
    if missing:
        lines.append(f"Skipped {len(missing)} path(s) that do not exist: " + ", ".join(missing))
    lines.append("")
    lines.append(f"{'rank':<5}{'identity':>10}{'palette':>10}{'hands':>7}  verdict / file")
    for rank, row in enumerate(ranked, start=1):
        palette_text = f"{row['palette']:+.4f}" if row["palette"] is not None else "n/a"
        hands_text = f"{row['hands']:.0f}" if row["hands"] is not None else "n/a"
        lines.append(f"{rank:<5}{row['identity']:>10.4f}{palette_text:>10}{hands_text:>7}  "
                     f"{row['verdict']} / {os.path.basename(row['path'])}")

    best = ranked[0]
    lines.append("")
    if best["verdict"] == "OFF-MODEL":
        lines.append(
            f"Every candidate is off-model. The best of them scores {best['identity']:.4f}, "
            f"below the {DINOV2_FAIL:.2f} floor where the calibration set stopped being the "
            "same character at all. Do not lock any of these. Regenerate -- and if the next "
            "round fails the same way, the conditioning is the problem, not the seed.")
    elif best["verdict"] == "borderline":
        lines.append(
            f"The best candidate scores {best['identity']:.4f}, between the {DINOV2_FAIL:.2f} "
            f"floor and the {DINOV2_PASS:.2f} pass mark. That is drift, not a match. Lockable "
            "if the schedule demands it, but put on the row that it was locked borderline.")
    else:
        lines.append(
            f"Best on identity: {os.path.basename(best['path'])} at {best['identity']:.4f}, "
            f"clear of the {DINOV2_PASS:.2f} pass mark. Full path: {best['path']}")

    flagged = [r for r in ranked if r["hands"] is not None and r["hands"] >= 2]
    if flagged:
        lines.append(
            "Hand detector fired on: "
            + ", ".join(os.path.basename(r["path"]) for r in flagged)
            + ". Advisory only -- in calibration this detector fired on good renders too "
              "and missed the known grip failures entirely. Worth a look if this design has "
              "no hands; ignore it otherwise.")

    for name, reason in errors.items():
        lines.append(f"({name} unavailable: {reason})")

    lines.append("")
    lines.append(
        "Measured, not seen. This is DINOv2 identity similarity against the reference -- "
        "the one metric that separated good from bad cleanly in calibration -- with palette "
        "and hand count as advisories. Nothing here judges composition, expression, lighting "
        "or whether the shot is any good."
        + (f" Criteria not assessed by these metrics: {criteria}" if criteria else ""))

    return "\n".join(lines)


# How a lock made on this critic's say-so should be described on the row. The
# locking tools stamp a marker when nothing looked at the image; a measurement
# is not a look, but it is not nothing either, and the row should record which
# of the three it was rather than collapsing to reviewed/unreviewed.
critique.review_kind = "metric"


if __name__ == "__main__":
    # Smoke path: point it at an anchor and some renders, get the raw numbers.
    #   python visual_critic.py anchor.png a.png b.png
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    print(json.dumps(measure(sys.argv[1], sys.argv[2:]), indent=2))
