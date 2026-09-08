"""
patch_ipadapter_flux.py — reconcile ComfyUI-IPAdapter-Flux with current core.

ComfyUI-IPAdapter-Flux works by replacing FLUX's `forward_orig` with its own
copy. That copy was written against an older core, and core has since added a
`timestep_zero_index` argument to the real function. Core passes it; the
replacement doesn't accept it; every render through the adapter dies inside
KSampler with:

    forward_orig_ipa() got an unexpected keyword argument 'timestep_zero_index'

Upstream has not shipped a fix (checked against Shakker-Labs/main), and the
error names KSampler rather than the custom node, so it reads like a sampler
problem rather than a version mismatch. That is a long way to travel for a
one-line signature change.

Core only ever populates that argument for Kontext-style reference latents,
which this adapter path does not use -- there it stays None. So the patch
accepts the argument and ignores it, but raises a clear error if it is ever
non-None rather than quietly rendering something wrong.

    python patch_ipadapter_flux.py --comfy-root <path>     apply
    python patch_ipadapter_flux.py --comfy-root <path> --revert

Idempotent, and backs the file up before touching it. Reverting or reinstalling
the custom node restores the original; re-run this afterwards.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

NODE_DIR = os.path.join("custom_nodes", "ComfyUI-IPAdapter-Flux")
TARGET = "utils.py"

ANCHOR = "    guidance: Tensor|None = None,\n    control=None,\n"
PATCHED = (
    "    guidance: Tensor|None = None,\n"
    "    control=None,\n"
    "    timestep_zero_index=None,  # added by harness: see patch_ipadapter_flux.py\n"
)

GUARD_ANCHOR = "    patches_replace = transformer_options.get(\"patches_replace\", {})\n"
GUARD = (
    "    if timestep_zero_index is not None:\n"
    "        # Core only sets this for reference-latent (Kontext) workflows,\n"
    "        # which this adapter's forward pass does not implement. Rendering\n"
    "        # anyway would produce a plausible but wrong image, so stop.\n"
    "        raise NotImplementedError(\n"
    "            \"ComfyUI-IPAdapter-Flux does not support reference-latent \"\n"
    "            \"(Kontext) conditioning. Remove the reference latents, or use \"\n"
    "            \"the IP-Adapter without them.\")\n"
    "    patches_replace = transformer_options.get(\"patches_replace\", {})\n"
)

MARKER = "added by harness"


def locate(comfy_root: str) -> str:
    path = os.path.join(comfy_root, NODE_DIR, TARGET)
    if not os.path.exists(path):
        raise SystemExit(
            f"Could not find {path}.\n"
            "Pass --comfy-root pointing at the folder that contains "
            "custom_nodes/ (usually ComfyUI_windows_portable/ComfyUI).")
    return path


def apply(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        return "Already patched. Nothing to do."

    if ANCHOR not in src:
        return ("Could not find the signature this patch expects. The custom "
                "node has changed -- check whether upstream has fixed it, and "
                "do not force this patch onto a version it wasn't written for.")

    backup = f"{path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)

    out = src.replace(ANCHOR, PATCHED, 1)
    if GUARD_ANCHOR in out:
        out = out.replace(GUARD_ANCHOR, GUARD, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(out)

    return (f"Patched {path}\n  backup: {backup}\n"
            "  Restart ComfyUI for it to take effect.")


def revert(path: str) -> str:
    d, name = os.path.split(path)
    backups = sorted(b for b in os.listdir(d) if b.startswith(name + ".bak."))
    if not backups:
        return "No backup found; nothing to revert to."
    newest = os.path.join(d, backups[-1])
    shutil.copy2(newest, path)
    return f"Restored {path} from {newest}\n  Restart ComfyUI."


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-root", required=True)
    p.add_argument("--revert", action="store_true")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    path = locate(a.comfy_root)
    print(revert(path) if a.revert else apply(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
