"""
flux_workflow_builder.py — build the FLUX + IP-Adapter keyframe workflow.

Exporting a workflow by hand is the one step in SETUP.md that nothing checks:
you wire it in the ComfyUI canvas, save it in API format, then type node IDs
into the Setup tab. Every part of that is silent when it goes wrong. Pick the
SD/SDXL IP-Adapter instead of the FLUX one and the graph still runs -- it just
quietly ignores your reference image, and you don't find out until a character
has drifted across a dozen shots.

So we build the graph from the server's own schemas instead. Every model name
comes out of /object_info, which is the same list the dropdowns are populated
from, so a filename that isn't on the machine can't be written into the file.
The node IDs come back as a mapping the app can use directly.

    python flux_workflow_builder.py --url http://127.0.0.1:8188 --write

This is a starting point, not a ceiling. It emits the smallest graph that gets
a correct identity-locked keyframe; open it in ComfyUI and add to it freely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_OUT = "workflows/flux_keyframe_api.json"


class BuildError(RuntimeError):
    """Something the machine is missing. The message says what to install."""


# --------------------------------------------------------------------------
# talking to the server
# --------------------------------------------------------------------------

def fetch_object_info(url: str, timeout: int = 60) -> dict:
    endpoint = url.rstrip("/") + "/object_info"
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        raise BuildError(
            f"Could not reach ComfyUI at {url} ({e}).\n"
            "Start ComfyUI first -- the workflow is built from the node list it "
            "serves, so the server has to be up."
        ) from e


def _choices(info: dict, node: str, field: str) -> list:
    """The dropdown options for one input, exactly as the UI would show them."""
    spec = info.get(node)
    if not spec:
        return []
    inputs = spec.get("input", {}) or {}
    entry = (inputs.get("required", {}) or {}).get(field)
    if entry is None:
        entry = (inputs.get("optional", {}) or {}).get(field)
    if not entry:
        return []
    first = entry[0]
    return list(first) if isinstance(first, list) else []


def _pick(options, *, must=(), prefer=(), what: str = "", fix: str = ""):
    """Choose a filename from a dropdown.

    `must` narrows to the ones that qualify at all; `prefer` ranks what's left.
    Ranking rather than requiring means an unusual filename still resolves --
    it just doesn't win over an obvious one.
    """
    pool = [o for o in options
            if all(m.lower() in str(o).lower() for m in must)]
    if not pool:
        raise BuildError(f"No {what} found on this machine.\n  {fix}")

    def score(name: str) -> tuple:
        low = str(name).lower()
        return (sum(1 for p in prefer if p.lower() in low), -len(str(name)))

    return sorted(pool, key=score, reverse=True)[0]


# ComfyUI-IPAdapter-Flux loads its weights with a bare torch.load, which reads
# pickle archives only. A .safetensors adapter therefore appears in the dropdown,
# is offered by the UI, and fails at render time with an unpickling error that
# names neither the file nor the format. The two distributions of these weights
# are byte-for-byte the same tensors, so the fix is only ever to choose the
# other file -- but nothing on screen tells you that. We choose it up front.
TORCH_LOADABLE = (".bin", ".pt", ".pth", ".ckpt")


def _pick_flux_adapter(options) -> str:
    loadable = [o for o in options
                if str(o).lower().endswith(TORCH_LOADABLE)]
    if loadable:
        return _pick(loadable, prefer=("instantx", "flux", "ip-adapter"),
                     what="FLUX IP-Adapter weights", fix="")
    if options:
        raise BuildError(
            "The FLUX IP-Adapter weights on this machine are in safetensors "
            f"format ({options[0]}), which ComfyUI-IPAdapter-Flux cannot read -- "
            "it loads with torch.load, so it needs the pickle build.\n"
            "  Download ip-adapter.bin from InstantX/FLUX.1-dev-IP-Adapter into "
            "ComfyUI/models/ipadapter-flux.\n"
            "  (Same weights, different container. Keeping only the "
            ".safetensors gives an 'invalid load key' error at render time.)")
    raise BuildError(
        "No FLUX IP-Adapter weights found.\n"
        "  Download ip-adapter.bin from InstantX/FLUX.1-dev-IP-Adapter into "
        "ComfyUI/models/ipadapter-flux.")


# --------------------------------------------------------------------------
# choosing the model files
# --------------------------------------------------------------------------

def resolve_models(info: dict) -> dict:
    """Work out which files on this machine make up a FLUX + IP-Adapter stack.

    FLUX ships two ways: an all-in-one fp8 checkpoint, or a bare diffusion model
    that needs CLIP and VAE loaded alongside it. Both are common, so we detect
    which one is here rather than insisting on either.
    """
    if "ApplyIPAdapterFlux" not in info:
        raise BuildError(
            "ApplyIPAdapterFlux is not installed.\n"
            "  Install ComfyUI-IPAdapter-Flux (Shakker-Labs) via ComfyUI Manager "
            "and restart.\n"
            "  Note the SD/SDXL pack (ComfyUI_IPAdapter_plus) will NOT work for "
            "FLUX -- they are different node families and are not interchangeable."
        )

    chosen: dict = {}

    ckpts = _choices(info, "CheckpointLoaderSimple", "ckpt_name")
    flux_ckpts = [c for c in ckpts if "flux" in str(c).lower()]
    if flux_ckpts:
        chosen["mode"] = "checkpoint"
        chosen["ckpt"] = _pick(flux_ckpts, prefer=("fp8", "dev"),
                               what="FLUX checkpoint", fix="")
    else:
        unets = _choices(info, "UNETLoader", "unet_name")
        chosen["mode"] = "unet"
        chosen["unet"] = _pick(
            unets, must=("flux",), prefer=("dev", "fp8"),
            what="FLUX diffusion model",
            fix="Put flux1-dev into ComfyUI/models/checkpoints (all-in-one fp8) "
                "or ComfyUI/models/unet (bare model).")
        clips = _choices(info, "DualCLIPLoader", "clip_name1")
        chosen["clip_l"] = _pick(
            clips, must=("clip_l",), what="clip_l text encoder",
            fix="FLUX needs clip_l and t5xxl in ComfyUI/models/clip.")
        chosen["t5"] = _pick(
            clips, must=("t5",), prefer=("fp8", "fp16"),
            what="t5xxl text encoder",
            fix="FLUX needs clip_l and t5xxl in ComfyUI/models/clip.")
        vaes = _choices(info, "VAELoader", "vae_name")
        chosen["vae"] = _pick(
            vaes, prefer=("flux", "ae"), what="VAE",
            fix="Put the FLUX ae.safetensors in ComfyUI/models/vae.")

    chosen["ipadapter"] = _pick_flux_adapter(
        _choices(info, "IPAdapterFluxLoader", "ipadapter"))

    chosen["clip_vision"] = _pick(
        _choices(info, "IPAdapterFluxLoader", "clip_vision"),
        prefer=("siglip",),
        what="SigLIP vision encoder",
        fix="The FLUX IP-Adapter needs google/siglip-so400m-patch14-384 in "
            "ComfyUI/models/clip_vision -- CLIP-ViT-H is for the SD adapter and "
            "will not substitute.")

    return chosen


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

def build_workflow(models: dict, *, reference_image: str, width: int,
                   height: int, steps: int, guidance: float,
                   weight: float, seed: int) -> tuple:
    """Emit the API-format graph, plus the node mapping the app drives."""
    wf: dict = {}

    def node(nid, class_type, inputs, title):
        wf[nid] = {"class_type": class_type, "inputs": inputs,
                   "_meta": {"title": title}}

    # --- base model -------------------------------------------------------
    if models["mode"] == "checkpoint":
        node("1", "CheckpointLoaderSimple", {"ckpt_name": models["ckpt"]},
             "FLUX checkpoint")
        model_src, clip_src, vae_src = ["1", 0], ["1", 1], ["1", 2]
    else:
        node("1", "UNETLoader",
             {"unet_name": models["unet"], "weight_dtype": "fp8_e4m3fn"},
             "FLUX model")
        node("1c", "DualCLIPLoader",
             {"clip_name1": models["clip_l"], "clip_name2": models["t5"],
              "type": "flux", "device": "default"}, "FLUX text encoders")
        node("1v", "VAELoader", {"vae_name": models["vae"]}, "FLUX VAE")
        model_src, clip_src, vae_src = ["1", 0], ["1c", 0], ["1v", 0]

    # --- identity lock ----------------------------------------------------
    # The reference goes into the MODEL path, not the latent: the character is
    # carried by conditioning, so the composition stays free to change. That is
    # what makes a new camera angle possible without losing the face.
    node("2", "LoadImage", {"image": reference_image, "upload": "image"},
         "Reference (character sheet)")
    node("3", "IPAdapterFluxLoader",
         {"ipadapter": models["ipadapter"], "clip_vision": models["clip_vision"],
          "provider": "cuda"}, "FLUX IP-Adapter")
    node("4", "ApplyIPAdapterFlux",
         {"model": model_src, "ipadapter_flux": ["3", 0], "image": ["2", 0],
          "weight": weight, "start_percent": 0.0, "end_percent": 1.0},
         "Lock identity")

    # --- prompt -----------------------------------------------------------
    node("5", "CLIPTextEncode", {"text": "", "clip": clip_src}, "Positive prompt")
    node("6", "FluxGuidance", {"conditioning": ["5", 0], "guidance": guidance},
         "FLUX guidance")
    # FLUX is a guidance-distilled model: it runs at cfg 1.0, where the negative
    # branch has no effect. KSampler still requires the socket to be connected,
    # so this stays empty by design rather than by oversight.
    node("7", "CLIPTextEncode", {"text": "", "clip": clip_src},
         "Negative (unused at cfg 1.0)")

    # --- sample -----------------------------------------------------------
    node("8", "EmptyLatentImage",
         {"width": width, "height": height, "batch_size": 1}, "Canvas")
    node("9", "KSampler",
         {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
          "latent_image": ["8", 0], "seed": seed, "steps": steps, "cfg": 1.0,
          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
         "Sampler")
    node("10", "VAEDecode", {"samples": ["9", 0], "vae": vae_src}, "Decode")
    node("11", "SaveImage",
         {"images": ["10", 0], "filename_prefix": "keyframe"}, "Save")

    mapping = {
        "positive_prompt_node": "5", "positive_prompt_input": "text",
        "negative_prompt_node": "7", "negative_prompt_input": "text",
        "seed_node": "9", "seed_input": "seed",
        "save_image_node": "11",
    }
    return wf, mapping


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _write_config(out_path: str, mapping: dict, url: str) -> str:
    """Fold the result into the app's config so Setup doesn't need retyping."""
    from config import load_config, save_config, NodeMapping, CONFIG_PATH

    cfg = load_config()
    cfg.workflow_json_path = os.path.abspath(out_path)
    cfg.node_mapping = NodeMapping(**mapping)
    cfg.comfyui_url = url
    save_config(cfg)
    return CONFIG_PATH


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build a FLUX + IP-Adapter keyframe workflow from the "
                    "node schemas of a running ComfyUI.")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--reference", default="example.png",
                   help="Placeholder reference filename; the app replaces it "
                        "per shot with the character sheet being locked to.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--weight", type=float, default=0.8,
                   help="IP-Adapter strength. Too high and every shot inherits "
                        "the reference's pose and lighting as well as its face.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true",
                   help="Also point toolchain_config.json at the result.")
    a = p.parse_args(argv)

    try:
        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

        info = fetch_object_info(a.url)
        models = resolve_models(info)
        wf, mapping = build_workflow(
            models, reference_image=a.reference, width=a.width, height=a.height,
            steps=a.steps, guidance=a.guidance, weight=a.weight, seed=a.seed)

        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2)

        print("Built from this machine's own node list:")
        for k, v in models.items():
            if k != "mode":
                print(f"    {k:<12} {v}")
        print(f"\nWrote {a.out}  ({len(wf)} nodes)")
        print("\nNode mapping:")
        for k, v in mapping.items():
            print(f"    {k:<22} {v}")

        if a.write:
            path = _write_config(a.out, mapping, a.url)
            print(f"\nConfigured {path}. Run doctor.py to confirm.")
        else:
            print("\nRe-run with --write to point the app at it, or set the "
                  "path and IDs by hand in the Setup tab.")
        return 0

    except BuildError as e:
        print(f"\nCannot build the workflow yet.\n\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
