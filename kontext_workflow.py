"""
kontext_workflow.py — in-context reference conditioning, the way ChatGPT does it.

Why this exists, in one paragraph:

IP-Adapter takes your reference, pushes it through a vision encoder, and hands
FLUX a *summary* -- a handful of embedding tokens injected through cross
attention. It is a description of the picture, not the picture. That is why the
character comes back recognisable but never exact: nothing downstream ever had
access to the pixels, so details it didn't think to summarise cannot survive.

FLUX.1 Kontext works the way people assume image references work, and the way
ChatGPT's image tool actually does: the reference is VAE-encoded into real
latent tokens and concatenated onto the sequence the model is denoising. The
model attends to the reference tokens directly, at full spatial resolution,
alongside the tokens it is generating. Nothing is summarised away.

Same idea in one line each:

    IP-Adapter   reference -> SigLIP -> ~few tokens -> cross-attention   (lossy)
    Kontext      reference -> VAE    -> full latent grid -> in sequence  (exact)

So this is not a better setting. It is a different mechanism, and the honest
expectation is that it holds costume detail, button count and comb lean that no
IP-Adapter weight will ever hold.

Kontext is an edit model: it wants an instruction ("turn him to face left"),
not a fresh description of the scene. Prompts that re-describe the character
tend to make it re-draw rather than transform.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux_workflow_builder import (  # noqa: E402
    BuildError, fetch_object_info, _choices, _pick,
)

KONTEXT_REQUIRED = ("ReferenceLatent", "FluxKontextImageScale")


def resolve_kontext_models(info: dict) -> dict:
    """Kontext ships as a bare diffusion model, so CLIP and VAE load separately."""
    missing = [n for n in KONTEXT_REQUIRED if n not in info]
    if missing:
        raise BuildError(
            f"This ComfyUI has no Kontext support ({', '.join(missing)} absent).\n"
            "  Update ComfyUI -- Kontext conditioning is in core, not a custom node.")

    unets = _choices(info, "UNETLoader", "unet_name")
    kontext = [u for u in unets if "kontext" in str(u).lower()]
    if not kontext:
        raise BuildError(
            "No FLUX Kontext model installed.\n"
            "  Download flux1-dev-kontext_fp8_scaled.safetensors from\n"
            "  Comfy-Org/flux1-kontext-dev_ComfyUI into ComfyUI/models/diffusion_models.\n"
            "  Note plain FLUX.1 dev cannot do this -- Kontext is a separate model.")

    clips = _choices(info, "DualCLIPLoader", "clip_name1")
    vaes = _choices(info, "VAELoader", "vae_name")
    return {
        "unet": _pick(kontext, prefer=("fp8", "scaled"), what="Kontext model", fix=""),
        "clip_l": _pick(clips, must=("clip_l",), what="clip_l", fix="Needs clip_l in models/clip."),
        "t5": _pick(clips, must=("t5",), prefer=("fp8", "fp16"), what="t5xxl",
                    fix="Needs t5xxl in models/clip."),
        "vae": _pick(vaes, must=("ae",), prefer=("ae.safetensors",), what="FLUX VAE",
                     fix="Needs the FLUX ae.safetensors in models/vae."),
    }


def build_kontext_workflow(models: dict, *, reference_image: str, steps: int,
                           guidance: float, seed: int, denoise: float = 1.0) -> tuple:
    wf: dict = {}

    def node(nid, cls, inputs, title):
        wf[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}

    node("1", "UNETLoader",
         {"unet_name": models["unet"], "weight_dtype": "default"}, "FLUX Kontext")
    node("2", "DualCLIPLoader",
         {"clip_name1": models["clip_l"], "clip_name2": models["t5"],
          "type": "flux", "device": "default"}, "Text encoders")
    node("3", "VAELoader", {"vae_name": models["vae"]}, "VAE")

    node("4", "LoadImage", {"image": reference_image, "upload": "image"}, "Reference")
    # Kontext was trained on a specific set of resolutions; feeding it anything
    # else degrades it in ways that look like the model being bad at its job.
    node("5", "FluxKontextImageScale", {"image": ["4", 0]}, "Fit to Kontext grid")
    node("6", "VAEEncode", {"pixels": ["5", 0], "vae": ["3", 0]}, "Encode reference")

    node("7", "CLIPTextEncode", {"text": "", "clip": ["2", 0]}, "Instruction")
    node("8", "FluxGuidance", {"conditioning": ["7", 0], "guidance": guidance}, "Guidance")
    # This is the whole trick: the reference latent is attached to the
    # conditioning, so its tokens ride along in the sequence being denoised.
    node("9", "ReferenceLatent", {"conditioning": ["8", 0], "latent": ["6", 0]},
         "Attach reference tokens")
    node("10", "ConditioningZeroOut", {"conditioning": ["7", 0]}, "Empty negative")

    # Starting from the reference's own latent rather than pure noise is what
    # makes this an edit of that image instead of a new picture that resembles it.
    node("11", "KSampler",
         {"model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0],
          "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0,
          "sampler_name": "euler", "scheduler": "simple", "denoise": denoise},
         "Sampler")
    node("12", "VAEDecode", {"samples": ["11", 0], "vae": ["3", 0]}, "Decode")
    node("13", "SaveImage", {"images": ["12", 0], "filename_prefix": "kontext"}, "Save")

    mapping = {
        "positive_prompt_node": "7", "positive_prompt_input": "text",
        "negative_prompt_node": "", "negative_prompt_input": "text",
        "seed_node": "11", "seed_input": "seed",
        "save_image_node": "13",
    }
    return wf, mapping


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--out", default="workflows/flux_kontext_api.json")
    p.add_argument("--reference", default="example.png")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    try:
        info = fetch_object_info(a.url)
        models = resolve_kontext_models(info)
        wf, mapping = build_kontext_workflow(
            models, reference_image=a.reference, steps=a.steps,
            guidance=a.guidance, seed=a.seed)

        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2)

        for k, v in models.items():
            print(f"    {k:<8} {v}")
        print(f"\nWrote {a.out}  ({len(wf)} nodes)")

        if a.write:
            from config import load_config, save_config, NodeMapping, CONFIG_PATH
            cfg = load_config()
            cfg.workflow_json_path = os.path.abspath(a.out)
            cfg.node_mapping = NodeMapping(**mapping)
            save_config(cfg)
            print(f"Configured {CONFIG_PATH}.")
        return 0
    except BuildError as e:
        print(f"\nCannot build the Kontext workflow yet.\n\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
