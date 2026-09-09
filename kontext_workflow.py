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
                           guidance: float, seed: int, denoise: float = 1.0,
                           extra_references=(), multi: str = "chain") -> tuple:
    """Wire a Kontext graph for one reference, or several.

    Two ways to give Kontext more than one reference, and they are not
    interchangeable:

      chain   each image gets its own encode and its own ReferenceLatent, and
              the conditioning passes through all of them. Every reference
              keeps its full resolution and its own identity. This is what you
              want for "this character, in this room".

      stitch  the images are joined into one picture before encoding, so the
              model reads them as a single scene. Useful when the relationship
              between them matters -- a size comparison, a before/after -- and
              wasteful otherwise, since each image gets a fraction of the
              token budget.

    A subtlety worth stating: the sampler's starting latent decides the output
    size. Under `stitch` that must come from the first reference alone, or a
    pair of 1024s would silently produce a 2048-wide frame.
    """
    wf: dict = {}

    def node(nid, cls, inputs, title):
        wf[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}

    node("1", "UNETLoader",
         {"unet_name": models["unet"], "weight_dtype": "default"}, "FLUX Kontext")
    node("2", "DualCLIPLoader",
         {"clip_name1": models["clip_l"], "clip_name2": models["t5"],
          "type": "flux", "device": "default"}, "Text encoders")
    node("3", "VAELoader", {"vae_name": models["vae"]}, "VAE")

    references = [reference_image] + [r for r in extra_references if r]
    if len(references) > 1 and multi not in ("chain", "stitch"):
        raise BuildError(f"Unknown multi-reference strategy '{multi}'. Use chain or stitch.")

    def load_and_encode(image, index):
        """LoadImage -> fit to Kontext's grid -> latent. Returns the encode id."""
        load_id, scale_id, enc_id = f"4_{index}", f"5_{index}", f"6_{index}"
        node(load_id, "LoadImage", {"image": image, "upload": "image"},
             f"Reference {index + 1}")
        # Kontext was trained on a specific set of resolutions; feeding it
        # anything else degrades it in ways that look like the model being bad
        # at its job.
        node(scale_id, "FluxKontextImageScale", {"image": [load_id, 0]},
             f"Fit reference {index + 1}")
        node(enc_id, "VAEEncode", {"pixels": [scale_id, 0], "vae": ["3", 0]},
             f"Encode reference {index + 1}")
        return load_id, enc_id

    node("7", "CLIPTextEncode", {"text": "", "clip": ["2", 0]}, "Instruction")
    node("8", "FluxGuidance", {"conditioning": ["7", 0], "guidance": guidance}, "Guidance")

    if len(references) == 1 or multi == "chain":
        conditioning = ["8", 0]
        canvas_encode = None
        for index, image in enumerate(references):
            _, enc_id = load_and_encode(image, index)
            if canvas_encode is None:
                canvas_encode = enc_id
            # This is the whole trick: the reference latent is attached to the
            # conditioning, so its tokens ride along in the sequence being
            # denoised. Chaining adds another without displacing the first.
            ref_id = f"9_{index}"
            node(ref_id, "ReferenceLatent",
                 {"conditioning": conditioning, "latent": [enc_id, 0]},
                 f"Attach reference {index + 1}")
            conditioning = [ref_id, 0]
    else:
        stitched = None
        for index, image in enumerate(references):
            load_id = f"4_{index}"
            node(load_id, "LoadImage", {"image": image, "upload": "image"},
                 f"Reference {index + 1}")
            if stitched is None:
                stitched = [load_id, 0]
                continue
            stitch_id = f"14_{index}"
            node(stitch_id, "ImageStitch",
                 {"image1": stitched, "image2": [load_id, 0], "direction": "right",
                  "match_image_size": True, "spacing_width": 0, "spacing_color": "white"},
                 f"Stitch reference {index + 1}")
            stitched = [stitch_id, 0]

        node("5_s", "FluxKontextImageScale", {"image": stitched}, "Fit stitched sheet")
        node("6_s", "VAEEncode", {"pixels": ["5_s", 0], "vae": ["3", 0]}, "Encode sheet")
        node("9_0", "ReferenceLatent",
             {"conditioning": ["8", 0], "latent": ["6_s", 0]}, "Attach stitched reference")
        conditioning = ["9_0", 0]

        # The canvas comes from the first reference on its own. Taking it from
        # the stitched sheet would make the output as wide as all the
        # references laid side by side.
        node("5_c", "FluxKontextImageScale", {"image": ["4_0", 0]}, "Fit output canvas")
        node("6_c", "VAEEncode", {"pixels": ["5_c", 0], "vae": ["3", 0]}, "Canvas latent")
        canvas_encode = "6_c"

    node("10", "ConditioningZeroOut", {"conditioning": ["7", 0]}, "Empty negative")

    # Starting from the reference's own latent rather than pure noise is what
    # makes this an edit of that image instead of a new picture that resembles it.
    node("11", "KSampler",
         {"model": ["1", 0], "positive": conditioning, "negative": ["10", 0],
          "latent_image": [canvas_encode, 0], "seed": seed, "steps": steps, "cfg": 1.0,
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
    p.add_argument("--extra", action="append", default=[],
                   help="An additional reference. Repeatable -- a backdrop, a prop.")
    p.add_argument("--multi", default="chain", choices=("chain", "stitch"),
                   help="chain keeps each reference whole; stitch joins them into one picture.")
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
        if a.extra and a.multi == "stitch" and "ImageStitch" not in info:
            raise BuildError("This ComfyUI has no ImageStitch node. Update ComfyUI, "
                             "or use --multi chain, which needs no extra nodes.")
        models = resolve_kontext_models(info)
        wf, mapping = build_kontext_workflow(
            models, reference_image=a.reference, steps=a.steps,
            guidance=a.guidance, seed=a.seed,
            extra_references=a.extra, multi=a.multi)

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
