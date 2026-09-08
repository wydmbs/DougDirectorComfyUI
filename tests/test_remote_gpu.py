"""Does the harness actually work when ComfyUI is on another machine?

The bug this guards against is quiet and expensive: a LoadImage node given an
absolute path from a *different* computer loads nothing, ComfyUI reports
success, and the render silently ignores the reference. Every character then
drifts while the logs say everything worked.

Also covers the ChatGPT handoff (prompt out, image in) and the video stage's
move to ComfyUI workflows.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod
import director_engine as engine
import draft_prompts
import model_pipeline as pipeline
import video_client
from harry_agent import build_registry
from reference_conditioning import apply_reference

FAILURES = []


def check(condition, label):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


def flux_workflow():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "3": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-plus.bin"}},
        "4": {"class_type": "IPAdapterAdvanced",
              "inputs": {"model": ["1", 0], "ipadapter": ["3", 0], "image": ["2", 0], "weight": 0.5}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a pig", "clip": ["1", 1]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": "scene.png"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "latent_image": ["7", 0], "seed": 1}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }


def scenario_remote_paths():
    print("\n[1] References survive ComfyUI being on another machine")
    cfg = cfgmod.ToolchainConfig()
    cfg.agent.reference_conditioning = "both"

    workdir = tempfile.mkdtemp(prefix="remote_")
    turnaround = os.path.join(workdir, "pig_turnaround.png")
    scene = os.path.join(workdir, "harbour.png")
    for path in (turnaround, scene):
        engine._mock_image("ref", "x", 1).save(path)

    uploaded = []

    def fake_upload(local_path, subfolder="director_harness", overwrite=True):
        """Stand in for ComfyUI's /upload/image: returns the name it would hand back."""
        uploaded.append(local_path)
        return f"director_harness/{os.path.basename(local_path)}"

    work, note = apply_reference(flux_workflow(), cfg, turnaround, scene, upload=fake_upload)
    check(note == "", f"both references applied (note: {note!r})")
    check(len(uploaded) == 2, f"both files were uploaded (got {len(uploaded)})")

    adapter_image = work["2"]["inputs"]["image"]
    scene_image = work["6"]["inputs"]["image"]
    check(not os.path.isabs(adapter_image),
          f"the turnaround is referenced by uploaded name, not a local path ({adapter_image!r})")
    check(not os.path.isabs(scene_image),
          f"the scene is referenced by uploaded name, not a local path ({scene_image!r})")
    check(adapter_image == "director_harness/pig_turnaround.png",
          "the adapter got the name ComfyUI returned")

    # Without an uploader we fall back to a local path -- correct only for a
    # same-machine ComfyUI, and the thing the uploader exists to replace.
    work_local, _ = apply_reference(flux_workflow(), cfg, turnaround, scene)
    check(os.path.isabs(work_local["2"]["inputs"]["image"]),
          "with no uploader it falls back to a local path (same-machine only)")

    # An upload failure must be reported, never silently ignored.
    def failing_upload(local_path, **kwargs):
        raise OSError("connection refused")

    _, warn = apply_reference(flux_workflow(), cfg, turnaround, upload=failing_upload)
    check("could not send" in warn.lower(),
          "an upload failure is reported rather than silently skipped")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_upload_caching():
    print("\n[2] One sheet isn't re-uploaded for every panel")
    from comfy_client import ComfyClient

    workdir = tempfile.mkdtemp(prefix="cache_")
    sheet = os.path.join(workdir, "sheet.png")
    engine._mock_image("sheet", "x", 1).save(sheet)

    client = ComfyClient("http://127.0.0.1:9")
    sent = []

    def fake_post(*args, **kwargs):
        sent.append(1)
        raise AssertionError("should not be reached in this test")

    # Prime the cache the way a successful upload would, then confirm a repeat
    # call is served from it rather than hitting the network again.
    stamp = os.path.getmtime(sheet)
    client._upload_cache[(os.path.abspath(sheet), stamp, "director_harness")] = "cached/sheet.png"
    for _ in range(18):
        reference = client.upload_image(sheet)
    check(reference == "cached/sheet.png", "the cached name is returned")
    check(not sent, "eighteen panels re-used one upload instead of eighteen")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_chatgpt_prompts():
    print("\n[3] The ChatGPT handoff produces a usable prompt")
    prompt = draft_prompts.build(
        "CHARACTER", "a stout farm pig in a tweed waistcoat",
        description="broad shouldered, calm", era="early 20th century maritime",
        palette="muted slate and oiled brass")
    for needed in ("front", "side profile", "behind", "Flat, even studio lighting",
                   "light grey", "early 20th century maritime", "muted slate"):
        check(needed.lower() in prompt.lower(), f"the character prompt covers {needed!r}")
    check("DO NOT INCLUDE" in prompt and "watermark" in prompt.lower(),
          "the character prompt excludes text and watermarks")

    scene = draft_prompts.build("BACKDROP", "a harbour at dawn",
                                lighting="low sun through fog", mood="uneasy calm")
    check("establishing" in scene.lower(), "the scene prompt asks for an establishing shot")
    check("low sun through fog" in scene, "the scene prompt carries the lighting")
    check("people or characters" in scene.lower(), "the scene prompt excludes characters")

    prop = draft_prompts.build("PROP", "a wooden shipping crate")
    check("front" in prop.lower() and "behind" in prop.lower(), "the prop prompt asks for angles")

    # A shot is composed, not drafted -- drafting one would produce a character
    # who doesn't match the sheet.
    try:
        draft_prompts.build("SHOT", "the pig on deck")
        check(False, "drafting a SHOT is refused")
    except ValueError as error:
        check("composed in FLUX" in str(error), "drafting a SHOT is refused, with the reason")


def scenario_video_via_comfy(cfg):
    print("\n[4] Video runs through ComfyUI, not three separate APIs")
    check(hasattr(video_client, "workflow_path_for"), "each model maps to a workflow")
    check(not hasattr(video_client, "generate_minimax"),
          "the direct Minimax HTTP client is gone")
    check(not hasattr(video_client, "generate_runway"),
          "the direct Runway HTTP client is gone")

    for key in (pipeline.CLIP_LTX.key, pipeline.CLIP_MINIMAX.key, pipeline.CLIP_RUNWAY.key):
        check(video_client.workflow_path_for(cfg, key) == "",
              f"{key} starts with no workflow configured")

    report = video_client.readiness(cfg)
    check(report["mock_mode"] is True, "readiness reports mock mode")
    check(set(report["models"]) == {pipeline.CLIP_LTX.key, pipeline.CLIP_MINIMAX.key,
                                    pipeline.CLIP_RUNWAY.key},
          "readiness covers all three routes")
    check(all(not m["ready"] for m in report["models"].values()),
          "nothing claims to be ready before a workflow is set")

    # A real run with no workflow must fail with something actionable.
    cfg_live = cfgmod.ToolchainConfig()
    cfg_live.mock_mode = False
    cfg_live.images_dir = cfg.images_dir
    try:
        video_client.generate_clip(cfg_live, "1.1", __file__, "pan right",
                                   model_key=pipeline.CLIP_LTX.key)
        check(False, "a missing workflow is refused")
    except video_client.VideoError as error:
        check("Save (API Format)" in str(error),
              "the error says exactly how to produce the workflow")


def scenario_workflow_patching():
    print("\n[5] A video workflow gets this shot's inputs")
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "old2.png"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "old prompt"}},
        "4": {"class_type": "LTXVImgToVideo", "inputs": {"duration": 3.0}},
        "5": {"class_type": "VHS_VideoCombine", "inputs": {}},
    }
    warnings = video_client._patch_workflow(
        workflow, "up/keyframe.png", "up/turnaround.png", "the pig turns to camera", 8.0)
    check(workflow["1"]["inputs"]["image"] == "up/keyframe.png", "the keyframe went to the first loader")
    check(workflow["2"]["inputs"]["image"] == "up/turnaround.png", "the turnaround went to the second")
    check(workflow["3"]["inputs"]["text"] == "the pig turns to camera", "the motion prompt landed")
    check(workflow["4"]["inputs"]["duration"] == 8.0, "the duration landed")
    check(not warnings, f"no warnings for a complete workflow (got {warnings})")

    # One loader means REF2VA has nothing to hold the face with -- say so.
    single = {"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}
    warnings = video_client._patch_workflow(single, "kf.png", "turn.png", "", 0)
    check(any("turnaround" in w for w in warnings),
          "a single-loader workflow warns that the turnaround wasn't passed")

    none = {"1": {"class_type": "KSampler", "inputs": {}}}
    warnings = video_client._patch_workflow(none, "kf.png", "", "", 0)
    check(any("LoadImage" in w for w in warnings),
          "a workflow with no image input warns that the keyframe wasn't applied")


def scenario_tools(cfg):
    print("\n[6] Harry has the handoff and the GPU check")
    registry = build_registry(cfg)
    for tool in ("draft_prompt", "gpu_readiness"):
        check(tool in registry.names(), f"tool '{tool}' is registered")

    result = registry.run("draft_prompt", {
        "entry_id": "CHARACTER:pig", "entry_type": "CHARACTER",
        "subject": "a stout farm pig", "era": "early 20th century"})
    payload = json.loads(result.content)
    check("prompt_for_chatgpt" in payload, "the tool returns a prompt for ChatGPT")
    check("attach_draft" in payload["next_step"], "it points at the next step")

    result = registry.run("draft_prompt", {
        "entry_id": "1.1", "entry_type": "SHOT", "subject": "the pig on deck"})
    check("composed in FLUX" in result.content, "it refuses to draft a shot")

    result = registry.run("gpu_readiness", {})
    payload = json.loads(result.content)
    check("comfyui_url" in payload and "models" in payload, "gpu_readiness reports the machine")
    check("Mock mode" in payload.get("note", ""), "it says mock mode is on")


def main():
    workdir = tempfile.mkdtemp(prefix="remote_gpu_")
    try:
        cfg = cfgmod.ToolchainConfig()
        cfg.storyboard_path = os.path.join(workdir, "storyboard.xlsx")
        cfg.images_dir = os.path.join(workdir, "images")
        cfg.mock_mode = True

        scenario_remote_paths()
        scenario_upload_caching()
        scenario_chatgpt_prompts()
        scenario_video_via_comfy(cfg)
        scenario_workflow_patching()
        scenario_tools(cfg)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nfailures: {len(FAILURES)}")
    for failure in FAILURES:
        print("   ", failure)
    print("RESULT:", "PASS" if not FAILURES else "FAIL")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
