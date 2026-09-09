"""Does the three-stage pipeline actually hold together?

Covers the parts that are testable without a GPU, an API key, or credits:
routing decisions, stage detection, IP-Adapter injection into a workflow, the
new registry columns, migration of an existing registry, and the full
draft -> keyframe -> clip walk through Harry's tools in mock mode.

What it deliberately cannot prove: that FLUX, Minimax, LTX or Runway produce
good output. That needs the real thing.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod
import director_engine as engine
import model_pipeline as pipeline
import storyboard_store as store
from harry_agent import build_registry
from reference_conditioning import apply_reference, inspect_workflow

FAILURES = []


def check(condition, label):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


def make_cfg(workdir):
    cfg = cfgmod.ToolchainConfig()
    cfg.storyboard_path = os.path.join(workdir, "storyboard.xlsx")
    cfg.images_dir = os.path.join(workdir, "images")
    cfg.mock_mode = True
    return cfg


def flux_workflow():
    """A minimal but realistic FLUX + IP-Adapter graph in API format."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "3": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-plus.bin"}},
        "4": {"class_type": "IPAdapterAdvanced",
              "inputs": {"model": ["1", 0], "ipadapter": ["3", 0], "image": ["2", 0], "weight": 0.5}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a pig", "clip": ["1", 1]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": "scene_placeholder.png"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "positive": ["5", 0], "latent_image": ["7", 0], "seed": 1}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }


def kontext_workflow(with_scale=True):
    workflow = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux1-dev-kontext_fp8_scaled.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "6": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["5" if with_scale else "4", 0], "vae": ["3", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "turn him to face left"}},
        "9": {"class_type": "ReferenceLatent",
              "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
        "11": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["9", 0],
                          "latent_image": ["6", 0], "seed": 1, "denoise": 1.0}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["11", 0]}},
    }
    if with_scale:
        workflow["5"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}}
    return workflow


def scenario_routing():
    print("\n[1] Shots route to the right model")
    cases = [
        ("Extreme close up of the pig squinting, dramatic side-lighting", pipeline.CLIP_MINIMAX.key, "dialogue"),
        ("The rooster speaks to the crew, reaction shot", pipeline.CLIP_MINIMAX.key, "dialogue"),
        ("The camera pans right following the cart, cuts to a wide tracking shot", pipeline.CLIP_LTX.key, "camera"),
        ("Wide establishing shot of the harbour at dawn", pipeline.CLIP_LTX.key, "camera"),
        ("The hull shatters, water splashes across the deck", pipeline.CLIP_RUNWAY.key, "action"),
        ("An explosion tears through the cargo hold, debris everywhere", pipeline.CLIP_RUNWAY.key, "action"),
    ]
    for description, expected_model, expected_type in cases:
        route = pipeline.route_shot(description)
        check(route.model_key == expected_model and route.shot_type == expected_type,
              f"{expected_type:8s} <- {description[:44]!r} (got {route.shot_type})")

    # Physics must win over performance when a shot has both.
    both = pipeline.route_shot("Close up of the pig's face as the hull explodes behind him")
    check(both.model_key == pipeline.CLIP_RUNWAY.key,
          "physics beats performance when a shot has both")

    override = pipeline.route_shot("anything at all", override="dialogue")
    check(override.model_key == pipeline.CLIP_MINIMAX.key and override.confidence == "certain",
          "the director's override wins")


def scenario_ipadapter():
    print("\n[2] IP-Adapter injection actually rewires the graph")
    cfg = cfgmod.ToolchainConfig()
    cfg.agent.reference_conditioning = "both"

    workdir = tempfile.mkdtemp(prefix="ipa_")
    turnaround = os.path.join(workdir, "turnaround.png")
    scene = os.path.join(workdir, "scene.png")
    for path in (turnaround, scene):
        engine._mock_image("ref", "x", 1).save(path)

    found = inspect_workflow(flux_workflow())
    check(found["has_ipadapter"], "the IP-Adapter node is detected")
    check(found["can_lock_identity"], "the workflow can lock identity")
    check(found["can_anchor_scene"], "the workflow can anchor a scene")

    work, note = apply_reference(flux_workflow(), cfg, turnaround, scene)
    check(note == "", f"both references applied cleanly (note: {note!r})")
    check(work["2"]["inputs"]["image"] == os.path.abspath(turnaround),
          "the turnaround landed on the IP-Adapter's own LoadImage")
    check(work["6"]["inputs"]["image"] == os.path.abspath(scene),
          "the scene landed on the VAEEncode's LoadImage")
    check(work["4"]["inputs"]["weight"] == cfg.agent.ipadapter_weight,
          "the adapter weight was set from config")

    # A workflow without the node must say so rather than pretend.
    bare = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    _, warn = apply_reference(bare, cfg, turnaround)
    check("IP-Adapter" in warn and "add" in warn.lower(),
          "a missing IP-Adapter node is reported, not silently ignored")

    cfg_off = cfgmod.ToolchainConfig()
    cfg_off.agent.reference_conditioning = "off"
    _, warn_off = apply_reference(flux_workflow(), cfg_off, turnaround)
    check("off" in warn_off.lower(),
          "conditioning switched off is reported honestly")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_kontext():
    print("\n[2b] Kontext injection reaches the loader through the whole chain")
    cfg = cfgmod.ToolchainConfig()
    check(cfg.agent.reference_conditioning == "kontext",
          "Kontext is the default, since IP-Adapter cannot hold costume detail")

    workdir = tempfile.mkdtemp(prefix="kontext_")
    reference = os.path.join(workdir, "reference.png")
    engine._mock_image("ref", "x", 1).save(reference)

    found = inspect_workflow(kontext_workflow())
    check(found["has_kontext"], "the ReferenceLatent node is detected")
    check(found["can_lock_identity"], "a Kontext graph can lock identity")
    check(found["identity_mechanism"] == "kontext", "the mechanism is named correctly")
    # A Kontext graph encodes its reference through a VAEEncode, which must not
    # be mistaken for img2img scene anchoring -- that mistake is what makes a
    # perfectly correct denoise of 1.0 look like a misconfiguration.
    check(not found["can_anchor_scene"],
          "its VAEEncode is not mistaken for scene anchoring")

    work, note = apply_reference(kontext_workflow(), cfg, reference)
    check(note == "", f"the reference applied cleanly (note: {note!r})")
    check(work["4"]["inputs"]["image"] == os.path.abspath(reference),
          "the filename landed on the LoadImage three nodes upstream of ReferenceLatent")

    # Without the scaler the render still runs, so this has to be a note rather
    # than a refusal -- but a silent one would leave quality loss unexplained.
    no_scale = kontext_workflow(with_scale=False)
    _, scale_note = apply_reference(no_scale, cfg, reference)
    check("FluxKontextImageScale" in scale_note,
          "a missing resolution scaler is mentioned, not silently tolerated")
    check(no_scale["4"]["inputs"]["image"] == os.path.abspath(reference),
          "and the reference is still applied")

    bare = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    _, warn = apply_reference(bare, cfg, reference)
    check("ReferenceLatent" in warn,
          "a workflow with no Kontext chain says so rather than pretending")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_stages(cfg):
    print("\n[3] Stage detection reads the registry, not a status field")
    check(pipeline.next_stage({}) == pipeline.STAGE_DRAFT, "nothing yet -> draft")
    check(pipeline.next_stage({"entry_type": "SHOT", "image_path": "a.png"}) == pipeline.STAGE_KEYFRAME,
          "shot with a draft -> keyframe")
    check(pipeline.next_stage({"entry_type": "SHOT", "image_path": "a.png",
                               "keyframe_path": "k.png"}) == pipeline.STAGE_CLIP,
          "shot with a keyframe -> clip")
    check(pipeline.next_stage({"entry_type": "SHOT", "image_path": "a.png",
                               "keyframe_path": "k.png", "clip_path": "c.mp4"}) == pipeline.STAGE_DONE,
          "shot with a clip -> done")
    check(pipeline.next_stage({"entry_type": "CHARACTER", "image_path": "a.png",
                               "composite_image_path": "s.png"}) == pipeline.STAGE_DONE,
          "character with a sheet -> done")


def scenario_registry_columns(cfg):
    print("\n[4] The new columns persist on the same sheet")
    store.set_keyframe(cfg.storyboard_path, "1.1", "kf.png", "a pig on deck", 42, "staged")
    store.set_clip(cfg.storyboard_path, "1.1", "clip.mp4", "pig turns to camera",
                   pipeline.CLIP_MINIMAX.key, "dialogue", "animated")
    row = store.get_entry(cfg.storyboard_path, "1.1")
    check(row.get("keyframe_path") == "kf.png", "keyframe_path persisted")
    check(row.get("clip_path") == "clip.mp4", "clip_path persisted")
    check(row.get("video_model") == pipeline.CLIP_MINIMAX.key, "video_model persisted")
    check(row.get("shot_type") == "dialogue", "shot_type persisted")
    check("staged" in (row.get("notes") or "") and "animated" in (row.get("notes") or ""),
          "both notes appended to the same row, nothing overwritten")

    entries = [r for r in store.list_entries(cfg.storyboard_path) if r.get("entry_id") == "1.1"]
    check(len(entries) == 1, "still exactly one row for the shot, not a duplicate")


def scenario_migration():
    print("\n[5] An existing registry gains the columns without losing data")
    workdir = tempfile.mkdtemp(prefix="migrate_")
    path = os.path.join(workdir, "old.xlsx")

    from openpyxl import Workbook
    workbook = Workbook()
    workbook.remove(workbook.active)
    sheet = workbook.create_sheet("Assets")
    old_columns = ["entry_id", "entry_type", "beat", "description", "prompt_positive",
                   "prompt_negative_add", "model", "seed", "reused_from", "locked",
                   "image_path", "notes", "template", "composite_image_path"]
    sheet.append(old_columns)
    sheet.append(["CHARACTER:old", "CHARACTER", "opening", "made before the video stage",
                  "p", "", "flux", 7, "", True, "old.png", "kept", "CHARACTER", "sheet.png"])
    workbook.save(path)

    store.set_keyframe(path, "CHARACTER:old", "new_kf.png", note="added later")
    row = store.get_entry(path, "CHARACTER:old")
    check(row.get("description") == "made before the video stage", "the original data survived")
    check(row.get("composite_image_path") == "sheet.png", "the original sheet path survived")
    check(row.get("keyframe_path") == "new_kf.png", "the new column was added and written")
    check("kept" in (row.get("notes") or ""), "the original note survived")
    shutil.rmtree(workdir, ignore_errors=True)


def scenario_tools_walk(cfg):
    print("\n[6] Harry can walk draft -> keyframe -> clip")
    registry = build_registry(cfg)
    names = registry.names()
    for tool in ("pipeline_map", "asset_stage", "attach_draft", "route_shot",
                 "plan_shot", "stage_keyframe", "lock_keyframe", "animate_shot",
                 "check_reference_setup"):
        check(tool in names, f"tool '{tool}' is registered")

    os.makedirs(cfg.images_dir, exist_ok=True)
    draft = os.path.join(cfg.images_dir, "chatgpt_pig_turnaround.png")
    engine._mock_image("turnaround", "pig, front side back", 1).save(draft)

    result = registry.run("attach_draft", {
        "entry_id": "CHARACTER:pig2", "entry_type": "CHARACTER",
        "image_path": draft, "description": "the lead"})
    check(result.ok and "Locked" in result.content, "the ChatGPT draft attached")

    result = registry.run("asset_stage", {"entry_id": "CHARACTER:pig2"})
    check(result.ok and json.loads(result.content)["has_draft"] is True,
          "asset_stage sees the draft")

    result = registry.run("route_shot", {
        "entry_id": "2.1",
        "description": "Close up of the pig speaking, dramatic side-lighting"})
    payload = json.loads(result.content)
    check(payload["model_key"] == pipeline.CLIP_MINIMAX.key,
          "a dialogue shot routes to Minimax H3")
    check("performance" in payload["why"], "the routing explains itself")

    result = registry.run("plan_shot", {
        "entry_id": "2.1", "shot_type": "dialogue",
        "video_model": pipeline.CLIP_MINIMAX.key,
        "motion_prompt": "the pig turns and speaks"})
    check(result.ok and "Minimax" in result.content, "the shot plan recorded")

    result = registry.run("stage_keyframe", {
        "entry_id": "2.1", "prompt_positive": "the pig on deck at dusk",
        "character_entry_id": "CHARACTER:pig2", "n_variants": 2})
    payload = json.loads(result.content)
    check(len(payload["variants"]) == 2, "two keyframe candidates came back")
    check(payload["identity_from"].endswith(".png"),
          "the character's draft was passed as the identity reference")

    winner = payload["variants"][0]["image_path"]
    result = registry.run("lock_keyframe", {
        "entry_id": "2.1", "image_path": winner, "seed": payload["variants"][0]["seed"]})
    check(result.ok and "Staged" in result.content, "the keyframe locked")

    result = registry.run("animate_shot", {
        "entry_id": "2.1", "motion_prompt": "the pig turns to camera and speaks",
        "character_entry_id": "CHARACTER:pig2"})
    payload = json.loads(result.content)
    check(payload["model"] == pipeline.CLIP_MINIMAX.key, "the clip used the routed model")
    check(os.path.exists(payload["clip_path"]), "a clip artefact was written")

    row = store.get_entry(cfg.storyboard_path, "2.1")
    check(row.get("clip_path") and row.get("keyframe_path"),
          "keyframe and clip both recorded on the shot's registry row")

    result = registry.run("animate_shot", {
        "entry_id": "9.9", "motion_prompt": "anything"})
    check("No entry" in result.content, "animating an unknown shot fails cleanly")


def scenario_guard_rails(cfg):
    print("\n[7] The pipeline refuses to skip a stage")
    registry = build_registry(cfg)
    store.set_shot_plan(cfg.storyboard_path, "3.1", "camera", pipeline.CLIP_LTX.key, "pan right")
    result = registry.run("animate_shot", {"entry_id": "3.1", "motion_prompt": "pan right"})
    check("no keyframe" in result.content.lower(),
          "a shot without a keyframe cannot be animated")


def main():
    workdir = tempfile.mkdtemp(prefix="pipeline_")
    try:
        cfg = make_cfg(workdir)
        scenario_routing()
        scenario_ipadapter()
        scenario_kontext()
        scenario_stages(cfg)
        scenario_registry_columns(cfg)
        scenario_migration()
        scenario_tools_walk(cfg)
        scenario_guard_rails(cfg)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nfailures: {len(FAILURES)}")
    for failure in FAILURES:
        print("   ", failure)
    print("RESULT:", "PASS" if not FAILURES else "FAIL")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
