"""Regressions for everything the design review turned up.

Each case here is a defect that was found by reading the code rather than by
running it, which is exactly the kind that survives a green test suite. They are
grouped by the review's numbering so a future reader can trace them back.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod
import director_engine as engine
import registry_safety
import storyboard_store as store
import video_client
from harry_agent import SUPERVISED, Decision, PermissionPolicy, build_registry
from harry_agent.safety import Verdict, analyze
from harry_agent.tools.registry import Tool, ToolRegistry
from reference_conditioning import apply_reference

FAILURES = []


def check(condition, label):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


def scenario_interpreters():
    print("\n[1] Interpreters are never waved through as read-only")
    # The worst case found: this classified as read_only and would have run
    # with no prompt at all.
    result = analyze('python -c "import shutil; shutil.rmtree(\'C:/Users\')"')
    check(result.verdict is not Verdict.READ_ONLY,
          "python -c is not read-only")
    check(analyze("py -c \"print(1)\"").verdict is Verdict.NEEDS_CONSENT,
          "py -c needs consent even when harmless")
    check(analyze("node evil.js").verdict is Verdict.NEEDS_CONSENT, "node needs consent")
    check(analyze("git -c alias.x='!powershell -c evil' x").verdict is Verdict.NEEDS_CONSENT,
          "a git alias smuggling a shell needs consent")
    check(analyze("Get-ChildItem C:\\ | Remove-Item -Recurse -Force").verdict
          is not Verdict.READ_ONLY,
          "a listing piped into a recursive delete is not read-only")
    check(analyze("git status").verdict is Verdict.READ_ONLY, "git status is still read-only")
    check(analyze("pip list").verdict is Verdict.READ_ONLY, "pip list is still read-only")


def scenario_always_confirm():
    print("\n[2] always_confirm really means always")
    install = Tool("run_setup_command", "d", {}, lambda: "ok",
                   mutating=True, always_confirm=True)
    routine = Tool("lock_panel", "d", {}, lambda: "ok", mutating=True, routine=True)
    policy = PermissionPolicy(posture=SUPERVISED)

    asked = []

    def ask_session(request):
        asked.append(request.tool_name)
        return Decision.ALLOW_SESSION

    from harry_agent.permissions import PermissionRequest
    request = PermissionRequest("run_setup_command", "install something")

    policy.check(install, request, ask=ask_session)
    policy.check(install, request, ask=ask_session)
    policy.check(install, request, ask=ask_session)
    check(len(asked) == 3,
          f"three installs asked three times, not once (asked {len(asked)})")
    check("run_setup_command" not in policy.session_grants,
          "an install grant is never remembered for the session")

    # An ordinary mutating tool may still be granted for the session. Under
    # `supervised` a routine tool isn't asked about at all, so this needs the
    # stricter posture to exercise the grant path.
    asked.clear()
    from harry_agent import ATTENDED
    strict = PermissionPolicy(posture=ATTENDED)
    strict.check(routine, PermissionRequest("lock_panel", "lock"), ask=ask_session)
    strict.check(routine, PermissionRequest("lock_panel", "lock"), ask=ask_session)
    check(len(asked) == 1, f"a routine tool asks once and is then remembered (asked {len(asked)})")

    # And under supervised, a routine tool isn't interrupted at all.
    asked.clear()
    policy.check(routine, PermissionRequest("lock_panel", "lock"), ask=ask_session)
    check(not asked, "supervised lets routine work through without asking")


def scenario_backups():
    print("\n[3] Backups are actually written")
    workdir = tempfile.mkdtemp(prefix="backups_")
    path = os.path.join(workdir, "storyboard.xlsx")

    store.lock_entry(path=path, entry_id="1.1", entry_type="SHOT", beat="", description="first",
                     prompt_positive="p", prompt_negative_add="", model="m", seed=1,
                     reused_from="", image_path="a.png", note="v1")
    store.lock_entry(path=path, entry_id="1.1", entry_type="SHOT", beat="", description="second",
                     prompt_positive="p", prompt_negative_add="", model="m", seed=2,
                     reused_from="", image_path="b.png", note="v2")

    folder = os.path.join(workdir, registry_safety.BACKUP_DIRNAME)
    backups = os.listdir(folder) if os.path.isdir(folder) else []
    check(len(backups) >= 1, f"a backup exists after a second write (found {len(backups)})")

    # And it can actually be restored.
    store.lock_entry(path=path, entry_id="1.1", entry_type="SHOT", beat="", description="third",
                     prompt_positive="p", prompt_negative_add="", model="m", seed=3,
                     reused_from="", image_path="c.png", note="v3")
    restored = registry_safety.restore_latest_backup(path)
    check(bool(restored), "the most recent backup restores")
    row = store.get_entry(path, "1.1")
    check(row.get("description") == "second",
          f"restoring rolled back the last write (got {row.get('description')!r})")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_lock_ownership():
    print("\n[4] A live writer's lock is never stolen")
    workdir = tempfile.mkdtemp(prefix="ownership_")
    target = os.path.join(workdir, "registry.xlsx")
    lock_file = registry_safety._lock_path(target)

    # A lock owned by THIS process, made to look ancient. Age alone must not be
    # enough to reclaim it -- a slow save over OneDrive looks exactly like this.
    with open(lock_file, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}:live")
    old = time.time() - 10_000
    os.utime(lock_file, (old, old))
    check(not registry_safety._reclaimable(lock_file, 1.0),
          "an old lock held by a living process is not reclaimable")

    # A lock owned by a process that cannot exist may be reclaimed.
    with open(lock_file, "w", encoding="utf-8") as handle:
        handle.write("999999999:dead")
    os.utime(lock_file, (old, old))
    check(registry_safety._reclaimable(lock_file, 1.0),
          "a lock held by a dead process is reclaimable")

    os.unlink(lock_file)

    # Two threads racing for a reclaimable lock must not both get inside.
    with open(lock_file, "w", encoding="utf-8") as handle:
        handle.write("999999998:dead")
    os.utime(lock_file, (old, old))

    inside, overlap = [], []
    guard = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(name):
        barrier.wait()
        try:
            with registry_safety.registry_lock(target, timeout_s=0.4, poll_s=0.01):
                with guard:
                    inside.append(name)
                    if len(inside) > 1:
                        overlap.append(tuple(inside))
                time.sleep(0.2)
                with guard:
                    inside.remove(name)
        except registry_safety.RegistryLockTimeout:
            pass

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(not overlap, f"two reclaimers never overlapped (overlap={overlap})")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_tool_failures():
    print("\n[5] A handler that returns a problem is reported as a failure")
    registry = ToolRegistry()
    registry.add(Tool("finder", "d", {}, lambda: "No entry called 'x'."))
    registry.add(Tool("worker", "d", {}, lambda: "Locked CHARACTER:pig."))
    registry.add(Tool("refuser", "d", {}, lambda: "REFUSED -- deletes from a drive root"))

    check(registry.run("finder", {}).ok is False, "a 'No entry' reply is a failure")
    check(registry.run("refuser", {}).ok is False, "a refusal is a failure")
    check(registry.run("worker", {}).ok is True, "a real result is still a success")


def scenario_tool_timeout():
    print("\n[6] A hung tool is given up on")
    registry = ToolRegistry()
    registry.add(Tool("hang", "d", {}, lambda: time.sleep(30), timeout_s=0.3))
    started = time.time()
    result = registry.run("hang", {})
    elapsed = time.time() - started
    check(result.ok is False, "a hung tool is a failure")
    check(elapsed < 5, f"it gave up quickly rather than blocking ({elapsed:.1f}s)")
    check("still running" in result.content, "the message says it may still be working")


def scenario_video_duration():
    print("\n[7] Seconds are not written into frame counts")
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, watermark"}},
        "4": {"class_type": "LTXVImgToVideo", "inputs": {"num_frames": 24, "fps": 24}},
        "5": {"class_type": "VHS_VideoCombine", "inputs": {}},
    }
    warnings, fatal = video_client._patch_workflow(workflow, "kf.png", "", "pig turns", 5.0)
    check(not fatal, f"a complete workflow is not fatal ({fatal})")
    check(workflow["4"]["inputs"]["num_frames"] == 120,
          f"5s at 24fps became 120 frames, not 5 (got {workflow['4']['inputs']['num_frames']})")
    check(workflow["2"]["inputs"]["text"] == "pig turns", "the positive prompt was set")
    check(workflow["3"]["inputs"]["text"] == "blurry, low quality, watermark",
          "the negative prompt was left alone")

    seconds_wf = {"1": {"class_type": "LoadImage", "inputs": {"image": "x"}},
                  "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
                  "3": {"class_type": "MinimaxVideo", "inputs": {"duration": 3.0}}}
    video_client._patch_workflow(seconds_wf, "kf.png", "", "x", 10.0)
    check(seconds_wf["3"]["inputs"]["duration"] == 10.0, "a seconds field still gets seconds")

    bare = {"1": {"class_type": "KSampler", "inputs": {}}}
    warnings, fatal = video_client._patch_workflow(bare, "kf.png", "", "x", 5)
    check(bool(fatal), "a workflow with no image input is fatal, not merely a warning")


def scenario_reference_tracing():
    print("\n[8] The turnaround reaches the loader, not a preprocessor")
    cfg = cfgmod.ToolchainConfig()
    cfg.agent.reference_conditioning = "both"

    workdir = tempfile.mkdtemp(prefix="tracing_")
    turnaround = os.path.join(workdir, "t.png")
    scene = os.path.join(workdir, "s.png")
    for p in (turnaround, scene):
        engine._mock_image("r", "x", 1).save(p)

    # Realistic shape: the adapter is fed through a preprocessing node, not
    # wired straight to LoadImage.
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "2": {"class_type": "PrepImageForClipVision", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "IPAdapterAdvanced", "inputs": {"image": ["2", 0], "weight": 0.5}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "scene_old.png"}},
        "5": {"class_type": "ImageScale", "inputs": {"image": ["4", 0]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0]}},
        "7": {"class_type": "KSampler", "inputs": {"latent_image": ["6", 0], "denoise": 1.0}},
    }
    work, note = apply_reference(workflow, cfg, turnaround, scene,
                                 upload=lambda p, **k: f"up/{os.path.basename(p)}")
    check(work["1"]["inputs"]["image"] == "up/t.png",
          "the turnaround was traced back through the preprocessor to LoadImage")
    check(work["4"]["inputs"]["image"] == "up/s.png",
          "the scene was traced back through the scale node to its LoadImage")
    check("image" not in (work["2"].get("inputs") or {}) or
          isinstance(work["2"]["inputs"]["image"], list),
          "no filename was written into the preprocessor, which would fail validation")
    check("denoise" in note and "0.5" in note,
          f"denoise 1.0 is flagged as discarding the scene (note: {note!r})")

    shutil.rmtree(workdir, ignore_errors=True)


def scenario_blind_locks(cfg):
    print("\n[9] Locking without a critic is recorded as unreviewed")
    registry = build_registry(cfg)
    result = registry.run("critique_variants", {
        "entry_id": "CHARACTER:pig", "image_paths": ["a.png", "b.png"]})
    check("not actually LOOKED" in result.content or "nobody has actually" in result.content.lower(),
          "the critic stub says plainly that nothing was seen")

    os.makedirs(cfg.images_dir, exist_ok=True)
    image = os.path.join(cfg.images_dir, "v.png")
    engine._mock_image("v", "x", 1).save(image)
    registry.run("lock_concept", {
        "entry_id": "CHARACTER:blind", "entry_type": "CHARACTER",
        "image_path": image, "seed": 1, "prompt_positive": "a pig", "note": "picked first"})
    row = store.get_entry(cfg.storyboard_path, "CHARACTER:blind")
    check("unreviewed" in (row.get("notes") or "").lower(),
          "the registry row records that no one saw the image")


def scenario_truncation():
    print("\n[10] A truncated answer is not reported as finished")
    from harry_agent import AgentLoop
    from harry_agent.providers.base import STOP_LENGTH, Turn
    from harry_agent.providers.mock import MockProvider

    loop = AgentLoop(provider=MockProvider([Turn(text="I was about to", stop_reason=STOP_LENGTH)]),
                     registry=ToolRegistry(), policy=PermissionPolicy(), max_steps=5)
    result = loop.run("do something")
    check(result.completed is False, "a length-truncated turn is not a completed run")
    check("ran out of room" in (result.error or ""), "the reason names the token limit")


def main():
    workdir = tempfile.mkdtemp(prefix="review_fixes_")
    try:
        cfg = cfgmod.ToolchainConfig()
        cfg.storyboard_path = os.path.join(workdir, "storyboard.xlsx")
        cfg.images_dir = os.path.join(workdir, "images")
        cfg.mock_mode = True

        scenario_interpreters()
        scenario_always_confirm()
        scenario_backups()
        scenario_lock_ownership()
        scenario_tool_failures()
        scenario_tool_timeout()
        scenario_video_duration()
        scenario_reference_tracing()
        scenario_blind_locks(cfg)
        scenario_truncation()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nfailures: {len(FAILURES)}")
    for failure in FAILURES:
        print("   ", failure)
    print("RESULT:", "PASS" if not FAILURES else "FAIL")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
