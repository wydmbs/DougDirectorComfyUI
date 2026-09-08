"""Does the doctor actually catch a half-configured machine?

An absent ComfyUI is easy to detect. The dangerous states are the in-between
ones: ComfyUI up but IP-Adapter never installed, nodes present but no model
file, a workflow that looks fine but can't hold a face. Those are the ones that
otherwise show up as "the renders look wrong" three days later.

Each profile starts a stand-in ComfyUI, runs the real doctor against it, and
checks the finding that matters is present.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PORT = 8207
FAKE = os.path.join(ROOT, "tests", "fake_comfy.py")
FAILURES = []


def check(condition, label):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        FAILURES.append(label)


def wait_for_port(timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/system_stats", timeout=1)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def full_workflow(path, denoise=0.6, with_ipadapter=True, flux_adapter=True):
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "turnaround.png"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a pig", "clip": ["1", 1]}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "scene.png"}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "KSampler",
               "inputs": {"positive": ["6", 0], "latent_image": ["9", 0],
                          "seed": 1, "denoise": denoise}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0]}},
    }
    if with_ipadapter:
        if flux_adapter:
            workflow["3"] = {"class_type": "IPAdapterFluxLoader",
                             "inputs": {"ipadapter": "instantx_flux1_dev_ip_adapter_bf16.safetensors"}}
            workflow["5"] = {"class_type": "ApplyIPAdapterFlux",
                             "inputs": {"model": ["1", 0], "ipadapter_flux": ["3", 0],
                                        "image": ["2", 0], "weight": 0.8}}
        else:
            workflow["3"] = {"class_type": "IPAdapterModelLoader",
                             "inputs": {"ipadapter_file": "ip-adapter-plus_sd15.safetensors"}}
            workflow["5"] = {"class_type": "IPAdapterAdvanced",
                             "inputs": {"model": ["1", 0], "ipadapter": ["3", 0],
                                        "image": ["2", 0], "weight": 0.8}}
        workflow["10"]["inputs"]["model"] = ["5", 0]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=2)


def run_doctor(workdir):
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "doctor.py"),
         "--url", f"http://127.0.0.1:{PORT}"],
        capture_output=True, text=True, cwd=workdir, timeout=180)
    return (result.stdout or "") + (result.stderr or "")


def with_fake(profile, body):
    process = subprocess.Popen(
        [sys.executable, FAKE, "--port", str(PORT), "--profile", profile],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_port():
            FAILURES.append(f"fake ComfyUI [{profile}] never came up")
            return ""
        return body()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def setup_workdir(**overrides):
    workdir = tempfile.mkdtemp(prefix="doctor_")
    import config as cfgmod
    cfg = cfgmod.ToolchainConfig()
    cfg.comfyui_url = f"http://127.0.0.1:{PORT}"
    cfg.mock_mode = False
    cfg.storyboard_path = os.path.join(workdir, "storyboard.xlsx")
    cfg.images_dir = os.path.join(workdir, "images")
    cfg.node_mapping.positive_prompt_node = "6"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfgmod.save_config(cfg, os.path.join(workdir, "toolchain_config.json"))
    return workdir


def main():
    print("\n[1] ComfyUI absent")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"))
    _patch_workflow_path(workdir, "wf.json")
    output = run_doctor(workdir)
    check("Not answering" in output, "an absent ComfyUI is the blocking finding")
    check("--listen 0.0.0.0" in output, "the fix mentions remote listening")
    check("Skipped" in output, "downstream checks are skipped rather than noisy")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[2] ComfyUI up, IP-Adapter never installed")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"), with_ipadapter=False)
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("no_ipadapter", lambda: run_doctor(workdir))
    check("No IP-Adapter nodes are installed at all" in output,
          "a total absence of IP-Adapter is reported")
    check("ComfyUI-IPAdapter-Flux" in output and "ComfyUI_IPAdapter_plus" in output,
          "the fix gives both repos, FLUX first")
    check("drift" in output.lower(), "it says why this matters, not just that it's missing")
    check("cannot lock a character's identity" in output,
          "the workflow is also reported as unable to hold a face")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[3] Nodes installed but no IP-Adapter model file")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"), flux_adapter=False)
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("no_model", lambda: run_doctor(workdir))
    check("no model file" in output.lower(), "the missing model file is called out")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[3b] SD/SDXL adapter present but FLUX one missing -- the real trap")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"), flux_adapter=False)
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("sd_only", lambda: run_doctor(workdir))
    check("FLUX needs a different one" in output,
          "having only the SD/SDXL adapter is reported as blocking, not fine")
    check("ComfyUI-IPAdapter-Flux" in output, "the fix names the FLUX adapter repo")
    check("ipadapter-flux" in output, "the fix names the model folder")
    check("easy to miss" in output, "it warns this looks correct while not working")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[3c] FLUX adapter node present but its model never downloaded")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"))
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("no_flux_adapter_model", lambda: run_doctor(workdir))
    check("no model file" in output.lower(), "the missing FLUX adapter model is caught")
    check("instantx_flux1_dev_ip_adapter" in output, "the fix names the exact file")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[3d] FLUX adapter model present but in the unreadable container")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"))
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("flux_adapter_wrong_format", lambda: run_doctor(workdir))
    check("cannot read" in output.lower(),
          "a safetensors-only FLUX adapter is reported as blocking, not fine")
    check("ip-adapter.bin" in output, "the fix names the file to download")
    check("torch.load" in output, "it says why safetensors will not work")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[4] No FLUX checkpoint")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"))
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("no_flux", lambda: run_doctor(workdir))
    check("No FLUX checkpoint" in output, "a missing FLUX checkpoint is noticed")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[5] A UI-format workflow instead of API format")
    workdir = setup_workdir()
    with open(os.path.join(workdir, "wf.json"), "w", encoding="utf-8") as handle:
        json.dump({"nodes": [], "links": [], "version": 0.4}, handle)
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("complete", lambda: run_doctor(workdir))
    check("UI export, not API format" in output, "a UI export is detected")
    check("Enable Dev mode options" in output, "the fix explains how to re-export")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[6] Node mapping points at a node that no longer exists")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"))
    _patch_workflow_path(workdir, "wf.json", positive_node="999")
    output = with_fake("complete", lambda: run_doctor(workdir))
    check("isn't in this workflow" in output, "a stale node mapping is caught")
    check("Node IDs change when a workflow is re-exported" in output,
          "the fix explains why it went stale")
    shutil.rmtree(workdir, ignore_errors=True)

    print("\n[7] A fully working setup reports nothing blocking")
    workdir = setup_workdir()
    full_workflow(os.path.join(workdir, "wf.json"), denoise=0.6)
    _patch_workflow_path(workdir, "wf.json")
    output = with_fake("complete", lambda: run_doctor(workdir))
    check("0 blocking" in output, "a complete setup has no blockers")
    check("Nothing is blocking a render" in output or "Everything checks out" in output,
          "it says so plainly")
    check("denoise" not in output.lower(), "denoise 0.6 is not flagged")
    shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nfailures: {len(FAILURES)}")
    for failure in FAILURES:
        print("   ", failure)
    print("RESULT:", "PASS" if not FAILURES else "FAIL")
    return 0 if not FAILURES else 1


def _patch_workflow_path(workdir, filename, positive_node="6"):
    path = os.path.join(workdir, "toolchain_config.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["workflow_json_path"] = os.path.join(workdir, filename)
    data["node_mapping"]["positive_prompt_node"] = positive_node
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
