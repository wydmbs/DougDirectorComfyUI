"""
doctor.py -- one command that tells you why the harness won't render yet.

Run this on the GPU machine, or on the laptop with the ComfyUI URL pointed at
it. It changes nothing: every check is a read, so it is safe to run at any time,
including mid-project.

    python doctor.py
    python doctor.py --url http://192.168.1.206:8188
    python doctor.py --verbose

The output is ordered by dependency, because the first real failure usually
explains every one after it -- there is no point reporting a missing IP-Adapter
node when ComfyUI itself isn't answering. Each finding carries the specific
thing to do about it, and the run ends with the single next action.
"""

import argparse
import json
import os
import sys

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

MARKS = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL ", SKIP: " --   "}

# What the keyframe stage needs present in ComfyUI.
#
# FLUX and SD/SDXL use different IP-Adapter implementations, and having the
# SD/SDXL ones installed does NOT mean a FLUX render can be conditioned. Both
# families are checked separately so "IP-Adapter is installed" can never be
# reported when the wrong one is present.
CORE_NODES = ("LoadImage", "VAEEncode")
IPADAPTER_FLUX_NODES = ("ApplyIPAdapterFlux", "IPAdapterFluxLoader")
IPADAPTER_SD_NODES = ("IPAdapterAdvanced", "IPAdapterModelLoader")
VIDEO_OUTPUT_NODES = ("VHS_VideoCombine", "SaveAnimatedWEBP", "SaveWEBM",
                      "SaveVideo", "SaveAnimatedPNG")


class Report:
    def __init__(self, verbose=False):
        self.rows = []
        self.verbose = verbose

    def add(self, section, status, title, detail="", fix=""):
        self.rows.append((section, status, title, detail, fix))
        return status

    def blocked(self) -> bool:
        return any(r[1] == FAIL for r in self.rows)

    def render(self) -> int:
        section = None
        for sect, status, title, detail, fix in self.rows:
            if sect != section:
                print(f"\n{sect}")
                section = sect
            print(f"  [{MARKS[status]}] {title}")
            if detail and (self.verbose or status in (FAIL, WARN)):
                for line in str(detail).splitlines():
                    print(f"            {line}")
            if fix and status in (FAIL, WARN):
                for line in fix.splitlines():
                    print(f"            -> {line}")

        fails = [r for r in self.rows if r[1] == FAIL]
        warns = [r for r in self.rows if r[1] == WARN]
        print("\n" + "-" * 68)
        print(f"{len(fails)} blocking, {len(warns)} worth fixing, "
              f"{sum(1 for r in self.rows if r[1] == OK)} fine")

        if fails:
            first = fails[0]
            print("\nDO THIS NEXT")
            print(f"  {first[2]}")
            for line in (first[4] or "See above.").splitlines():
                print(f"    {line}")
            return 1
        if warns:
            print("\nNothing is blocking a render. Worth tidying:")
            for warn in warns[:3]:
                print(f"  - {warn[2]}")
            return 0
        print("\nEverything checks out. Go and stage a keyframe.")
        return 0


# ------------------------------------------------------------------ checks


def check_python(report):
    section = "Python"
    report.add(section, OK, f"Python {sys.version.split()[0]}")
    missing = []
    for module, why in (("gradio", "the app shell"), ("openpyxl", "the registry"),
                        ("PIL", "image handling")):
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} ({why})")
    if missing:
        report.add(section, FAIL, "Required packages are missing",
                   ", ".join(missing),
                   "pip install -r requirements.txt\n"
                   "If you use conda, run it with the full interpreter path so it lands\n"
                   "in the same environment the app runs from.")
    else:
        report.add(section, OK, "Core packages are installed")

    for module, why in (("docx", "reading .docx sources"), ("pypdf", "reading PDFs")):
        try:
            __import__(module)
        except ImportError:
            report.add(section, WARN, f"{module} is missing", why,
                       f"pip install {'python-docx' if module == 'docx' else module}")


def check_config(report, cfg):
    section = "Configuration"
    report.add(section, OK, f"Registry: {cfg.storyboard_path}")
    report.add(section, OK, f"Images:   {cfg.images_dir}")

    if cfg.mock_mode:
        report.add(section, WARN, "Mock mode is ON -- nothing renders for real",
                   "Every generation returns a placeholder image.",
                   "Turn it off in Setup once ComfyUI is up. Leave it on if you're\n"
                   "only exercising the flow.")
    else:
        report.add(section, OK, "Mock mode is off -- renders go to ComfyUI")

    folder = os.path.dirname(os.path.abspath(cfg.storyboard_path)) or "."
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, ".doctor_write_probe")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("x")
        os.unlink(probe)
        report.add(section, OK, "The registry folder is writable")
    except OSError as error:
        report.add(section, FAIL, "The registry folder is not writable", str(error),
                   f"Check permissions on {folder}. If it's a synced OneDrive folder,\n"
                   f"make sure the files are available offline rather than cloud-only.")


def check_harry_provider(report, cfg):
    section = "Harry's model"
    try:
        import harry_advisor
        status = harry_advisor.provider_status(cfg.harry_provider)
    except Exception as error:  # noqa: BLE001
        report.add(section, WARN, "Could not read the provider status", str(error))
        return
    ready = "ready" in status.lower() or "locally" in status.lower()
    report.add(section, OK if ready else WARN,
               f"{cfg.harry_provider}: {status}",
               fix="" if ready else
               "Set the API key in this app's environment and restart it.\n"
               "Harry can't run without a model, though rendering still works.")


def check_comfy(report, cfg):
    """Everything downstream depends on this, so a failure here skips the rest."""
    section = "ComfyUI"
    from comfy_client import ComfyClient, ComfyClientError

    client = ComfyClient(cfg.comfyui_url)
    if not client.ping():
        report.add(section, FAIL, f"Not answering at {cfg.comfyui_url}", "",
                   "Start ComfyUI on the GPU machine.\n"
                   "If it's on another box, launch it with --listen 0.0.0.0 so it\n"
                   "accepts connections, and set that machine's URL in Setup.\n"
                   "Check a firewall isn't blocking port 8188.")
        return None, set()

    report.add(section, OK, f"Answering at {cfg.comfyui_url}")

    try:
        stats = client.system_stats()
        devices = stats.get("devices") or []
        if devices:
            device = devices[0]
            total = device.get("vram_total") or 0
            free = device.get("vram_free") or 0
            gb = total / (1024 ** 3) if total else 0
            free_gb = free / (1024 ** 3) if free else 0
            name = device.get("name", "unknown GPU")
            if gb and gb < 15:
                report.add(section, WARN, f"{name} -- {gb:.0f}GB VRAM",
                           f"{free_gb:.0f}GB free right now.",
                           "FLUX.1 Dev at FP8 wants ~12-16GB and LTX-2.5 ~14GB.\n"
                           "It may still work, but expect to close other GPU apps.")
            else:
                report.add(section, OK, f"{name} -- {gb:.0f}GB VRAM, {free_gb:.0f}GB free")
    except ComfyClientError as error:
        report.add(section, WARN, "Could not read GPU stats", str(error))

    try:
        classes = set(client.object_info().keys())
        report.add(section, OK, f"{len(classes)} node classes installed")
    except ComfyClientError as error:
        report.add(section, FAIL, "Could not list installed nodes", str(error),
                   "ComfyUI answered but /object_info failed. Check its console for errors.")
        return client, set()

    missing_core = [n for n in CORE_NODES if n not in classes]
    if missing_core:
        report.add(section, FAIL, "Core nodes are missing", ", ".join(missing_core),
                   "This ComfyUI looks incomplete. Reinstall or repair it.")

    has_flux = all(n in classes for n in IPADAPTER_FLUX_NODES)
    has_sd = all(n in classes for n in IPADAPTER_SD_NODES)

    if has_flux:
        report.add(section, OK, "FLUX IP-Adapter nodes are installed (ApplyIPAdapterFlux)")
    elif has_sd:
        report.add(section, FAIL,
                   "Only the SD/SDXL IP-Adapter is installed -- FLUX needs a different one",
                   "Found IPAdapterAdvanced, but not ApplyIPAdapterFlux.",
                   "The SD/SDXL adapter cannot condition a FLUX render. Install:\n"
                   "  cd ComfyUI/custom_nodes\n"
                   "  git clone https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux\n"
                   "and put instantx_flux1_dev_ip_adapter_bf16.safetensors in\n"
                   "ComfyUI/models/ipadapter-flux, then restart ComfyUI.\n"
                   "This is easy to miss: a workflow can look correct and still not hold a face.")
    else:
        report.add(section, FAIL, "No IP-Adapter nodes are installed at all", "",
                   "For FLUX:\n"
                   "  git clone https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux\n"
                   "For SD/SDXL:\n"
                   "  git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus\n"
                   "into ComfyUI/custom_nodes, then restart ComfyUI.\n"
                   "Without one, characters drift between every single shot.")

    if not any(n in classes for n in VIDEO_OUTPUT_NODES):
        report.add(section, WARN, "No video output node found",
                   "Looked for: " + ", ".join(VIDEO_OUTPUT_NODES),
                   "Install ComfyUI-VideoHelperSuite when you get to the clip stage.\n"
                   "Not needed for keyframes.")
    else:
        report.add(section, OK, "A video output node is available")

    return client, classes


def check_models(report, client, classes):
    section = "Models"
    if client is None or not classes:
        report.add(section, SKIP, "Skipped -- ComfyUI wasn't reachable")
        return
    try:
        info = client.object_info()
    except Exception as error:  # noqa: BLE001
        report.add(section, WARN, "Could not read installed models", str(error))
        return

    def options(node_name, field):
        node = info.get(node_name) or {}
        required = (node.get("input") or {}).get("required") or {}
        value = required.get(field)
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        return []

    checkpoints = options("CheckpointLoaderSimple", "ckpt_name") + options("UNETLoader", "unet_name")
    flux = [c for c in checkpoints if "flux" in str(c).lower()]
    if flux:
        report.add(section, OK, f"FLUX checkpoint found: {flux[0]}")
    elif checkpoints:
        report.add(section, WARN, "No FLUX checkpoint found",
                   f"{len(checkpoints)} other checkpoints are installed.",
                   "Download FLUX.1 Dev (FP8 fits a 4090 comfortably) into\n"
                   "ComfyUI/models/checkpoints or models/unet.")
    else:
        report.add(section, FAIL, "No checkpoints installed at all", "",
                   "Put FLUX.1 Dev FP8 in ComfyUI/models/checkpoints and restart ComfyUI.")

    flux_adapters = options("IPAdapterFluxLoader", "ipadapter")
    sd_adapters = options("IPAdapterModelLoader", "ipadapter_file")
    if flux_adapters:
        report.add(section, OK, f"FLUX IP-Adapter model found: {flux_adapters[0]}")
    elif "IPAdapterFluxLoader" in classes:
        report.add(section, FAIL,
                   "The FLUX IP-Adapter node is installed but has no model file", "",
                   "Download instantx_flux1_dev_ip_adapter_bf16.safetensors into\n"
                   "ComfyUI/models/ipadapter-flux. Without it the node loads and still\n"
                   "cannot hold a character's face.")
    elif sd_adapters:
        report.add(section, WARN, f"Only an SD/SDXL adapter model is present: {sd_adapters[0]}",
                   "", "Fine for SD/SDXL work; FLUX needs its own adapter model.")
    elif "IPAdapterModelLoader" in classes:
        report.add(section, WARN,
                   "The SD/SDXL IP-Adapter node is installed but has no model file", "",
                   "Put an IP-Adapter model in ComfyUI/models/ipadapter, or install the\n"
                   "FLUX adapter instead if this machine is for FLUX work.")

    vision = options("CLIPVisionLoader", "clip_name")
    if vision:
        report.add(section, OK, f"CLIP vision model found: {vision[0]}")
    elif "CLIPVisionLoader" in classes:
        report.add(section, WARN, "No CLIP vision model installed",
                   "IP-Adapter needs one to read the turnaround sheet.",
                   "Download a CLIP-ViT vision model into ComfyUI/models/clip_vision.")


def check_workflow(report, cfg, classes):
    section = "Keyframe workflow"
    path = cfg.workflow_json_path
    if not path:
        report.add(section, FAIL, "No workflow is configured", "",
                   "Build a FLUX + IP-Adapter graph in ComfyUI, then:\n"
                   "  enable Settings > Enable Dev mode options\n"
                   "  use Workflow > Save (API Format)\n"
                   "and point Setup at that file. See SETUP.md for the graph shape.")
        return
    if not os.path.exists(path):
        report.add(section, FAIL, f"Workflow file is missing: {path}", "",
                   "Re-export it, or fix the path in Setup.")
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        report.add(section, FAIL, "Workflow file could not be read", str(error),
                   "Re-export it with Save (API Format).")
        return

    if "nodes" in workflow or "links" in workflow:
        report.add(section, FAIL, "This is a UI export, not API format", "",
                   "In ComfyUI: Settings > Enable Dev mode options, then\n"
                   "Workflow > Save (API Format). The two files look similar but only\n"
                   "the API one can be submitted programmatically.")
        return

    report.add(section, OK, f"{os.path.basename(path)} is API format ({len(workflow)} nodes)")

    if classes:
        needed = {n.get("class_type") for n in workflow.values()
                  if isinstance(n, dict) and n.get("class_type")}
        missing = sorted(needed - classes)
        if missing:
            report.add(section, FAIL, "The workflow uses nodes this ComfyUI doesn't have",
                       ", ".join(missing),
                       "Install the custom nodes that provide them, or rebuild the\n"
                       "workflow using what's installed.")
        else:
            report.add(section, OK, "Every node the workflow uses is installed")

    from reference_conditioning import inspect_workflow
    found = inspect_workflow(workflow)
    if found["can_lock_identity"]:
        family = found.get("ipadapter_family") or "?"
        report.add(section, OK,
                   f"It can lock a character's identity ({family.upper()} IP-Adapter + LoadImage)")
        if family == "sd" and classes and "ApplyIPAdapterFlux" in classes:
            report.add(section, WARN,
                       "The workflow uses the SD/SDXL adapter, but a FLUX one is available",
                       "",
                       "If this workflow drives FLUX, swap to ApplyIPAdapterFlux --\n"
                       "the SD/SDXL adapter cannot condition a FLUX model.")
    else:
        report.add(section, FAIL, "It cannot lock a character's identity",
                   f"IP-Adapter node: {found['has_ipadapter']}, "
                   f"LoadImage nodes: {found['load_image_nodes']}",
                   "Add an ApplyIPAdapterFlux node (for FLUX) fed by a LoadImage, wired\n"
                   "between the model loader and the sampler. Without it, every panel of\n"
                   "a character sheet will be a different pig.")

    if found["can_anchor_scene"]:
        report.add(section, OK, "It can anchor a scene (VAEEncode present)")
    else:
        report.add(section, WARN, "It cannot anchor a scene to a concept",
                   "No VAEEncode node.",
                   "Optional. Add LoadImage -> VAEEncode -> the sampler's latent_image\n"
                   "if you want backdrops to hold their look.")

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        denoise = (node.get("inputs") or {}).get("denoise")
        if isinstance(denoise, (int, float)) and denoise >= 0.95 and found["can_anchor_scene"]:
            report.add(section, WARN, f"Sampler denoise is {denoise:g}",
                       "At 1.0 the sampler discards the scene reference entirely.",
                       "Drop it to about 0.5-0.7 so the background actually holds.\n"
                       "Leave it at 1.0 if you aren't using scene anchoring.")
            break

    mapping = cfg.node_mapping
    if not mapping.positive_prompt_node:
        report.add(section, FAIL, "No positive-prompt node is mapped", "",
                   "In Setup, set which node ID carries the positive prompt.\n"
                   "Open the API JSON and find your CLIPTextEncode node's key.")
    elif mapping.positive_prompt_node not in workflow:
        report.add(section, FAIL,
                   f"Mapped prompt node '{mapping.positive_prompt_node}' isn't in this workflow",
                   f"Available: {', '.join(list(workflow)[:12])}",
                   "Re-map it in Setup. Node IDs change when a workflow is re-exported.")
    else:
        report.add(section, OK, f"Prompt node '{mapping.positive_prompt_node}' resolves")

    if mapping.seed_node and mapping.seed_node not in workflow:
        report.add(section, WARN, f"Mapped seed node '{mapping.seed_node}' isn't in this workflow",
                   "", "Re-map it in Setup, or clear it to let the workflow's own seed stand.")


def check_video(report, cfg, classes):
    section = "Video workflows"
    from video_client import workflow_path_for
    import model_pipeline as pipeline

    any_set = False
    for key in (pipeline.CLIP_LTX.key, pipeline.CLIP_MINIMAX.key, pipeline.CLIP_RUNWAY.key):
        spec = pipeline.MODELS[key]
        path = workflow_path_for(cfg, key)
        if not path:
            report.add(section, SKIP, f"{spec.name}: not configured yet")
            continue
        any_set = True
        if not os.path.exists(path):
            report.add(section, WARN, f"{spec.name}: file missing", path,
                       "Re-export it, or clear the path in Setup.")
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                work = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            report.add(section, WARN, f"{spec.name}: unreadable", str(error))
            continue
        if "nodes" in work:
            report.add(section, WARN, f"{spec.name}: UI export, not API format", "",
                       "Re-export with Save (API Format).")
            continue
        if classes:
            missing = sorted({n.get("class_type") for n in work.values()
                              if isinstance(n, dict) and n.get("class_type")} - classes)
            if missing:
                report.add(section, WARN, f"{spec.name}: missing nodes", ", ".join(missing),
                           "Install the custom nodes it needs.")
                continue
        report.add(section, OK, f"{spec.name}: ready")

    if not any_set:
        report.add(section, SKIP, "None configured -- that's fine until you reach the clip stage")


def check_registry(report, cfg):
    section = "Registry"
    try:
        import storyboard_store as store
        rows = store.list_entries(cfg.storyboard_path)
    except Exception as error:  # noqa: BLE001
        report.add(section, FAIL, "The registry could not be read", str(error),
                   "If it's corrupt, restore a backup:\n"
                   "  python -c \"import registry_safety as r; "
                   "print(r.restore_latest_backup('storyboard.xlsx'))\"")
        return

    if not rows:
        report.add(section, OK, "Empty -- nothing locked yet")
        return

    report.add(section, OK, f"{len(rows)} entries")
    drafted = [r for r in rows if r.get("image_path")]
    missing_files = [r["entry_id"] for r in drafted
                     if r.get("image_path") and not os.path.exists(str(r["image_path"]))]
    if missing_files:
        report.add(section, WARN, f"{len(missing_files)} entries point at missing images",
                   ", ".join(missing_files[:5]),
                   "Image paths are absolute, so moving the project folder breaks them.\n"
                   "Re-lock those entries, or move the folder back.")

    unreviewed = [r["entry_id"] for r in rows
                  if "unreviewed" in str(r.get("notes") or "").lower()]
    if unreviewed:
        report.add(section, WARN, f"{len(unreviewed)} entries were locked unseen",
                   ", ".join(unreviewed[:5]),
                   "Harry has no vision critic, so nobody looked at these.\n"
                   "Worth reviewing them in the Registry tab before building on them.")


def main():
    # Windows consoles default to cp1252; force UTF-8 where we can so
    # the report is readable rather than full of replacement characters.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Check why the harness won't render yet.")
    parser.add_argument("--url", help="Override the ComfyUI URL for this check")
    parser.add_argument("--verbose", action="store_true", help="Show detail for passing checks too")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfgmod

    cfg = cfgmod.load_config()
    if args.url:
        cfg.comfyui_url = args.url

    print("=" * 68)
    print("  ComfyUI Director Harness -- preflight")
    print("=" * 68)

    report = Report(verbose=args.verbose)
    check_python(report)
    check_config(report, cfg)
    check_harry_provider(report, cfg)
    client, classes = check_comfy(report, cfg)
    check_models(report, client, classes)
    check_workflow(report, cfg, classes)
    check_video(report, cfg, classes)
    check_registry(report, cfg)
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
