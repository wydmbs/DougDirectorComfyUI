"""
comfy_tools.py -- goal 2: help the director get a workflow actually running.

Three jobs, in increasing order of risk:

  1. ask a live ComfyUI what it has installed        (read-only)
  2. compare a workflow against that and report gaps (read-only)
  3. install what's missing                          (always needs consent)

Step 3 runs real commands on the director's PC, so every proposed command goes
through the shell guard first and is shown before it runs. Nothing here installs
anything on its own initiative.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request

from ..safety import Verdict, analyze
from .registry import Tool

OBJECT_INFO_TIMEOUT = 30


def _get_json(url: str, timeout: int = OBJECT_INFO_TIMEOUT):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_comfy_tools(cfg, run_command=None) -> list:
    """run_command is injectable so tests never touch a real shell."""

    def _runner(command: str, cwd: str = "") -> str:
        if run_command is not None:
            return run_command(command, cwd)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=cwd or None, capture_output=True, text=True, timeout=1800,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return f"exit {completed.returncode}\n{output[:4000]}"

    def comfy_environment() -> str:
        """What this ComfyUI actually has: node classes and model files."""
        base = cfg.comfyui_url.rstrip("/")
        try:
            info = _get_json(f"{base}/object_info")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            return (f"Could not reach ComfyUI at {base}: {error}. "
                    f"Is it running? Mock mode is {'on' if cfg.mock_mode else 'off'}.")
        node_classes = sorted(info.keys())
        checkpoints, loras = [], []
        for key, holder in (("CheckpointLoaderSimple", "ckpt_name"), ("LoraLoader", "lora_name")):
            node = info.get(key) or {}
            required = (node.get("input") or {}).get("required") or {}
            options = required.get(holder)
            if isinstance(options, list) and options and isinstance(options[0], list):
                (checkpoints if key.startswith("Checkpoint") else loras).extend(options[0])
        return json.dumps({
            "reachable": True,
            "node_class_count": len(node_classes),
            "checkpoints": checkpoints[:60],
            "loras": loras[:60],
            "sample_node_classes": node_classes[:40],
        }, indent=2)

    def workflow_requirements(workflow_path: str = "") -> str:
        """What a workflow needs, and which of it is missing here."""
        path = workflow_path or cfg.workflow_json_path
        if not path or not os.path.exists(path):
            return f"No workflow file at '{path or '(none configured)'}'."
        try:
            with open(path, "r", encoding="utf-8") as handle:
                workflow = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            return f"Could not read the workflow: {error}"

        if not isinstance(workflow, dict) or "nodes" in workflow:
            return ("This looks like a UI-format workflow, not API format. In ComfyUI turn on "
                    "'Enable Dev mode options', then use 'Save (API Format)' and upload that file.")

        needed_classes, models = set(), set()
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type"):
                needed_classes.add(node["class_type"])
            for key, value in (node.get("inputs") or {}).items():
                if isinstance(value, str) and value.lower().endswith(
                        (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")):
                    models.add(value)

        report = {"workflow": os.path.basename(path),
                  "node_classes_required": sorted(needed_classes),
                  "model_files_referenced": sorted(models)}

        base = cfg.comfyui_url.rstrip("/")
        try:
            info = _get_json(f"{base}/object_info")
            available = set(info.keys())
            report["missing_node_classes"] = sorted(needed_classes - available)
            report["comfy_reachable"] = True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            report["comfy_reachable"] = False
            report["missing_node_classes"] = "unknown — ComfyUI wasn't reachable"
        return json.dumps(report, indent=2)

    def check_command(command: str) -> str:
        """Judge a command before proposing it. Always available, never runs anything."""
        result = analyze(command)
        return json.dumps({
            "verdict": result.verdict.value,
            "reason": result.reason,
            "will_run": result.verdict is not Verdict.BLOCKED,
        }, indent=2)

    def run_setup_command(command: str, reason: str, cwd: str = "") -> str:
        """Run one install/setup command. Refuses anything the guard blocks."""
        result = analyze(command)
        if result.verdict is Verdict.BLOCKED:
            return (f"REFUSED — {result.reason}. This command will not be run: {command}\n"
                    f"Propose a narrower command that targets a specific path.")
        try:
            return f"[{reason}]\n{_runner(command, cwd)}"
        except subprocess.TimeoutExpired:
            return f"'{command}' was still running after 30 minutes and was stopped."
        except OSError as error:
            return f"Could not run '{command}': {error}"

    def obj(**properties):
        return {"type": "object", "properties": properties,
                "required": [k for k, v in properties.items() if v.pop("_required", False)]}

    def s(desc, required=False):
        return {"type": "string", "description": desc, "_required": required}

    return [
        Tool("comfy_environment",
             "Ask the live ComfyUI what node classes, checkpoints and LoRAs it has installed.",
             obj(), comfy_environment),
        Tool("workflow_requirements",
             "Read a workflow and report the node classes and model files it needs, and which are missing here.",
             obj(workflow_path=s("Defaults to the configured workflow")), workflow_requirements),
        Tool("check_command",
             "Check whether a shell command would be permitted, without running it. Use before proposing one.",
             obj(command=s("The command to check", True)), check_command),
        Tool("run_setup_command",
             "Run one setup command on the director's PC — installing a custom node or downloading a model. "
             "Always needs the director's approval. Say plainly what it does and why.",
             obj(command=s("The exact command", True),
                 reason=s("Why this is needed", True),
                 cwd=s("Directory to run it in")),
             run_setup_command, mutating=True, always_confirm=True, timeout_s=1800),
    ]
