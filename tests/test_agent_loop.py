"""End-to-end exercise of the assistant director with no API key and no GPU.

This drives the real loop, the real tool registry, the real permission policy
and the real Excel registry. Only the model and the image generator are
substituted. What it proves:

  * tools dispatch and their results reach the model
  * a lock written by the agent actually lands in the registry
  * the supervised posture stops an install until the director agrees
  * a refusal is reported to the model rather than crashing the run
  * a failing tool doesn't end the run
  * the step limit holds
  * the audit log records what happened
  * progress is resumable from registry state alone
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod
import director_engine as engine
import storyboard_store as store
from harry_agent import (SUPERVISED, AgentLoop, Decision, PermissionPolicy,
                         build_registry)
from harry_agent.audit import AuditLog
from harry_agent.providers.base import STOP_END_TURN, ToolCall, Turn
from harry_agent.providers.mock import MockProvider

FAILURES = []


def check(condition, label):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def make_cfg(workdir):
    cfg = cfgmod.ToolchainConfig()
    cfg.storyboard_path = os.path.join(workdir, "storyboard.xlsx")
    cfg.images_dir = os.path.join(workdir, "images")
    cfg.mock_mode = True
    cfg.agent.variants_per_generation = 3
    return cfg


def tool_turn(name, arguments, text=""):
    return Turn(text=text,
                tool_calls=[ToolCall(id=f"c{abs(hash((name, str(arguments)))) % 9999}",
                                     name=name, arguments=arguments)],
                stop_reason="tool_use")


def run_loop(cfg, script, posture=SUPERVISED, ask=None, max_steps=20, audit=None):
    registry = build_registry(cfg)
    loop = AgentLoop(
        provider=MockProvider(script),
        registry=registry,
        policy=PermissionPolicy(posture=posture),
        audit=audit,
        max_steps=max_steps,
        ask=ask,
    )
    return loop.run("test goal"), registry


def scenario_generate_and_lock(cfg):
    print("\n[1] Harry generates, picks a winner, and locks it")
    store.create_character(cfg.storyboard_path, "CHARACTER:pig", "The Pig", "lead")

    # A locked concept is the reference every panel is built against.
    os.makedirs(cfg.images_dir, exist_ok=True)
    concept_path = os.path.join(cfg.images_dir, "CHARACTER_pig__concept.png")
    engine._mock_image("CHARACTER:pig", "a pig", 1).save(concept_path)
    engine.lock_concept(cfg, "CHARACTER:pig", "CHARACTER", "opening", "the lead",
                        "a pig", "", 1, concept_path, note="concept")

    script = [
        tool_turn("project_overview", {}),
        tool_turn("sheet_status", {"entry_id": "CHARACTER:pig", "entry_type": "CHARACTER"}),
        tool_turn("generate_variants", {
            "entry_id": "CHARACTER:pig", "entry_type": "CHARACTER",
            "panel_key": "fullbody_front", "prompt_positive": "a pig, front view",
            "n_variants": 3, "base_seed": 100}),
        tool_turn("lock_panel", {
            "entry_id": "CHARACTER:pig", "panel_key": "fullbody_front",
            "image_path": os.path.join(cfg.images_dir, "CHARACTER_pig__fullbody_front__100.png"),
            "seed": 100, "prompt_positive": "a pig, front view",
            "note": "cleanest silhouette"}),
        Turn(text="Front panel locked.", stop_reason=STOP_END_TURN),
    ]
    result, _ = run_loop(cfg, script)

    check(result.completed, "run completed")
    check(result.steps == 5, f"took 5 steps (got {result.steps})")

    panels = store.get_panel_images(cfg.storyboard_path, "CHARACTER:pig")
    check("fullbody_front" in panels, "panel landed in the registry")

    status = engine.sheet_status(cfg, "CHARACTER:pig", "CHARACTER")
    check(len(status["locked_panels"]) == 1, "sheet_status sees exactly one locked panel")
    check(status["total_panels"] == 18, f"CHARACTER template is 18 panels (got {status['total_panels']})")

    tool_events = [e for e in result.events if e.kind == "tool"]
    check(len(tool_events) == 4, f"four tool calls recorded (got {len(tool_events)})")


def scenario_consent(cfg):
    print("\n[2] An install waits for the director")
    asked = {}

    def ask_yes(request):
        asked["request"] = request
        return Decision.ALLOW_ONCE

    def fake_shell(command, cwd=""):
        return "exit 0\ninstalled"

    registry = build_registry(cfg, run_command=fake_shell)
    loop = AgentLoop(
        provider=MockProvider([
            tool_turn("run_setup_command", {
                "command": "git clone https://github.com/x/comfyui-node",
                "reason": "the workflow needs this node"}),
            Turn(text="Node installed.", stop_reason=STOP_END_TURN),
        ]),
        registry=registry, policy=PermissionPolicy(posture=SUPERVISED), ask=ask_yes, max_steps=10)
    result = loop.run("install a node")

    check("request" in asked, "the director was asked before installing")
    check(asked.get("request") and asked["request"].tool_name == "run_setup_command",
          "the request named the install tool")
    check(result.completed, "run completed after approval")


def scenario_refusal(cfg):
    print("\n[3] A refusal is handled, not fatal")

    def ask_no(request):
        return Decision.DENY

    registry = build_registry(cfg, run_command=lambda c, w="": "should not run")
    loop = AgentLoop(
        provider=MockProvider([
            tool_turn("run_setup_command", {"command": "pip install something", "reason": "x"}),
            Turn(text="Understood, skipping that.", stop_reason=STOP_END_TURN),
        ]),
        registry=registry, policy=PermissionPolicy(posture=SUPERVISED), ask=ask_no, max_steps=10)
    result = loop.run("install a node")

    check(result.completed, "run continued after refusal")
    check(any(e.kind == "blocked" for e in result.events), "refusal surfaced as an event")


def scenario_dangerous_command(cfg):
    print("\n[4] A destructive command is refused even with approval")
    registry = build_registry(cfg, run_command=lambda c, w="": "SHOULD NEVER RUN")
    result = registry.run("run_setup_command", {
        "command": "Remove-Item -Recurse -Force C:\\", "reason": "cleanup"})
    check("REFUSED" in result.content, "the shell guard refused it")
    check("SHOULD NEVER RUN" not in result.content, "the command never reached the shell")


def scenario_tool_failure(cfg):
    print("\n[5] A failing tool doesn't end the run")
    script = [
        tool_turn("get_entry", {"entry_id": "CHARACTER:does_not_exist"}),
        tool_turn("sheet_status", {"entry_id": "nope", "entry_type": "NOT_A_TYPE"}),
        Turn(text="Recovered and stopped.", stop_reason=STOP_END_TURN),
    ]
    result, _ = run_loop(cfg, script)
    check(result.completed, "run completed despite a failing tool")
    check(result.steps == 3, f"all three steps ran (got {result.steps})")


def scenario_step_limit(cfg):
    print("\n[6] The step limit holds")
    script = [tool_turn("project_overview", {}) for _ in range(50)]
    result, _ = run_loop(cfg, script, max_steps=4)
    check(not result.completed, "run stopped short")
    check(result.steps == 4, f"stopped at the limit (got {result.steps})")
    check("limit" in (result.error or "").lower(), "the reason names the limit")


def scenario_resumability(cfg):
    print("\n[7] Progress is resumable from the registry alone")
    first = engine.next_unfinished_panel(cfg)
    check(first is not None, "an unfinished panel is identified")
    if first:
        check(first["entry_id"] == "CHARACTER:pig", "it points at the right entry")
        check(first["panel_key"] != "fullbody_front",
              "the already-locked panel is not offered again")
        remaining_before = first["remaining"]
        engine.lock_panel(cfg, "CHARACTER:pig", first["panel_key"], "p", "", 7, "x.png")
        after = engine.next_unfinished_panel(cfg)
        check(after and after["remaining"] == remaining_before - 1,
              "the remaining count drops after a lock, with no run state kept")


def scenario_audit(cfg, workdir):
    print("\n[8] The audit log records decisions")
    audit = AuditLog(os.path.join(workdir, "audit.sqlite3"))
    script = [
        tool_turn("project_overview", {}),
        Turn(text="Done.", stop_reason=STOP_END_TURN),
    ]
    result, _ = run_loop(cfg, script, audit=audit)
    events = audit.events(result.run_id)
    kinds = {e["kind"] for e in events}
    check("run_start" in kinds, "the run start was recorded")
    check("tool_call" in kinds, "the tool call was recorded")
    check("run_done" in kinds, "the run completion was recorded")


def main():
    workdir = tempfile.mkdtemp(prefix="harry_agent_")
    try:
        cfg = make_cfg(workdir)
        scenario_generate_and_lock(cfg)
        scenario_consent(cfg)
        scenario_refusal(cfg)
        scenario_dangerous_command(cfg)
        scenario_tool_failure(cfg)
        scenario_step_limit(cfg)
        scenario_resumability(cfg)
        scenario_audit(cfg, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nfailures: {len(FAILURES)}")
    for failure in FAILURES:
        print("   ", failure)
    print("RESULT:", "PASS" if not FAILURES else "FAIL")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
