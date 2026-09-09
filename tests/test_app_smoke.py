"""Does the app actually build with Harry wired in?

Imports app.py with launch() stubbed out, which constructs every Gradio
component and every event binding. A typo in the wiring, a missing component,
or a mismatched output list all fail here rather than in front of the director.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Keep the smoke test out of the real project's files.
WORK = tempfile.mkdtemp(prefix="app_smoke_")
os.environ["TOOLCHAIN_CONFIG_PATH"] = os.path.join(WORK, "toolchain_config.json")
os.chdir(WORK)

import gradio as gr

_launched = {}
_real_launch = gr.Blocks.launch
gr.Blocks.launch = lambda self, *a, **k: _launched.setdefault("ok", True)

failures = []
try:
    sys.path.insert(0, ROOT)
    import app  # noqa: F401  -- importing is the test
    print("  PASS  app.py imported and built its interface")
except Exception as error:  # noqa: BLE001
    import traceback
    failures.append(f"app.py failed to build: {type(error).__name__}: {error}")
    traceback.print_exc()

if not failures:
    checks = [
        ("harry_rail_html", callable(getattr(app, "harry_rail_html", None))),
        ("harry_agent_start", callable(getattr(app, "harry_agent_start", None))),
        ("harry_agent_poll", callable(getattr(app, "harry_agent_poll", None))),
        ("harry_agent_stop", callable(getattr(app, "harry_agent_stop", None))),
        ("director_engine imported", getattr(app, "engine", None) is not None),
        ("harry_ui imported", getattr(app, "harry_ui", None) is not None),
        ("launch() was reached", _launched.get("ok") is True),
    ]
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    # The rail must render without a project, which is the first-run case.
    try:
        html = app.harry_rail_html()
        ok = "harry-rail" in html and len(html) > 200
        print(f"  {'PASS' if ok else 'FAIL'}  rail renders on an empty project")
        if not ok:
            failures.append("rail render")
    except Exception as error:  # noqa: BLE001
        print(f"  FAIL  rail raised: {error}")
        failures.append("rail raised")

    # Starting with no goal must warn, not crash or start a run.
    try:
        result = app.harry_agent_start("", "supervised", 10, "Claude")
        ok = isinstance(result, tuple) and "⚠️" in str(result[2])
        print(f"  {'PASS' if ok else 'FAIL'}  empty goal is refused politely")
        if not ok:
            failures.append("empty goal handling")
    except Exception as error:  # noqa: BLE001
        print(f"  FAIL  empty goal raised: {error}")
        failures.append("empty goal raised")

    # Saving Setup must not discard the settings the form doesn't ask about.
    # Azure credentials, video workflow paths and the active project are all
    # invisible on that form, which is exactly why losing them went unnoticed.
    try:
        import config as cfgmod
        before = cfgmod.ToolchainConfig()
        before.azure.endpoint = "https://example.openai.azure.com"
        before.video.ltx_workflow_path = "ltx.json"
        before.active_project_id = "pig_and_rooster"
        before.agent.max_steps = 77
        app.CFG = before

        app.setup_save("http://127.0.0.1:8188", None, "6", "text", "", "text",
                       "3", "seed", "storyboard.xlsx", True, "Claude",
                       "IP-Adapter — holds the character approximately")
        after = cfgmod.load_config(os.environ["TOOLCHAIN_CONFIG_PATH"])

        survives = [
            ("the Azure endpoint survives a Setup save",
             after.azure.endpoint == "https://example.openai.azure.com"),
            ("the video workflow paths survive", after.video.ltx_workflow_path == "ltx.json"),
            ("the active project survives", after.active_project_id == "pig_and_rooster"),
            ("the agent settings survive", after.agent.max_steps == 77),
            ("the conditioning mode the form owns is written",
             after.agent.reference_conditioning == "ipadapter"),
        ]
        app.CFG = before
        app.setup_save("http://127.0.0.1:8188", None, "6", "text", "", "text",
                       "3", "seed", "storyboard.xlsx", True, "Claude",
                       "Kontext — holds the character exactly (recommended)")
        survives.append((
            "the dropdown label round-trips back to the mode key",
            cfgmod.load_config(os.environ["TOOLCHAIN_CONFIG_PATH"])
            .agent.reference_conditioning == "kontext"))

        for label, ok in survives:
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
            if not ok:
                failures.append(label)
    except Exception as error:  # noqa: BLE001
        print(f"  FAIL  setup_save raised: {error}")
        failures.append("setup_save raised")

gr.Blocks.launch = _real_launch
print(f"\nfailures: {len(failures)}")
for failure in failures:
    print("   ", failure)
print("RESULT:", "PASS" if not failures else "FAIL")
raise SystemExit(0 if not failures else 1)
