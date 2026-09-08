# BUILD_LIST — Harry, the assistant director

Harry began as a one-shot analyser: read the source, propose a call sheet, stop.
This is the plan for the rest of the job — executing that call sheet — and a
record of what has already been built.

Status keys: **[done]** shipped in this branch · **[next]** ready to start ·
**[open]** needs a decision first.

---

## 0. Decisions

| | Decision | Status |
|---|---|---|
| D1 | **Image + video model chain** | **[done]** — see below |
| D2 | Registry storage | **[done]** keep `.xlsx`, add write safety |
| D3 | Where the agent runs | **[done]** worker thread inside the Gradio shell |
| D4 | Autonomy posture | **[done]** three postures; installs always ask |

## The model chain — baked in

```
DRAFT              KEYFRAME                    CLIP
ChatGPT/DALL-E 3   FLUX.1 Dev FP8 (local)      Minimax H3   performance / dialogue
concept sheets  →  IP-Adapter locks the    →   LTX-2.5      camera moves, cuts
turnarounds        face; img2img anchors       Runway Gen-4 physics, destruction
scene concepts     the scene
```

**D1 resolved: FLUX.1 Dev + IP-Adapter, not Nano Banana Pro.** Reference
conditioning now defaults to `ipadapter` because without it every character
quietly drifts between shots.

| Stage | Model | Where | Cost | Notes |
|---|---|---|---|---|
| Draft | ChatGPT / DALL-E 3 | manual | subscription | Attached via `attach_draft`; no API or seed control |
| Keyframe | FLUX.1 Dev FP8 | local | free, ~12–16GB | Needs `IPAdapterAdvanced` + `LoadImage` in the workflow |
| Clip A | Minimax H3 REF2VA | cloud | credits, ≤15s | Takes keyframe **and** turnaround — why a head turn holds |
| Clip B | LTX-2.5 | local | free, ~14GB | The workhorse: multi-shot cuts, tracking |
| Clip C | Runway Gen-4 Turbo | cloud | credits, ≤5s | Only for physics local models fumble |

**Routing** is automatic but advisory — `route_shot` returns a recommendation
*and its reasoning*, because a wrong call wastes either credits or a take.
Priority is physics → performance → camera: a melted face ruins a shot in a way
a duller camera move does not.

**Registry mapping** follows SUITE rule 1 — columns added to the same sheet, no
side database: `keyframe_path`, `clip_path`, `motion_prompt`, `video_model`,
`shot_type`. Existing registries migrate automatically (verified).

---

## 0. Decisions (original)

---

## 1. Foundations — **[done]**

1. **[done] Cut the OpenScout coupling.** Harry read another application's
   `config.yaml` off disk for Azure settings, so Azure silently failed on any
   machine without that app installed. The harness now owns `AzureConfig`.
   All 11 references gone, including two UI strings.
2. **[done] `director_engine.py`** — headless core. No Gradio import, no module
   globals, config passed as an argument. The Build buttons and Harry's tools
   call the *same* functions, so the agent cannot drift from what the UI does.
3. **[done] Registry write safety** (`registry_safety.py`) — cross-process lock,
   atomic replace, rolling backups. Measured: 8 threads × 6 writes lands 48/48
   with the guard, **5/48 with 42 errors without it**.
4. **[done] Offline test harness** — mock provider + mock ComfyUI, four suites.
5. **[next] One real ComfyUI generation, end to end.** Still the highest-value
   unproven step. `comfy_client.py` is real and complete; what's missing is an
   API-format workflow to feed it. Until this exists, everything downstream is
   theory.

---

## 2. Agent spine — **[done]**

6. **[done] Providers with tool-calling** (`harry_agent/providers/`). The old
   `_chat()` could only return prose. Anthropic is the internal format;
   Azure/OpenAI/Grok share an adapter; Ollama gets a JSON-decision fallback
   because local models vary in tool support.
7. **[done] Tool registry** (`tools/registry.py`) — dataclass contract, dispatch,
   argument filtering, failures returned to the model rather than raised.
8. **[done] Permissions** (`permissions.py`) — attended / supervised / unattended,
   plus `always_confirm` for anything touching the machine.
9. **[done] Shell guard** (`safety.py`) — 34 cases green, including base64-encoded
   PowerShell. Blocks recursive deletes at roots and kill-by-name outright.
10. **[done] Agent loop** (`loop.py`) — plan/act/observe, bounded steps, cancellable.
11. **[done] Audit log** (`audit.py`) + trust boundary (`trust.py`).

### The load-bearing design choice

**The registry is Harry's memory, not the conversation.**

He re-reads project state each step instead of accumulating a transcript. Three
consequences, all good:

- **Resumable** — progress lives in the workbook, so "what's left" is a query.
  Crash, cancel, or close the browser; he picks up exactly where he stopped.
- **Bounded context** — none of the history-compaction machinery a chat agent
  needs. That's ~1,200 lines of the hardest, buggiest code simply not written.
- **Collaborative** — lock a panel by hand mid-run and Harry just sees it.

---

## 3. Domain tools — **[done]** *(goals 1 & 3)*

12. **[done]** Registry tools: overview, list, get, sheet status, panel template,
    beats, characters.
13. **[done]** `generate_variants` with the reference seam wired through.
14. **[done]** `lock_concept`, `lock_panel`, `render_sheet`, `print_sheet`,
    `register_character` — all `[HARRY]`-tagged in the registry notes.
15. **[open] `critique_variants` is a stub.** The tool and its call path exist,
    but with no critic configured it returns "pick on the evidence you have".
    **This is the gap between goals 1 and 3.** Now unblocked by D1: a real critic
    is a vision model scoring variants against the locked draft. Next up.

---

## 3b. The pipeline — **[done]** *(the model chain)*

| | | |
|---|---|---|
| **[done]** | `model_pipeline.py` | stages, model specs, shot routing, stage detection |
| **[done]** | `reference_conditioning.py` | real IP-Adapter + img2img injection, rewritten for FLUX |
| **[done]** | `video_client.py` | Minimax H3 / LTX-2.5 / Runway behind one interface |
| **[done]** | registry columns | keyframe, clip, motion prompt, model, shot type |
| **[done]** | `pipeline_tools.py` | 9 tools: map, stage, attach draft, route, plan, keyframe, lock, animate, check setup |

Harry's system prompt now carries the pipeline, so he knows a clip is grown from
a keyframe and a keyframe is anchored to a draft — and the tools enforce it
(`animate_shot` refuses a shot with no keyframe, verified).

**[next]** The cloud clients are written against each vendor's documented
submit-and-poll shape but have **never been run against the live services**. The
LTX path needs an exported API-format workflow. Both are structurally sound and
factually unproven — treat first contact as a debugging session, not a formality.

---

## 4. ComfyUI build assist — **[done]** *(goal 2)*

16. **[done]** `comfy_environment` — asks a live ComfyUI what nodes/checkpoints/
    LoRAs it has, via `/object_info`.
17. **[done]** `workflow_requirements` — reads a workflow, reports required node
    classes and model files, diffs against what's installed. Detects a UI-format
    workflow and explains how to export API format.
18. **[done]** `check_command` / `run_setup_command` — guarded install path.
19. **[next]** Websocket progress, replacing the 2-second poll.
20. **[next]** Versioned workflow library.

---

## 5. End-to-end production — **[next]** *(goal 4)*

21. **[done]** Threaded runner with cancellation and a consent queue.
22. **[next]** Batch/overnight queue against TheBeast.
23. **[done]** Resume-after-crash — falls out of the registry-as-memory design.

---

## 6. Optional

MCP exposure (drive the harness from Scout/Copilot without opening Gradio) ·
per-asset subagents · project-bible memory · video module.

---

## UI: where Harry lives

**Recommendation, implemented: in-context, not a separate menu.**

A second "Harry Agent" tab would split him in two — the Harry who reads the
script over there, the Harry who does the work over here. The tab strip already
*is* the pipeline (Welcome → Setup → Harry → Build → Registry), so Harry rides
along it instead:

- **The rail** (`harry_ui.py`) — a persistent strip on Build and Registry:
  mood, one honest status line, a panel-progress meter, per-sheet chips, and
  what's next. Driven by the same `project_overview()` his tools call, so the
  rail and the assistant can never disagree.
- **"On set with Harry"** — a collapsible panel inside Build: goal, posture,
  step budget, **Action!** / **Cut**, a live feed, and an inline consent card
  showing the exact command before it runs.
- **His own tab stays** as pre-production — the read-through.

Feedback follows the house rule: one entrance animation on a real state change,
never anything looping. Manual locks move the rail too — it reflects the
production, not just Harry's work.

---

## Verification

| Suite | Covers | Result |
|---|---|---|
| `test_registry_concurrency` | 8 threads × 6 writes | 48/48, 0 errors |
| `test_shell_safety` | 34 command shapes | 0 failures |
| `test_agent_loop` | 26 checks, 8 scenarios | 0 failures |
| `test_pipeline` | 51 checks, 7 scenarios | 0 failures |
| `test_app_smoke` | app builds + wiring | 0 failures |

Run: `python tests\test_pipeline.py` (each is standalone).

**What is *not* verified:** no live LLM call, no live ComfyUI, no FLUX render, no
Minimax / LTX / Runway call, no browser session. The handover's open item 1 —
that no Harry prompt change has been validated against a real model — is still
true, and now applies to the agent loop and the whole model chain too.

The three things that convert this from sound to proven, in order:
1. One real FLUX keyframe with IP-Adapter holding a face.
2. One real clip from each of the three video routes.
3. A real Harry run on the corrected poem.
