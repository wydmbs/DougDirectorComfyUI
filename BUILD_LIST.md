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
DRAFT (manual)          KEYFRAME + CLIP (GPU machine, all via ComfyUI)
ChatGPT / DALL-E 3      FLUX.1 Dev      IP-Adapter locks the face
  app writes prompt  →  Minimax H3      dialogue / performance
  you paste + save   →  LTX-2.5         camera moves, cuts
  attach the image   →  Runway Gen-4    physics, destruction
```

**D1 resolved: FLUX.1 Dev + IP-Adapter, not Nano Banana Pro.** Reference
conditioning defaults to `ipadapter` because without it every character quietly
drifts between shots.

**Drafting is a manual handoff.** ChatGPT has no API here and no seed control,
so `draft_prompts.py` writes an opinionated prompt (plain grey background, flat
studio lighting, four angles on one sheet) and the director carries the image
back. Automating it would add a dependency and a failure mode for nothing.

**Everything after the draft runs on the GPU machine, through ComfyUI.** LTX-2.5
is local weights; Minimax H3 and Runway Gen-4 arrive through ComfyUI's API
nodes. So all three are workflows rather than three REST clients: one connection
to configure, credentials living in ComfyUI where they belong, and the render
happening where the GPU is. Each route needs its own API-format export.

| Stage | Model | Where | Cost | Notes |
|---|---|---|---|---|
| Draft | ChatGPT / DALL-E 3 | manual | subscription | Prompt out, image back via `attach_draft` |
| Keyframe | FLUX.1 Dev FP8 | GPU box | free, ~12–16GB | Needs `IPAdapterAdvanced` + `LoadImage` |
| Clip A | Minimax H3 REF2VA | GPU box → cloud | credits, ≤15s | Takes keyframe **and** turnaround |
| Clip B | LTX-2.5 | GPU box | free, ~14GB | The workhorse |
| Clip C | Runway Gen-4 Turbo | GPU box → cloud | credits, ≤5s | Physics only |

**Remote is the default assumption.** Images are uploaded to ComfyUI and
referenced by the returned name, so the app can run on a laptop while rendering
happens on the 4090. This fixed a real bug: a local absolute path handed to a
remote ComfyUI loads nothing, the render reports success, and the reference is
silently ignored — characters drift while the logs look clean.

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
| `test_shell_safety` | 44 command shapes | 0 failures |
| `test_agent_loop` | 26 checks, 8 scenarios | 0 failures |
| `test_pipeline` | 51 checks, 7 scenarios | 0 failures |
| `test_remote_gpu` | 40 checks, 6 scenarios | 0 failures |
| `test_review_fixes` | 32 checks, 10 scenarios | 0 failures |
| `test_doctor` | 18 checks, 7 machine states | 0 failures |
| `test_app_smoke` | app builds + wiring | 0 failures |

Run: `python tests\test_doctor.py` (each is standalone).

`tests\fake_comfy.py` is a stand-in ComfyUI with profiles for the half-configured
states that matter — `complete`, `no_ipadapter`, `no_model`, `no_flux`. An absent
ComfyUI is easy to detect; the dangerous states are the in-between ones, and
those are what it exists to reproduce.

---

## What a design review found

Worth recording, because several of these were invisible to a green test suite.

**Fixed:**

| Severity | Defect |
|---|---|
| Critical | `python -c "shutil.rmtree(...)"` classified as **read-only** — it would have run with no prompt. Interpreters and multi-tools are never read-only now. |
| Critical | An "allow for this session" grant defeated `always_confirm`, so installs asked once rather than always — contradicting the documented guarantee. |
| Critical | `backup_registry()` was written and **never called**. Atomic replace protects against a crash mid-write; nothing protected against a write that succeeded and was wrong. |
| High | Stale locks were reclaimed on **age alone**, which doesn't prove the owner died. A slow save could have its lock stolen mid-write. |
| High | Clip duration in seconds was written into **frame-count** fields: 5s became 5 frames. |
| High | The motion prompt overwrote **every** text node, including the negative prompt. |
| High | The turnaround was written into whatever node the IP-Adapter linked to — usually a preprocessor, which ComfyUI rejects. Now traced back to the real loader. |
| High | `Tool.timeout_s` was decorative; a hung render blocked the worker thread forever. |
| High | Handlers reporting failure by *returning* a message were marked `ok=True`, so the model saw a refusal as success. |
| Medium | A response truncated by the token limit was reported as a completed run. |
| Medium | Backups sorted by `mtime`, which ties on Windows for rapid writes — "restore the latest" could restore the wrong one. Found as a 1-in-8 test flake. |

**Open, and documented in the README rather than quietly carried:**

- **Harry cannot see.** No vision critic; he gets paths, not pictures. Locks made
  without one are now stamped `unreviewed` in the registry, and his system prompt
  forbids claiming one variant looks better.
- **"The registry is the memory" was overclaimed.** He re-reads state only when
  the model calls a read tool. Candidate paths from a generation live in the
  transcript, so a crash between generating and locking loses them. The
  resumability claim holds for *locked* work, not for work in flight.
- One run per process; a second browser tab shares it.
- Cancel stops between steps, not mid-render.
- No stale-read detection (no compare-and-set on locking).
- `save_image_node` is configured but not honoured when picking outputs.

**What is *not* verified:** no live LLM call, no live ComfyUI, no FLUX render, no
Minimax / LTX / Runway clip, no browser session. The handover's open item 1 —
that no Harry prompt change has been validated against a real model — is still
true, and now applies to the agent loop and the whole model chain too.

The three things that convert this from sound to proven, in order:
1. **One real FLUX keyframe with IP-Adapter holding a face**, driven from this
   app against the GPU machine. This proves the upload path *and* the continuity
   claim in one go.
2. One real clip from each of the three video routes.
3. A real Harry run on the corrected poem.
