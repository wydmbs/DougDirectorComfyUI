# SETUP — getting to your first real render

This is the path from a fresh GPU machine to one staged FLUX keyframe with a
character's face actually held. That single render is the thing worth chasing:
it proves the upload path, the IP-Adapter wiring, and the continuity claim the
whole harness rests on. Everything after it is easier.

**Run the doctor whenever you're unsure.** It changes nothing and tells you the
one next thing to fix:

```powershell
python doctor.py                                  # local ComfyUI
python doctor.py --url http://192.168.1.206:8188  # ComfyUI on the GPU box
```

---

## 1. The app

```powershell
pip install -r requirements.txt
python app.py
```

If you use conda, call the interpreter by its full path so the packages land in
the environment the app actually runs from — installing into a different Python
than the one you launch with is a recurring way to lose an hour.

`python doctor.py` should now show Python and Configuration as fine.

---

## 2. ComfyUI on the GPU machine

Install ComfyUI, then add what the keyframe stage needs.

**Custom nodes** — via ComfyUI-Manager, or by hand:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite   # for clips later
```

Restart ComfyUI afterwards; it only scans custom nodes at startup.

**Models:**

| What | Where it goes |
|---|---|
| FLUX.1 Kontext Dev, FP8 | `ComfyUI/models/diffusion_models` |
| `clip_l` + `t5xxl` text encoders | `ComfyUI/models/clip` |
| FLUX `ae.safetensors` VAE | `ComfyUI/models/vae` |
| FLUX.1 Dev, FP8 | `ComfyUI/models/checkpoints` (or `models/unet`) |
| IP-Adapter Plus | `ComfyUI/models/ipadapter` |
| CLIP-ViT-H vision | `ComfyUI/models/clip_vision` |

FP8 is the variant to use on a 4090 — around 12–16GB, which leaves room to work.

Kontext is a **separate model**, not a setting on FLUX.1 Dev, and it ships as a
bare diffusion model — so the text encoders and VAE that an all-in-one checkpoint
bundles have to be present on their own. Get it from
`Comfy-Org/flux1-kontext-dev_ComfyUI` (`flux1-dev-kontext_fp8_scaled.safetensors`,
11GB, no gate, no token needed).

The last three rows are only needed for the IP-Adapter fallback. If you are
starting fresh and only want the recommended path, skip them.

**If ComfyUI is on a different machine from the app**, start it so it accepts
connections from elsewhere:

```powershell
python main.py --listen 0.0.0.0
```

Then put that machine's URL into Setup → The GPU machine. Remote works exactly
like local: images are uploaded to ComfyUI and referenced by the name it returns,
rather than by a local path that would mean nothing on the other box.

`python doctor.py --url http://<gpu-box>:8188` should now show ComfyUI and
Models as fine.

---

## 3. Build the keyframe workflow

You don't have to draw this by hand. With ComfyUI running:

```powershell
python kontext_workflow.py --write        # recommended
python flux_workflow_builder.py --write   # IP-Adapter fallback
```

Both read the live server's node list, so every filename they write is one this
machine actually has, and `--write` points the app's config at the result. What
follows is what they build, and why it is shaped that way.

### How a character survives from shot to shot

This is the single decision that makes or breaks continuity, and it is a choice
between two **different mechanisms** rather than two quality settings:

| | What reaches the model | Holds |
|---|---|---|
| **Kontext** | the reference, VAE-encoded into a full latent grid, concatenated onto the sequence being denoised | costume detail, button count, the exact shape of a comb |
| **IP-Adapter** | a handful of embedding tokens from a vision encoder — a *summary* of the reference | "roughly this character" |

IP-Adapter cannot hold what it never received. Turning its weight up makes the
picture more contrasty, not more faithful; measured on the Pig & Rooster
reference, weight 1.0 and 1.3 both replaced a distinctive floppy two-lobed comb
with a generic serrated one. Kontext kept it. That is the whole reason Kontext
is the default.

### The Kontext graph

```
UNETLoader (kontext) ─────────────────────────────> KSampler ──> VAEDecode ──> SaveImage
                                                      ▲  ▲
LoadImage ──> FluxKontextImageScale ──> VAEEncode ────┴──┤
                                            │            │
CLIPTextEncode (instruction) ──> FluxGuidance ──> ReferenceLatent
                             └─> ConditioningZeroOut ────────> (negative)
```

What matters, and why:

- **`ReferenceLatent` is the node doing the work.** It attaches the encoded
  reference to the conditioning, so the reference's tokens travel alongside the
  ones being generated. It is core ComfyUI, not a custom node — if it is absent,
  update ComfyUI.
- **`FluxKontextImageScale` is not optional in practice.** Kontext was trained on
  a fixed set of resolutions; anything else degrades in a way that reads as the
  model being bad rather than the image being the wrong shape.
- **The sampler starts from the reference's own latent.** That is what makes this
  an edit of that image rather than a new picture that resembles it.
- **`denoise` stays at 1.0 here.** The 0.5–0.7 rule below belongs to img2img,
  which is a different thing.
- **Guidance is low, around 2.5.** Kontext is transforming an image it can see,
  not inventing one from a description.

**Write instructions, not descriptions.** Kontext is an edit model. "Turn him to
face left" transforms the reference; re-describing the character makes it redraw
from scratch and you lose the thing you were trying to keep. Anything you don't
mention is inherited, which is the point — so mentioning the wardrobe or the
lighting is how they drift.

### The IP-Adapter graph, if you use the fallback

```
CheckpointLoaderSimple ──┐
                         ├──> IPAdapterAdvanced ──> KSampler ──> VAEDecode ──> SaveImage
LoadImage (turnaround) ──┤          ▲                  ▲
                         │          │                  │
IPAdapterModelLoader ────┘          │            VAEEncode
CLIPVisionLoader ───────────────────┘                  ▲
                                                       │
CLIPTextEncode (positive) ─────────────────────> LoadImage (scene concept)
CLIPTextEncode (negative)
```

What matters, and why:

- **`IPAdapterAdvanced` sits between the model loader and the sampler.** This is
  what carries the character's identity. Without it every panel of a sheet is a
  different pig, no matter how carefully the prompt is worded.
- **A `LoadImage` feeds the adapter.** The harness traces the adapter's image
  input back to whichever node actually reads a file, so a preprocessing node in
  between (`PrepImageForClipVision`, a resize) is fine.
- **A second `LoadImage` → `VAEEncode` → the sampler's `latent_image`** anchors a
  scene to its concept art. Optional, but it's what keeps a location looking like
  itself across shots.
- **Set the sampler's `denoise` to about 0.5–0.7** if you use scene anchoring. At
  1.0 the sampler throws the encoded scene away entirely — the reference is in
  the graph and absent from the result, which looks exactly like conditioning not
  working. The doctor flags this.
- **On FLUX, use `ApplyIPAdapterFlux`, not the SD/SDXL nodes.** They are separate
  implementations and the SD one cannot condition FLUX at all. The FLUX adapter
  also loads its weights with a bare `torch.load`, so it needs `ip-adapter.bin`
  — the safetensors build of the same weights appears in the dropdown and fails
  at render time with an unpickling error naming neither file nor format.

Then export it:

1. Settings → **Enable Dev mode options**
2. Workflow → **Save (API Format)**

The API export and the normal one look similar and are not interchangeable; only
the API form can be submitted programmatically. The doctor detects a UI export
and says so.

---

## 4. Point the app at it

In **Setup**:

- ComfyUI URL → your GPU machine
- Workflow JSON → the API-format file you just exported
- Node mapping → the node **IDs** carrying the positive prompt, negative prompt
  and seed. Open the API JSON: the keys (`"6"`, `"7"`, `"10"`) are the IDs.
- Turn **Mock mode off**

> Node IDs change every time you re-export a workflow. If renders start failing
> after you edit the graph, re-check this mapping first — the doctor catches a
> stale one.

`python doctor.py` should now report **0 blocking**.

---

## 5. Draft, stage, and check the face

**Draft it.** Harry's tab → *Draft it in ChatGPT*. Describe the character, copy
the prompt, paste it into ChatGPT, save the image you like, attach it back. Ask
for the **reference** prompt, not the turnaround: one character, one view, plain
grey background, flat studio lighting. A busy background or dramatic lighting is
inherited along with the character, so a scenic reference poisons every render
downstream. Keep the four-view turnaround too — Minimax H3 genuinely wants it
later — but it is the wrong shape for conditioning a still.

**Stage a keyframe.** Build tab → name a shot, write a cinematic prompt, and
generate with the character selected as the identity reference.

**Then do the check that actually matters.** Put the generated keyframe next to
the reference and ask: *is this the same character?* Not "is it good" — is it the
**same**. Pick one small, distinctive feature and check that specifically; a
comb, a button count, a collar. Broad impressions are too forgiving. If it isn't
the same, in likely order:

| Symptom | Cause |
|---|---|
| Recognisable but details all slightly wrong | Conditioning is on IP-Adapter. Switch to Kontext — no weight fixes this |
| Character redrawn rather than transformed | The Kontext prompt describes the scene instead of instructing an edit |
| Wardrobe or lighting changed on its own | The instruction mentioned them. Anything named gets regenerated |
| Face unrelated to the reference | The reference never reached the graph — check the doctor's *Keyframe workflow* section |
| Softer than expected | No `FluxKontextImageScale`, so Kontext got an untrained resolution |
| Background ignored (img2img) | `denoise` at 1.0 |
| ComfyUI rejects the prompt | Node mapping stale, or a missing custom node |

On the fallback path only: raise `ipadapter_weight` (Setup, default 1.0 — the
node's own baseline) if identity is loose, drop it if the pose is too rigidly
copied. Do not expect much; the ceiling is low by construction.

---

## 6. Clips, when you get there

Each video route needs its own API-format workflow, built the same way:

| Route | For | Needs |
|---|---|---|
| LTX-2.5 | camera moves, cuts | local weights; start here, it's free |
| Minimax H3 | dialogue, head turns | ComfyUI API node; **two** `LoadImage` nodes — keyframe *and* turnaround |
| Runway Gen-4 | physics, destruction | ComfyUI API node; short bursts only |

Minimax needs the second image input specifically: REF2VA uses the turnaround's
side and back profiles, which is why a head turn holds together instead of
melting. With only one loader the harness warns you rather than silently
producing a drifting face.

Set the paths in Setup → The GPU machine. `gpu_readiness` and the doctor both
report which routes are ready.

---

## When something breaks

| | |
|---|---|
| Anything at all | `python doctor.py` — it names the one next thing |
| Registry looks wrong | `python -c "import registry_safety as r; print(r.restore_latest_backup('storyboard.xlsx'))"` |
| Harry did something odd | Check the audit log: `harry_agent_audit.sqlite3` |
| Renders ignore the reference | Doctor's *Keyframe workflow* section, then `denoise` |

Two things worth remembering:

- **Harry cannot see.** He gets file paths, not pictures. Any lock he makes
  without a vision critic is stamped `unreviewed` in the registry — treat those
  as drafts, not decisions.
- **Cancel stops between steps.** A render already in flight finishes and, on a
  cloud route, is paid for.
