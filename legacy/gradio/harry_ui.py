"""
harry_ui.py -- how the assistant director appears in the shell.

The design call: Harry is *in context*, not a separate destination.

A second "Harry Agent" tab would have split him in two -- the Harry who reads
the script over there, the Harry who does the work over here -- and the tab
strip already is the production pipeline. So instead he shows up as a rail that
rides along the existing tabs, plus an "On set" panel inside Build where the
actual work is commissioned and watched.

All rendering is HTML strings so it works on any Gradio version, matching how
the UI graphics are embedded elsewhere. Motion follows the house rule: a single
entrance animation on a real state change, never anything looping.
"""

import html

MOOD_READY = "ready"
MOOD_WORKING = "working"
MOOD_ASKING = "asking"
MOOD_WON = "won"
MOOD_STUCK = "stuck"

# Harry has a voice. It's the same assistant director in every state, so the
# copy stays dry and practical rather than chirpy.
MOODS = {
    MOOD_READY:   ("🎬", "Standing by", "navy"),
    MOOD_WORKING: ("🎞️", "On set", "amber"),
    MOOD_ASKING:  ("✋", "Needs you", "amber"),
    MOOD_WON:     ("🏆", "That's a take", "gold"),
    MOOD_STUCK:   ("⚠️", "Stuck", "warn"),
}

RAIL_CSS = """
<style>
.harry-rail{border:1px solid var(--hr-border,#d9d2c5);border-radius:12px;
  background:linear-gradient(180deg,#fdfcf9 0%,#f7f4ed 100%);padding:14px 16px;margin:8px 0 14px 0;}
.harry-rail-head{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.harry-badge{font-size:20px;line-height:1;}
.harry-name{font-weight:700;color:#172534;letter-spacing:.01em;}
.harry-state{margin-left:auto;font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;
  background:#eceadf;color:#5c6c77;}
.harry-state.amber{background:#f6e2c8;color:#8a5a1f;}
.harry-state.gold{background:#f9edc8;color:#7a5c12;}
.harry-state.warn{background:#f3dcda;color:#8c3b33;}
.harry-line{color:#3b4a56;font-size:13px;margin:2px 0 10px 0;line-height:1.5;}
.harry-meter{height:9px;border-radius:999px;background:#e6e1d6;overflow:hidden;margin:10px 0 6px 0;}
.harry-meter > span{display:block;height:100%;border-radius:999px;
  background:linear-gradient(90deg,#e8ae72 0%,#f6be45 100%);
  animation:harry-fill .5s ease-out;}
@keyframes harry-fill{from{width:0%!important;}}
.harry-meter-label{display:flex;justify-content:space-between;font-size:11px;color:#5c6c77;}
.harry-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;}
.harry-chip{font-size:11px;padding:3px 9px;border-radius:999px;background:#eef1f2;color:#3b4a56;
  border:1px solid #e0e4e6;}
.harry-chip.done{background:#e8f0e6;color:#3d5c33;border-color:#d5e4d1;}
.harry-chip.next{background:#f6e2c8;color:#8a5a1f;border-color:#eed4b2;font-weight:600;}
.harry-feed{max-height:320px;overflow-y:auto;border:1px solid #e2ddd2;border-radius:10px;
  background:#fffdf9;padding:8px 10px;margin-top:4px;}
.harry-ev{display:flex;gap:8px;padding:5px 0;border-bottom:1px dashed #efe9dd;font-size:12.5px;}
.harry-ev:last-child{border-bottom:none;}
.harry-ev-ic{width:18px;flex:0 0 18px;text-align:center;}
.harry-ev-tx{color:#33414c;line-height:1.45;word-break:break-word;}
.harry-ev.err .harry-ev-tx{color:#8c3b33;}
.harry-ev.win .harry-ev-tx{color:#3d5c33;}
.harry-ev.tool .harry-ev-tx{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:#5c6c77;}
.harry-consent{border:1px solid #eed4b2;background:#fdf6ec;border-radius:10px;padding:12px 14px;
  margin-top:10px;animation:harry-rise .35s ease-out;}
@keyframes harry-rise{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:none;}}
.harry-consent h4{margin:0 0 6px 0;color:#8a5a1f;font-size:13px;}
.harry-consent pre{background:#fffdf9;border:1px solid #efe3d2;border-radius:8px;padding:8px;
  font-size:11.5px;max-height:150px;overflow:auto;margin:8px 0 0 0;}
.harry-empty{color:#6d7b85;font-size:12.5px;font-style:italic;padding:10px 2px;}
</style>
"""


def _esc(text) -> str:
    return html.escape(str(text or ""))


def rail(overview: dict, mood: str = MOOD_READY, line: str = "", next_action: str = "") -> str:
    """The persistent strip: where the production stands, in one glance.

    Deliberately driven by the same project_overview() the agent calls, so the
    rail can never show something different from what Harry believes.
    """
    badge, state_text, tone = MOODS.get(mood, MOODS[MOOD_READY])
    locked = overview.get("locked_panels", 0)
    total = overview.get("total_panels", 0)
    pct = int(round((overview.get("panel_progress", 0.0) or 0.0) * 100))

    if not line:
        if total == 0:
            line = "No sheets to build yet. Approve a call sheet on Harry's tab and I'll get to work."
        elif locked == total:
            line = "Every panel is printed. The reels are ready in the screening room."
        else:
            line = f"{total - locked} panel{'s' if total - locked != 1 else ''} still to shoot."

    meter = ""
    if total:
        meter = (
            f'<div class="harry-meter"><span style="width:{pct}%"></span></div>'
            f'<div class="harry-meter-label"><span>{locked} of {total} panels printed</span>'
            f'<span>{pct}%</span></div>'
        )

    chips = []
    for sheet in overview.get("sheets", [])[:8]:
        done = sheet.get("complete")
        name = sheet.get("entry_id", "")
        count = f"{len(sheet.get('locked_panels', []))}/{sheet.get('total_panels', 0)}"
        chips.append(
            f'<span class="harry-chip{" done" if done else ""}">{_esc(name)} {count}</span>')
    if next_action:
        chips.append(f'<span class="harry-chip next">next: {_esc(next_action)}</span>')

    return (
        f'{RAIL_CSS}<div class="harry-rail">'
        f'<div class="harry-rail-head"><span class="harry-badge">{badge}</span>'
        f'<span class="harry-name">Harry</span>'
        f'<span class="harry-state {tone}">{state_text}</span></div>'
        f'<div class="harry-line">{_esc(line)}</div>'
        f'{meter}'
        f'<div class="harry-chips">{"".join(chips)}</div>'
        f'</div>'
    )


EVENT_STYLE = {
    "step":        ("•", ""),
    "text":        ("💬", ""),
    "tool":        ("⚙️", "tool"),
    "tool_result": ("↩️", "tool"),
    "consent":     ("✋", ""),
    "blocked":     ("🚫", "err"),
    "error":       ("⚠️", "err"),
    "done":        ("✅", "win"),
}


def feed(events, limit: int = 60) -> str:
    """The live account of what Harry is doing, newest last so it reads
    like a running log rather than a stack."""
    if not events:
        return (f'{RAIL_CSS}<div class="harry-feed"><div class="harry-empty">'
                f'Nothing yet. Give Harry a goal and hit Action.</div></div>')

    rows = []
    for event in list(events)[-limit:]:
        icon, cls = EVENT_STYLE.get(event.kind, ("·", ""))
        message = event.message or ""
        if event.kind == "tool":
            arguments = (event.detail or {}).get("arguments", {})
            hint = arguments.get("entry_id") or arguments.get("panel_key") or ""
            message = f"{message}({_esc(hint)})" if hint else message
        if event.kind == "step":
            continue  # the step counter is noise in the feed; it shows in the rail
        rows.append(
            f'<div class="harry-ev {cls}"><span class="harry-ev-ic">{icon}</span>'
            f'<span class="harry-ev-tx">{_esc(message)[:600]}</span></div>')

    return f'{RAIL_CSS}<div class="harry-feed">{"".join(rows)}</div>'


def consent_card(request) -> str:
    """Harry asking permission. Shows exactly what will happen before it does."""
    if request is None:
        return ""
    detail = f'<pre>{_esc(request.detail)[:1500]}</pre>' if request.detail else ""
    warning = ""
    if getattr(request, "irreversible", False):
        warning = ('<div style="margin-top:8px;font-size:12px;color:#8c3b33;">'
                   'This one runs a command on your PC. Read it before you agree.</div>')
    return (
        f'{RAIL_CSS}<div class="harry-consent">'
        f'<h4>✋ Harry needs your go-ahead</h4>'
        f'<div style="font-size:13px;color:#3b4a56;">{_esc(request.summary)}</div>'
        f'{warning}{detail}</div>'
    )


def wrap_banner(entry_id: str, banner_uri: str = "") -> str:
    """A finished sheet is worth marking."""
    image = (f'<img src="{banner_uri}" alt="" style="width:100%;border-radius:10px;'
             f'display:block;margin-bottom:10px;" />' if banner_uri else "")
    return (f'<div class="celebration-banner">{image}🎬 <strong>That\'s a wrap!</strong> '
            f'Harry printed every panel for {_esc(entry_id)}.</div>')


def mood_for(snapshot: dict) -> str:
    """Pick Harry's state from what the runner is doing."""
    if snapshot.get("awaiting_consent"):
        return MOOD_ASKING
    if snapshot.get("running"):
        return MOOD_WORKING
    result = snapshot.get("result")
    if result is not None:
        return MOOD_WON if getattr(result, "completed", False) else MOOD_STUCK
    return MOOD_READY


def status_line(snapshot: dict, overview: dict) -> str:
    """One honest sentence about what just happened."""
    if snapshot.get("awaiting_consent"):
        return "Waiting on your call before I go any further."
    if snapshot.get("running"):
        events = snapshot.get("events") or []
        for event in reversed(events):
            if event.kind in ("tool", "text") and event.message:
                return f"Working — {event.message[:120]}"
        return "Getting my bearings."
    result = snapshot.get("result")
    if result is not None:
        if getattr(result, "completed", False):
            return getattr(result, "summary", "") or "Finished."
        return getattr(result, "error", "") or "Stopped before finishing."
    return ""
