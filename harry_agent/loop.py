"""
loop.py -- the assistant director's plan / act / observe cycle.

The important design choice: **the registry is the memory, not the transcript.**

Harry re-reads project state from the workbook at each step rather than
accumulating everything he has ever done in the conversation. That has three
consequences worth the trade:

  * a run survives a crash, because progress lives in the registry, and
    "what's left" is a query rather than a replay
  * the context stays small and bounded, so none of the history-compaction
    machinery a chat agent needs is required here
  * a person can lock a panel by hand mid-run and Harry simply sees it

The conversation is kept only as far back as it needs to be for the model to
follow its own recent tool calls; older exchanges are dropped, and their effect
is already visible in the registry anyway.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .errors import Cancelled, PermissionDenied, ProviderError, StepLimitReached
from .permissions import PermissionRequest
from .providers.base import STOP_TOOL_USE
from .trust import TRUST_RULE

SYSTEM_PROMPT = """You are Harry, the assistant director on an AI-assisted film production.

You are not a chat assistant. You do real work: you generate images, judge them, and
lock the good ones into the project registry. You have tools; use them rather than
describing what you would do.

HOW YOU WORK
- Start by getting your bearings with project_overview. Never assume state.
- Work one asset at a time. Finish it before starting another.
- After anything that changes the project, move on. Don't re-read what you just wrote.
- When you judge variants, say plainly why one wins. That reasoning is kept.
- If a tool fails, read the error and adapt. Don't repeat the same call unchanged.
- When the goal is met, say so clearly and stop calling tools.

WHAT MATTERS TO THIS PRODUCTION
- Continuity is the whole job. An asset must match its locked concept and its
  continuity note, not merely look good on its own.
- The era/setting is a hard constraint on every visual detail.
- Anything marked "inferred, not explicit in source" is a suggestion awaiting the
  director's decision. Never lock one without being told to.
- entry_ids are CHARACTER:name, BACKDROP:name, PROP:name, or a shot id like 1.1.
  Use those exactly; never invent a different scheme.

%s
""" % TRUST_RULE


@dataclass
class AgentEvent:
    """Something worth showing the director as it happens."""
    kind: str                 # step | text | tool | tool_result | done | error | blocked
    message: str = ""
    detail: Optional[dict] = None
    step: int = 0


@dataclass
class RunResult:
    run_id: str
    completed: bool
    steps: int
    summary: str
    events: List[AgentEvent] = field(default_factory=list)
    error: str = ""


class AgentLoop:
    def __init__(self, provider, registry, policy, audit=None, max_steps: int = 40,
                 ask=None, on_event: Optional[Callable[[AgentEvent], None]] = None,
                 history_turns: int = 12):
        self.provider = provider
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.max_steps = max_steps
        self.ask = ask
        self.on_event = on_event
        self.history_turns = history_turns
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the run to stop. Checked between steps, so an in-flight
        generation finishes rather than leaving a half-written registry row."""
        self._cancelled = True

    # ------------------------------------------------------------------ events

    def _emit(self, kind: str, message: str = "", detail=None, step: int = 0) -> AgentEvent:
        event = AgentEvent(kind, message, detail, step)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001 - a broken listener must not kill the run
                pass
        return event

    def _record(self, run_id: str, kind: str, **kwargs) -> None:
        if self.audit:
            try:
                self.audit.record(run_id, kind, **kwargs)
            except Exception:  # noqa: BLE001 - logging must never break the work
                pass

    # -------------------------------------------------------------------- run

    def run(self, goal: str, context: str = "", run_id: str = "") -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        events: List[AgentEvent] = []
        messages = [{"role": "user", "content": [{"type": "text", "text": self._opening(goal, context)}]}]
        tool_schemas = self.registry.schemas()

        self._record(run_id, "run_start", summary=goal)
        summary, completed, error = "", False, ""
        step = 0

        while step < self.max_steps:
            if self._cancelled:
                error = "Stopped by the director."
                events.append(self._emit("done", error, step=step))
                self._record(run_id, "cancelled", summary=error)
                return RunResult(run_id, False, step, error, events, error)

            step += 1
            events.append(self._emit("step", f"Step {step}", step=step))

            try:
                turn = self.provider.complete(SYSTEM_PROMPT, self._trim(messages), tool_schemas)
            except ProviderError as exc:
                error = str(exc)
                events.append(self._emit("error", error, step=step))
                self._record(run_id, "provider_error", summary=error, ok=False)
                return RunResult(run_id, False, step, "", events, error)

            if turn.text:
                events.append(self._emit("text", turn.text, step=step))

            if not turn.wants_tools or turn.stop_reason != STOP_TOOL_USE:
                summary = turn.text or "Finished."
                completed = True
                events.append(self._emit("done", summary, step=step))
                self._record(run_id, "run_done", summary=summary)
                break

            assistant_blocks = ([{"type": "text", "text": turn.text}] if turn.text else [])
            assistant_blocks += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in turn.tool_calls
            ]
            messages.append({"role": "assistant", "content": assistant_blocks})

            result_blocks = []
            for call in turn.tool_calls:
                block = self._invoke(run_id, call, events, step)
                result_blocks.append(block)
            messages.append({"role": "user", "content": result_blocks})
        else:
            error = f"Reached the {self.max_steps}-step limit before finishing."
            events.append(self._emit("error", error, step=step))
            self._record(run_id, "step_limit", summary=error, ok=False)
            return RunResult(run_id, False, step, summary, events, error)

        return RunResult(run_id, completed, step, summary, events, error)

    # ------------------------------------------------------------------ pieces

    @staticmethod
    def _opening(goal: str, context: str) -> str:
        parts = [f"GOAL: {goal}"]
        if context:
            parts.append(f"\nCONTEXT:\n{context}")
        parts.append("\nGet your bearings first, then work the goal. Begin.")
        return "\n".join(parts)

    def _trim(self, messages: list) -> list:
        """Keep the opening brief plus the most recent exchanges.

        Tool calls and their results must stay paired or providers reject the
        request, so the cut is made on a user message boundary and the first
        message is always kept for the goal.
        """
        if len(messages) <= self.history_turns:
            return messages
        head = messages[:1]
        tail = messages[-(self.history_turns - 1):]
        while tail and not (tail[0].get("role") == "assistant"):
            tail = tail[1:]
        return head + tail

    def _invoke(self, run_id: str, call, events: list, step: int) -> dict:
        """Run one tool call, honouring the permission policy, and return the
        result block to hand back to the model."""
        tool = self.registry.get(call.name)
        events.append(self._emit(
            "tool", f"{call.name}", detail={"arguments": call.arguments}, step=step))

        if tool is None:
            message = f"No tool named '{call.name}'. Available: {', '.join(self.registry.names())}"
            events.append(self._emit("tool_result", message, step=step))
            self._record(run_id, "tool_unknown", name=call.name, summary=message, ok=False)
            return self._result_block(call.id, message, error=True)

        entry_id = str(call.arguments.get("entry_id", "") or "")

        try:
            self.policy.check(
                tool,
                PermissionRequest(
                    tool_name=tool.name,
                    summary=self._describe_call(tool, call),
                    detail=json.dumps(call.arguments, indent=2, default=str),
                    entry_id=entry_id,
                    irreversible=tool.always_confirm,
                ),
                ask=self.ask,
            )
        except PermissionDenied as denied:
            message = str(denied)
            events.append(self._emit("blocked", message, step=step))
            self._record(run_id, "permission_denied", name=tool.name, entry_id=entry_id,
                         summary=message, ok=False)
            # Reported as a tool result so the model can choose something else
            # rather than treating a refusal as a crash.
            return self._result_block(call.id, f"Declined by the director: {message}", error=True)

        started = time.time()
        result = self.registry.run(call.name, call.arguments)
        events.append(self._emit(
            "tool_result",
            (result.content or "")[:400],
            detail={"ok": result.ok, "tool": call.name, "seconds": round(time.time() - started, 2)},
            step=step,
        ))
        self._record(run_id, "tool_call", name=call.name, entry_id=entry_id,
                     summary=(result.content or "")[:500],
                     detail={"arguments": call.arguments}, ok=result.ok)
        return self._result_block(call.id, result.content, error=not result.ok)

    @staticmethod
    def _describe_call(tool, call) -> str:
        entry = call.arguments.get("entry_id") or call.arguments.get("panel_key") or ""
        return f"{tool.name} {entry}".strip()

    @staticmethod
    def _result_block(call_id: str, content: str, error: bool = False) -> dict:
        block = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": (content or "")[:8000],
        }
        if error:
            block["is_error"] = True
        return block
