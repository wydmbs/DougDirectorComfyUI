"""
runner.py -- run the assistant director on a worker thread.

SUITE.md rule 3 says one shell, more tabs, not more apps. So the agent runs
inside the Gradio process on a background thread and reports progress through
the same UI, rather than becoming a second thing to launch and babysit.

The runner owns the parts a UI needs and a loop shouldn't care about: a live
event feed the On set panel can poll, cancellation, and the consent queue that
lets a tool call pause until the director answers.
"""

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .audit import AuditLog
from .loop import AgentEvent, AgentLoop
from .permissions import Decision, PermissionPolicy, PermissionRequest


@dataclass
class PendingConsent:
    request: PermissionRequest
    answered: threading.Event = field(default_factory=threading.Event)
    decision: Decision = Decision.DENY


class AgentRunner:
    """One run at a time, which is the right constraint: two agents writing to
    the same registry would be a race even with the write lock protecting the
    file, because they'd be making contradictory creative decisions."""

    def __init__(self, cfg, provider, registry, posture: str, audit_path: str = "",
                 max_steps: int = 0):
        self.cfg = cfg
        self.provider = provider
        self.registry = registry
        self.policy = PermissionPolicy(posture=posture)
        self.audit = AuditLog(audit_path or "harry_agent_audit.sqlite3")
        self.max_steps = max_steps or cfg.agent.max_steps

        self.events = deque(maxlen=500)
        self.pending: Optional[PendingConsent] = None
        self.result = None
        self.run_id = ""
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[AgentLoop] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "run_id": self.run_id,
                "events": list(self.events),
                "awaiting_consent": self.pending.request if self.pending else None,
                "result": self.result,
            }

    # -------------------------------------------------------------- lifecycle

    def start(self, goal: str, context: str = "") -> str:
        if self.running:
            raise RuntimeError("Harry is already working. Let him finish, or stop him first.")

        self.run_id = uuid.uuid4().hex[:12]
        self.events.clear()
        self.result = None
        self.pending = None

        self._loop = AgentLoop(
            provider=self.provider,
            registry=self.registry,
            policy=self.policy,
            audit=self.audit,
            max_steps=self.max_steps,
            ask=self._ask,
            on_event=self._on_event,
        )

        def work():
            try:
                self.result = self._loop.run(goal, context, run_id=self.run_id)
            except Exception as error:  # noqa: BLE001 - surfaced, never silent
                self._on_event(AgentEvent("error", f"{type(error).__name__}: {error}"))
                self.result = None
            finally:
                self.pending = None

        self._thread = threading.Thread(target=work, name=f"harry-{self.run_id}", daemon=True)
        self._thread.start()
        return self.run_id

    def cancel(self) -> None:
        if self._loop:
            self._loop.cancel()
        # A run paused for consent would otherwise wait forever.
        if self.pending and not self.pending.answered.is_set():
            self.answer(Decision.DENY)

    # ---------------------------------------------------------------- consent

    def _ask(self, request: PermissionRequest) -> Decision:
        """Called from the worker thread; blocks until the director answers."""
        pending = PendingConsent(request=request)
        with self._lock:
            self.pending = pending
        self._on_event(AgentEvent(
            "consent", f"Harry needs your go-ahead: {request.summary}",
            {"tool": request.tool_name, "detail": request.detail, "entry_id": request.entry_id}))

        # Timeout so an abandoned browser tab can't wedge the thread forever.
        if not pending.answered.wait(timeout=1800):
            pending.decision = Decision.DENY
        with self._lock:
            self.pending = None
        return pending.decision

    def answer(self, decision: Decision) -> None:
        with self._lock:
            pending = self.pending
        if pending and not pending.answered.is_set():
            pending.decision = decision
            pending.answered.set()

    # ----------------------------------------------------------------- events

    def _on_event(self, event: AgentEvent) -> None:
        with self._lock:
            self.events.append(event)
