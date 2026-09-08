"""
permissions.py -- how much the assistant director may do without asking.

Three postures, chosen by the director:

  attended    -- ask before anything that changes the project or the machine
  supervised  -- generate freely, ask before locking or installing (the default)
  unattended  -- run the plan; still always ask before touching the machine

Installing custom nodes and downloading models runs commands on Doug's PC, so
that class of action requires explicit consent under every posture. There is no
setting that turns it off, because "the agent installed something overnight and
broke my ComfyUI" is not a recoverable kind of surprise.
"""

from dataclasses import dataclass, field
from enum import Enum

from .errors import PermissionDenied

ATTENDED = "attended"
SUPERVISED = "supervised"
UNATTENDED = "unattended"
POSTURES = (ATTENDED, SUPERVISED, UNATTENDED)


class Decision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass
class PermissionRequest:
    tool_name: str
    summary: str
    detail: str = ""
    entry_id: str = ""
    irreversible: bool = False


@dataclass
class PermissionPolicy:
    """Decides, per tool call, whether to run, ask, or refuse."""
    posture: str = SUPERVISED
    session_grants: set = field(default_factory=set)

    def needs_confirmation(self, tool) -> bool:
        if getattr(tool, "always_confirm", False):
            return True
        if not getattr(tool, "mutating", False):
            return False
        if self.posture == ATTENDED:
            return True
        if self.posture == SUPERVISED:
            return not getattr(tool, "routine", False)
        return False  # unattended: mutating-but-not-always-confirm runs freely

    def check(self, tool, request: PermissionRequest, ask=None) -> None:
        """Raise PermissionDenied unless this call may proceed.

        `ask` is a callable the shell supplies to put the question to the
        director. When confirmation is required and no asker is wired up, the
        call is refused rather than assumed -- silence is never consent.
        """
        if not self.needs_confirmation(tool):
            return

        # always_confirm has to mean *always*. A session grant earned by one
        # harmless install must not silently cover every later command, or
        # "installs always ask" becomes "installs ask once", which is exactly
        # the assumption someone would be relying on when they walked away.
        always = getattr(tool, "always_confirm", False)
        if not always and tool.name in self.session_grants:
            return

        if ask is None:
            raise PermissionDenied(
                f"'{tool.name}' needs the director's approval and nothing is available to ask. "
                f"Run this from the On set panel, or switch posture."
            )
        decision = ask(request)
        if decision == Decision.ALLOW_SESSION:
            if always:
                # Honour it for this one call, but don't remember it.
                return
            self.session_grants.add(tool.name)
            return
        if decision == Decision.ALLOW_ONCE:
            return
        raise PermissionDenied(f"The director declined: {request.summary}")


def describe(posture: str) -> str:
    return {
        ATTENDED: "Attended — Harry checks with you before every change.",
        SUPERVISED: "Supervised — Harry generates freely and checks in before locking.",
        UNATTENDED: "Unattended — Harry works the plan alone. Installs still need you.",
    }.get(posture, f"Unknown posture '{posture}'.")
