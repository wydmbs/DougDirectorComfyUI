"""
harry_agent -- Harry the assistant director.

Harry started as a one-shot analyser: read the source, propose a call sheet,
stop. This package is the rest of the job -- executing that call sheet. He
generates, judges, locks, assembles sheets, and helps get a real ComfyUI
workflow running, with the registry as his memory and the director's approval
gating anything that changes the project or the machine.
"""

from .errors import (AgentError, Cancelled, PermissionDenied, ProviderError,
                     StepLimitReached, ToolError)
from .loop import AgentEvent, AgentLoop, RunResult
from .permissions import (ATTENDED, POSTURES, SUPERVISED, UNATTENDED, Decision,
                          PermissionPolicy, PermissionRequest, describe)
from .runner import AgentRunner
from .tools.registry import Tool, ToolRegistry, ToolResult

__all__ = [
    "AgentError", "ProviderError", "ToolError", "PermissionDenied", "Cancelled",
    "StepLimitReached", "AgentLoop", "AgentEvent", "RunResult", "AgentRunner",
    "PermissionPolicy", "PermissionRequest", "Decision", "describe",
    "ATTENDED", "SUPERVISED", "UNATTENDED", "POSTURES",
    "Tool", "ToolRegistry", "ToolResult",
]


def build_registry(cfg, generate_fn=None, critique_fn=None, run_command=None) -> ToolRegistry:
    """The full tool set, bound to one config."""
    from .tools.comfy_tools import build_comfy_tools
    from .tools.director_tools import build_tools

    registry = ToolRegistry()
    registry.extend(build_tools(cfg, generate_fn=generate_fn, critique_fn=critique_fn))
    registry.extend(build_comfy_tools(cfg, run_command=run_command))
    return registry
