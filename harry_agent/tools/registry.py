"""
registry.py -- the tool contract and dispatcher.

A tool is a plain Python callable plus the metadata a model needs to decide
when to call it. Keeping the contract this thin means the same function the
Build tab calls can be exposed to the assistant director without a wrapper
layer that can drift out of step.

Two flags carry the safety semantics:

  mutating       -- changes the project or the machine
  routine        -- mutating, but the ordinary business of building
                    (locking a panel), so the supervised posture can let it
                    through while still gating the unusual
  always_confirm -- ask every time regardless of posture (installs)
"""

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..errors import ToolError


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                    # JSON Schema for the arguments
    handler: Callable[..., Any]
    mutating: bool = False
    routine: bool = False
    always_confirm: bool = False
    timeout_s: float = 300.0

    def schema(self) -> dict:
        """Anthropic-shaped tool definition; providers adapt from this."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters or {"type": "object", "properties": {}},
        }


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    content: str
    data: Any = None
    duration_s: float = 0.0

    def as_text(self) -> str:
        return self.content


@dataclass
class ToolRegistry:
    tools: Dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self.tools[tool.name] = tool

    def extend(self, tools: List[Tool]) -> None:
        for tool in tools:
            self.add(tool)

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def schemas(self) -> List[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def names(self) -> List[str]:
        return list(self.tools)

    def run(self, name: str, arguments: dict) -> ToolResult:
        """Invoke a tool and normalise whatever it returns into text.

        A tool raising is not a crash -- it is a result the model is told about
        so it can adapt, which is the whole point of a loop. Only genuinely
        unknown tool names are treated as a caller error.
        """
        tool = self.get(name)
        if tool is None:
            raise ToolError(f"No tool named '{name}'. Available: {', '.join(self.names())}")

        started = time.time()
        try:
            filtered = self._filter_arguments(tool, arguments or {})
            value = tool.handler(**filtered)
            content = value if isinstance(value, str) else json.dumps(value, default=str, indent=2)
            return ToolResult(name, True, content, value, time.time() - started)
        except Exception as error:  # noqa: BLE001 - reported to the model, not swallowed
            return ToolResult(
                name, False,
                f"{type(error).__name__}: {error}",
                None, time.time() - started,
            )

    @staticmethod
    def _filter_arguments(tool: Tool, arguments: dict) -> dict:
        """Drop arguments the handler doesn't accept.

        Models occasionally invent an extra field. Dropping it is far better
        than a TypeError that reads like a bug in the harness.
        """
        try:
            signature = inspect.signature(tool.handler)
        except (TypeError, ValueError):
            return arguments
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return arguments
        accepted = set(signature.parameters)
        return {k: v for k, v in arguments.items() if k in accepted}
