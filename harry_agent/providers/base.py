"""
base.py -- one shape for every model provider.

Harry's original _chat() could only ask a question and read prose back. An
assistant director has to be able to call tools, so this interface returns a
structured turn instead: some text, and zero or more tool calls the loop should
execute and feed back.

Deliberately synchronous. The agent runs on a worker thread and its slow part
is image generation, not token streaming, so async would add cancellation and
generator-cleanup problems for no benefit here.
"""

from dataclasses import dataclass, field
from typing import List, Optional

STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_LENGTH = "length"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    """One model response."""
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = STOP_END_TURN
    raw: Optional[dict] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Provider:
    """Implemented once per model vendor."""

    name = "provider"
    supports_tools = True
    supports_vision = False

    def complete(self, system: str, messages: list, tools: Optional[list] = None,
                 max_tokens: int = 4096) -> Turn:
        """Send a conversation and return one turn.

        `messages` uses the Anthropic content-block shape as the internal
        format, because it represents tool calls and tool results explicitly.
        Providers that use a different wire format convert on the way in and
        out, so the loop only ever deals with one shape.
        """
        raise NotImplementedError

    def describe(self) -> str:
        return self.name
