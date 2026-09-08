"""
mock.py -- a scripted provider for testing without an API key.

The handover manifest flags that every Harry change was verified with mocked
calls and never a real one. That is a fair criticism of *only* testing this way,
but a deterministic provider is still exactly what you want for proving loop
mechanics: tool dispatch, permission gating, retries, step limits and
cancellation, none of which should depend on a live model's mood.

Scripts are lists of Turn objects, or callables that receive the conversation
so far and decide what to do next.
"""

import itertools
from typing import Callable, List, Optional, Union

from .base import STOP_END_TURN, STOP_TOOL_USE, Provider, ToolCall, Turn


class MockProvider(Provider):
    name = "mock"

    def __init__(self, script: Optional[List[Union[Turn, Callable]]] = None):
        self.script = list(script or [])
        self.calls = []
        self._ids = itertools.count(1)

    def complete(self, system: str, messages: list, tools=None, max_tokens: int = 4096) -> Turn:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.script:
            return Turn(text="Nothing left to do.", stop_reason=STOP_END_TURN)
        step = self.script.pop(0)
        if callable(step):
            step = step(messages)
        return step

    def tool_turn(self, name: str, arguments: dict, text: str = "") -> Turn:
        return Turn(
            text=text,
            tool_calls=[ToolCall(id=f"call_{next(self._ids)}", name=name, arguments=arguments)],
            stop_reason=STOP_TOOL_USE,
        )
