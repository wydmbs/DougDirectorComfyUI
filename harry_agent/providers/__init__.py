"""Model providers, normalised to one tool-calling interface."""

from .base import Provider, ToolCall, Turn
from .mock import MockProvider

__all__ = ["Provider", "Turn", "ToolCall", "MockProvider", "build_provider"]


def build_provider(provider_name: str, cfg=None):
    from .live import build_provider as _build
    return _build(provider_name, cfg)
