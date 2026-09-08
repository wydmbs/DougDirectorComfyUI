"""Failure modes for the assistant director, each with a reason worth showing."""


class AgentError(Exception):
    """Base for anything the assistant director can't do."""


class ProviderError(AgentError):
    """The model call failed: unreachable, rejected, or unusable response."""


class ToolError(AgentError):
    """A tool ran and failed. The loop reports this back to the model to retry."""


class PermissionDenied(AgentError):
    """The director declined, or the posture forbids this without asking."""


class Cancelled(AgentError):
    """The run was stopped on purpose. Not a failure."""


class StepLimitReached(AgentError):
    """The run hit its step budget before finishing."""
