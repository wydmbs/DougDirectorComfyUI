"""
safety.py -- decide whether a shell command is safe to run on the director's PC.

Phase 4 lets the assistant director install ComfyUI custom nodes and download
models, which means it proposes real commands. This module is the gate. It is
deliberately conservative: it classifies a command as READ_ONLY, NEEDS_CONSENT,
or BLOCKED, and anything it does not positively recognise gets NEEDS_CONSENT
rather than a pass. Unknown is never treated as harmless.

Two categories are refused outright, because they are the ways an automated
agent realistically destroys a machine:

  * recursive deletes aimed at a root, home, or drive
  * killing processes by name rather than by a specific PID

Commands are split on separators and each part judged on its own, so appending
a dangerous clause after a harmless one doesn't slip through. Base64-encoded
PowerShell is decoded and re-examined rather than waved past.
"""

import base64
import re
import shlex
from enum import Enum

MAX_UNWRAP_DEPTH = 5


class Verdict(str, Enum):
    READ_ONLY = "read_only"
    NEEDS_CONSENT = "needs_consent"
    BLOCKED = "blocked"


class SafetyResult:
    def __init__(self, verdict: Verdict, reason: str = "", command: str = ""):
        self.verdict = verdict
        self.reason = reason
        self.command = command

    @property
    def allowed(self) -> bool:
        return self.verdict is not Verdict.BLOCKED

    def __repr__(self):
        return f"SafetyResult({self.verdict.value}, {self.reason!r})"


# Commands that only look at the world.
READ_ONLY_HEADS = {
    "dir", "ls", "pwd", "cd", "type", "cat", "head", "tail", "findstr", "grep",
    "where", "which", "echo", "get-content", "get-childitem", "get-location",
    "test-path", "select-string", "measure-object", "python", "py", "nvidia-smi",
    "git", "pip", "conda",
}

# Sub-commands that make an otherwise read-only head mutating.
MUTATING_SUBCOMMANDS = {
    "git": {"clone", "pull", "push", "checkout", "reset", "clean", "rm", "merge", "rebase"},
    "pip": {"install", "uninstall", "download"},
    "conda": {"install", "remove", "update", "create"},
}

DESTRUCTIVE_ROOTS = re.compile(
    # The prefix allows ':' so a drive-qualified system path (C:\Windows) is
    # caught as well as a separator-prefixed one.
    r"(?:^|[\s'\"=:])(?:"
    r"[a-zA-Z]:[\\/]?(?=\s|$|['\"])"       # a drive root: C:  or  C:\
    r"|[\\/](?=\s|$)"                       # a bare slash
    r"|~[\\/]?(?=\s|$|['\"])"               # home
    r"|\$env:USERPROFILE"
    r"|\$HOME"
    r"|%USERPROFILE%"
    r"|[\\/](?:windows|winnt|system32|program files|programdata|users|home)(?:[\\/]|\s|$)"
    r")",
    re.IGNORECASE,
)

RECURSIVE_DELETE = re.compile(
    r"\b(rm|del|erase|rmdir|rd|remove-item|ri)\b", re.IGNORECASE)
RECURSIVE_FLAG = re.compile(
    r"(^|\s)(-r\b|-rf\b|-fr\b|--recursive\b|-recurse\b|/s\b)", re.IGNORECASE)
FORCE_FLAG = re.compile(r"(^|\s)(-f\b|--force\b|-force\b|/q\b|/f\b)", re.IGNORECASE)

KILL_BY_NAME = re.compile(
    r"\b(taskkill\s+[^\n]*/im|stop-process\s+[^\n]*-name|killall|pkill)\b", re.IGNORECASE)
STOP_PROCESS = re.compile(r"\bstop-process\b", re.IGNORECASE)
HAS_PID = re.compile(r"(-id\s+\d+|/pid\s+\d+)", re.IGNORECASE)

PIPE_TO_SHELL = re.compile(
    r"\b(curl|wget|iwr|invoke-webrequest)\b[^|]*\|\s*(sh|bash|zsh|iex|invoke-expression|powershell|pwsh)\b",
    re.IGNORECASE)

# No trailing \b here: these patterns end on ':' or '=', and a word boundary
# after a non-word character never matches, which silently defeated the rule.
FORMAT_OR_WIPE = re.compile(
    r"(\bformat\s+[a-z]:|\bdiskpart\b|\bmkfs|\bdd\s+if=|\bcipher\s+/w|\bvssadmin\s+delete)",
    re.IGNORECASE)

# Heads that are never legitimate for this tool to run, as a backstop for
# anything the pattern rules above miss.
BLOCKED_HEADS = {"format", "diskpart", "dd", "mkfs", "fdisk", "vssadmin", "cipher", "shutdown"}

REGISTRY_WRITE = re.compile(r"\b(reg\s+(add|delete)|remove-itemproperty|set-itemproperty)\b", re.IGNORECASE)

ENCODED_PS = re.compile(r"-(?:e|ec|enc|encoded|encodedcommand)\s+([A-Za-z0-9+/=]{16,})", re.IGNORECASE)

SEPARATORS = re.compile(r"(?:&&|\|\||;|\n|\|)")


def _split_commands(command: str) -> list:
    """Split on separators, respecting quotes so a semicolon inside a string
    doesn't create a phantom second command."""
    parts, buf, quote, i = [], [], None, 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        two = command[i:i + 2]
        if two in ("&&", "||"):
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in ";\n|":
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _decode_encoded(command: str, depth: int = 0) -> list:
    """Return any base64 PowerShell payloads hidden in the command."""
    if depth >= MAX_UNWRAP_DEPTH:
        return []
    found = []
    for blob in ENCODED_PS.findall(command):
        for encoding in ("utf-16-le", "utf-8"):
            try:
                decoded = base64.b64decode(blob + "=" * (-len(blob) % 4)).decode(encoding)
            except (ValueError, UnicodeDecodeError):
                continue
            if decoded.isprintable() or "\n" in decoded:
                found.append(decoded)
                found.extend(_decode_encoded(decoded, depth + 1))
                break
    return found


def _head_of(part: str):
    try:
        tokens = shlex.split(part, posix=False)
    except ValueError:
        tokens = part.split()
    if not tokens:
        return "", []
    return tokens[0].strip("'\"").lower().lstrip("./\\"), tokens[1:]


def _judge_one(part: str) -> SafetyResult:
    lowered = part.lower()

    if FORMAT_OR_WIPE.search(lowered):
        return SafetyResult(Verdict.BLOCKED, "formats or wipes storage", part)

    if PIPE_TO_SHELL.search(lowered):
        return SafetyResult(Verdict.BLOCKED, "pipes a download straight into a shell", part)

    if KILL_BY_NAME.search(lowered):
        return SafetyResult(Verdict.BLOCKED,
                            "kills processes by name, which can hit unrelated programs", part)
    if STOP_PROCESS.search(lowered) and not HAS_PID.search(lowered):
        return SafetyResult(Verdict.BLOCKED,
                            "stops processes without naming a specific PID", part)

    if RECURSIVE_DELETE.search(lowered):
        recursive = bool(RECURSIVE_FLAG.search(lowered))
        forced = bool(FORCE_FLAG.search(lowered))
        if DESTRUCTIVE_ROOTS.search(part):
            return SafetyResult(Verdict.BLOCKED,
                                "deletes from a drive root, home or system directory", part)
        if recursive or forced:
            return SafetyResult(Verdict.NEEDS_CONSENT, "recursive or forced delete", part)
        return SafetyResult(Verdict.NEEDS_CONSENT, "deletes files", part)

    if REGISTRY_WRITE.search(lowered):
        return SafetyResult(Verdict.NEEDS_CONSENT, "writes to the registry", part)

    head, rest = _head_of(part)
    if not head:
        return SafetyResult(Verdict.NEEDS_CONSENT, "empty or unparsable command", part)

    if head in BLOCKED_HEADS:
        return SafetyResult(Verdict.BLOCKED, f"'{head}' can destroy storage or force a shutdown", part)

    if head in MUTATING_SUBCOMMANDS:
        sub = next((token.lower() for token in rest if not token.startswith("-")), "")
        if sub in MUTATING_SUBCOMMANDS[head]:
            return SafetyResult(Verdict.NEEDS_CONSENT, f"{head} {sub} changes this machine", part)
        return SafetyResult(Verdict.READ_ONLY, f"{head} query", part)

    if head in READ_ONLY_HEADS:
        return SafetyResult(Verdict.READ_ONLY, "reads only", part)

    return SafetyResult(Verdict.NEEDS_CONSENT, f"'{head}' isn't a recognised safe command", part)


def analyze(command: str) -> SafetyResult:
    """Judge a whole command line. The strictest verdict among its parts wins."""
    if not (command or "").strip():
        return SafetyResult(Verdict.BLOCKED, "empty command", command)

    # Pipeline-shaped attacks have to be judged before the line is split, since
    # splitting on '|' is exactly what would hide them.
    whole = command.lower()
    if PIPE_TO_SHELL.search(whole):
        return SafetyResult(Verdict.BLOCKED, "pipes a download straight into a shell", command)
    if KILL_BY_NAME.search(whole):
        return SafetyResult(Verdict.BLOCKED,
                            "kills processes by name, which can hit unrelated programs", command)

    segments = _split_commands(command)
    payloads = _decode_encoded(command)
    for payload in payloads:
        if PIPE_TO_SHELL.search(payload.lower()) or KILL_BY_NAME.search(payload.lower()):
            return SafetyResult(Verdict.BLOCKED, "encoded command hides a destructive pipeline", command)
        segments.extend(_split_commands(payload))

    worst = SafetyResult(Verdict.READ_ONLY, "reads only", command)
    rank = {Verdict.READ_ONLY: 0, Verdict.NEEDS_CONSENT: 1, Verdict.BLOCKED: 2}
    for segment in segments:
        result = _judge_one(segment)
        if rank[result.verdict] > rank[worst.verdict]:
            worst = result
    worst.command = command
    return worst
