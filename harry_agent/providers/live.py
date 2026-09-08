"""
live.py -- real providers with tool-calling.

Anthropic uses content blocks natively, so it is the internal format and needs
almost no translation. The OpenAI-shaped vendors (Azure OpenAI, OpenAI, Grok)
share one adapter because their chat-completions tool protocol is the same
apart from auth and the model field. Ollama gets a prompt-level fallback: local
models vary in whether they support tool schemas at all, so it asks for a JSON
decision and parses it rather than assuming.

urllib is used rather than each vendor's SDK to keep the dependency list as it
is; these are ordinary JSON POSTs.
"""

import json
import os
import re
import urllib.error
import urllib.request

from ..errors import ProviderError
from .base import STOP_END_TURN, STOP_LENGTH, STOP_TOOL_USE, Provider, ToolCall, Turn

TIMEOUT_S = 180


def _post(url: str, headers: dict, payload: dict, timeout: int = TIMEOUT_S) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:800]
        raise ProviderError(f"HTTP {error.code} from the model: {detail}") from error
    except urllib.error.URLError as error:
        raise ProviderError(f"Could not reach the model: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise ProviderError("The model returned something that wasn't JSON.") from error


class AnthropicProvider(Provider):
    name = "Claude"
    supports_vision = True

    def __init__(self, model: str = "", api_key: str = ""):
        self.model = model or os.environ.get("HARRY_CLAUDE_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, system, messages, tools=None, max_tokens=4096) -> Turn:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not available to this app process.")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        body = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            payload,
        )
        text, calls = [], []
        for block in body.get("content", []):
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block.get("id", ""), block.get("name", ""),
                                      block.get("input", {}) or {}))
        stop = body.get("stop_reason") or STOP_END_TURN
        if calls:
            stop = STOP_TOOL_USE
        elif stop == "max_tokens":
            stop = STOP_LENGTH
        return Turn("".join(text).strip(), calls, stop, body)


class OpenAICompatibleProvider(Provider):
    """Azure OpenAI, OpenAI and Grok all speak the same tool protocol."""

    def __init__(self, name: str, url: str, model: str, api_key: str, auth_header: str = "Authorization",
                 auth_prefix: str = "Bearer "):
        self.name = name
        self.url = url
        self.model = model
        self.api_key = api_key
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix

    @staticmethod
    def _to_openai(system: str, messages: list) -> list:
        """Content blocks in, chat-completions messages out."""
        converted = [{"role": "system", "content": system}]
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue
            text_parts, tool_calls, tool_results = [], [], []
            for block in content or []:
                kind = block.get("type")
                if kind == "text":
                    text_parts.append(block.get("text", ""))
                elif kind == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {"name": block.get("name"),
                                     "arguments": json.dumps(block.get("input", {}))},
                    })
                elif kind == "tool_result":
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": block.get("content", ""),
                    })
            if role == "assistant":
                entry = {"role": "assistant", "content": "\n".join(text_parts).strip() or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                converted.append(entry)
            else:
                if text_parts:
                    converted.append({"role": "user", "content": "\n".join(text_parts).strip()})
                converted.extend(tool_results)
        return converted

    @staticmethod
    def _tools_to_openai(tools: list) -> list:
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t.get("description", ""),
                              "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
                for t in tools or []]

    def complete(self, system, messages, tools=None, max_tokens=4096) -> Turn:
        if not self.api_key:
            raise ProviderError(f"No API key is available for {self.name}.")
        payload = {
            "model": self.model,
            "messages": self._to_openai(system, messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self._tools_to_openai(tools)
            payload["tool_choice"] = "auto"
        body = _post(self.url,
                     {self.auth_header: f"{self.auth_prefix}{self.api_key}",
                      "Content-Type": "application/json"},
                     payload)
        try:
            choice = body["choices"][0]
        except (KeyError, IndexError) as error:
            raise ProviderError("The model returned no choices.") from error
        message = choice.get("message", {})
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(call.get("id", ""), function.get("name", ""), arguments))
        stop = STOP_TOOL_USE if calls else (
            STOP_LENGTH if choice.get("finish_reason") == "length" else STOP_END_TURN)
        return Turn((message.get("content") or "").strip(), calls, stop, body)


class OllamaProvider(Provider):
    """Local models, with a prompt-level fallback when tools aren't supported.

    Rather than assume a given local model implements the tool API, this asks
    for a JSON decision and parses it. Slower to reason with, but it works on
    whatever the director happens to have pulled.
    """

    name = "Ollama"

    TOOL_INSTRUCTIONS = (
        "\n\nYou can use tools. To call one, reply with ONLY this JSON and nothing else:\n"
        '{"tool": "<tool_name>", "arguments": {...}}\n'
        "When the work is finished, reply with ONLY:\n"
        '{"done": true, "summary": "what you did"}\n'
    )

    def __init__(self, model: str = "", url: str = ""):
        self.model = model or os.environ.get("HARRY_OLLAMA_MODEL", "qwen2.5:7b")
        self.url = (url or os.environ.get("HARRY_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")

    def complete(self, system, messages, tools=None, max_tokens=4096) -> Turn:
        system_prompt = system
        if tools:
            catalogue = "\n".join(
                f"- {t['name']}: {t.get('description','')} args={json.dumps(t.get('input_schema',{}).get('properties',{}))}"
                for t in tools)
            system_prompt = f"{system}\n\nAVAILABLE TOOLS:\n{catalogue}{self.TOOL_INSTRUCTIONS}"

        flat = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                flat.append({"role": message["role"], "content": content})
                continue
            parts = []
            for block in content or []:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool result] {block.get('content','')}")
                elif block.get("type") == "tool_use":
                    parts.append(f"[called {block.get('name')} with {json.dumps(block.get('input', {}))}]")
            flat.append({"role": message["role"], "content": "\n".join(parts).strip()})

        body = _post(f"{self.url}/api/chat",
                     {"Content-Type": "application/json"},
                     {"model": self.model, "stream": False,
                      "messages": [{"role": "system", "content": system_prompt}] + flat})
        content = (body.get("message", {}) or {}).get("content", "") or ""
        return self._parse(content, body)

    @staticmethod
    def _parse(content: str, raw: dict) -> Turn:
        blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        try:
            decision = json.loads(blob)
        except json.JSONDecodeError:
            return Turn(content.strip(), [], STOP_END_TURN, raw)
        if isinstance(decision, dict) and decision.get("tool"):
            return Turn("", [ToolCall("local_call", decision["tool"], decision.get("arguments", {}) or {})],
                        STOP_TOOL_USE, raw)
        if isinstance(decision, dict) and decision.get("done"):
            return Turn(str(decision.get("summary", "Finished.")), [], STOP_END_TURN, raw)
        return Turn(content.strip(), [], STOP_END_TURN, raw)


def build_provider(provider_name: str, cfg=None) -> Provider:
    """Turn the Setup tab's provider name into a live provider."""
    provider_name = provider_name or "Claude"

    if provider_name == "Claude":
        return AnthropicProvider()

    if provider_name == "Azure OpenAI":
        import harry_advisor

        azure = harry_advisor._azure_settings()
        if not azure["endpoint"] or not azure["deployment"]:
            raise ProviderError("Azure endpoint and deployment aren't configured yet — set them in Setup.")
        return OpenAICompatibleProvider(
            "Azure OpenAI",
            f"{azure['endpoint']}/openai/v1/chat/completions",
            azure["deployment"],
            azure["key"],
        )

    if provider_name == "OpenAI":
        return OpenAICompatibleProvider(
            "OpenAI", "https://api.openai.com/v1/chat/completions",
            os.environ.get("HARRY_OPENAI_MODEL", "gpt-4.1"),
            os.environ.get("OPENAI_API_KEY", ""))

    if provider_name == "Grok":
        return OpenAICompatibleProvider(
            "Grok", "https://api.x.ai/v1/chat/completions",
            os.environ.get("HARRY_GROK_MODEL", "grok-3"),
            os.environ.get("XAI_API_KEY", ""))

    if provider_name == "Ollama":
        return OllamaProvider()

    raise ProviderError(f"Unknown provider '{provider_name}'.")
