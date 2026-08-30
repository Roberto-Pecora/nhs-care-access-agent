"""Model adapters with a single tool-calling contract.

OpenAI-compatible endpoints cover llama.cpp, vLLM, and many hosted providers.
Gemini uses its native function-calling protocol so tool-call IDs and thought
signatures are preserved correctly across turns.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Protocol

from .types import ChatMessage, ModelReply, ToolCall, ToolDefinition


class ModelProvider(Protocol):
    model_name: str

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelReply: ...


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleModel:
    """Tool calling for local and hosted OpenAI-compatible APIs."""

    def __init__(self, model_name: str, base_url: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelReply:
        payload = {
            "model": self.model_name,
            "messages": [_openai_message(message) for message in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ],
            "tool_choice": "auto",
            "temperature": 0,
        }
        response = await _post_json(self._url, payload, self._api_key)
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"Malformed OpenAI-compatible response: {response}") from error

        calls = tuple(_openai_tool_call(item) for item in message.get("tool_calls", []))
        provider_message = {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls", []),
        }
        usage = response.get("usage", {})
        return ModelReply(
            content=message.get("content") or "",
            tool_calls=calls,
            provider_payload=provider_message,
            model_name=response.get("model", self.model_name),
            usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
        )


class GeminiModel:
    """Gemini native function calling using the public GenerateContent endpoint."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self._api_key = api_key

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelReply:
        system_messages = [message.content for message in messages if message.role == "system"]
        payload: dict[str, Any] = {
            "contents": [_gemini_content(message) for message in messages if message.role != "system"],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                        for tool in tools
                    ]
                }
            ],
            "generationConfig": {"temperature": 0},
        }
        if system_messages:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_messages)}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        response = await _post_json(url, payload, self._api_key, header_name="x-goog-api-key")
        try:
            content = response["candidates"][0]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"Malformed Gemini response: {response}") from error

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for part in content.get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
            function_call = part.get("functionCall")
            if function_call:
                calls.append(
                    ToolCall(
                        id=function_call.get("id") or uuid.uuid4().hex,
                        name=function_call["name"],
                        arguments=dict(function_call.get("args", {})),
                    )
                )
        usage = response.get("usageMetadata", {})
        return ModelReply(
            content="\n".join(text_parts).strip(),
            tool_calls=tuple(calls),
            provider_payload=content,
            model_name=response.get("modelVersion", self.model_name),
            usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
        )


class AnthropicModel:
    """Claude tool use via the native Messages API (raw HTTP, no SDK dependency).

    Kept parallel to :class:`GeminiModel` so the adapter surface stays uniform;
    the whole point of the project is one agent loop over interchangeable models.
    """

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self._api_key = api_key

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition]
    ) -> ModelReply:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        # `temperature` is deprecated on current Claude models (Sonnet 5, Opus
        # 4.7+) and returns a 400, so it is deliberately omitted here.
        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [_anthropic_message(m) for m in messages if m.role != "system"],
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
        response = await _post_json(
            "https://api.anthropic.com/v1/messages", payload, None, extra_headers=headers
        )
        blocks = response.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(f"Malformed Anthropic response: {response}")

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.get("id") or uuid.uuid4().hex,
                        name=block["name"],
                        arguments=dict(block.get("input", {})),
                    )
                )
        usage = response.get("usage", {})
        return ModelReply(
            content="\n".join(p for p in text_parts if p).strip(),
            tool_calls=tuple(calls),
            provider_payload={"role": "assistant", "content": blocks},
            model_name=response.get("model", self.model_name),
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )


def _anthropic_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.provider_payload:
        # Verbatim replay keeps tool_use ids aligned with the tool_result blocks.
        return dict(message.provider_payload)
    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
            ],
        }
    if message.role == "assistant" and message.tool_calls:
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        return {"role": "assistant", "content": content}
    return {"role": message.role, "content": message.content}


def _openai_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if message.role == "assistant" and message.provider_payload:
        return dict(message.provider_payload)
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return result


def _openai_tool_call(value: dict[str, Any]) -> ToolCall:
    try:
        function = value["function"]
        arguments = json.loads(function.get("arguments", "{}"))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ProviderError(f"Malformed tool call: {value}") from error
    if not isinstance(arguments, dict):
        raise ProviderError(f"Tool arguments must be an object: {value}")
    return ToolCall(
        id=value.get("id") or uuid.uuid4().hex,
        name=function["name"],
        arguments=arguments,
    )


def _gemini_content(message: ChatMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.provider_payload:
        # Returning this response verbatim preserves Gemini 3 thought signatures.
        return dict(message.provider_payload)
    if message.role == "tool":
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": message.tool_call_id,
                        "name": message.tool_name,
                        "response": {"result": _parse_json_or_text(message.content)},
                    }
                }
            ],
        }
    role = "model" if message.role == "assistant" else "user"
    parts: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
    for call in message.tool_calls:
        parts.append({"functionCall": {"id": call.id, "name": call.name, "args": call.arguments}})
    return {"role": role, "parts": parts}


def _parse_json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"text": value}


_RETRY_STATUS = frozenset({429, 500, 503})
_MAX_ATTEMPTS = 5
_MAX_RETRY_WAIT = 65.0  # cap a single wait so a run cannot stall indefinitely.


def _retry_delay(error: urllib.error.HTTPError, body: str, attempt: int) -> float:
    """Seconds to wait before a retry: the server's hint if given, else backoff.

    Google returns the hint as a ``retryDelay: "50s"`` field in the JSON body;
    other providers use the ``Retry-After`` header. Both are capped so a
    misreported delay cannot hang the run.
    """
    header = error.headers.get("Retry-After") if error.headers else None
    if header and header.strip().isdigit():
        return min(float(header.strip()), _MAX_RETRY_WAIT)
    match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', body)
    if match:
        return min(float(match.group(1)), _MAX_RETRY_WAIT)
    return min(2.0**attempt, _MAX_RETRY_WAIT)


async def _post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    *,
    header_name: str = "Authorization",
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers[header_name] = api_key if header_name != "Authorization" else f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    def send() -> dict[str, Any]:
        # Retry throttling and transient server errors. Free tiers rate-limit
        # aggressively (a whole eval run makes many calls), and the server tells
        # us how long to wait via Retry-After or a retryDelay in the body, which
        # is honoured over the exponential fallback when present.
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if error.code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_retry_delay(error, body, attempt))
                    continue
                raise ProviderError(f"Provider returned HTTP {error.code}: {body}") from error
            except urllib.error.URLError as error:
                raise ProviderError(f"Provider request failed: {error.reason}") from error
            if not isinstance(parsed, dict):
                raise ProviderError("Provider response was not a JSON object")
            return parsed
        raise ProviderError("Provider retries exhausted")  # unreachable: loop returns or raises

    return await asyncio.to_thread(send)
