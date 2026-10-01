"""Gemini (Vertex AI) chat adapter built on the google-genai SDK."""
from __future__ import annotations

import os
import atexit
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from src.config.catalog import EndpointSpec, ModelSpec
from src.inference.base import ChatModel
from src.inference.types import (
    ChatResult,
    ImagePart,
    InferenceError,
    Message,
    Part,
    PdfPart,
    TextPart,
    ToolCall,
    ToolSpec,
    Usage,
)

# Gemini does not always return call ids; ids we invent are never sent back to the API.
SYNTHETIC_ID_PREFIX = "gemini-call-"

_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER_LAST_REQUEST = 0.0
_SCHEDULER_METRICS = {"calls": 0, "429s": 0, "retries": 0, "malformed_function_calls": 0}


def _scheduler_enabled() -> bool:
    return os.getenv("EVISEARCH_VERTEX_SCHEDULER", "").strip().lower() in {"1", "true", "yes"}


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _scheduler_metrics_at_exit() -> None:
    if _scheduler_enabled():
        print(f"[gemini-scheduler] metrics={_SCHEDULER_METRICS}", flush=True)


atexit.register(_scheduler_metrics_at_exit)


# ---- Vertex authentication -----------------------------------------------------------------------

def vertex_api_key(endpoint: EndpointSpec) -> str:
    return os.getenv(endpoint.api_key_env or "VERTEX_API_KEY", "").strip()


def vertex_project(endpoint: EndpointSpec) -> str:
    return (os.getenv(endpoint.project_env or "GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID") or "").strip()


def vertex_location(endpoint: EndpointSpec) -> str:
    return (
        os.getenv(endpoint.location_env or "GOOGLE_CLOUD_LOCATION") or os.getenv("GCP_LOCATION") or endpoint.default_location
    ).strip()


def has_vertex_auth(endpoint: EndpointSpec) -> bool:
    return bool(vertex_api_key(endpoint) or vertex_project(endpoint))


def vertex_auth_error_message(endpoint: EndpointSpec) -> str:
    return (
        "Vertex AI Gemini authentication is not configured. "
        f"Set {endpoint.api_key_env or 'VERTEX_API_KEY'} for local development, or configure ADC/service-account "
        f"auth with {endpoint.project_env or 'GOOGLE_CLOUD_PROJECT'} and {endpoint.location_env or 'GOOGLE_CLOUD_LOCATION'}. "
        f"Current values: project={vertex_project(endpoint) or '<unset>'}, location={vertex_location(endpoint)}."
    )


def create_vertex_client(endpoint: EndpointSpec) -> genai.Client:
    timeout_s = float(os.getenv("EVISEARCH_VERTEX_TIMEOUT_S", str(endpoint.timeout_s)))
    max_retries = int(os.getenv("EVISEARCH_VERTEX_MAX_RETRIES", str(endpoint.max_retries)))
    http_options = types.HttpOptions(
        api_version="v1",
        timeout=int(timeout_s * 1000),
        retry_options=types.HttpRetryOptions(attempts=1 if _scheduler_enabled() else max_retries + 1),
    )
    api_key = vertex_api_key(endpoint)
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
    project = vertex_project(endpoint)
    if not project:
        raise InferenceError(vertex_auth_error_message(endpoint))
    return genai.Client(vertexai=True, project=project, location=vertex_location(endpoint), http_options=http_options)


# ---- Chat adapter ----------------------------------------------------------------------------------

class GeminiChat(ChatModel):
    def __init__(self, key: str, spec: ModelSpec, client: Any):
        super().__init__(key, spec)
        self.client = client

    def _chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        tool_choice: str,
        response_schema: Optional[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        system, contents = to_gemini_contents(messages)
        thinking_budget = os.getenv("EVISEARCH_GEMINI_THINKING_BUDGET")
        config = build_config(
            system,
            tools,
            tool_choice,
            response_schema,
            temperature,
            max_tokens,
            self.spec.thinking,
            int(thinking_budget) if thinking_budget else None,
        )
        max_retries = int(os.getenv("EVISEARCH_VERTEX_MAX_RETRIES", "0"))
        retry_delays = (5.0, 10.0, 20.0, 40.0, 60.0)
        attempt = 0
        while True:
            global _SCHEDULER_LAST_REQUEST
            try:
                if _scheduler_enabled():
                    with _SCHEDULER_LOCK:
                        interval = float(os.getenv("EVISEARCH_VERTEX_MIN_INTERVAL_S", "3"))
                        wait = interval - (time.monotonic() - _SCHEDULER_LAST_REQUEST)
                        if wait > 0:
                            time.sleep(wait)
                        _SCHEDULER_METRICS["calls"] += 1
                        try:
                            response = self.client.models.generate_content(
                                model=self.spec.name, contents=contents, config=config
                            )
                        finally:
                            _SCHEDULER_LAST_REQUEST = time.monotonic()
                else:
                    response = self.client.models.generate_content(model=self.spec.name, contents=contents, config=config)
                break
            except Exception as exc:
                if not (_scheduler_enabled() and _is_rate_limit_error(exc) and attempt < max_retries):
                    raise InferenceError(f"{self.key} ({self.spec.name}) request failed: {exc}") from exc
                _SCHEDULER_METRICS["429s"] += 1
                _SCHEDULER_METRICS["retries"] += 1
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                time.sleep(delay + random.uniform(0, delay * 0.25))
                attempt += 1
        return parse_gemini_response(response, self.key)


def build_config(
    system: Optional[str],
    tools: List[ToolSpec],
    tool_choice: str,
    response_schema: Optional[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    thinking: Optional[bool] = None,
    thinking_budget: Optional[int] = None,
) -> types.GenerateContentConfig:
    kwargs: Dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    elif thinking is False:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    if system:
        kwargs["system_instruction"] = system
    if tools:
        kwargs["tools"] = [
            types.Tool(
                function_declarations=[
                    {"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools
                ]
            )
        ]
        mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(tool_choice, "AUTO")
        kwargs["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
    elif response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


def to_gemini_contents(messages: List[Message]) -> Tuple[Optional[str], List[types.Content]]:
    system_texts: List[str] = []
    contents: List[types.Content] = []
    for message in messages:
        if message.role == "system":
            system_texts.append(message.text)
        elif message.role == "user":
            parts = [_part(p) for p in message.parts]
            if parts:
                contents.append(types.Content(role="user", parts=parts))
        elif message.role == "assistant":
            if isinstance(message.provider_state, types.Content):
                contents.append(message.provider_state)
                continue
            parts = [types.Part.from_text(text=message.text)] if message.text else []
            parts.extend(
                types.Part(function_call=types.FunctionCall(name=call.name, args=call.arguments, id=_api_id(call.id)))
                for call in message.tool_calls
            )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif message.role == "tool":
            # Vertex requires function responses to be the final parts of a user turn. Keep the logical
            # follow-up and page attachments in the same user turn, but serialize them before responses.
            parts = []
            for result in message.tool_results:
                parts.extend(_part(p) for p in result.attachments)
            parts.extend(_part(p) for p in message.parts)
            parts.extend(
                types.Part(
                    function_response=types.FunctionResponse(name=result.name, response=result.content, id=_api_id(result.call_id))
                )
                for result in message.tool_results
            )
            contents.append(types.Content(role="user", parts=parts))
        else:
            raise InferenceError(f"Unknown message role '{message.role}'")
    system = "\n\n".join(text for text in system_texts if text) or None
    return system, contents


def parse_gemini_response(response: Any, key: str) -> ChatResult:
    metadata = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
        output_tokens=(getattr(metadata, "candidates_token_count", 0) or 0) + (getattr(metadata, "thoughts_token_count", 0) or 0),
        api_calls=1,
        cached_input_tokens=getattr(metadata, "cached_content_token_count", 0) or 0,
    )
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise InferenceError(f"{key}: no candidates returned (prompt_feedback={getattr(response, 'prompt_feedback', None)})")

    candidate = candidates[0]
    content = candidate.content
    parts = list(content.parts or []) if content is not None else []
    finish = candidate.finish_reason
    finish_name = getattr(finish, "name", None) or (str(finish) if finish is not None else None)

    texts: List[str] = []
    calls: List[ToolCall] = []
    for part in parts:
        if part.function_call is not None:
            call = part.function_call
            calls.append(
                ToolCall(
                    id=call.id or f"{SYNTHETIC_ID_PREFIX}{uuid.uuid4().hex[:12]}",
                    name=call.name or "",
                    arguments=dict(call.args or {}),
                )
            )
        elif part.text and not part.thought:
            texts.append(part.text)
    if not parts and finish_name not in (None, "STOP", "MAX_TOKENS"):
        if finish_name == "MALFORMED_FUNCTION_CALL" and _scheduler_enabled():
            _SCHEDULER_METRICS["malformed_function_calls"] += 1
        raise InferenceError(f"{key}: empty response (finish_reason={finish_name})")

    text = "".join(texts).strip()
    message = Message(
        role="assistant",
        parts=[TextPart(text)] if text else [],
        tool_calls=calls,
        provider_state=content if parts else None,
    )
    return ChatResult(text=text, tool_calls=calls, usage=usage, message=message, model=key, finish_reason=finish_name)


def _part(part: Part) -> types.Part:
    if isinstance(part, TextPart):
        return types.Part.from_text(text=part.text)
    if isinstance(part, ImagePart):
        return types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
    if isinstance(part, PdfPart):
        return types.Part.from_bytes(data=part.data, mime_type="application/pdf")
    raise InferenceError(f"Unsupported part type {type(part).__name__}")


def _api_id(call_id: str) -> Optional[str]:
    return None if not call_id or call_id.startswith(SYNTHETIC_ID_PREFIX) else call_id
