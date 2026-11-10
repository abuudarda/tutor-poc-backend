"""
services/llm_service.py
NVIDIA Nemotron-3.5-Bolt-E4B Reasoning & Structured JSON Workflow Engine for Socratica Studio.

Responsibilities:
1. Calls `https://integrate.api.nvidia.com/v1/chat/completions` with model
   `nvidia/nemotron-3.5-bolt-e4b` and `chat_template_kwargs={"enable_thinking": True}`.
2. Implements `extract_clean_json(raw_text: str) -> dict` to strip `<think>...</think>`
   reasoning blocks, markdown code fences (```json ... ```), and leading/trailing chatter
   before parsing the JSON object.
3. Validates the extracted dictionary against the target Pydantic v2 schema.
4. Automatically retries once with a structured correction prompt (`build_json_correction_messages`)
   if JSON parsing or Pydantic validation fails on the first attempt.
5. Supports both synchronous JSON responses and Server-Sent Events (SSE) token streaming.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Generator, List, Type, TypeVar

import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from core.prompts import (
    build_checkpoint_evaluation_messages,
    build_json_correction_messages,
    build_lesson_generation_messages,
    build_tutor_chat_messages,
)
from models.schemas import (
    EvaluateCheckpointRequest,
    EvaluateCheckpointResponse,
    GenerateLessonRequest,
    LessonResponse,
    TutorChatRequest,
    TutorChatResponse,
)

load_dotenv()
logger = logging.getLogger("socratica.llm_service")

NEMOTRON_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-3.5-bolt-e4b"

T = TypeVar("T", bound=BaseModel)

# Pre-compiled regexes for stripping <think>...</think> and markdown code fences
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"^.*?<think>.*?(?=\{)", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE
)


def extract_clean_json(raw_text: str) -> Dict[str, Any]:
    """
    Robustly extract and parse a JSON object from an LLM response that may include
    `<think>...</think>` blocks, markdown code fences (```json ... ```), or conversational
    preamble/postscript text.

    Raises `ValueError` if no valid JSON object can be extracted.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned an empty response string.")

    # Step 1: Strip closed <think>...</think> blocks
    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()

    # Step 2: Handle stray closing </think> tag if opening tag was omitted
    if "</think>" in cleaned.lower():
        parts = re.split(r"</think>", cleaned, flags=re.IGNORECASE)
        cleaned = parts[-1].strip()

    # Step 3: Handle unclosed <think> tag followed by a JSON object
    if "<think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK_RE.sub("", cleaned).strip()

    # Step 4: Try direct JSON parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Step 5: Check for ```json ... ``` markdown code fences
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        fenced_candidate = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced_candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Step 6: Balanced-brace extraction for the outermost `{ ... }` JSON object
    first_brace = cleaned.find("{")
    if first_brace == -1:
        raise ValueError(
            f"No JSON object found after stripping <think> blocks. Cleaned text: {cleaned[:300]!r}"
        )

    depth = 0
    in_string = False
    escape = False
    end_brace = -1

    for idx in range(first_brace, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_brace = idx
                    break

    candidate = (
        cleaned[first_brace : end_brace + 1]
        if end_brace != -1
        else cleaned[first_brace:]
    )

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Expected a JSON object (dict), got {type(parsed).__name__}.")
    except json.JSONDecodeError as exc:
        # Step 7: Fallback to first `{` through last `}` if nested brace scan encountered unescaped quotes
        last_brace = cleaned.rfind("}")
        if last_brace > first_brace:
            fallback_candidate = cleaned[first_brace : last_brace + 1]
            try:
                parsed = json.loads(fallback_candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Failed to decode JSON payload: {exc}. Candidate snippet: {candidate[:300]!r}"
        ) from exc


class LLMService:
    """Client for `nvidia/nemotron-3.5-bolt-e4b` with thinking-mode JSON extraction and Pydantic validation."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.invoke_url = NEMOTRON_CHAT_URL
        self.model = NEMOTRON_MODEL

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("NVIDIA_API_KEY", "")
        return key.strip()

    # ------------------------------------------------------------------------
    # Core NVIDIA Chat Completions Invocation
    # ------------------------------------------------------------------------

    def invoke_nemotron(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> requests.Response:
        """
        Invoke `nvidia/nemotron-3.5-bolt-e4b` using the exact required NVIDIA NIM payload format.
        """
        nvidia_api_key = self._get_api_key()
        if not nvidia_api_key or nvidia_api_key == "nvapi-your-nvidia-api-key-here":
            raise RuntimeError(
                "NVIDIA_API_KEY environment variable is missing or set to placeholder."
            )

        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,  # 4096 for full lesson JSON generation, 1024 for chat/checkpoints
            "seed": seed,
            "stream": stream,
            "parallel_tool_calls": True,
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
        }

        response = requests.post(
            self.invoke_url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=120,
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _extract_message_content(api_json: Dict[str, Any]) -> str:
        """Extract assistant content from OpenAI-compatible NVIDIA chat completion response."""
        choices = api_json.get("choices", [])
        if not choices:
            raise ValueError(f"NVIDIA LLM response contained no choices: {api_json}")

        message = choices[0].get("message", {})
        content = message.get("content") or ""
        return content

    # ------------------------------------------------------------------------
    # Validated Structured JSON Execution + Automatic 1-Time Correction Retry
    # ------------------------------------------------------------------------

    def generate_structured_workflow(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        max_tokens: int = 4096,
    ) -> T:
        """
        Execute a non-streaming LLM call, strip `<think>...</think>`, parse JSON,
        and validate against `response_model`. If JSON parsing or Pydantic validation
        fails, automatically retry once with a correction prompt.
        """
        raw_content = ""
        first_error_msg = ""

        # Attempt 1: Primary invocation
        try:
            response = self.invoke_nemotron(
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
            )
            raw_content = self._extract_message_content(response.json())
            parsed_dict = extract_clean_json(raw_content)
            return response_model.model_validate(parsed_dict)

        except (ValueError, ValidationError, KeyError) as first_err:
            first_error_msg = str(first_err)
            logger.warning(
                "Initial JSON/Pydantic validation failed for %s (%s). Retrying once with correction prompt.",
                response_model.__name__,
                first_error_msg,
            )
        except requests.RequestException as http_err:
            logger.error("NVIDIA API HTTP error: %s", http_err)
            raise HTTPException(
                status_code=502,
                detail=f"Upstream NVIDIA LLM API request failed: {http_err}",
            ) from http_err
        except RuntimeError as config_err:
            raise HTTPException(
                status_code=500,
                detail=str(config_err),
            ) from config_err

        # Attempt 2: Automatic 1-time retry with correction prompt
        correction_messages = build_json_correction_messages(
            original_messages=messages,
            malformed_output=raw_content,
            validation_error=first_error_msg,
            schema_json=response_model.model_json_schema(),
        )

        try:
            retry_response = self.invoke_nemotron(
                messages=correction_messages,
                max_tokens=max_tokens,
                stream=False,
                temperature=0.4,  # Lower temperature on repair pass for deterministic schema adherence
            )
            retry_raw_content = self._extract_message_content(retry_response.json())
            retry_parsed_dict = extract_clean_json(retry_raw_content)
            return response_model.model_validate(retry_parsed_dict)

        except Exception as retry_err:
            logger.error(
                "Retry for %s failed after correction prompt: %s",
                response_model.__name__,
                retry_err,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"LLM output failed schema validation for {response_model.__name__} "
                    f"after automatic retry. Error: {retry_err}"
                ),
            ) from retry_err

    # ------------------------------------------------------------------------
    # SSE Streaming Helper (`text/event-stream`)
    # ------------------------------------------------------------------------

    def stream_raw_sse(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """
        Stream SSE chunks from `nvidia/nemotron-3.5-bolt-e4b` while filtering out
        `<think>...</think>` blocks and yielding a final validated `done` event.
        """
        response = self.invoke_nemotron(
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )

        accumulated = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk_obj = json.loads(data_str)
                delta = (
                    chunk_obj.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if delta:
                    accumulated += delta
                    yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"
            except json.JSONDecodeError:
                continue

        # Emit final cleaned JSON payload event once stream completes
        try:
            clean_dict = extract_clean_json(accumulated)
            yield f"event: complete\ndata: {json.dumps(clean_dict)}\n\n"
        except ValueError as err:
            yield f"event: error\ndata: {json.dumps({'error': str(err), 'raw': accumulated})}\n\n"

    # ------------------------------------------------------------------------
    # High-Level Socratica Studio Workflow Entrypoints
    # ------------------------------------------------------------------------

    def generate_lesson(self, request: GenerateLessonRequest) -> LessonResponse:
        """Workflow 1: Synthesize a multi-module interactive lesson (`max_tokens=4096`)."""
        messages = build_lesson_generation_messages(request)
        return self.generate_structured_workflow(
            messages=messages,
            response_model=LessonResponse,
            max_tokens=10256,
        )

    def evaluate_checkpoint(
        self, request: EvaluateCheckpointRequest
    ) -> EvaluateCheckpointResponse:
        """Workflow 2: Semantically evaluate a learner's checkpoint answer (`max_tokens=1024`)."""
        messages = build_checkpoint_evaluation_messages(request)
        return self.generate_structured_workflow(
            messages=messages,
            response_model=EvaluateCheckpointResponse,
            max_tokens=1024,
        )

    def tutor_chat(self, request: TutorChatRequest) -> TutorChatResponse:
        """Workflow 3: Generate a multimodal live tutor reply (`max_tokens=1024`)."""
        messages = build_tutor_chat_messages(request)
        return self.generate_structured_workflow(
            messages=messages,
            response_model=TutorChatResponse,
            max_tokens=10240,
        )


llm_service = LLMService()
