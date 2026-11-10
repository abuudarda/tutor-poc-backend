"""
main.py
Production FastAPI Application Entrypoint for Socratica Studio Backend.

Exposes:
- POST /api/v1/parse-document       (NVIDIA `nvidia/nemotron-3.5-parse` + local document extraction)
- POST /api/v1/generate-lesson      (NVIDIA `nvidia/nemotron-3.5-bolt-e4b` curriculum synthesis)
- POST /api/v1/evaluate-checkpoint  (NVIDIA `nvidia/nemotron-3.5-bolt-e4b` semantic understanding gate)
- POST /api/v1/tutor-chat           (NVIDIA `nvidia/nemotron-3.5-bolt-e4b` live multimodal tutor + quick actions)
- POST /api/v1/tutor-chat/stream    (Optional SSE streaming endpoint for live tutor tokens)
- GET  /health                      (Service health & configuration readiness probe)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from core.prompts import build_tutor_chat_messages
from models.schemas import (
    EvaluateCheckpointRequest,
    EvaluateCheckpointResponse,
    GenerateLessonRequest,
    LessonResponse,
    ParseDocumentResponse,
    TutorChatRequest,
    TutorChatResponse,
)
from services.llm_service import llm_service
from services.parser_service import parser_service

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("socratica.main")

app = FastAPI(
    title="Socratica Studio AI Tutor Backend",
    description=(
        "Multimodal AI Tutor Backend powered by NVIDIA Nemotron 3.5 Parse "
        "(`nvidia/nemotron-3.5-parse`) and NVIDIA Nemotron 3.5 Bolt "
        "(`nvidia/nemotron-3.5-bolt-e4b`)."
    ),
    version="1.0.0",
)

# Configure CORS middleware so the Socratica Studio frontend can connect from any local or hosted origin
raw_origins = os.getenv("CORS_ORIGINS", "*")
allowed_origins = (
    ["*"]
    if raw_origins.strip() == "*"
    else [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Readiness Check
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Verify service health and NVIDIA_API_KEY configuration status."""
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    key_configured = bool(
        api_key and api_key != "nvapi-your-nvidia-api-key-here"
    )
    return {
        "status": "ok",
        "service": "socratica-studio-backend",
        "models": {
            "document_parser": "nvidia/nemotron-3.5-parse",
            "reasoning_llm": "nvidia/nemotron-3.5-bolt-e4b",
        },
        "nvidia_api_key_configured": key_configured,
    }


# ============================================================================
# Endpoint 1: POST /api/v1/parse-document
# ============================================================================

@app.post(
    "/api/v1/parse-document",
    response_model=ParseDocumentResponse,
    tags=["View 1: Setup Studio"],
    summary="Parse uploaded study document (.pdf, .png, .jpg, .txt, .md, .docx)",
)
async def parse_document(
    file: UploadFile = File(
        ...,
        description="Study material file uploaded in Socratica Studio (.pdf, .png, .jpg, .jpeg, .txt, .md, .docx).",
    ),
) -> ParseDocumentResponse:
    """
    Ingests a learner's uploaded study file:
    - `.pdf`, `.png`, `.jpg`/`.jpeg`: Renders pages to in-memory JPEG bytes and calls
      `nvidia/nemotron-3.5-parse`.
    - `.txt`, `.md`, `.docx`: Extracts structured text directly.
    """
    logger.info(
        "Parsing uploaded document: filename=%s, content_type=%s",
        file.filename,
        file.content_type,
    )
    return await parser_service.parse_uploaded_file(file)


# ============================================================================
# Endpoint 2: POST /api/v1/generate-lesson
# ============================================================================

@app.post(
    "/api/v1/generate-lesson",
    response_model=LessonResponse,
    tags=["View 1: Setup Studio"],
    summary="Synthesize an interactive multi-module lesson curriculum via nvidia/nemotron-3.5-bolt-e4b",
)
async def generate_lesson(
    payload: GenerateLessonRequest,
) -> LessonResponse:
    """
    Accepts the `GenerateLessonRequest` payload from Socratica Studio's Setup Studio
    and uses `nvidia/nemotron-3.5-bolt-e4b` (`max_tokens=4096`, `enable_thinking=True`)
    to generate a validated `LessonResponse` JSON curriculum.
    """
    logger.info(
        "Generating lesson for session_id=%s | file=%s | methodology=%s | simplicity=%d",
        payload.session_id,
        payload.source_material.file_name,
        payload.tutor_config.methodology,
        payload.tutor_config.simplicity_level,
    )
    return llm_service.generate_lesson(payload)


# ============================================================================
# Endpoint 3: POST /api/v1/evaluate-checkpoint
# ============================================================================

@app.post(
    "/api/v1/evaluate-checkpoint",
    response_model=EvaluateCheckpointResponse,
    tags=["View 2: Interactive Classroom"],
    summary="Evaluate learner understanding at 'The True Understanding Gate'",
)
async def evaluate_checkpoint(
    payload: EvaluateCheckpointRequest,
) -> EvaluateCheckpointResponse:
    """
    Evaluates the user's spoken or typed explanation at "The True Understanding Gate"
    semantically using `nvidia/nemotron-3.5-bolt-e4b` (`max_tokens=1024`, `enable_thinking=True`).
    """
    logger.info(
        "Evaluating checkpoint for session_id=%s | module_id=%s",
        payload.session_id,
        payload.module_id,
    )
    return llm_service.evaluate_checkpoint(payload)


# ============================================================================
# Endpoint 4: POST /api/v1/tutor-chat (JSON + Optional SSE Stream)
# ============================================================================

@app.post(
    "/api/v1/tutor-chat",
    response_model=TutorChatResponse,
    tags=["View 2: Interactive Classroom"],
    summary="Multimodal AI Tutor live reply and Quick-Action handler",
)
async def tutor_chat(
    payload: TutorChatRequest,
    stream: bool = Query(
        default=False,
        description="Set to true to receive Server-Sent Events (text/event-stream) token streaming.",
    ),
) -> Any:
    """
    Powers the Right Panel Multimodal AI Tutor & Live Comms (both freeform spoken/written
    questions and quick-action buttons: `explain_simpler`, `real_world_example`, `test_me`, `freeform`).
    """
    logger.info(
        "Tutor chat request: session_id=%s | module=%s | quick_action=%s | stream=%s",
        payload.session_id,
        payload.active_module_id,
        payload.quick_action,
        stream,
    )
    if stream:
        messages = build_tutor_chat_messages(payload)
        return StreamingResponse(
            llm_service.stream_raw_sse(messages=messages, max_tokens=1024),
            media_type="text/event-stream",
        )

    return llm_service.tutor_chat(payload)


@app.post(
    "/api/v1/tutor-chat/stream",
    tags=["View 2: Interactive Classroom"],
    summary="SSE Streaming endpoint for Live Multimodal Tutor responses",
)
async def tutor_chat_stream(payload: TutorChatRequest) -> StreamingResponse:
    """Dedicated Server-Sent Events (`text/event-stream`) endpoint for live tutor streaming."""
    messages = build_tutor_chat_messages(payload)
    return StreamingResponse(
        llm_service.stream_raw_sse(messages=messages, max_tokens=1024),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
