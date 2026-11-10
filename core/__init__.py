"""Core prompt library and pedagogical prompt builders for Socratica Studio."""

from .prompts import (
    CHECKPOINT_EVALUATION_SYSTEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    LESSON_SYNTHESIS_SYSTEM_PROMPT,
    LIVE_TUTOR_CHAT_SYSTEM_PROMPT,
    build_checkpoint_evaluation_messages,
    build_json_correction_messages,
    build_lesson_generation_messages,
    build_tutor_chat_messages,
)

__all__ = [
    "CHECKPOINT_EVALUATION_SYSTEM_PROMPT",
    "JSON_REPAIR_SYSTEM_PROMPT",
    "LESSON_SYNTHESIS_SYSTEM_PROMPT",
    "LIVE_TUTOR_CHAT_SYSTEM_PROMPT",
    "build_checkpoint_evaluation_messages",
    "build_json_correction_messages",
    "build_lesson_generation_messages",
    "build_tutor_chat_messages",
]
