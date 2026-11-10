"""
models/schemas.py
Strict Pydantic v2 Request and Response Contracts for Socratica Studio.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enumerations & Shared Pedagogical Types
# ============================================================================

class EducationLevel(str, Enum):
    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    PROFESSIONAL = "professional"


class Methodology(str, Enum):
    SOCRATIC = "socratic"
    FEYNMAN = "feynman"
    DIRECT = "direct"


class Persona(str, Enum):
    WARM_ENCOURAGING = "warm_encouraging"
    SOCRATIC_CHALLENGER = "socratic_challenger"
    RIGOROUS_PROFESSOR = "rigorous_professor"
    WITTY_MENTOR = "witty_mentor"
    CALM_EXPLAINER = "calm_explainer"


class QuickAction(str, Enum):
    EXPLAIN_SIMPLER = "explain_simpler"
    REAL_WORLD_EXAMPLE = "real_world_example"
    TEST_ME = "test_me"
    FREEFORM = "freeform"


class TutorConfig(BaseModel):
    """Pedagogical configuration chosen by the learner in View 1 (Setup Studio)."""

    education_level: str = Field(
        default="undergraduate",
        description="Target academic level (e.g., high_school, undergraduate, graduate).",
    )
    simplicity_level: int = Field(
        default=2,
        ge=1,
        le=5,
        description="1 = ELI5 / pure intuition, 3 = balanced conceptual, 5 = rigorous mathematical/technical depth.",
    )
    methodology: Literal["socratic", "feynman", "direct"] = Field(
        default="feynman",
        description="Teaching methodology governing narration and checkpoint style.",
    )
    focus_areas: Optional[List[str]] = Field(
        default_factory=lambda: ["core_intuition", "real_world_examples"],
        description="Specific learning priorities selected in Setup Studio.",
    )
    persona: str = Field(
        default="warm_encouraging",
        description="Voice and interpersonal style of the AI Tutor.",
    )


# ============================================================================
# Endpoint 1: POST /api/v1/parse-document
# ============================================================================

class ParseDocumentResponse(BaseModel):
    """Response returned after parsing an uploaded document or image."""

    file_name: str = Field(..., description="Original uploaded file name.")
    content_type: str = Field(..., description="MIME type of the uploaded file.")
    page_count: int = Field(..., ge=1, description="Total number of pages processed.")
    word_count: int = Field(..., ge=0, description="Total extracted word count.")
    estimated_reading_time_minutes: int = Field(
        ...,
        ge=1,
        description="Estimated reading time in minutes (at ~200 words per minute).",
    )
    raw_text_extracted: str = Field(
        ...,
        description="Complete concatenated text/markdown extracted from the document.",
    )


# ============================================================================
# Endpoint 2: POST /api/v1/generate-lesson
# ============================================================================

class SourceMaterial(BaseModel):
    """Uploaded study material metadata and extracted text."""

    file_name: str = Field(
        default="study_notes.txt",
        description="Name of the source document.",
    )
    content_type: str = Field(
        default="text/plain",
        description="MIME type of the source document.",
    )
    raw_text_extracted: str = Field(
        ...,
        min_length=10,
        description="Full or chunked text extracted from the uploaded material.",
    )


class GenerateLessonRequest(BaseModel):
    """Incoming payload from Socratica Studio's Setup Studio (View 1)."""

    session_id: str = Field(..., description="Unique client session UUID-v4 string.")
    source_material: SourceMaterial = Field(
        ...,
        description="Extracted text and metadata from the uploaded file.",
    )
    tutor_config: TutorConfig = Field(
        ...,
        description="Learner's pedagogical preferences.",
    )


class VisualAnalogy(BaseModel):
    """Side-by-side mental model comparison rendered on the classroom whiteboard."""

    target_concept: str = Field(
        ...,
        description="Technical concept name from the source text.",
    )
    everyday_parallel: str = Field(
        ...,
        description="Concrete everyday mental model comparison tailored to simplicity_level.",
    )


class UnderstandingCheck(BaseModel):
    """'The True Understanding Gate' checkpoint at the end of a module."""

    question_spoken: str = Field(
        ...,
        description="Feynman/Socratic checkpoint question spoken aloud to the user.",
    )
    expected_concepts: List[str] = Field(
        ...,
        min_length=1,
        description="Key conceptual ideas or mechanisms the learner should articulate.",
    )
    hint: str = Field(
        ...,
        description="Helpful nudge if the user struggles without spoiling the full answer.",
    )


class LessonModule(BaseModel):
    """Single interactive learning module in the Socratica Studio classroom."""

    module_id: str = Field(..., description="Sequential identifier such as 'mod_1'.")
    title: str = Field(..., description="Clear, engaging module title.")
    core_concept: str = Field(
        ...,
        description="The Big Idea (TL;DR) callout string summarizing the module's essence.",
    )
    spoken_narration: str = Field(
        ...,
        description=(
            "Natural spoken script for Web Speech TTS—conversational, clear, "
            "free of markdown symbols, and paced for audio listening."
        ),
    )
    visual_analogy: VisualAnalogy = Field(
        ...,
        description="Side-by-side technical concept vs. everyday mental model.",
    )
    key_points: List[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Concise visual bullets (highlighted sequentially while spoken_narration plays).",
    )
    understanding_check: UnderstandingCheck = Field(
        ...,
        description="Interactive voice/text checkpoint gating progression to the next module.",
    )


class LessonResponse(BaseModel):
    """Complete structured lesson curriculum generated by nvidia/nemotron-3.5-bolt-e4b."""

    lesson_id: str = Field(..., description="Unique lesson identifier (e.g., 'lesson-001').")
    chapter_title: str = Field(..., description="Overarching title of the synthesized lesson.")
    estimated_duration_minutes: int = Field(
        ...,
        ge=1,
        description="Estimated minutes to complete all modules and checkpoints.",
    )
    welcome_spoken_intro: str = Field(
        ...,
        description="Warm conversational intro meant for Web Speech TTS when entering the classroom.",
    )
    modules: List[LessonModule] = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "Dynamically sized sequence of interactive pedagogical modules (typically 2 to 8+ modules "
            "based on source material length, conceptual density, and simplicity_level; never hardcoded to 4)."
        ),
    )


# ============================================================================
# Endpoint 3: POST /api/v1/evaluate-checkpoint
# ============================================================================

class EvaluateCheckpointRequest(BaseModel):
    """Incoming payload for evaluating a learner's answer at The True Understanding Gate."""

    session_id: str = Field(..., description="Active session UUID-v4 string.")
    module_id: str = Field(..., description="ID of the module being evaluated (e.g., 'mod_1').")
    module_title: str = Field(..., description="Title of the active module.")
    core_concept: str = Field(..., description="Core concept statement for the module.")
    question_asked: str = Field(..., description="The checkpoint question posed to the user.")
    expected_concepts: List[str] = Field(
        ...,
        description="Expected conceptual anchors or keywords for this checkpoint.",
    )
    user_response: str = Field(
        ...,
        min_length=1,
        description="Transcribed voice or typed answer from the user.",
    )
    tutor_config: TutorConfig = Field(
        ...,
        description="Pedagogical configuration governing tone and evaluation strictness.",
    )


class EvaluateCheckpointResponse(BaseModel):
    """LLM-generated semantic evaluation of the user's checkpoint explanation."""

    passed: bool = Field(
        ...,
        description="True if the learner demonstrated genuine conceptual understanding.",
    )
    mastery_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Numeric mastery score from 0 to 100.",
    )
    concepts_matched: List[str] = Field(
        default_factory=list,
        description="Expected concepts that the learner successfully captured (semantically or verbatim).",
    )
    concepts_missed: List[str] = Field(
        default_factory=list,
        description="Expected concepts that were absent or misunderstood.",
    )
    spoken_feedback: str = Field(
        ...,
        description=(
            "Natural TTS voice response praising what they got right or gently guiding "
            "them toward the missing piece without giving away the whole answer if passed is false."
        ),
    )
    written_summary: str = Field(
        ...,
        description="Concise markdown summary for the transcript bubble.",
    )
    follow_up_hint: Optional[str] = Field(
        default=None,
        description="Optional Socratic hint when passed is false; null when passed is true.",
    )


# ============================================================================
# Endpoint 4: POST /api/v1/tutor-chat
# ============================================================================

class ActiveModuleContext(BaseModel):
    """Whiteboard state of the currently displayed module in View 2."""

    title: str = Field(..., description="Active module title.")
    core_concept: str = Field(..., description="Active module Big Idea (TL;DR).")
    key_points: List[str] = Field(
        default_factory=list,
        description="List of visual key_points currently rendered on the whiteboard.",
    )


class ChatMessage(BaseModel):
    """Single message in the tutor conversation history."""

    role: Literal["assistant", "user", "system"] = Field(
        ...,
        description="Speaker role in the conversation.",
    )
    content: str = Field(..., description="Text content of the message.")


class TutorChatRequest(BaseModel):
    """Incoming payload from the Right Panel Multimodal AI Tutor & Live Comms."""

    session_id: str = Field(..., description="Active session UUID-v4 string.")
    active_module_id: str = Field(..., description="Currently active module ID (e.g., 'mod_1').")
    active_module_context: ActiveModuleContext = Field(
        ...,
        description="Context of the module currently on the user's whiteboard.",
    )
    tutor_config: TutorConfig = Field(
        ...,
        description="Active pedagogical settings.",
    )
    quick_action: Literal[
        "explain_simpler", "real_world_example", "test_me", "freeform"
    ] = Field(
        default="freeform",
        description="Quick action trigger or 'freeform' for open questions.",
    )
    user_message: str = Field(
        ...,
        min_length=1,
        description="What the user said via microphone, typed, or triggered via quick action.",
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns in the module's tutor conversation.",
    )


class TutorChatResponse(BaseModel):
    """Structured multimodal tutor reply synchronizing TTS speech, transcript markdown, and whiteboard highlights."""

    spoken_reply: str = Field(
        ...,
        description=(
            "Conversational, TTS-friendly explanation (zero markdown symbols, LaTeX, "
            "code fences, or punctuation that breaks speech synthesis)."
        ),
    )
    written_markdown: str = Field(
        ...,
        description="Richly formatted markdown version for the Synchronized Transcript Stream.",
    )
    highlight_key_point_index: Optional[int] = Field(
        default=None,
        description="0-based index of the key_point card on the whiteboard to highlight while speaking, or null.",
    )
    suggested_follow_ups: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="2-3 clickable follow-up questions tailored to the current discussion.",
    )

    @field_validator("highlight_key_point_index")
    @classmethod
    def validate_non_negative_index(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            return None
        return value
