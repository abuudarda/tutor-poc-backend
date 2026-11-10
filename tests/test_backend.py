"""
tests/test_backend.py
Unit test suite for the Socratica Studio Python FastAPI backend:
- Tests `extract_clean_json` with `<think>...</think>`, markdown fences, and malformed payloads.
- Tests Pydantic schemas (`models/schemas.py`).
- Tests prompt builders (`core/prompts.py`).
- Tests `LLMService.generate_structured_workflow` automatic retry logic when initial JSON fails.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from core.prompts import (
    build_checkpoint_evaluation_messages,
    build_lesson_generation_messages,
    build_tutor_chat_messages,
)
from models.schemas import (
    ActiveModuleContext,
    EvaluateCheckpointRequest,
    EvaluateCheckpointResponse,
    GenerateLessonRequest,
    LessonResponse,
    ParseDocumentResponse,
    SourceMaterial,
    TutorChatRequest,
    TutorChatResponse,
    TutorConfig,
)
from services.llm_service import LLMService, extract_clean_json


class TestExtractCleanJson(unittest.TestCase):
    """Verify `<think>...</think>` and code fence stripping in `extract_clean_json`."""

    def test_strips_think_tags_and_fences(self) -> None:
        raw = """<think>
Let me plan the tutor response carefully.
The user asked for a simpler explanation of gradient descent.
</think>
```json
{
  "spoken_reply": "Imagine walking down a foggy hill one careful step at a time.",
  "written_markdown": "**Gradient Descent**: Stepping downhill along the steepest slope.",
  "highlight_key_point_index": 1,
  "suggested_follow_ups": ["What if the step is too big?", "Give me a numeric example"]
}
```"""
        result = extract_clean_json(raw)
        self.assertEqual(result["highlight_key_point_index"], 1)
        self.assertIn("foggy hill", result["spoken_reply"])

    def test_handles_multiple_think_blocks_and_surrounding_text(self) -> None:
        raw = """<think>First thought</think>Here is your JSON:
        {"passed": true, "mastery_score": 95, "concepts_matched": ["overshoot"], "concepts_missed": [], "spoken_feedback": "Spot on!", "written_summary": "Great job.", "follow_up_hint": null}
        Hope this helps!"""
        result = extract_clean_json(raw)
        self.assertTrue(result["passed"])
        self.assertEqual(result["mastery_score"], 95)

    def test_raises_value_error_on_empty_or_non_json(self) -> None:
        with self.assertRaises(ValueError):
            extract_clean_json("<think>Only thinking, no JSON output</think>")


class TestSchemasAndPrompts(unittest.TestCase):
    """Verify Pydantic validation and prompt generation."""

    def test_lesson_generation_prompt_builder(self) -> None:
        req = GenerateLessonRequest(
            session_id="123e4567-e89b-12d3-a456-426614174000",
            source_material=SourceMaterial(
                file_name="chapter_4_neural_networks.pdf",
                content_type="application/pdf",
                raw_text_extracted="Neural networks adjust weights via backpropagation and gradient descent.",
            ),
            tutor_config=TutorConfig(
                education_level="undergraduate",
                simplicity_level=2,
                methodology="feynman",
                focus_areas=["core_intuition", "real_world_examples"],
                persona="warm_encouraging",
            ),
        )
        messages = build_lesson_generation_messages(req)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("feynman", messages[1]["content"])

    def test_automatic_retry_on_malformed_first_llm_output(self) -> None:
        service = LLMService(api_key="nvapi-test-key")

        bad_response = MagicMock()
        bad_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "<think>Thinking...</think>Not valid JSON at all!"
                    }
                }
            ]
        }

        good_payload = {
            "passed": True,
            "mastery_score": 90,
            "concepts_matched": ["overshoot"],
            "concepts_missed": [],
            "spoken_feedback": "You nailed the intuition behind overshooting the minimum.",
            "written_summary": "**Passed (90%)**: Correctly explained step-size overshoot.",
            "follow_up_hint": None,
        }
        good_response = MagicMock()
        good_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": f"<think>Repairing JSON</think>```json\n{json.dumps(good_payload)}\n```"
                    }
                }
            ]
        }

        req = EvaluateCheckpointRequest(
            session_id="uuid-1",
            module_id="mod_1",
            module_title="Learning Rate Dynamics",
            core_concept="A learning rate too high causes divergence.",
            question_asked="What happens if alpha is too large?",
            expected_concepts=["overshoot", "diverge"],
            user_response="It jumps right past the valley and bounces higher.",
            tutor_config=TutorConfig(),
        )

        with patch.object(
            service, "invoke_nemotron", side_effect=[bad_response, good_response]
        ) as mock_invoke:
            result = service.evaluate_checkpoint(req)
            self.assertEqual(mock_invoke.call_count, 2)
            self.assertIsInstance(result, EvaluateCheckpointResponse)
            self.assertTrue(result.passed)
            self.assertEqual(result.mastery_score, 90)


if __name__ == "__main__":
    unittest.main()
