import json

import pytest
from pydantic import ValidationError

from app.prompts import FEW_SHOT_EXAMPLES, build_messages, correction_message
from app.schema import AnalyzeRequest, FeedbackAnalysis
from tests.conftest import good_json


def test_valid_payload():
    a = FeedbackAnalysis.model_validate_json(good_json())
    assert a.urgency == "Medium"


def test_canonicalises_case_and_punctuation():
    a = FeedbackAnalysis.model_validate_json(
        good_json(sentiment="negative", urgency=" HIGH ", category="fraud / security", suggested_team="risk")
    )
    assert (a.sentiment, a.urgency, a.category, a.suggested_team) == ("Negative", "High", "Fraud & Security", "Risk")


@pytest.mark.parametrize(
    "override",
    [
        {"category": "Weather"},
        {"urgency": "Urgent"},
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"key_issues": list("abcdef")},
        {"reasoning_summary": "x" * 201},
        {"reasoning_summary": ""},
        {"escalate_to_human": "maybe"},
        {"suggested_team": "Marketing"},
    ],
)
def test_invalid_values_rejected(override):
    with pytest.raises(ValidationError):
        FeedbackAnalysis.model_validate_json(good_json(**override))


def test_key_issues_string_is_wrapped_and_blank_dropped():
    assert FeedbackAnalysis.model_validate_json(good_json(key_issues="one")).key_issues == ["one"]
    assert FeedbackAnalysis.model_validate_json(good_json(key_issues=["a", " ", ""])).key_issues == ["a"]


def test_extra_keys_ignored():
    data = json.loads(good_json())
    data["notes"] = "extra"
    assert FeedbackAnalysis.model_validate(data)


def test_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="   ")


def test_few_shot_examples_are_valid_against_schema():
    assert len(FEW_SHOT_EXAMPLES) == 6
    for _, payload in FEW_SHOT_EXAMPLES:
        FeedbackAnalysis.model_validate_json(payload)


def test_prompt_modes():
    zero = build_messages("hello", "zero_shot")
    few = build_messages("hello", "few_shot")
    full = build_messages("hello", "full")
    assert len(zero) == 2 and zero[-1]["content"].startswith("<customer_feedback>")
    assert len(few) == len(full) == 2 + 2 * len(FEW_SHOT_EXAMPLES)
    assert "senior customer-experience analyst" in few[0]["content"]
    assert "ESCALATION POLICY" not in few[0]["content"]
    assert "ESCALATION POLICY" in full[0]["content"]
    assert "ESCALATION POLICY" not in zero[0]["content"] and "analyst" not in zero[0]["content"]
    assert all("JSON" in m[0]["content"] for m in (zero, few, full))  # required by JSON mode


def test_correction_message_contains_error():
    assert "confidence" in correction_message("confidence: too big")["content"]
