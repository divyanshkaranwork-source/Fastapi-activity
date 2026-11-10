import json
import re

from pydantic import ValidationError

from . import llm
from .prompts import build_messages
from .schema import FeedbackAnalysis

MAX_CHARS = 4000


class InvalidModelOutput(Exception):
    """The model did not return valid JSON for the schema, even after one retry."""


def clean_text(text: str) -> str:
    """Tidy whitespace, mask basic private data, and cap the length before sending to the LLM."""
    text = re.sub(r"</?customer_feedback>", " ", text, flags=re.I)  # no forged prompt tags
    text = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "[EMAIL]", text)
    text = re.sub(r"\b(?:\d[ -]?){13,19}\b", "[CARD_NUMBER]", text)
    text = re.sub(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)", "[PHONE]", text)
    text = re.sub(r"(?<!\d)\d{9,18}(?!\d)", "[ACCOUNT_NUMBER]", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


def parse(reply: str) -> FeedbackAnalysis:
    """Schema validation: JSON text -> FeedbackAnalysis, or ValueError with a readable reason."""
    reply = re.sub(r"^```(?:json)?|```$", "", reply.strip()).strip()
    try:
        return FeedbackAnalysis.model_validate(json.loads(reply))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(str(exc)[:500]) from None


def analyze(text: str, call=None) -> FeedbackAnalysis:
    call = call or llm.call_llm
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("text is empty")
    messages = build_messages(cleaned)
    reply = call(messages)
    try:
        return parse(reply)
    except ValueError as err:
        # One retry, telling the model what was wrong.
        messages += [
            {"role": "assistant", "content": reply[:2000]},
            {"role": "user", "content": f"That reply was invalid: {err}\nReturn the corrected JSON only."},
        ]
        try:
            return parse(call(messages))
        except ValueError as err2:
            raise InvalidModelOutput(str(err2)) from None
