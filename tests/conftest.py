import json

import pytest

from feedbackiq.analyzer import Analyzer
from feedbackiq.config import Settings
from feedbackiq.llm import LLMResponse
from feedbackiq.storage import Store


def good_json(**overrides) -> str:
    base = {
        "sentiment": "Negative",
        "urgency": "Medium",
        "category": "Digital Banking",
        "key_issues": ["App crash"],
        "recommended_action": "Open a bug ticket.",
        "suggested_team": "Tech",
        "escalate_to_human": False,
        "confidence": 0.9,
        "reasoning_summary": "App fault with a workaround.",
    }
    base.update(overrides)
    return json.dumps(base)


class ScriptedClient:
    """Returns queued replies (str) or raises queued exceptions; records every call."""

    model = "scripted-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        item = self.replies.pop(0) if self.replies else good_json()
        if isinstance(item, Exception):
            raise item
        return LLMResponse(content=item, prompt_tokens=1000, completion_tokens=100, latency_ms=50, model=self.model)


@pytest.fixture
def settings(tmp_path):
    return Settings(provider="mock", db_path=str(tmp_path / "t.db"), batch_workers=2)


@pytest.fixture
def make_analyzer(settings):
    def _make(*replies):
        client = ScriptedClient(*replies)
        return Analyzer(client, settings, Store(settings.db_path)), client

    return _make
