"""Offline keyword-heuristic stand-in for the LLM.

Only for tests, CI and key-less demos (LLM_PROVIDER=mock). It is NOT a model: its accuracy says
nothing about GPT-4.1 mini, and every record it produces carries model="mock-heuristic".
"""

from __future__ import annotations

import json
import re

from .llm import LLMResponse
from .prompts import CLOSE_TAG, OPEN_TAG

_FRAUD = re.compile(
    r"unauthori[sz]ed|fraud|phishing|scam|stolen|lost my (?:wallet|card)|did not (?:make|do|authori)|"
    r"never (?:made|authori)|hacked|compromised|without my (?:consent|knowledge)|ombudsman|police|legal action",
    re.I,
)
_CHURN = re.compile(r"clos(?:e|ing) my (?:savings |salary |current )?account|switch(?:ing)? (?:to|banks)|leav(?:e|ing) (?:your|the) bank|taking my business", re.I)
_MONEY_STUCK = re.compile(r"debited|charged twice|deducted|not (?:been )?credited|stuck|frozen|blocked", re.I)
_NEGATIVE = re.compile(r"not happy|disappoint|pathetic|terrible|worst|angry|unacceptable|crash|fail|declin|delay|still not|no reply|complain|unfair|disgusting|problem|issue", re.I)
_POSITIVE = re.compile(r"thank|great|excellent|love|happy with|smooth|impressed|appreciate|wonderful", re.I)


class MockClient:
    model = "mock-heuristic"

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        text = _extract(messages[-1]["content"])
        return LLMResponse(
            content=json.dumps(self._classify(text)),
            prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
            completion_tokens=120,
            latency_ms=1,
            model=self.model,
        )

    @staticmethod
    def _classify(text: str) -> dict:
        fraud, churn = bool(_FRAUD.search(text)), bool(_CHURN.search(text))
        stuck, neg, pos = bool(_MONEY_STUCK.search(text)), bool(_NEGATIVE.search(text)), bool(_POSITIVE.search(text))

        if fraud:
            urgency, category, team = "Critical", "Fraud & Security", "Risk"
        elif churn or stuck:
            urgency = "High"
            category = "Card & Payments" if stuck else "Branch & Service"
            team = "Retention" if churn else "Payments"
        elif neg:
            urgency, category, team = "Medium", "Digital Banking" if re.search(r"app|login|net ?banking", text, re.I) else "Branch & Service", "Support"
        else:
            urgency, category, team = "Low", "Other", "Support"

        if fraud or churn or stuck:
            sentiment = "Negative"
        elif neg and pos:
            sentiment = "Neutral"
        elif neg:
            sentiment = "Negative"
        elif pos:
            sentiment = "Positive"
        else:
            sentiment = "Neutral"
        return {
            "sentiment": sentiment,
            "urgency": urgency,
            "category": category,
            "key_issues": [text[:60]] if text else [],
            "recommended_action": "Review the message and follow up with the customer.",
            "suggested_team": team,
            "escalate_to_human": urgency == "Critical" or churn,
            "confidence": 0.9 if (fraud or churn or pos) else 0.75,
            "reasoning_summary": "Keyword heuristic (mock provider, not an LLM).",
        }


def _extract(user_content: str) -> str:
    start, end = user_content.find(OPEN_TAG), user_content.rfind(CLOSE_TAG)
    if start != -1 and end > start:
        return user_content[start + len(OPEN_TAG) : end].strip()
    return user_content.strip()
