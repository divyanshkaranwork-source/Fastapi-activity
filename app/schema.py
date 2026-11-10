from typing import Literal

from pydantic import BaseModel, Field


class FeedbackAnalysis(BaseModel):
    """The JSON the LLM must return (and the API returns)."""

    sentiment: Literal["Positive", "Neutral", "Negative"]
    urgency: Literal["Low", "Medium", "High", "Critical"]
    category: Literal[
        "Card & Payments", "Fraud & Security", "Account Services", "Loans & Credit",
        "Fees & Charges", "Digital Banking", "Branch & Service", "Other",
    ]
    key_issues: list[str] = Field(max_length=5)
    recommended_action: str
    suggested_team: Literal["Support", "Payments", "Tech", "Risk", "Retention", "Lending"]
    escalate_to_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(max_length=200)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
