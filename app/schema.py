from typing import Literal

from pydantic import BaseModel, Field


class FeedbackAnalysis(BaseModel):
    """Expected JSON output (from the problem statement). The LLM reply is validated against this."""

    sentiment: Literal["Positive", "Neutral", "Negative"]
    urgency: Literal["Low", "Medium", "High"]
    recommended_action: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
