"""Pydantic models: the LLM output contract plus API request/response shapes."""

from __future__ import annotations

import re
from typing import Literal, Optional, get_args

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["Positive", "Neutral", "Negative"]
Urgency = Literal["Low", "Medium", "High", "Critical"]
Category = Literal[
    "Card & Payments",  # cards, UPI, IMPS/NEFT/RTGS, failed or double debits, refunds
    "Fraud & Security",  # unauthorized activity, phishing, lost/stolen cards, account takeover
    "Account Services",  # KYC, statements, cheque books, nominations, salary/savings accounts
    "Loans & Credit",  # loans, EMIs, credit limits, collections, foreclosure
    "Fees & Charges",  # annual/overdraft/processing fees, penalties
    "Digital Banking",  # mobile app and net-banking functionality
    "Branch & Service",  # branch staff, call centre, wait times, complaint handling
    "Other",
]
Team = Literal["Support", "Payments", "Tech", "Risk", "Retention", "Lending"]
Mode = Literal["zero_shot", "few_shot", "full"]

SENTIMENTS = get_args(Sentiment)
URGENCIES = get_args(Urgency)
CATEGORIES = get_args(Category)
TEAMS = get_args(Team)
MODES = get_args(Mode)


def _canonicaliser(options: tuple[str, ...]):
    """Map 'fraud / security', 'NEGATIVE', ' high ' ... onto the canonical literal."""

    def squash(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    table = {squash(o): o for o in options}

    def canon(value):
        if isinstance(value, str):
            return table.get(squash(value), value.strip())
        return value

    return canon


_canon_sentiment = _canonicaliser(SENTIMENTS)
_canon_urgency = _canonicaliser(URGENCIES)
_canon_category = _canonicaliser(CATEGORIES)
_canon_team = _canonicaliser(TEAMS)


class FeedbackAnalysis(BaseModel):
    """What the LLM must return for each customer message."""

    sentiment: Sentiment
    urgency: Urgency
    category: Category
    key_issues: list[str] = Field(default_factory=list, max_length=5)
    recommended_action: str = Field(min_length=1)
    suggested_team: Team
    escalate_to_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1, max_length=200)

    # Tolerate harmless formatting drift (case, punctuation) but never invented values.
    @field_validator("sentiment", mode="before")
    @classmethod
    def _v_sentiment(cls, v):
        return _canon_sentiment(v)

    @field_validator("urgency", mode="before")
    @classmethod
    def _v_urgency(cls, v):
        return _canon_urgency(v)

    @field_validator("category", mode="before")
    @classmethod
    def _v_category(cls, v):
        return _canon_category(v)

    @field_validator("suggested_team", mode="before")
    @classmethod
    def _v_team(cls, v):
        return _canon_team(v)

    @field_validator("key_issues", mode="before")
    @classmethod
    def _v_issues(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return v

    @field_validator("recommended_action", "reasoning_summary", mode="before")
    @classmethod
    def _v_strip(cls, v):
        return v.strip() if isinstance(v, str) else v


class Ticket(BaseModel):
    ticket_id: str
    team: Team
    priority: Literal["P1", "P2", "P3", "P4"]
    sla_minutes: int
    due_at: str
    status: Literal["open", "resolved"] = "open"


class AnalysisRecord(BaseModel):
    """Full API result for one piece of feedback."""

    id: str
    created_at: str
    status: Literal["ok", "needs_human_review"]
    mode: Mode
    model: str
    masked_input: str
    analysis: FeedbackAnalysis
    routed_to: Team
    flags: list[str] = Field(default_factory=list)
    routing_reasons: list[str] = Field(default_factory=list)
    ticket: Optional[Ticket] = None
    retries: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str = Field(max_length=50_000)
    mode: Mode = "full"

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class BatchItem(BaseModel):
    row: int  # 1-based data row in the uploaded CSV
    status: Literal["ok", "needs_human_review", "skipped", "error"]
    record: Optional[AnalysisRecord] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    total: int
    ok: int
    needs_human_review: int
    skipped: int
    error: int
    results: list[BatchItem]
