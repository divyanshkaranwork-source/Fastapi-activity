"""Stage 5: deterministic business rules applied on top of the validated model output.

The LLM suggests; these rules decide. They are plain code so they are auditable and testable,
and every override is recorded in `routing_reasons`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import FeedbackAnalysis

# urgency -> (ticket priority, SLA minutes). Mirrors the policy text injected into the prompt.
PRIORITY = {
    "Critical": ("P1", 15),
    "High": ("P2", 60),
    "Medium": ("P3", 8 * 60),
    "Low": ("P4", 48 * 60),
}

_CHURN = re.compile(
    r"\b(?:clos(?:e|ing)\s+(?:my|the|all my)\s+(?:\w+\s+){0,2}account"
    r"|switch(?:ing)?\s+(?:to\s+(?:another|a different|some other)\s+bank|banks)"
    r"|mov(?:e|ing)\s+(?:my\s+(?:money|account|salary|business)|to\s+(?:another|a different)\s+bank)"
    r"|leav(?:e|ing)\s+(?:your|the)\s+bank"
    r"|take\s+my\s+(?:business|salary(?:\s+account)?|account)\s+elsewhere"
    r"|(?:never|won'?t|will not)\s+(?:bank|use)\s+(?:with|your)\b)",
    re.IGNORECASE,
)

_FRAUD_WORDS = re.compile(
    r"unauthori[sz]ed|fraud|phishing|scam|stolen|hacked|compromised|did\s*n[o']t\s+(?:make|do|authori[sz]e)|"
    r"without\s+my\s+(?:consent|knowledge|permission)",
    re.IGNORECASE,
)


def has_churn_signal(text: str) -> bool:
    return bool(_CHURN.search(text))


@dataclass
class RoutingDecision:
    analysis: FeedbackAnalysis  # final analysis after rule overrides
    routed_to: str
    reasons: list[str] = field(default_factory=list)
    needs_ticket: bool = True
    priority: str = "P4"
    sla_minutes: int = 48 * 60


def apply_business_rules(
    analysis: FeedbackAnalysis,
    *,
    text: str,
    flags: list[str],
    needs_human_review: bool,
    confidence_threshold: float,
) -> RoutingDecision:
    reasons: list[str] = []
    urgency = analysis.urgency
    team = analysis.suggested_team
    escalate = analysis.escalate_to_human

    churn = analysis.sentiment == "Negative" and has_churn_signal(text)

    # 1. Fraud / security at High or Critical urgency goes straight to Risk.
    if analysis.category == "Fraud & Security" and urgency in ("High", "Critical"):
        if team != "Risk":
            reasons.append("Fraud/security issue at High/Critical urgency routed to Risk")
        team = "Risk"
    # 2. Negative sentiment with churn signals goes to Retention (never overrides Risk).
    elif churn:
        if urgency in ("Low", "Medium"):
            urgency = "High"
            reasons.append("Explicit churn threat raised urgency to High")
        if team != "Retention":
            reasons.append("Negative sentiment with churn signal routed to Retention")
        team = "Retention"

    # 3. Escalation to a human.
    def escalate_because(reason: str) -> None:
        nonlocal escalate
        if not escalate:
            reasons.append(reason)
        escalate = True

    if urgency == "Critical":
        escalate_because("Critical urgency requires human escalation")
    if churn:
        escalate_because("Churn threat requires human escalation")
    if analysis.confidence < confidence_threshold:
        escalate_because(f"Confidence {analysis.confidence:.2f} below threshold {confidence_threshold:.2f}")
    if "prompt_injection_suspected" in flags:
        escalate_because("Possible prompt-injection attempt in the message")
    if needs_human_review:
        escalate_because("Automated analysis failed; manual review required")

    final = analysis.model_copy(update={"urgency": urgency, "suggested_team": team, "escalate_to_human": escalate})
    priority, sla = PRIORITY[urgency]
    # No ticket for happy customers with nothing to do.
    needs_ticket = not (final.sentiment == "Positive" and urgency == "Low" and not escalate)
    return RoutingDecision(
        analysis=final,
        routed_to=team,
        reasons=reasons,
        needs_ticket=needs_ticket,
        priority=priority,
        sla_minutes=sla,
    )


def fallback_analysis(text: str, reason: str) -> FeedbackAnalysis:
    """Safe record used when the LLM output cannot be obtained or validated.

    A tiny keyword safety net keeps obvious fraud reports from landing in a generic queue.
    """
    fraud = bool(_FRAUD_WORDS.search(text))
    return FeedbackAnalysis(
        sentiment="Negative" if fraud else "Neutral",
        urgency="High" if fraud else "Medium",
        category="Fraud & Security" if fraud else "Other",
        key_issues=["Automated analysis unavailable"] + (["Possible fraud keywords detected"] if fraud else []),
        recommended_action="Route to a human agent for manual review.",
        suggested_team="Risk" if fraud else "Support",
        escalate_to_human=True,
        confidence=0.0,
        reasoning_summary=f"Needs human review: {reason}"[:200],
    )
