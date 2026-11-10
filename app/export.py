"""CSV export of analysis records (shared by the API and the dashboard)."""

from __future__ import annotations

import csv
import io
from typing import Iterable

from .schema import AnalysisRecord

COLUMNS = [
    "id", "created_at", "status", "sentiment", "urgency", "category", "suggested_team", "routed_to",
    "escalate_to_human", "confidence", "recommended_action", "key_issues", "reasoning_summary",
    "ticket_id", "priority", "due_at", "ticket_status", "flags", "corrected", "model", "cost_usd", "masked_input",
]


def _safe(value) -> str:
    """Neutralise spreadsheet formula injection (cells starting with = + - @)."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def records_to_csv(records: Iterable[AnalysisRecord]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    for r in records:
        a, t = r.analysis, r.ticket
        writer.writerow(
            [_safe(v) for v in (
                r.id, r.created_at, r.status, a.sentiment, a.urgency, a.category, a.suggested_team, r.routed_to,
                a.escalate_to_human, a.confidence, a.recommended_action, "; ".join(a.key_issues),
                a.reasoning_summary, t.ticket_id if t else "", t.priority if t else "", t.due_at if t else "",
                t.status if t else "", ", ".join(r.flags), r.corrected, r.model, r.cost_usd, r.masked_input,
            )]
        )
    return buf.getvalue()
