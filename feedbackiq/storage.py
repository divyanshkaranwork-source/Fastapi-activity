"""SQLite persistence for analyses and tickets (upgrade path: swap this module for PostgreSQL).

Only the masked text is stored, never the customer's raw message.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .schema import AnalysisRecord, FeedbackAnalysis, Ticket

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL,
    masked_input TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    urgency TEXT NOT NULL,
    category TEXT NOT NULL,
    key_issues TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    suggested_team TEXT NOT NULL,
    escalate_to_human INTEGER NOT NULL,
    confidence REAL NOT NULL,
    reasoning_summary TEXT NOT NULL,
    routed_to TEXT NOT NULL,
    flags TEXT NOT NULL,
    routing_reasons TEXT NOT NULL,
    retries INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analyses(id),
    team TEXT NOT NULL,
    priority TEXT NOT NULL,
    sla_minutes INTEGER NOT NULL,
    due_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status, priority);
"""


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: safe across FastAPI worker threads.
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes
    def save(self, rec: AnalysisRecord) -> None:
        a = rec.analysis
        with self._conn() as c:
            c.execute(
                "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.id, rec.created_at, rec.status, rec.mode, rec.model, rec.masked_input,
                    a.sentiment, a.urgency, a.category, json.dumps(a.key_issues, ensure_ascii=False),
                    a.recommended_action, a.suggested_team, int(a.escalate_to_human), a.confidence,
                    a.reasoning_summary, rec.routed_to, json.dumps(rec.flags),
                    json.dumps(rec.routing_reasons, ensure_ascii=False), rec.retries, rec.latency_ms,
                    rec.prompt_tokens, rec.completion_tokens, rec.cost_usd, rec.error,
                ),
            )
            if rec.ticket:
                t = rec.ticket
                c.execute(
                    "INSERT INTO tickets (ticket_id, analysis_id, team, priority, sla_minutes, due_at, status, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (t.ticket_id, rec.id, t.team, t.priority, t.sla_minutes, t.due_at, t.status, rec.created_at),
                )

    def resolve_ticket(self, ticket_id: str, resolved_at: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE tickets SET status='resolved', resolved_at=? WHERE ticket_id=? AND status='open'",
                (resolved_at, ticket_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------- reads
    _SELECT = (
        "SELECT a.*, t.ticket_id, t.team AS t_team, t.priority, t.sla_minutes, t.due_at, t.status AS t_status "
        "FROM analyses a LEFT JOIN tickets t ON t.analysis_id = a.id"
    )

    def get(self, analysis_id: str) -> Optional[AnalysisRecord]:
        with self._conn() as c:
            row = c.execute(self._SELECT + " WHERE a.id = ?", (analysis_id,)).fetchone()
        return _row_to_record(row) if row else None

    def list_records(self, limit: int = 50, offset: int = 0) -> list[AnalysisRecord]:
        with self._conn() as c:
            rows = c.execute(
                self._SELECT + " ORDER BY a.created_at DESC, a.rowid DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def escalation_queue(self, limit: int = 100) -> list[AnalysisRecord]:
        """Open tickets that need a human, most urgent first, then oldest first."""
        with self._conn() as c:
            rows = c.execute(
                self._SELECT
                + " WHERE a.escalate_to_human = 1 AND t.status = 'open'"
                " ORDER BY t.priority ASC, a.created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            totals = c.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, COALESCE(AVG(cost_usd),0) AS avg_cost,"
                " COALESCE(AVG(latency_ms),0) AS avg_latency,"
                " COALESCE(SUM(escalate_to_human),0) AS escalated,"
                " COALESCE(SUM(status='needs_human_review'),0) AS needs_review FROM analyses"
            ).fetchone()
            avg_conf = c.execute("SELECT AVG(confidence) AS v FROM analyses WHERE status='ok'").fetchone()["v"]
            open_tickets = c.execute("SELECT COUNT(*) AS n FROM tickets WHERE status='open'").fetchone()["n"]

            def dist(col: str) -> dict[str, int]:
                rows = c.execute(f"SELECT {col}, COUNT(*) FROM analyses GROUP BY {col} ORDER BY 2 DESC").fetchall()
                return {r[0]: r[1] for r in rows}

            result = {
                "total": totals["n"],
                "escalated": totals["escalated"],
                "needs_human_review": totals["needs_review"],
                "open_tickets": open_tickets,
                "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
                "avg_latency_ms": round(totals["avg_latency"]),
                "avg_cost_usd": round(totals["avg_cost"], 6),
                "total_cost_usd": round(totals["cost"], 6),
                "sentiment": dist("sentiment"),
                "urgency": dist("urgency"),
                "category": dist("category"),
                "team": dist("routed_to"),
                "sentiment_trend": [
                    dict(r)
                    for r in c.execute(
                        "SELECT substr(created_at,1,10) AS day, sentiment, COUNT(*) AS n FROM analyses"
                        " GROUP BY day, sentiment ORDER BY day"
                    )
                ],
            }
        return result

    def ping(self) -> bool:
        try:
            with self._conn() as c:
                c.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False


def _row_to_record(r: sqlite3.Row) -> AnalysisRecord:
    ticket = None
    if r["ticket_id"]:
        ticket = Ticket(
            ticket_id=r["ticket_id"], team=r["t_team"], priority=r["priority"],
            sla_minutes=r["sla_minutes"], due_at=r["due_at"], status=r["t_status"],
        )
    return AnalysisRecord(
        id=r["id"], created_at=r["created_at"], status=r["status"], mode=r["mode"], model=r["model"],
        masked_input=r["masked_input"],
        analysis=FeedbackAnalysis(
            sentiment=r["sentiment"], urgency=r["urgency"], category=r["category"],
            key_issues=json.loads(r["key_issues"]), recommended_action=r["recommended_action"],
            suggested_team=r["suggested_team"], escalate_to_human=bool(r["escalate_to_human"]),
            confidence=r["confidence"], reasoning_summary=r["reasoning_summary"],
        ),
        routed_to=r["routed_to"], flags=json.loads(r["flags"]), routing_reasons=json.loads(r["routing_reasons"]),
        ticket=ticket, retries=r["retries"], latency_ms=r["latency_ms"], prompt_tokens=r["prompt_tokens"],
        completion_tokens=r["completion_tokens"], cost_usd=r["cost_usd"], error=r["error"],
    )
