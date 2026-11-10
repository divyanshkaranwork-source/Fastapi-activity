"""SQLite persistence for analyses, tickets and human corrections (upgrade path: PostgreSQL).

Only the masked text is stored, never the customer's raw message.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
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
    text_hash TEXT NOT NULL,
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
    error TEXT,
    corrected INTEGER NOT NULL DEFAULT 0
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
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL REFERENCES analyses(id),
    created_at TEXT NOT NULL,
    changes TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_hash ON analyses(text_hash, mode);
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
    def save(self, rec: AnalysisRecord, text_hash: str = "") -> None:
        a = rec.analysis
        row = {
            "id": rec.id, "created_at": rec.created_at, "status": rec.status, "mode": rec.mode,
            "model": rec.model, "text_hash": text_hash, "masked_input": rec.masked_input,
            "sentiment": a.sentiment, "urgency": a.urgency, "category": a.category,
            "key_issues": json.dumps(a.key_issues, ensure_ascii=False),
            "recommended_action": a.recommended_action, "suggested_team": a.suggested_team,
            "escalate_to_human": int(a.escalate_to_human), "confidence": a.confidence,
            "reasoning_summary": a.reasoning_summary, "routed_to": rec.routed_to,
            "flags": json.dumps(rec.flags), "routing_reasons": json.dumps(rec.routing_reasons, ensure_ascii=False),
            "retries": rec.retries, "latency_ms": rec.latency_ms, "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens, "cost_usd": rec.cost_usd, "error": rec.error,
            "corrected": int(rec.corrected),
        }
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with self._conn() as c:
            c.execute(f"INSERT INTO analyses ({cols}) VALUES ({marks})", row)
            if rec.ticket:
                self._insert_ticket(c, rec.id, rec.ticket, rec.created_at)

    @staticmethod
    def _insert_ticket(c: sqlite3.Connection, analysis_id: str, t: Ticket, created_at: str) -> None:
        c.execute(
            "INSERT INTO tickets (ticket_id, analysis_id, team, priority, sla_minutes, due_at, status, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (t.ticket_id, analysis_id, t.team, t.priority, t.sla_minutes, t.due_at, t.status, created_at),
        )

    def save_correction(self, rec: AnalysisRecord, changes: dict, note: Optional[str], now: str) -> None:
        """Persist a human correction: updated labels, updated/created ticket, audit row."""
        a = rec.analysis
        with self._conn() as c:
            c.execute(
                "UPDATE analyses SET sentiment=?, urgency=?, category=?, suggested_team=?, routed_to=?,"
                " escalate_to_human=?, routing_reasons=?, corrected=1 WHERE id=?",
                (a.sentiment, a.urgency, a.category, a.suggested_team, rec.routed_to, int(a.escalate_to_human),
                 json.dumps(rec.routing_reasons, ensure_ascii=False), rec.id),
            )
            if rec.ticket:
                t = rec.ticket
                cur = c.execute(
                    "UPDATE tickets SET team=?, priority=?, sla_minutes=?, due_at=? WHERE analysis_id=?",
                    (t.team, t.priority, t.sla_minutes, t.due_at, rec.id),
                )
                if cur.rowcount == 0:
                    self._insert_ticket(c, rec.id, t, now)
            c.execute(
                "INSERT INTO corrections (analysis_id, created_at, changes, note) VALUES (?,?,?,?)",
                (rec.id, now, json.dumps(changes), note),
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

    def find_cached(self, text_hash: str, mode: str, model_prefix: str) -> Optional[AnalysisRecord]:
        """Latest successful analysis of an identical masked message with the same mode and model."""
        with self._conn() as c:
            row = c.execute(
                self._SELECT + " WHERE a.text_hash = ? AND a.mode = ? AND a.status = 'ok' AND a.model LIKE ?"
                " ORDER BY a.created_at DESC, a.rowid DESC LIMIT 1",
                (text_hash, mode, model_prefix + "%"),
            ).fetchone()
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

    def correction_examples(self, limit: int) -> list[tuple[str, str]]:
        """(masked text, corrected analysis JSON) pairs from the latest human corrections.

        Messages flagged as possible prompt injection are never reused as examples.
        """
        if limit <= 0:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT a.*, t.ticket_id, t.team AS t_team, t.priority, t.sla_minutes, t.due_at, t.status AS t_status"
                " FROM (SELECT analysis_id, MAX(id) AS cid FROM corrections GROUP BY analysis_id) k"
                " JOIN analyses a ON a.id = k.analysis_id LEFT JOIN tickets t ON t.analysis_id = a.id"
                " WHERE a.flags NOT LIKE '%prompt_injection%' ORDER BY k.cid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            analysis = _row_to_record(r).analysis
            out.append((r["masked_input"], json.dumps(analysis.model_dump(), ensure_ascii=False)))
        return out

    def stats(self) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            totals = c.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, COALESCE(AVG(cost_usd),0) AS avg_cost,"
                " COALESCE(AVG(latency_ms),0) AS avg_latency,"
                " COALESCE(SUM(escalate_to_human),0) AS escalated,"
                " COALESCE(SUM(status='needs_human_review'),0) AS needs_review,"
                " COALESCE(SUM(corrected),0) AS corrected FROM analyses"
            ).fetchone()
            avg_conf = c.execute("SELECT AVG(confidence) AS v FROM analyses WHERE status='ok'").fetchone()["v"]
            open_tickets = c.execute("SELECT COUNT(*) AS n FROM tickets WHERE status='open'").fetchone()["n"]
            overdue = c.execute(
                "SELECT COUNT(*) AS n FROM tickets WHERE status='open' AND due_at < ?", (now,)
            ).fetchone()["n"]

            def dist(col: str) -> dict[str, int]:
                rows = c.execute(f"SELECT {col}, COUNT(*) FROM analyses GROUP BY {col} ORDER BY 2 DESC").fetchall()
                return {r[0]: r[1] for r in rows}

            issues: Counter[str] = Counter()
            for (raw,) in c.execute("SELECT key_issues FROM analyses WHERE status='ok'"):
                for issue in json.loads(raw):
                    issues[issue.strip().lower()] += 1

            result = {
                "total": totals["n"],
                "escalated": totals["escalated"],
                "needs_human_review": totals["needs_review"],
                "corrected": totals["corrected"],
                "open_tickets": open_tickets,
                "overdue_tickets": overdue,
                "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
                "avg_latency_ms": round(totals["avg_latency"]),
                "avg_cost_usd": round(totals["avg_cost"], 6),
                "total_cost_usd": round(totals["cost"], 6),
                "sentiment": dist("sentiment"),
                "urgency": dist("urgency"),
                "category": dist("category"),
                "team": dist("routed_to"),
                "top_issues": dict(issues.most_common(10)),
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
        corrected=bool(r["corrected"]),
    )
