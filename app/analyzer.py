"""The pipeline: preprocess -> prompt -> LLM -> validate/retry -> business rules -> store."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import ValidationError

from .config import Settings
from .llm import LLMClient, LLMConfigError, LLMError, LLMResponse, estimate_cost
from .preprocess import EmptyInputError, Preprocessed, preprocess
from .prompts import build_messages, correction_message, draft_messages
from .rules import PRIORITY, apply_business_rules, fallback_analysis
from .schema import (
    MODES, AnalysisRecord, BatchItem, BatchResponse, CompareResponse, CorrectionRequest, DraftReply,
    FeedbackAnalysis, Mode, ModeResult, Ticket,
)
from .storage import Store

log = logging.getLogger("assessment1.analyzer")


@dataclass
class ClassifyResult:
    """Outcome of the LLM + validation stage (before business rules)."""

    analysis: Optional[FeedbackAnalysis]
    retries: int
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    model: str
    error: Optional[str] = None


class OutputParseError(ValueError):
    pass


def parse_llm_json(content: str) -> FeedbackAnalysis:
    """Parse the model reply into a FeedbackAnalysis. Raises OutputParseError with a readable reason."""
    text = (content or "").strip()
    if not text:
        raise OutputParseError("reply was empty")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise OutputParseError("reply is not valid JSON") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OutputParseError(f"reply is not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict):
        raise OutputParseError("reply must be a JSON object")
    try:
        return FeedbackAnalysis.model_validate(data)
    except ValidationError as exc:
        raise OutputParseError(_summarise(exc)) from None


def _summarise(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)[:600]


class DraftError(Exception):
    """A customer reply could not be drafted."""


_UNSAFE_ASK = re.compile(
    r"(?:share|send|provide|give|tell|reply with|enter|confirm)\s+(?:us\s+|me\s+)?(?:your|the)\s+"
    r"(?:\w+\s+){0,2}(?:otp|pin|cvv|cvc|password|passcode|card number)",
    re.IGNORECASE,
)


_NEGATION = re.compile(r"(?:not|never|n't|nobody|no one|avoid)\W+(?:\w+\W+){0,3}$", re.IGNORECASE)


def _asks_for_credentials(text: str) -> bool:
    """True if the text asks for credentials. 'Do not share your OTP' is advice, not a request."""
    for m in _UNSAFE_ASK.finditer(text):
        before = text[max(0, m.start() - 40) : m.start()]
        if not _NEGATION.search(before) or re.search(r"hesitate|forget|fail", before, re.IGNORECASE):
            return True
    return False


def parse_draft(content: str) -> DraftReply:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip(), flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
        draft = DraftReply.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        reason = _summarise(exc) if isinstance(exc, ValidationError) else "reply is not a valid JSON object"
        raise OutputParseError(reason) from None
    if _asks_for_credentials(draft.body) or _asks_for_credentials(draft.subject):
        raise OutputParseError("the draft asks the customer for credentials (OTP/PIN/CVV/password/card number)")
    return draft


def text_fingerprint(masked_text: str) -> str:
    return hashlib.sha256(" ".join(masked_text.lower().split()).encode("utf-8")).hexdigest()


class Analyzer:
    def __init__(self, client: LLMClient, settings: Settings, store: Optional[Store] = None) -> None:
        self.client = client
        self.settings = settings
        self.store = store

    # ------------------------------------------------------- LLM + validation
    def classify(self, pre: Preprocessed, mode: Mode = "full") -> ClassifyResult:
        """Call the LLM and validate. One corrective retry, then give up (analysis=None).

        LLMConfigError is deliberately not caught: misconfiguration must be loud, not hidden
        behind a queue full of 'needs human review' records.
        """
        extra = self.store.correction_examples(self.settings.correction_examples) if (self.store and mode != "zero_shot") else []
        messages = build_messages(pre.text, mode, extra)
        totals = {"latency": 0, "prompt": 0, "completion": 0}
        model = self.client.model
        last_error = ""
        retries = 0

        for attempt in range(2):
            try:
                resp: LLMResponse = self.client.complete(messages)
            except LLMConfigError:
                raise
            except LLMError as exc:
                last_error = f"LLM call failed: {exc}"
                break
            totals["latency"] += resp.latency_ms
            totals["prompt"] += resp.prompt_tokens
            totals["completion"] += resp.completion_tokens
            model = resp.model or model
            try:
                analysis = parse_llm_json(resp.content)
                return ClassifyResult(analysis, retries, totals["latency"], totals["prompt"], totals["completion"], model)
            except OutputParseError as exc:
                last_error = f"Schema validation failed: {exc}"
                log.warning("LLM output invalid (attempt %d): %s", attempt + 1, exc)
                if attempt == 0:
                    retries = 1
                    messages = messages + [
                        {"role": "assistant", "content": resp.content[:2000]},
                        correction_message(str(exc)),
                    ]
        return ClassifyResult(None, retries, totals["latency"], totals["prompt"], totals["completion"], model, last_error)

    # ------------------------------------------------------------- full flow
    def analyze(
        self, raw_text: str, mode: Mode = "full", persist: bool = True, use_cache: bool = True
    ) -> AnalysisRecord:
        """Analyze one message. Raises EmptyInputError (bad input) or LLMConfigError (bad setup)."""
        pre = preprocess(raw_text, self.settings.max_input_chars)
        fingerprint = text_fingerprint(pre.text)
        if use_cache and persist and self.store is not None:
            hit = self.store.find_cached(fingerprint, mode, self.client.model)
            if hit is not None:
                log.info("analysis cache hit id=%s", hit.id)
                return hit.model_copy(
                    update={"cached": True, "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
                )
        result = self.classify(pre, mode)

        failed = result.analysis is None
        analysis = result.analysis or fallback_analysis(pre.text, result.error or "unknown error")
        decision = apply_business_rules(
            analysis,
            text=pre.text,
            flags=pre.flags,
            needs_human_review=failed,
            confidence_threshold=self.settings.confidence_threshold,
        )

        now = datetime.now(timezone.utc)
        ticket = None
        if decision.needs_ticket:
            ticket = Ticket(
                ticket_id="TKT-" + uuid.uuid4().hex[:8].upper(),
                team=decision.routed_to,
                priority=decision.priority,
                sla_minutes=decision.sla_minutes,
                due_at=(now + timedelta(minutes=decision.sla_minutes)).isoformat(timespec="seconds"),
            )
        record = AnalysisRecord(
            id=uuid.uuid4().hex,
            created_at=now.isoformat(timespec="seconds"),
            status="needs_human_review" if failed else "ok",
            mode=mode,
            model=result.model,
            masked_input=pre.text,
            analysis=decision.analysis,
            routed_to=decision.routed_to,
            flags=pre.flags,
            routing_reasons=decision.reasons,
            ticket=ticket,
            retries=result.retries,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=estimate_cost(self.settings, result.prompt_tokens, result.completion_tokens),
            error=result.error,
        )
        if persist and self.store is not None:
            self.store.save(record, fingerprint)
        # Never log customer text: ids, timings and counters only.
        log.info(
            "analysis id=%s status=%s mode=%s urgency=%s team=%s retries=%d latency_ms=%d tokens=%d/%d",
            record.id, record.status, mode, decision.analysis.urgency, record.routed_to,
            record.retries, record.latency_ms, record.prompt_tokens, record.completion_tokens,
        )
        return record

    # ------------------------------------------------------- compare modes
    def compare(self, raw_text: str) -> CompareResponse:
        """Run the same message through zero-shot, few-shot and full prompts (nothing is stored)."""
        pre = preprocess(raw_text, self.settings.max_input_chars)

        def one(mode: Mode) -> ModeResult:
            res = self.classify(pre, mode)
            return ModeResult(
                mode=mode, ok=res.analysis is not None, analysis=res.analysis, error=res.error,
                retries=res.retries, latency_ms=res.latency_ms, prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                cost_usd=estimate_cost(self.settings, res.prompt_tokens, res.completion_tokens),
            )

        with ThreadPoolExecutor(max_workers=len(MODES)) as pool:
            results = list(pool.map(one, MODES))
        return CompareResponse(masked_input=pre.text, results=results)

    # ----------------------------------------------------------- draft reply
    def draft_reply(self, record: AnalysisRecord) -> DraftReply:
        """Draft a customer email for human approval. One corrective retry, like classify()."""
        messages = draft_messages(
            record.masked_input,
            json.dumps(record.analysis.model_dump(), ensure_ascii=False),
            record.ticket.sla_minutes if record.ticket else None,
        )
        last = ""
        for attempt in range(2):
            try:
                resp = self.client.complete(messages)
            except LLMConfigError:
                raise
            except LLMError as exc:
                raise DraftError(f"LLM call failed: {exc}") from None
            try:
                return parse_draft(resp.content)
            except OutputParseError as exc:
                last = str(exc)
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": resp.content[:2000]},
                        correction_message(last),
                    ]
        raise DraftError(f"Could not produce a valid draft: {last}")

    # ----------------------------------------------------- human correction
    def correct(self, analysis_id: str, req: CorrectionRequest) -> Optional[AnalysisRecord]:
        """Apply a human correction, re-derive priority/ticket, and keep it as a future few-shot example."""
        if self.store is None:
            raise RuntimeError("corrections need a Store")
        rec = self.store.get(analysis_id)
        if rec is None:
            return None
        changes = req.model_dump(exclude={"note"}, exclude_none=True)
        team = changes.get("suggested_team", rec.routed_to)
        analysis = rec.analysis.model_copy(update={**changes, "suggested_team": team})
        if analysis.urgency == "Critical":
            analysis = analysis.model_copy(update={"escalate_to_human": True})
        priority, sla = PRIORITY[analysis.urgency]
        now = datetime.now(timezone.utc)
        needs_ticket = not (analysis.sentiment == "Positive" and analysis.urgency == "Low" and not analysis.escalate_to_human)
        ticket = rec.ticket
        if ticket is not None or needs_ticket:
            created = datetime.fromisoformat(rec.created_at)
            ticket = Ticket(
                ticket_id=ticket.ticket_id if ticket else "TKT-" + uuid.uuid4().hex[:8].upper(),
                team=team, priority=priority, sla_minutes=sla,
                due_at=(created + timedelta(minutes=sla)).isoformat(timespec="seconds"),
                status=ticket.status if ticket else "open",
            )
        reasons = [*rec.routing_reasons, "Labels corrected by a human reviewer"]
        updated = rec.model_copy(
            update={"analysis": analysis, "routed_to": team, "ticket": ticket, "routing_reasons": reasons, "corrected": True}
        )
        self.store.save_correction(updated, changes, req.note, now.isoformat(timespec="seconds"))
        return updated

    # ----------------------------------------------------------------- batch
    def analyze_many(self, texts: list[str], mode: Mode = "full") -> BatchResponse:
        """Analyze many messages concurrently. Row problems are reported, not fatal.

        A configuration error (bad API key, no quota) aborts the whole batch by re-raising
        LLMConfigError, since every remaining row would fail the same way.
        """
        abort = threading.Event()
        config_errors: list[LLMConfigError] = []

        def work(item: tuple[int, str]) -> BatchItem:
            row, text = item
            if abort.is_set():
                return BatchItem(row=row, status="error", error="batch aborted")
            try:
                rec = self.analyze(text, mode)
            except EmptyInputError as exc:
                return BatchItem(row=row, status="skipped", error=str(exc))
            except LLMConfigError as exc:
                config_errors.append(exc)
                abort.set()
                return BatchItem(row=row, status="error", error=str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad row must not sink the batch
                log.exception("row %d failed", row)
                return BatchItem(row=row, status="error", error=f"{type(exc).__name__}: {exc}")
            return BatchItem(row=row, status=rec.status, record=rec)

        with ThreadPoolExecutor(max_workers=self.settings.batch_workers) as pool:
            items = list(pool.map(work, enumerate(texts, start=1)))
        if config_errors:
            raise config_errors[0]

        count = lambda s: sum(1 for i in items if i.status == s)  # noqa: E731
        return BatchResponse(
            total=len(items), ok=count("ok"), needs_human_review=count("needs_human_review"),
            skipped=count("skipped"), error=count("error"), results=items,
        )
