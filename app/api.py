"""REST API (FastAPI)."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from . import __version__
from .analyzer import Analyzer, DraftError
from .config import ConfigError, get_settings
from .csv_input import CsvInputError, parse_feedback_csv
from .export import records_to_csv
from .llm import LLMConfigError, get_llm_client
from .preprocess import EmptyInputError
from .schema import (
    AnalysisRecord, AnalyzeRequest, BatchResponse, CompareResponse, CorrectionRequest, DraftReply, Mode,
)
from .storage import Store

log = logging.getLogger("assessment1.api")


def build_analyzer() -> Analyzer:
    settings = get_settings()
    return Analyzer(get_llm_client(settings), settings, Store(settings.db_path))


def get_analyzer(request: Request) -> Analyzer:
    return request.app.state.analyzer


def require_key(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    """If ASSESSMENT_API_KEY is set, every endpoint except /health needs a matching X-API-Key header."""
    expected = request.app.state.analyzer.settings.service_api_key
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing X-API-Key header")


def create_app(analyzer: Optional[Analyzer] = None) -> FastAPI:
    if analyzer is not None and analyzer.store is None:
        raise ValueError("The API needs an Analyzer constructed with a Store")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if analyzer is not None:
            app.state.analyzer = analyzer
        else:
            try:
                app.state.analyzer = build_analyzer()
            except ConfigError as exc:
                # Fail at startup with a readable message instead of a 500 on every request.
                raise RuntimeError(f"Invalid configuration: {exc}") from exc
        yield

    app = FastAPI(
        title="Assessment-1",
        version=__version__,
        description="Banking customer-feedback triage: raw text in, validated routing decision out.",
        lifespan=lifespan,
    )

    @app.exception_handler(LLMConfigError)
    async def _config_error(_: Request, exc: LLMConfigError):
        log.error("LLM configuration error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": f"LLM not available: {exc}"})

    @app.exception_handler(EmptyInputError)
    async def _empty_input(_: Request, exc: EmptyInputError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health(a: Analyzer = Depends(get_analyzer)):
        s = a.settings
        return {
            "status": "ok",
            "version": __version__,
            "provider": s.provider,
            "model": a.client.model,
            "llm_configured": s.provider == "mock" or bool(s.api_key),
            "auth_required": bool(s.service_api_key),
            "database_ok": a.store.ping() if a.store else False,
        }

    router = APIRouter(dependencies=[Depends(require_key)])

    @router.post("/analyze", response_model=AnalysisRecord)
    def analyze(body: AnalyzeRequest, a: Analyzer = Depends(get_analyzer)):
        return a.analyze(body.text, body.mode, use_cache=body.use_cache)

    @router.post("/compare", response_model=CompareResponse)
    def compare(body: AnalyzeRequest, a: Analyzer = Depends(get_analyzer)):
        """Same message through zero-shot, few-shot and full prompts. Nothing is stored."""
        return a.compare(body.text)

    @router.post("/analyze-batch", response_model=BatchResponse)
    def analyze_batch(
        file: UploadFile = File(..., description="CSV with a 'feedback' (or text/message/comment) column"),
        mode: Mode = Query("full"),
        a: Analyzer = Depends(get_analyzer),
    ):
        s = a.settings
        raw = file.file.read(s.max_upload_bytes + 1)
        try:
            texts = parse_feedback_csv(raw, s.max_upload_bytes, s.max_batch_rows)
        except CsvInputError as exc:
            raise HTTPException(exc.status_code, exc.message) from None
        return a.analyze_many(texts, mode)

    @router.get("/analyses", response_model=list[AnalysisRecord])
    def list_analyses(
        limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), a: Analyzer = Depends(get_analyzer)
    ):
        return a.store.list_records(limit, offset)

    @router.get("/analyses/{analysis_id}", response_model=AnalysisRecord)
    def get_analysis(analysis_id: str, a: Analyzer = Depends(get_analyzer)):
        rec = a.store.get(analysis_id)
        if rec is None:
            raise HTTPException(404, "Analysis not found")
        return rec

    @router.post("/analyses/{analysis_id}/draft-reply", response_model=DraftReply)
    def draft_reply(analysis_id: str, a: Analyzer = Depends(get_analyzer)):
        """AI-drafted reply email for a human to review. Never sent automatically."""
        rec = a.store.get(analysis_id)
        if rec is None:
            raise HTTPException(404, "Analysis not found")
        try:
            return a.draft_reply(rec)
        except DraftError as exc:
            raise HTTPException(502, str(exc)) from None

    @router.post("/analyses/{analysis_id}/correction", response_model=AnalysisRecord)
    def correct(analysis_id: str, body: CorrectionRequest, a: Analyzer = Depends(get_analyzer)):
        """A human fixes the labels. Priority/ticket are re-derived and the case becomes a few-shot example."""
        rec = a.correct(analysis_id, body)
        if rec is None:
            raise HTTPException(404, "Analysis not found")
        return rec

    @router.get("/escalation-queue", response_model=list[AnalysisRecord])
    def escalation_queue(limit: int = Query(100, ge=1, le=500), a: Analyzer = Depends(get_analyzer)):
        return a.store.escalation_queue(limit)

    @router.post("/tickets/{ticket_id}/resolve")
    def resolve_ticket(ticket_id: str, a: Analyzer = Depends(get_analyzer)):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not a.store.resolve_ticket(ticket_id, now):
            raise HTTPException(404, "Open ticket not found")
        return {"ticket_id": ticket_id, "status": "resolved", "resolved_at": now}

    @router.get("/stats")
    def stats(a: Analyzer = Depends(get_analyzer)):
        return a.store.stats()

    @router.get("/export.csv")
    def export_csv(limit: int = Query(5000, ge=1, le=20000), a: Analyzer = Depends(get_analyzer)):
        data = records_to_csv(a.store.list_records(limit))
        return Response(
            data, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="assessment-1-analyses.csv"'}
        )

    app.include_router(router)
    return app


app = create_app()
