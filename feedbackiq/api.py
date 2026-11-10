"""Stage 1 + 5: REST API (FastAPI)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .analyzer import Analyzer
from .config import ConfigError, get_settings
from .csv_input import CsvInputError, parse_feedback_csv
from .llm import LLMConfigError, get_llm_client
from .preprocess import EmptyInputError
from .schema import AnalysisRecord, AnalyzeRequest, BatchResponse, Mode
from .storage import Store

log = logging.getLogger("feedbackiq.api")


def build_analyzer() -> Analyzer:
    settings = get_settings()
    return Analyzer(get_llm_client(settings), settings, Store(settings.db_path))


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
        title="FeedbackIQ",
        version=__version__,
        description="Banking customer-feedback triage: raw text in, validated routing decision out.",
        lifespan=lifespan,
    )

    def get_analyzer(request: Request) -> Analyzer:
        return request.app.state.analyzer

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
            "database_ok": a.store.ping() if a.store else False,
        }

    @app.post("/analyze", response_model=AnalysisRecord)
    def analyze(body: AnalyzeRequest, a: Analyzer = Depends(get_analyzer)):
        return a.analyze(body.text, body.mode)

    @app.post("/analyze-batch", response_model=BatchResponse)
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

    @app.get("/analyses", response_model=list[AnalysisRecord])
    def list_analyses(
        limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), a: Analyzer = Depends(get_analyzer)
    ):
        return a.store.list_records(limit, offset)

    @app.get("/analyses/{analysis_id}", response_model=AnalysisRecord)
    def get_analysis(analysis_id: str, a: Analyzer = Depends(get_analyzer)):
        rec = a.store.get(analysis_id)
        if rec is None:
            raise HTTPException(404, "Analysis not found")
        return rec

    @app.get("/escalation-queue", response_model=list[AnalysisRecord])
    def escalation_queue(limit: int = Query(100, ge=1, le=500), a: Analyzer = Depends(get_analyzer)):
        return a.store.escalation_queue(limit)

    @app.post("/tickets/{ticket_id}/resolve")
    def resolve_ticket(ticket_id: str, a: Analyzer = Depends(get_analyzer)):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not a.store.resolve_ticket(ticket_id, now):
            raise HTTPException(404, "Open ticket not found")
        return {"ticket_id": ticket_id, "status": "resolved", "resolved_at": now}

    @app.get("/stats")
    def stats(a: Analyzer = Depends(get_analyzer)):
        return a.store.stats()

    return app


app = create_app()
