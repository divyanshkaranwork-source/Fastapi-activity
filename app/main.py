import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import llm
from .analyzer import InvalidModelOutput, analyze
from .schema import AnalyzeRequest, FeedbackAnalysis

log = logging.getLogger("assessment-1")


app = FastAPI(title="Assessment-1", description="Analyze bank customer feedback and get structured JSON.")


@app.exception_handler(llm.LLMError)
async def llm_error(_, exc: llm.LLMError):
    log.error("LLM error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "model": llm.MODEL}


@app.post("/analyze", response_model=FeedbackAnalysis)
def analyze_feedback(body: AnalyzeRequest):
    try:
        return analyze(body.text)
    except InvalidModelOutput as exc:
        raise HTTPException(502, f"Model returned invalid output: {exc}") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
