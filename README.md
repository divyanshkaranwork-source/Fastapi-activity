# Assessment-1

Analyze bank customer feedback with `gpt-4.1-mini` and get validated JSON:

```json
{
  "sentiment": "Negative",
  "urgency": "High",
  "recommended_action": "Block the card immediately and start a fraud investigation.",
  "confidence": 0.97
}
```

Plain HTTP calls (`httpx`) to OpenAI. No LangChain, no LangGraph, no OpenAI SDK.
Pydantic validates every model reply.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                  # put your OPENAI_API_KEY in it
streamlit run dashboard.py            # dashboard at http://localhost:8501
uvicorn app.main:app --reload         # API (optional), docs at http://localhost:8000/docs
pytest                                # tests need no key or network
```

API example:

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"text": "My card was charged twice for Rs 3,200."}'
```

## How it maps to the problem statement

| Stage | Where |
|---|---|
| 1. User query | `dashboard.py` text box, `POST /analyze` |
| 2. LLM fundamentals | `app/llm.py`: model `gpt-4.1-mini`, temperature 0, 500 max tokens, JSON mode, retries |
| 3. Prompt techniques | `app/prompts.py`: role prompt, clear rules, bank policy as context, 3 few-shot examples |
| 4. Structured outputs | `app/schema.py` Pydantic model; `app/analyzer.py` validates, retries once with the error |
| 5. Business application | Streamlit dashboard (sentiment and urgency charts, average confidence, history) |

## Errors

| Status | When |
|---|---|
| 422 | Empty or too-long text |
| 502 | Model returned invalid JSON twice |
| 503 | Missing or bad API key, no quota, or OpenAI unreachable |

The dashboard shows the same errors as messages. Emails, phone, card and account numbers are masked before the text is sent.
