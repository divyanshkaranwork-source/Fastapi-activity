# Assessment-1

Send a bank customer's message, get back structured JSON. Uses OpenAI `gpt-4.1-mini` over plain
HTTP (`httpx`). No LangChain, no LangGraph, no OpenAI SDK.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                 # put your OPENAI_API_KEY in it
uvicorn app.main:app --reload        # docs at http://localhost:8000/docs
pytest                               # tests need no key or network
```

## Use

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"text": "My card was charged twice for Rs 3,200 and your app keeps crashing."}'
```

Response:

```json
{
  "sentiment": "Negative",
  "urgency": "High",
  "category": "Card & Payments",
  "key_issues": ["Duplicate card charge", "App crashing"],
  "recommended_action": "Reverse the duplicate charge and fix the app issue.",
  "suggested_team": "Payments",
  "escalate_to_human": false,
  "confidence": 0.92,
  "reasoning_summary": "Double debit plus an app fault."
}
```

(Example of the shape only; real output comes from the model.)

Endpoints: `POST /analyze`, `GET /health`.

## How it works

1. `app/analyzer.py` cleans the text: masks emails, phones, card and account numbers, and caps the length.
2. `app/prompts.py` builds the prompt: a bank-analyst role, rules, and 4 worked examples.
3. `app/llm.py` calls the model (temperature 0, JSON mode, retries on timeouts/429/5xx).
4. `app/schema.py` validates the reply. If invalid, the model gets one retry with the error.
5. If confidence is below 0.7, `escalate_to_human` is set to `true`.

## Errors

| Status | When |
|---|---|
| 422 | Empty or too-long text |
| 502 | Model returned invalid JSON twice |
| 503 | Missing or bad API key, no quota, or OpenAI unreachable |
