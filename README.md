# Assessment-1

Banking customer-feedback triage. Paste or upload customer messages and get validated JSON
(sentiment, urgency, recommended action, confidence, plus routing detail), tickets with SLAs, an
escalation queue, analytics, and an AI-drafted reply for a human to approve.

* **Model:** OpenAI `gpt-4.1-mini` over plain HTTPS (`httpx`)
* **No LLM frameworks:** no LangChain, no LangGraph, no OpenAI SDK
* **Validation:** every model reply is checked against Pydantic models; one corrective retry, then a safe fallback
* **Domain:** retail banking only

```
Raw feedback -> preprocess (clean, mask PII, flag injection, truncate)
  -> cache lookup (identical message already analyzed?)
  -> prompt (role + rules + bank policy + few-shot + human-corrected examples)
  -> LLM (temperature 0, JSON mode, retries) -> Pydantic validation -> corrective retry -> fallback
  -> business rules (route, escalate, SLA priority) -> SQLite + ticket
  -> API + Streamlit dashboard (analytics, queue, draft reply, corrections, export)
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # put your OPENAI_API_KEY in it
streamlit run dashboard.py                 # dashboard  http://localhost:8501
uvicorn app.api:app --reload               # API docs   http://localhost:8000/docs
pytest                                     # 131 tests, no key or network needed
python -m evaluation.run_eval              # zero-shot vs few-shot vs full on 52 labelled messages
```

No key? `LLM_PROVIDER=mock` runs an offline keyword heuristic (labelled `mock-heuristic`; it is not an
LLM and says nothing about GPT-4.1 mini's accuracy).

## Features

**Core**
* Strict output schema: `sentiment`, `urgency`, `category`, `key_issues`, `recommended_action`, `suggested_team`, `escalate_to_human`, `confidence`, `reasoning_summary`
* Prompt techniques as switchable modes: zero-shot baseline, few-shot, and full (adds the bank escalation policy)
* PII masking before the model sees anything (email, UPI, phone, card, PAN, Aadhaar, account numbers, OTP/CVV); only masked text is stored
* Prompt-injection flagging, forged-delimiter stripping, truncation, non-English flagging
* Business rules: fraud to Risk, churn threats to Retention, confidence below 0.7 escalates, SLA priority P1 (15 min) to P4 (48 h)
* SQLite storage, tickets, escalation queue, batch CSV upload, concurrent processing

**Premium**
* **Dashboard** with seven tabs: Analyze, Prompt Lab, Batch, Analytics, Escalation Queue, Evaluation, About
* **Prompt Lab**: one message through all three techniques side by side, with latency, tokens and cost
* **AI-drafted reply** for human approval (never sent automatically). Guardrails: no requests for OTP/PIN/CVV/password/card number, no promised refunds
* **Human corrections**: fix a label in the UI or API; the ticket and priority are re-derived and the case becomes a few-shot example for future prompts (messages flagged as injection are never reused)
* **Response cache**: an identical message returns the earlier result with no LLM call, no cost and no duplicate ticket (toggle or `"use_cache": false`)
* **Analytics**: sentiment trend, urgency, categories, teams, top recurring issues, SLA overdue count, spend
* **Evaluation tab**: run the labelled evaluation from the UI and chart the techniques
* **CSV export** (spreadsheet-formula safe) and JSON/draft downloads
* **Optional API key** (`ASSESSMENT_API_KEY`) on every endpoint except `/health`

## API

| Endpoint | Purpose |
|---|---|
| `POST /analyze` `{"text": "...", "mode": "full", "use_cache": true}` | Analyze one message |
| `POST /compare` | Same message through all three prompt modes (not stored) |
| `POST /analyze-batch?mode=full` (multipart CSV) | Analyze up to `MAX_BATCH_ROWS` rows |
| `GET /analyses`, `GET /analyses/{id}` | History |
| `POST /analyses/{id}/draft-reply` | Draft a customer email for human review |
| `POST /analyses/{id}/correction` | Human correction (sentiment, urgency, category, suggested_team, note) |
| `GET /escalation-queue`, `POST /tickets/{id}/resolve` | Work the queue |
| `GET /stats`, `GET /export.csv`, `GET /health` | Aggregates, export, readiness |

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"text":"My card was charged twice for Rs 3,200 and your app keeps crashing. I am closing my account today."}'
```

## Reliability

| Concern | Behaviour |
|---|---|
| Invalid JSON / schema | Retry once with the validator's error fed back; then a `needs_human_review` record (still ticketed). Obvious fraud keywords route that fallback to Risk |
| Timeouts, 429, 5xx | 3 attempts with backoff, honours `Retry-After`; then the same fallback |
| Bad or missing key, unknown model, no quota | Not hidden: API returns 503 with the reason; the dashboard shows it |
| Empty input, bad CSV (encoding, columns, size, rows) | 4xx without calling the LLM |
| Unsafe drafts | Rejected and retried once, then 502 |
| Concurrency | Worker pool for batches; one SQLite connection per operation (WAL) |

## Evaluation

`evaluation/dataset.jsonl`: 52 hand-labelled messages (9 Critical; sarcasm, mixed sentiment, Hindi,
Hinglish, injection attempts). Labels are the author's judgement, so treat accuracy as indicative.
Metrics (schema validity, sentiment/urgency/category accuracy, Critical recall, category macro-F1,
latency, cost) are computed on raw model output so the table isolates prompt design.

> **Status:** the harness and all features are tested, but GPT-4.1 mini results have **not** been
> produced: the development environment had no OpenAI access. Run the evaluation with your key.

## Layout

```
app/          config, schema, preprocess, prompts, llm, mock_llm, analyzer, rules, storage, csv_input, export, api
dashboard.py  Streamlit UI          evaluation/  labelled data + harness          tests/  pytest suite
```

## Known limits

SQLite and in-process batching suit a single node (swap `storage.py` for PostgreSQL and a queue to scale).
No rate limiting. SLA times are wall-clock, not business hours. Drafts are never sent: wiring an email
provider is deliberately out of scope. Romanised Hindi is not auto-detected (the model still handles it).
Prices default to GPT-4.1 mini list prices; verify them, since `cost_usd` is an estimate.
