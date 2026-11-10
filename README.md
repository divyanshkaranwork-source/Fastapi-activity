# FeedbackIQ: banking customer-feedback triage

Turns free-text customer messages (emails, app reviews, chat logs, survey comments) into a
validated JSON record, then routes it: ticket, SLA, team, and a human-escalation queue.

* **Model:** OpenAI `gpt-4.1-mini`, called over plain HTTPS with `httpx`
* **No LLM frameworks.** No LangChain, no LangGraph, no OpenAI SDK. The prompt builder, retry loop,
  validation and routing are ordinary, readable Python in `feedbackiq/`.
* **Domain:** retail banking only (cards and payments, fraud, accounts, loans, fees, digital banking, branch service)

```
Raw feedback -> preprocess (clean, mask PII, flag injection, truncate)
  -> prompt builder (role + rules + bank policy + few-shot + schema)
  -> LLM call (temperature 0, JSON mode, retries with backoff)
  -> Pydantic validation -> one corrective retry -> "needs human review" fallback
  -> business rules (route, escalate, SLA priority) -> SQLite + ticket
  -> FastAPI + Streamlit dashboard
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put your key in OPENAI_API_KEY
uvicorn feedbackiq.api:app --reload      # API  -> http://localhost:8000/docs
streamlit run dashboard.py               # UI   -> http://localhost:8501
pytest                                   # 98 tests, no network or key needed
```

No key? Set `LLM_PROVIDER=mock` for an offline keyword heuristic (clearly labelled
`mock-heuristic`; it is not an LLM and its accuracy says nothing about GPT-4.1 mini).

## API

| Endpoint | Purpose |
|---|---|
| `POST /analyze` `{"text": "...", "mode": "full"}` | Analyze one message |
| `POST /analyze-batch?mode=full` (multipart CSV, column `feedback`/`text`/`message`/`comment`/`review`/`content`) | Analyze up to `MAX_BATCH_ROWS` rows |
| `GET /analyses`, `GET /analyses/{id}` | History |
| `GET /escalation-queue` | Open tickets needing a human, P1 first |
| `POST /tickets/{ticket_id}/resolve` | Close a ticket |
| `GET /stats`, `GET /health` | Aggregates / readiness |

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"text":"My card was charged twice for Rs 3,200 and your app keeps crashing. I am closing my account today."}'
```

`mode` is `zero_shot` (baseline), `few_shot` (role + rules + 6 examples) or `full` (default, adds the bank escalation policy).

## Output schema (`feedbackiq/schema.py`)

`sentiment` (Positive/Neutral/Negative), `urgency` (Low/Medium/High/Critical), `category`
(Card & Payments, Fraud & Security, Account Services, Loans & Credit, Fees & Charges, Digital Banking,
Branch & Service, Other), `key_issues` (max 5), `recommended_action`, `suggested_team`
(Support, Payments, Tech, Risk, Retention, Lending), `escalate_to_human`, `confidence` (0 to 1),
`reasoning_summary` (max 200 chars). Enumerated fields are `Literal`s, so the model cannot invent a
value the downstream system does not understand. Harmless drift (`"negative"`, `"Fraud / Security"`) is
normalised; anything else fails validation.

## Reliability

| Concern | Behaviour |
|---|---|
| Invalid JSON / schema | Retry once with the validator's error fed back; then a `needs_human_review` record (still ticketed and escalated). Obvious fraud keywords route that fallback to Risk |
| Transient LLM errors (timeout, 429, 5xx) | 3 attempts with exponential backoff, honours `Retry-After`; then the same fallback |
| Misconfiguration (no key, 401/403, unknown model, no quota) | **Not** hidden behind fallbacks: API returns **503** with the reason |
| Low confidence (< 0.7), Critical urgency, churn threat, injection suspected | `escalate_to_human = true`, each override listed in `routing_reasons` |
| PII | Emails, UPI IDs, phones, card numbers, PAN, Aadhaar, labelled account numbers, OTP/CVV/PIN masked **before** the LLM call; only masked text is stored or logged |
| Prompt injection | Customer text is wrapped in `<customer_feedback>` tags and declared untrusted; forged tags are stripped; 12 injection patterns raise a flag that forces human review; the output schema limits what a successful injection could do |
| Edge inputs | Empty/markup-only -> 422 without an LLM call; long text truncated (`MAX_INPUT_CHARS`); non-Latin scripts flagged and handled by the model; CSV: bad encoding, missing column, size and row limits |
| Concurrency | Batch uses a worker pool; SQLite connection per operation (WAL) |

Business rules (`feedbackiq/rules.py`): Fraud & Security at High/Critical -> Risk; Negative + churn
signal -> Retention (urgency raised to High); priority/SLA P1 15 min, P2 1 h, P3 8 h, P4 48 h (wall-clock); no ticket for Positive + Low + no escalation.

## Evaluation

`evaluation/dataset.jsonl` holds 52 hand-labelled banking messages (9 Critical; sarcasm, mixed
sentiment, Hindi and Hinglish, injection attempts). Labels are the author's judgement, so treat
accuracy as indicative, not definitive.

```bash
python -m evaluation.run_eval               # zero_shot vs few_shot vs full, needs OPENAI_API_KEY
python -m evaluation.run_eval --limit 10    # quick run
```

It prints schema validity, sentiment/urgency/category accuracy, Critical recall, weighted urgency
accuracy, category macro-F1, latency and cost per mode, and saves details to `evaluation/results/`.
It measures raw model output (before business rules) to isolate the effect of prompt design, and
refuses to run on the mock provider unless `--allow-mock` is passed.

> **Status:** the harness is tested, but the GPT-4.1 mini numbers have **not** been produced yet; the
> development environment had no OpenAI access. Run the command above with your key to get them.

## Configuration

See `.env.example`. Prices default to GPT-4.1 mini list prices ($0.40 / $1.60 per 1M tokens);
verify them against current OpenAI pricing, since `cost_usd` is an estimate from the API's token usage.

## 5-minute demo

1. Dashboard -> Analyze -> sample "Double charge + churn": Negative, High, Card & Payments, routed to Retention, escalated, ticket P2.
2. "Fraud" sample: Critical, Risk, P1 ticket, and the phone number shown as `[PHONE]` in the text sent to the model.
3. "Prompt injection" sample: the relabel request is ignored and the record is flagged for human review.
4. Batch upload -> `evaluation/dataset.jsonl` converted to CSV, or your own file; watch the Dashboard tab update.
5. Escalation queue -> resolve a ticket. 6. Run the evaluation and show the zero-shot vs full table.

## Layout

```
feedbackiq/  config, schema, preprocess, prompts, llm, mock_llm, analyzer, rules, storage, csv_input, api
dashboard.py Streamlit UI        evaluation/  labelled data + harness        tests/  pytest suite
```

## Known limits / next steps

SQLite and in-process batching suit a single node (swap `storage.py` for PostgreSQL and a queue to scale).
No API authentication or rate limiting yet. SLA times are wall-clock, not business hours. Romanised Hindi is not auto-detected (the model still handles it).
