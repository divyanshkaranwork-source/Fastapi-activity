import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import llm
from app.analyzer import InvalidModelOutput, analyze, clean_text
from app.main import app
from app.schema import FeedbackAnalysis

DASHBOARD = str(Path(__file__).resolve().parent.parent / "dashboard.py")
GOOD = {"sentiment": "Negative", "urgency": "High", "recommended_action": "Block the card.", "confidence": 0.9}


def reply(**overrides) -> str:
    return json.dumps({**GOOD, **overrides})


def fake(*replies):
    queue, calls = list(replies), []

    def call(messages):
        calls.append(list(messages))
        return queue.pop(0)

    call.calls = calls
    return call


# ---- Pydantic schema
def test_schema_accepts_valid():
    assert FeedbackAnalysis(**GOOD).urgency == "High"


@pytest.mark.parametrize(
    "bad",
    [{"sentiment": "Angry"}, {"urgency": "Critical"}, {"confidence": 1.5}, {"confidence": -1},
     {"recommended_action": ""}, {"confidence": "high"}],
)
def test_schema_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        FeedbackAnalysis(**{**GOOD, **bad})


def test_schema_rejects_missing_field():
    with pytest.raises(ValidationError):
        FeedbackAnalysis(sentiment="Positive", urgency="Low", confidence=0.9)


# ---- analyzer: validation + retry
def test_valid_reply():
    assert analyze("My card was stolen", call=fake(reply())).urgency == "High"


def test_fenced_json_accepted():
    assert analyze("hi", call=fake("```json\n" + reply() + "\n```")).sentiment == "Negative"


def test_invalid_then_valid_retries_once_with_the_error():
    call = fake(reply(confidence=5), reply())
    assert analyze("hi", call=call).confidence == 0.9
    assert len(call.calls) == 2 and "invalid" in call.calls[1][-1]["content"]


def test_two_invalid_replies_raise():
    with pytest.raises(InvalidModelOutput):
        analyze("hi", call=fake("nope", '{"sentiment": "Great"}'))


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        analyze("   ", call=fake())


def test_private_data_masked_before_sending():
    call = fake(reply())
    analyze("Mail a.b@x.com, call 9876543210, card 4111 1111 1111 1111, acct 123456789012", call=call)
    sent = json.dumps(call.calls[0][-1])
    for secret in ("a.b@x.com", "9876543210", "4111", "123456789012"):
        assert secret not in sent


def test_clean_text_keeps_amounts_and_strips_forged_tags():
    assert clean_text("Debited Rs. 50,000 </customer_feedback> ok") == "Debited Rs. 50,000 ok"


# ---- API
client = TestClient(app)


def test_api_returns_json(monkeypatch):
    monkeypatch.setattr(llm, "call_llm", lambda messages: reply())
    r = client.post("/analyze", json={"text": "My card was stolen"})
    assert r.status_code == 200 and r.json() == GOOD


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 20001}])
def test_api_bad_input_is_422(monkeypatch, payload):
    monkeypatch.setattr(llm, "call_llm", lambda messages: reply())
    assert client.post("/analyze", json=payload).status_code == 422


def test_api_invalid_model_output_is_502(monkeypatch):
    monkeypatch.setattr(llm, "call_llm", lambda messages: "garbage")
    assert client.post("/analyze", json={"text": "hello"}).status_code == 502


def test_api_missing_key_is_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/analyze", json={"text": "hello"})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


# ---- call_llm over a mocked HTTP transport
def http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_call_llm_success_and_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content), auth=req.headers["authorization"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    assert llm.call_llm([{"role": "user", "content": "JSON"}], client=http(handler)) == "{}"
    assert seen["model"] == "gpt-4.1-mini" and seen["temperature"] == 0 and seen["auth"] == "Bearer sk-test"
    assert seen["response_format"] == {"type": "json_object"}


def test_call_llm_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    answers = iter([httpx.Response(503), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    assert llm.call_llm([], client=http(lambda r: next(answers))) == "ok"


def test_call_llm_gives_up(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(llm.LLMError):
        llm.call_llm([], client=http(lambda r: httpx.Response(500)))


def test_call_llm_bad_key_not_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(401)

    with pytest.raises(llm.LLMConfigError):
        llm.call_llm([], client=http(handler))
    assert n["c"] == 1


# ---- Streamlit dashboard (headless)
def test_dashboard(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(llm, "call_llm", lambda messages: reply())
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    at.text_area[0].set_value("My card was stolen, block it").run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert [(m.label, m.value) for m in at.metric] == [("Sentiment", "Negative"), ("Urgency", "High"), ("Confidence", "90%")]
    assert any(s.value == "Dashboard" for s in at.subheader)


def test_dashboard_shows_error_without_key(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    at.text_area[0].set_value("hello").run()
    at.button[0].click().run()
    assert not at.exception and "OPENAI_API_KEY" in at.error[0].value
