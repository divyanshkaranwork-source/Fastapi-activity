import json

import httpx
import pytest

from app.config import Settings
from app.llm import (
    LLMBadRequestError, LLMConfigError, LLMTransientError, OpenAIChatClient, estimate_cost,
)

MSGS = [{"role": "user", "content": "hi JSON"}]


def ok_body(content="{}"):
    return {"model": "gpt-4.1-mini-2025-04-14", "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


def make(handler, **kw):
    s = Settings(api_key="sk-test", max_retries=kw.pop("max_retries", 3), **kw)
    sleeps: list[float] = []
    client = OpenAIChatClient(s, httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleeps.append)
    return client, sleeps


def test_success_and_payload():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=ok_body('{"a":1}'))

    client, _ = make(handler)
    r = client.complete(MSGS)
    assert r.content == '{"a":1}' and r.prompt_tokens == 120 and r.completion_tokens == 30
    assert seen["body"]["model"] == "gpt-4.1-mini" and seen["body"]["temperature"] == 0.0
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["auth"] == "Bearer sk-test"


def test_missing_key_is_config_error():
    client = OpenAIChatClient(Settings(api_key=""), httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        client.complete(MSGS)


def test_retries_429_then_succeeds_honouring_retry_after():
    calls = iter([httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}),
                  httpx.Response(200, json=ok_body())])
    client, sleeps = make(lambda r: next(calls))
    assert client.complete(MSGS).content == "{}"
    assert sleeps == [2.0]


def test_5xx_exhausts_retries_to_transient_error():
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(503, text="down")

    client, sleeps = make(handler)
    with pytest.raises(LLMTransientError):
        client.complete(MSGS)
    assert n["c"] == 3 and len(sleeps) == 2


def test_timeout_is_retried():
    calls = {"c": 0}

    def handler(req):
        calls["c"] += 1
        if calls["c"] == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json=ok_body())

    client, _ = make(handler)
    assert client.complete(MSGS).content == "{}"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_and_model_errors_not_retried(status):
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(status, json={"error": {"message": "nope"}})

    client, _ = make(handler)
    with pytest.raises(LLMConfigError):
        client.complete(MSGS)
    assert n["c"] == 1


def test_quota_exhausted_is_config_error():
    client, _ = make(lambda r: httpx.Response(429, json={"error": {"message": "quota", "code": "insufficient_quota"}}))
    with pytest.raises(LLMConfigError, match="quota"):
        client.complete(MSGS)


def test_400_is_bad_request():
    client, _ = make(lambda r: httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(LLMBadRequestError):
        client.complete(MSGS)


def test_malformed_success_body():
    client, _ = make(lambda r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMTransientError):
        client.complete(MSGS)


def test_null_content_and_missing_usage_tolerated():
    client, _ = make(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    r = client.complete(MSGS)
    assert r.content == "" and r.prompt_tokens == 0


def test_cost():
    assert estimate_cost(Settings(), 1_000_000, 1_000_000) == 2.0
