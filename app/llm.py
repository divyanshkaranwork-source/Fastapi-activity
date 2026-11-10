"""Plain-HTTP call to the OpenAI chat completions API (no SDK, no framework)."""

import os
import time

import httpx

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


class LLMError(Exception):
    """The LLM could not be reached or returned an error."""


class LLMConfigError(LLMError):
    """Bad or missing API key / model. Retrying will not help."""


def call_llm(messages: list[dict], client: httpx.Client | None = None, retries: int = 3) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise LLMConfigError("OPENAI_API_KEY is not set (add it to .env)")
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    http = client or httpx.Client(timeout=30)
    last = "unknown error"
    for attempt in range(retries):
        try:
            resp = http.post(f"{BASE_URL}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    raise LLMError("Unexpected response from the LLM") from None
            last = f"HTTP {resp.status_code}"
            if resp.status_code in (401, 403, 404):
                raise LLMConfigError(f"OpenAI rejected the request ({last}): check OPENAI_API_KEY / OPENAI_MODEL")
            if resp.status_code == 429 and "insufficient_quota" in resp.text:
                raise LLMConfigError("OpenAI quota exhausted")
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise LLMError(f"OpenAI error ({last})")
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise LLMError(f"LLM unavailable after {retries} attempts: {last}")
