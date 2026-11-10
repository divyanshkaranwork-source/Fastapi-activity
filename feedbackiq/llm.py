"""Stage 2: a thin, provider-swappable LLM wrapper (plain HTTPS via httpx, no SDK/framework)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx

from .config import Settings

log = logging.getLogger("feedbackiq.llm")


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMConfigError(LLMError):
    """Misconfiguration (missing/invalid key, unknown model, no quota). Retrying cannot help."""


class LLMTransientError(LLMError):
    """Timeouts, 429/5xx or network errors that persisted through all retries."""


class LLMBadRequestError(LLMError):
    """The provider rejected this specific request (HTTP 400/422)."""


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse: ...


def estimate_cost(settings: Settings, prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens * settings.price_input_per_1m / 1_000_000
        + completion_tokens * settings.price_output_per_1m / 1_000_000,
        6,
    )


class OpenAIChatClient:
    """Calls POST {base_url}/chat/completions in JSON mode with retries and backoff."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.model = settings.model
        self._sleep = sleep
        self._http = http_client or httpx.Client(timeout=httpx.Timeout(settings.request_timeout))

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        s = self.settings
        if not s.api_key:
            raise LLMConfigError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file "
                "(or set LLM_PROVIDER=mock for the offline demo)."
            )
        payload = {
            "model": s.model,
            "messages": messages,
            "temperature": s.temperature,
            "max_tokens": s.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {s.api_key}", "Content-Type": "application/json"}
        url = f"{s.base_url}/chat/completions"

        last_error = "unknown error"
        for attempt in range(1, s.max_retries + 1):
            started = time.perf_counter()
            retry_after: float | None = None
            try:
                resp = self._http.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                latency_ms = int((time.perf_counter() - started) * 1000)
                status = resp.status_code
                if status == 200:
                    return self._parse(resp, latency_ms)
                detail = _error_detail(resp)
                last_error = f"HTTP {status}: {detail}"
                if status in (401, 403):
                    raise LLMConfigError(f"OpenAI rejected the API key ({last_error})")
                if status == 404:
                    raise LLMConfigError(f"Model or endpoint not found ({last_error})")
                if status == 429:
                    if "insufficient_quota" in resp.text or "billing" in detail.lower():
                        raise LLMConfigError(f"OpenAI quota exhausted ({last_error})")
                    retry_after = _retry_after(resp)
                elif status in (400, 422):
                    raise LLMBadRequestError(last_error)
                elif status < 500:
                    raise LLMError(last_error)
                else:
                    retry_after = _retry_after(resp)

            if attempt < s.max_retries:
                delay = retry_after if retry_after is not None else min(2 ** (attempt - 1), 8)
                delay = min(delay, 30.0)
                log.warning("LLM call failed (%s); retry %d/%d in %.1fs", last_error, attempt, s.max_retries - 1, delay)
                self._sleep(delay)

        raise LLMTransientError(f"LLM unavailable after {s.max_retries} attempts: {last_error}")

    def _parse(self, resp: httpx.Response, latency_ms: int) -> LLMResponse:
        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMTransientError(f"Unexpected response shape from LLM: {exc!r}") from exc
        return LLMResponse(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            model=str(data.get("model") or self.settings.model),
        )

    def close(self) -> None:
        self._http.close()


def _error_detail(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)[:300]
        if err:
            return str(err)[:300]
    except (ValueError, AttributeError):
        pass
    return (resp.text or resp.reason_phrase or "")[:300]


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.provider == "mock":
        from .mock_llm import MockClient

        return MockClient()
    return OpenAIChatClient(settings)
