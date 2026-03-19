"""
llm_client.py
-------------
OpenAI client wrapper for the Enterprise RAG Platform.

Provides:
  - Retry logic with exponential backoff for transient API errors
  - Token usage tracking per request (for cost control)
  - Structured logging for audit trails
  - Clean interface that abstracts OpenAI SDK details
    from the rest of the codebase

Usage:
    client = OpenAIClient()
    response = client.complete(
        messages=[{"role": "user", "content": "What is RAG?"}],
        system_prompt="You are a helpful assistant."
    )
    print(response.content)
    print(response.usage)
"""

import logging
import os
import time
from dataclasses import dataclass

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response returned for every completion call."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: float

    @property
    def usage(self) -> dict:
        """Token usage summary — used by cost_tracker.py."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class OpenAIClient:
    """
    Thin wrapper around the OpenAI SDK.

    Handles:
      - Authentication via OPENAI_API_KEY environment variable
      - Automatic retry with exponential backoff on transient errors
      - Token usage logging for cost tracking
      - Consistent error handling so callers get clean exceptions
    """

    RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError)

    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        """
        Args:
            model:               OpenAI model to use. Default is gpt-4o.
            max_retries:         Retry attempts on transient errors.
            retry_delay_seconds: Base delay between retries (doubles each attempt).
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

        self._client = OpenAI(api_key=api_key)

        logger.info("OpenAIClient initialised", extra={"model": self.model})

    def complete(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Send a chat completion request to OpenAI.

        Args:
            messages:      List of {"role": ..., "content": ...} dicts.
                           Do NOT include the system message here —
                           pass it via system_prompt instead.
            system_prompt: Optional system instruction prepended to messages.
            temperature:   0.0 = deterministic (best for RAG responses).
            max_tokens:    Maximum tokens in the completion.

        Returns:
            LLMResponse with content, token counts, and latency.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
            ValueError:   If messages list is empty.

        Example:
            response = client.complete(
                messages=[{"role": "user", "content": "What is RAG?"}],
                system_prompt="Answer in 2 sentences.",
            )
            print(response.content)
            print(response.usage)
        """
        if not messages:
            raise ValueError("messages list cannot be empty")

        full_messages = self._build_messages(messages, system_prompt)

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call(full_messages, temperature, max_tokens)

            except self.RETRYABLE_ERRORS as e:
                wait = self.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable error — retrying",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "wait_seconds": wait,
                        "error": str(e),
                    },
                )
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenAI call failed after {self.max_retries} attempts: {e}"
                    ) from e
                time.sleep(wait)

            except Exception as e:
                logger.error(
                    "Non-retryable error",
                    extra={"error": str(e), "error_type": type(e).__name__},
                )
                raise

    def _call(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Make the actual API call and parse the response."""
        start = time.monotonic()

        raw = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.monotonic() - start) * 1000
        choice = raw.choices[0]

        response = LLMResponse(
            content=choice.message.content or "",
            model=raw.model,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            total_tokens=raw.usage.total_tokens,
            finish_reason=choice.finish_reason,
            latency_ms=round(latency_ms, 1),
        )

        logger.info(
            "OpenAI completion",
            extra={
                "model": response.model,
                "total_tokens": response.total_tokens,
                "latency_ms": response.latency_ms,
            },
        )

        return response

    @staticmethod
    def _build_messages(
        messages: list[dict],
        system_prompt: str | None,
    ) -> list[dict]:
        """Prepend the system message if provided."""
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + messages
        return messages