from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: float


class ModelError(Exception):
    """A model call failed. ``retryable`` errors (throttling, outages) trigger retry and fallback."""

    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable


class ModelProvider(Protocol):
    def converse(self, model_id: str, messages: list[dict], system: str | None, max_tokens: int,
                 temperature: float, guardrail: dict | None = None) -> Completion: ...
