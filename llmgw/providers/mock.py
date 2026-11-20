"""Offline stand-in for Bedrock with fault injection.

It records every prompt it receives, which is how the tests and the PII evaluation prove that no
raw PII value ever reaches the model.
"""
from __future__ import annotations

import random
import threading
import time

from .base import Completion, ModelError


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class MockProvider:
    def __init__(self, latency_ms: float = 0.0, fail_rate: float = 0.0, down_models: set[str] | None = None,
                 seed: int = 7, reply_words: int = 12):
        self.latency_ms = latency_ms
        self.fail_rate = fail_rate
        self.down_models = set(down_models or ())
        self.reply_words = reply_words
        self.received: list[dict] = []
        self.calls = 0
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def converse(self, model_id, messages, system, max_tokens, temperature, guardrail=None) -> Completion:
        start = time.perf_counter()
        with self._lock:
            self.calls += 1
            self.received.append({"model_id": model_id, "system": system, "messages": messages})
            fail = model_id in self.down_models or self._rng.random() < self.fail_rate
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        if fail:
            raise ModelError("ThrottlingException", f"Too many requests for {model_id}", retryable=True)
        last = messages[-1]["content"]
        placeholders = [w.strip(".,;:") for w in last.split() if w.startswith("[") and "_" in w]
        body = f"[mock reply] Re: {last[:70]}"
        if placeholders:
            body += f" | I will follow up with {placeholders[0]}."
        filler = " ".join(["lorem"] * self.reply_words)
        text = (body + " " + filler)[: max_tokens * 4]
        prompt = (system or "") + " ".join(m["content"] for m in messages)
        return Completion(text=text, input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(text),
                          stop_reason="end_turn", latency_ms=(time.perf_counter() - start) * 1000)
