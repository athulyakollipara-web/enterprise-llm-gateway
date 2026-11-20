"""Amazon Bedrock through the Converse API, so every model family uses one request shape."""
from __future__ import annotations

import time

from .base import Completion, ModelError

RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "ModelTimeoutException",
             "InternalServerException", "ServiceQuotaExceededException"}


class BedrockProvider:
    def __init__(self, client=None, region: str | None = None):
        import boto3
        from botocore.config import Config

        # The gateway does its own retries and fallback, so the SDK's are turned off.
        self.client = client or boto3.client("bedrock-runtime", region_name=region,
                                             config=Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=120))

    def converse(self, model_id, messages, system, max_tokens, temperature, guardrail=None) -> Completion:
        from botocore.exceptions import ClientError

        kwargs = {
            "modelId": model_id,
            "messages": [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if guardrail:
            kwargs["guardrailConfig"] = {"guardrailIdentifier": guardrail["id"], "guardrailVersion": guardrail["version"]}
        start = time.perf_counter()
        try:
            out = self.client.converse(**kwargs)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ModelError(code, e.response["Error"].get("Message", ""), retryable=code in RETRYABLE) from e
        text = "".join(block.get("text", "") for block in out["output"]["message"]["content"])
        return Completion(text=text, input_tokens=out["usage"]["inputTokens"], output_tokens=out["usage"]["outputTokens"],
                          stop_reason=out["stopReason"], latency_ms=(time.perf_counter() - start) * 1000)
