"""Settings from environment variables.

    GATEWAY_MODE            local (default: SQLite state, mock model, regex PII) | aws
    GATEWAY_HOME            local state, dev signing keys and audit log (default ~/.llmgw)
    GATEWAY_MODELS_FILE     model catalog JSON (default: the bundled llmgw/models.json)
    GATEWAY_MAX_ATTEMPTS    model calls per request across retries and fallbacks (default 3)

AWS mode:
    AWS_REGION, COGNITO_USER_POOL_ID, STATE_TABLE, AUDIT_TABLE,
    PII_DETECTOR            comprehend (default in aws mode) | regex
    PII_MIN_SCORE           Comprehend confidence threshold (default 0.5)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    input_per_m: float
    output_per_m: float
    fallback: tuple[str, ...] = ()

    def cost_micro_usd(self, input_tokens: int, output_tokens: int) -> int:
        """Cost in millionths of a dollar. Integers keep budget arithmetic exact."""
        return round(input_tokens * self.input_per_m + output_tokens * self.output_per_m)


@dataclass(frozen=True)
class Settings:
    mode: str = "local"
    home: Path = Path.home() / ".llmgw"
    models: dict[str, ModelSpec] = field(default_factory=dict)
    max_attempts: int = 3
    region: str = "us-east-1"
    user_pool_id: str | None = None
    state_table: str | None = None
    audit_table: str | None = None
    pii_detector: str = "regex"
    pii_min_score: float = 0.5
    retry_base_s: float = 0.2

    @property
    def issuer(self) -> str:
        if self.mode == "aws":
            return f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}"
        return "http://localhost/llmgw-dev"


def load_models(path: Path) -> dict[str, ModelSpec]:
    raw = json.loads(path.read_text())["models"]
    return {alias: ModelSpec(alias=alias, model_id=m["model_id"], input_per_m=m["input_per_m"],
                             output_per_m=m["output_per_m"], fallback=tuple(m.get("fallback", ())))
            for alias, m in raw.items()}


def get_settings(**overrides) -> Settings:
    mode = os.getenv("GATEWAY_MODE", "local")
    values = dict(
        mode=mode,
        home=Path(os.getenv("GATEWAY_HOME", Path.home() / ".llmgw")),
        models=load_models(Path(os.getenv("GATEWAY_MODELS_FILE", PACKAGE_DIR / "models.json"))),
        max_attempts=int(os.getenv("GATEWAY_MAX_ATTEMPTS", "3")),
        region=os.getenv("AWS_REGION", "us-east-1"),
        user_pool_id=os.getenv("COGNITO_USER_POOL_ID"),
        state_table=os.getenv("STATE_TABLE"),
        audit_table=os.getenv("AUDIT_TABLE"),
        pii_detector=os.getenv("PII_DETECTOR", "comprehend" if mode == "aws" else "regex"),
        pii_min_score=float(os.getenv("PII_MIN_SCORE", "0.5")),
        retry_base_s=float(os.getenv("GATEWAY_RETRY_BASE_S", "0.2")),
    )
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
