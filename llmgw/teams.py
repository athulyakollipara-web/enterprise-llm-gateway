"""Per-team policy. One record per team; onboarding a team means writing one of these."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

PII_MODES = ("pseudonymize", "redact", "block", "off")


@dataclass
class TeamPolicy:
    team_id: str
    client_ids: list[str]
    allowed_models: list[str]
    monthly_budget_usd: float
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    max_output_tokens: int = 2048
    pii_mode: str = "pseudonymize"
    cache_ttl_s: int = 3600          # 0 disables the response cache
    guardrail_id: str | None = None   # optional Bedrock Guardrail applied to this team's calls
    guardrail_version: str = "DRAFT"
    description: str = ""
    owner: str = ""
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pii_mode not in PII_MODES:
            raise ValueError(f"pii_mode must be one of {PII_MODES}")
        if self.monthly_budget_usd < 0 or self.requests_per_minute < 1:
            raise ValueError("budget must be >= 0 and requests_per_minute >= 1")

    @property
    def budget_micro_usd(self) -> int:
        return round(self.monthly_budget_usd * 1_000_000)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TeamPolicy":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


DEMO_TEAMS = [
    TeamPolicy(team_id="risk-analytics", client_ids=["dev-risk-analytics"], allowed_models=["claude-sonnet", "nova-pro", "claude-haiku"],
               monthly_budget_usd=200, requests_per_minute=60, pii_mode="pseudonymize", owner="risk-analytics-leads",
               description="Credit risk analysts: memo drafting and portfolio Q&A"),
    TeamPolicy(team_id="customer-support", client_ids=["dev-customer-support"], allowed_models=["claude-haiku", "nova-lite"],
               monthly_budget_usd=50, requests_per_minute=120, max_output_tokens=800, pii_mode="pseudonymize",
               owner="support-platform", description="Agent-assist reply drafts"),
    TeamPolicy(team_id="marketing", client_ids=["dev-marketing"], allowed_models=["nova-lite"], monthly_budget_usd=25,
               requests_per_minute=10, max_output_tokens=400, pii_mode="block", owner="marketing-ops",
               description="Campaign copy; must never send customer data"),
]
