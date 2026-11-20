"""HTTP API (FastAPI). Runs the same way under uvicorn locally and on AWS Lambda via the Lambda Web Adapter."""
from __future__ import annotations

import csv
import io
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .auth import ADMIN_SCOPE, INVOKE_SCOPE, AuthError, Caller, TokenVerifier
from .config import Settings, get_settings
from .service import Gateway, GatewayError
from .teams import DEMO_TEAMS, TeamPolicy


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = Field(examples=["claude-haiku"])
    messages: list[Message]
    system: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=64_000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class TeamIn(BaseModel):
    client_ids: list[str]
    allowed_models: list[str]
    monthly_budget_usd: float = Field(ge=0)
    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=100_000, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)
    pii_mode: Literal["pseudonymize", "redact", "block", "off"] = "pseudonymize"
    cache_ttl_s: int = Field(default=3600, ge=0)
    guardrail_id: str | None = None
    guardrail_version: str = "DRAFT"
    description: str = ""
    owner: str = ""


def build_gateway(settings: Settings, **kwargs) -> Gateway:
    """Wire the local or AWS implementations from settings."""
    if settings.mode == "aws":
        from .pii import ComprehendDetector, RegexDetector
        from .providers.bedrock import BedrockProvider
        from .store.dynamo import DynamoStore

        store = DynamoStore(settings.state_table, settings.audit_table, region=settings.region)
        provider = BedrockProvider(region=settings.region)
        detector = (ComprehendDetector(region=settings.region, min_score=settings.pii_min_score)
                    if settings.pii_detector == "comprehend" else RegexDetector())
    else:
        from .pii import RegexDetector
        from .providers.mock import MockProvider
        from .store.sqlite import SqliteStore

        store = SqliteStore(settings.home / "state.db")
        provider = MockProvider()
        detector = RegexDetector()
        if not store.list_teams():
            for team in DEMO_TEAMS:
                store.put_team(team)
    return Gateway(settings, kwargs.pop("store", store), kwargs.pop("provider", provider),
                   kwargs.pop("detector", detector), **kwargs)


def create_app(gateway: Gateway | None = None, verifier: TokenVerifier | None = None) -> FastAPI:
    settings = gateway.settings if gateway else get_settings()
    gateway = gateway or build_gateway(settings)
    verifier = verifier or TokenVerifier.for_settings(settings)

    app = FastAPI(title="Enterprise LLM Gateway", version=__version__,
                  description="One governed door from every internal team to Amazon Bedrock.")
    app.state.gateway = gateway

    def error(status: int, type_: str, message: str, request_id: str, headers: dict | None = None, extra: dict | None = None):
        return JSONResponse({"error": {"type": type_, "message": message, "request_id": request_id, **(extra or {})}},
                            status_code=status, headers={"x-request-id": request_id, **(headers or {})})

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        try:
            response = await call_next(request)
        except AuthError as e:
            return error(401, "unauthorized", str(e), request.state.request_id, {"WWW-Authenticate": "Bearer"})
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, e: GatewayError):
        return error(e.status, e.type, e.message, request.state.request_id, e.headers, e.extra)

    @app.exception_handler(AuthError)
    async def auth_error(request: Request, e: AuthError):
        return error(401, "unauthorized", str(e), request.state.request_id, {"WWW-Authenticate": "Bearer"})

    def caller(authorization: str | None = Header(default=None)) -> Caller:
        return verifier.verify(authorization)

    def team_of(c: Caller = Depends(caller)) -> TeamPolicy:
        if INVOKE_SCOPE not in c.scopes:
            raise GatewayError(403, "forbidden", f"Token lacks the {INVOKE_SCOPE} scope.")
        team = gateway.store.team_for_client(c.client_id)
        if team is None:
            raise GatewayError(403, "unknown_client", "This client is not onboarded to any team.")
        return team

    def admin(c: Caller = Depends(caller)) -> Caller:
        if ADMIN_SCOPE not in c.scopes:
            raise GatewayError(403, "forbidden", f"Admin endpoints need the {ADMIN_SCOPE} scope.")
        return c

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "version": __version__, "mode": settings.mode}

    @app.post("/v1/chat")
    def chat(body: ChatRequest, request: Request, team: TeamPolicy = Depends(team_of)):
        result = gateway.chat(team, model=body.model, messages=[m.model_dump() for m in body.messages], system=body.system,
                              max_tokens=body.max_tokens, temperature=body.temperature, request_id=request.state.request_id)
        return JSONResponse(result, headers={"x-cache": result["cache"],
                                             "x-ratelimit-limit-requests": str(team.requests_per_minute),
                                             "x-ratelimit-remaining-requests": str(result["rate_limit"]["remaining"])})

    @app.get("/v1/models")
    def models(team: TeamPolicy = Depends(team_of)):
        return {"team_id": team.team_id, "models": [
            {"model": a, "model_id": s.model_id, "input_usd_per_m": s.input_per_m, "output_usd_per_m": s.output_per_m,
             "fallback": [f for f in s.fallback if f in team.allowed_models]}
            for a, s in settings.models.items() if a in team.allowed_models]}

    @app.get("/v1/usage")
    def my_usage(month: str | None = None, team: TeamPolicy = Depends(team_of)):
        return gateway.usage(team.team_id, month)["teams"][0]

    @app.get("/admin/teams", dependencies=[Depends(admin)])
    def list_teams():
        return {"teams": [t.to_dict() for t in gateway.store.list_teams()]}

    @app.put("/admin/teams/{team_id}", dependencies=[Depends(admin)])
    def put_team(team_id: str, body: TeamIn):
        unknown = sorted(set(body.allowed_models) - set(settings.models))
        if unknown:
            raise GatewayError(400, "unknown_model", f"Not in the model catalog: {unknown}.")
        policy = TeamPolicy(team_id=team_id, **body.model_dump())
        gateway.store.put_team(policy)
        return policy.to_dict()

    @app.get("/admin/usage", dependencies=[Depends(admin)])
    def all_usage(month: str | None = None):
        return gateway.usage(None, month)

    @app.get("/admin/usage.csv", dependencies=[Depends(admin)], response_class=PlainTextResponse)
    def usage_csv(month: str | None = None):
        rows = gateway.usage_rows(month)
        buf = io.StringIO()
        fields = ["date", "team_id", "model", "outcome", "requests", "input_tokens", "output_tokens", "cost_usd",
                  "cache_hits", "fallbacks", "pii_entities"]
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"content-disposition": f"attachment; filename=llm-usage-{month or 'current'}.csv"})

    return app


def __getattr__(name):   # ``uvicorn llmgw.app:app`` builds the app lazily from the environment
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
