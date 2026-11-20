"""Command line: run the server, mint local tokens, onboard teams, run the scripted demo."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import get_settings


def _demo() -> None:
    """A scripted tour through the real HTTP API with the mock model, printing each call."""
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient

    from .app import build_gateway, create_app
    from .auth import ADMIN_SCOPE, INVOKE_SCOPE, mint_dev_token

    home = Path(tempfile.mkdtemp(prefix="llmgw-demo-"))
    settings = get_settings(home=home, mode="local", retry_base_s=0.0)
    gateway = build_gateway(settings)
    client = TestClient(create_app(gateway))
    tok = lambda cid, scopes=(INVOKE_SCOPE,): {"Authorization": "Bearer " + mint_dev_token(home, settings.issuer, cid, list(scopes))}  # noqa: E731
    risk, support, marketing = tok("dev-risk-analytics"), tok("dev-customer-support"), tok("dev-marketing")

    def show(title, resp, keys=None):
        body = resp.json()
        if keys:
            body = {k: body[k] for k in keys if k in body}
        if isinstance(body.get("error"), dict):
            body["error"].pop("request_id", None)
        print((f"\n# {title}\n" if title else "") + f"< {resp.status_code}  {json.dumps(body)}")

    ask = lambda text, model: {"model": model, "messages": [{"role": "user", "content": text}]}  # noqa: E731
    keys = ["model", "output", "usage", "cache", "pii", "fallback_used"]
    show("risk-analytics: prompt with an email and a card number (model only sees placeholders)",
         client.post("/v1/chat", headers=risk, json=ask("Draft a reply to jane.doe@example.com about card 4111 1111 1111 1111.", "claude-sonnet")), keys)
    print("  model received:", json.dumps(gateway.provider.received[-1]["messages"][-1]["content"]))
    show("same prompt for a different customer: cache hit, $0, values restored",
         client.post("/v1/chat", headers=risk, json=ask("Draft a reply to sam.lee@example.com about card 4111 1111 1111 1111.", "claude-sonnet")),
         ["output", "usage", "cache"])
    show("customer-support asks for a model its policy does not allow",
         client.post("/v1/chat", headers=support, json=ask("Summarise this ticket.", "claude-sonnet")))
    show("marketing sends customer data: its policy blocks PII",
         client.post("/v1/chat", headers=marketing, json=ask("Email offer to 415-555-0132 and ann@example.com", "nova-lite")))
    gateway.provider.down_models = {settings.models["claude-haiku"].model_id}
    show("Claude Haiku is throttling: support's request falls back to Nova Lite",
         client.post("/v1/chat", headers=support, json={**ask("Suggest a reply about a late fee.", "claude-haiku"), "temperature": 0.3}),
         ["model", "fallback_used", "usage"])
    gateway.provider.down_models = set()
    statuses = [client.post("/v1/chat", headers=marketing, json=ask(f"Tagline idea {i} for a savings account.", "nova-lite")).status_code
                for i in range(12)]
    print(f"\n# marketing is limited to 10 requests per minute; 12 requests -> {statuses.count(200)} served, then:")
    show("", client.post("/v1/chat", headers=marketing, json=ask("One more tagline.", "nova-lite")))
    month = __import__("llmgw.service", fromlist=["month_of"]).month_of(__import__("time").time())
    gateway.store.adjust_spend("risk-analytics", month, 199_985_000)   # simulate a month of earlier usage
    print("\n# risk-analytics has already spent $199.99 of its $200 budget this month (simulated)")
    show("a request whose worst-case cost would cross the budget is refused before the model is called",
         client.post("/v1/chat", headers=risk, json=ask("Summarise the Q3 credit memo.", "claude-sonnet") | {"max_tokens": 2048}))
    usage = client.get("/admin/usage", headers=tok("platform-admin", [ADMIN_SCOPE])).json()
    print("\n# admin usage report (from the audit log)")
    for t in usage["teams"]:
        print(f"  {t['team_id']:<17} requests={t['requests']:<3} spent=${t['spent_usd']:.4f} of ${t['budget_usd']:g} "
              f"outcomes={t['outcomes']} pii={t['pii_detected']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="llmgw", description="Enterprise LLM gateway for Amazon Bedrock.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    t = sub.add_parser("token", help="mint a local development access token")
    t.add_argument("--client", default="dev-risk-analytics", help="client id (local demo clients: dev-risk-analytics, "
                   "dev-customer-support, dev-marketing)")
    t.add_argument("--admin", action="store_true", help="admin scope instead of invoke")
    sub.add_parser("teams", help="list team policies")
    sub.add_parser("demo", help="scripted tour of the gateway with the mock model")
    o = sub.add_parser("onboard", help="AWS: create a team's Cognito client and policy")
    o.add_argument("--team", required=True)
    o.add_argument("--models", required=True, help="comma list of model aliases")
    o.add_argument("--budget", type=float, required=True, help="monthly budget in USD")
    o.add_argument("--rpm", type=int, default=60)
    o.add_argument("--tpm", type=int, default=100_000)
    o.add_argument("--max-output-tokens", type=int, default=2048)
    o.add_argument("--pii", default="pseudonymize", choices=["pseudonymize", "redact", "block", "off"])
    o.add_argument("--owner", default="")
    o.add_argument("--description", default="")
    args = p.parse_args(argv)

    settings = get_settings()
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("llmgw.app:app", host=args.host, port=args.port)
    elif args.cmd == "token":
        if settings.mode != "local":
            sys.exit("Tokens come from Cognito in aws mode (client-credentials grant on the user pool domain).")
        from .auth import ADMIN_SCOPE, INVOKE_SCOPE, mint_dev_token

        print(mint_dev_token(settings.home, settings.issuer, args.client, [ADMIN_SCOPE if args.admin else INVOKE_SCOPE]))
    elif args.cmd == "teams":
        from .app import build_gateway

        for team in build_gateway(settings).store.list_teams():
            print(json.dumps(team.to_dict()))
    elif args.cmd == "demo":
        _demo()
    elif args.cmd == "onboard":
        if settings.mode != "aws":
            sys.exit("onboard creates a Cognito app client: set GATEWAY_MODE=aws and COGNITO_USER_POOL_ID, STATE_TABLE, AUDIT_TABLE.")
        import boto3

        from .onboard import onboard
        from .store.dynamo import DynamoStore
        from .teams import TeamPolicy

        unknown = set(args.models.split(",")) - set(settings.models)
        if unknown:
            sys.exit(f"Unknown models: {sorted(unknown)}")
        policy = TeamPolicy(team_id=args.team, client_ids=[], allowed_models=args.models.split(","), monthly_budget_usd=args.budget,
                            requests_per_minute=args.rpm, tokens_per_minute=args.tpm, max_output_tokens=args.max_output_tokens,
                            pii_mode=args.pii, owner=args.owner, description=args.description)
        store = DynamoStore(settings.state_table, settings.audit_table, region=settings.region)
        policy, secret = onboard(store, boto3.client("cognito-idp", region_name=settings.region), settings.user_pool_id, policy)
        print(json.dumps({"team_id": policy.team_id, "client_id": policy.client_ids[0], "client_secret": secret,
                          "note": "Shown once. Share it through your secrets process."}, indent=2))


if __name__ == "__main__":
    main()
