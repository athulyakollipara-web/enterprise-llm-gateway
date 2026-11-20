import jwt

from llmgw.auth import ADMIN_SCOPE

from .conftest import ask


def test_health(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_requires_a_valid_token(client, auth, settings):
    assert client.post("/v1/chat", json=ask("hi")).status_code == 401
    assert client.post("/v1/chat", json=ask("hi"), headers={"Authorization": "Bearer nonsense"}).status_code == 401
    expired = auth(ttl_s=-10)
    r = client.post("/v1/chat", json=ask("hi"), headers=expired)
    assert r.status_code == 401 and r.json()["error"]["type"] == "unauthorized"


def test_rejects_tokens_from_another_issuer(client, settings):
    from llmgw.auth import _dev_key
    forged = jwt.encode({"iss": "https://evil.example", "client_id": "dev-risk-analytics", "token_use": "access",
                         "scope": "gateway/invoke", "exp": 9_999_999_999}, _dev_key(settings.home), algorithm="RS256")
    assert client.post("/v1/chat", json=ask("hi"), headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_rejects_tokens_signed_with_another_key(client, settings, tmp_path):
    from llmgw.auth import mint_dev_token
    other = mint_dev_token(tmp_path / "other", settings.issuer, "dev-risk-analytics", ["gateway/invoke"])
    assert client.post("/v1/chat", json=ask("hi"), headers={"Authorization": f"Bearer {other}"}).status_code == 401


def test_scopes_and_unknown_clients(client, auth):
    assert client.post("/v1/chat", json=ask("hi"), headers=auth(scopes=[ADMIN_SCOPE])).status_code == 403
    r = client.post("/v1/chat", json=ask("hi"), headers=auth("not-onboarded"))
    assert r.status_code == 403 and r.json()["error"]["type"] == "unknown_client"
    assert client.get("/admin/teams", headers=auth()).status_code == 403


def test_chat_roundtrip_headers(client, auth):
    r = client.post("/v1/chat", json=ask("Email jane@example.com about APR"), headers={**auth(), "x-request-id": "req-123"})
    body = r.json()
    assert r.status_code == 200 and r.headers["x-request-id"] == "req-123" == body["id"]
    assert r.headers["x-cache"] == "MISS" and int(r.headers["x-ratelimit-remaining-requests"]) == 59
    assert body["pii"]["found"] == {"EMAIL": 1} and "jane@example.com" in body["output"]


def test_errors_are_structured(client, auth):
    r = client.post("/v1/chat", json=ask("hi", model="claude-sonnet"), headers=auth("dev-customer-support"))
    assert r.status_code == 403 and set(r.json()["error"]) == {"type", "message", "request_id"}
    assert client.post("/v1/chat", json={"model": "claude-sonnet"}, headers=auth()).status_code == 422


def test_models_and_own_usage(client, auth):
    models = client.get("/v1/models", headers=auth("dev-customer-support")).json()
    assert {m["model"] for m in models["models"]} == {"claude-haiku", "nova-lite"}
    client.post("/v1/chat", json=ask("hi", model="nova-lite"), headers=auth("dev-customer-support"))
    usage = client.get("/v1/usage", headers=auth("dev-customer-support")).json()
    assert usage["team_id"] == "customer-support" and usage["requests"] == 1


def test_admin_onboards_a_team_that_can_call_immediately(client, auth):
    admin = auth("platform", scopes=[ADMIN_SCOPE])
    body = {"client_ids": ["fraud-ops-client"], "allowed_models": ["claude-haiku"], "monthly_budget_usd": 100}
    assert client.put("/admin/teams/fraud-ops", json=body, headers=admin).status_code == 200
    r = client.post("/v1/chat", json=ask("hi", model="claude-haiku"), headers=auth("fraud-ops-client"))
    assert r.status_code == 200
    bad = {**body, "allowed_models": ["gpt-9"]}
    assert client.put("/admin/teams/x", json=bad, headers=admin).status_code == 400


def test_usage_csv_for_tableau(client, auth):
    client.post("/v1/chat", json=ask("hi"), headers=auth())
    csv_text = client.get("/admin/usage.csv", headers=auth("platform", scopes=[ADMIN_SCOPE])).text
    header, row = csv_text.strip().splitlines()[:2]
    assert header.startswith("date,team_id,model,outcome,requests") and ",risk-analytics,claude-sonnet,ok,1," in row
