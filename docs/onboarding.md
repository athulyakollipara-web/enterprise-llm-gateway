# Onboarding a team

This is the runbook a platform engineer follows when a new team asks for LLM access. It usually takes a few
minutes. Most of the work is agreeing the policy with the team, not the commands.

## 1. Agree the policy (15-minute call with the team)

| Question | Becomes |
|---|---|
| What are you building, and who owns it? | `description`, `owner` |
| Which models do you need? Start with the cheapest that works. | `allowed_models` |
| Will prompts contain customer data? | `pii_mode`: `pseudonymize` (default), `redact`, or `block` if they should never send any |
| Expected volume per day, and peak per minute? | `requests_per_minute`, `tokens_per_minute` |
| Monthly budget, and who approves overruns? | `monthly_budget_usd` |
| Longest answer you need? | `max_output_tokens` |
| Any content rules beyond PII (topics, tone)? | optional Bedrock Guardrail, `guardrail_id` |

Rules of thumb:
- Budget about 1.5x the expected monthly cost. The gateway refuses requests that would cross the budget, so too tight a budget looks like an outage to the team.
- Requests per minute about 2x the peak.

## 2. Create the team

```bash
export GATEWAY_MODE=aws AWS_REGION=us-east-1
export COGNITO_USER_POOL_ID=<UserPoolId output> STATE_TABLE=<StateTable output> AUDIT_TABLE=<AuditTable output>

llmgw onboard --team fraud-ops --models claude-haiku,nova-lite --budget 300 \
  --rpm 60 --tpm 100000 --max-output-tokens 1024 --pii pseudonymize \
  --owner fraud-leads --description "Fraud analyst case summaries"
```

The command does two things:
- It creates a Cognito app client named `llmgw-fraud-ops`, allowed only the `client_credentials` flow and the `gateway/invoke` scope, with 60-minute tokens.
- It writes the team policy to DynamoDB.

It prints the client ID and secret **once**. Send the secret through your secrets process, never in chat or email.

## 3. What the team does

```bash
# get a token (valid 60 minutes; cache it and refresh before expiry)
TOKEN=$(curl -s -X POST "$TOKEN_URL" -u "$CLIENT_ID:$CLIENT_SECRET" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&scope=gateway/invoke" | python -c "import json,sys;print(json.load(sys.stdin)['access_token'])")

curl -s "$API_URL/v1/models" -H "Authorization: Bearer $TOKEN"     # what they may use
curl -s "$API_URL/v1/chat" -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d '{"model": "claude-haiku", "messages": [{"role": "user", "content": "Summarise case 4411 in three bullets."}]}'
curl -s "$API_URL/v1/usage" -H "Authorization: Bearer $TOKEN"      # spend so far this month
```

Tell the team how to handle each error type:

| Status | `error.type` | What the client should do |
|---|---|---|
| 429 | `rate_limited`, `token_rate_limited` | Wait `Retry-After` seconds, then retry |
| 402 | `budget_exceeded` | Stop and contact the platform team (budget owner approves an increase) |
| 422 | `pii_blocked` | Remove customer data from the prompt; this team's policy forbids it |
| 403 | `model_not_allowed` | Use a model from `/v1/models` or request a policy change |
| 503 | `upstream_unavailable` | All models in the fallback chain failed; retry with backoff |

## 4. Change or offboard

- **Change limits or models:** `PUT /admin/teams/{team_id}` with an admin token. It takes effect on the next request.
- **Rotate the secret:** create a new app client, add its ID to `client_ids`, let the team switch over, then delete the old client.
- **Offboard:** delete the Cognito app client. Its tokens stop working within 60 minutes, or at once if they are revoked. Keep the policy for the audit trail.

## 5. Monthly review

Pull `GET /admin/usage.csv?month=YYYY-MM` into Tableau or a spreadsheet and check four things:
- **Spend against budget:** teams above 80% may need a budget increase.
- **Error mix:** many `rate_limited` errors means the limits are too low; many `pii_blocked` means the team needs guidance.
- **Cache hit rate:** a high rate suggests an FAQ the team could serve without a model.
- **Fallback share:** a high share means the primary model is unhealthy in this region.
