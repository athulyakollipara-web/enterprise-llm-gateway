import json
import threading

import pytest

from llmgw.providers.base import ModelError
from llmgw.providers.mock import MockProvider
from llmgw.service import GatewayError, month_of
from llmgw.teams import TeamPolicy

PII_PROMPT = "Reply to jane.doe@example.com, card 4111 1111 1111 1111, SSN 123-45-6789, phone (415) 555-0132."
RAW_VALUES = ["jane.doe@example.com", "4111 1111 1111 1111", "123-45-6789", "(415) 555-0132"]


def team(gw, team_id="risk-analytics", **changes):
    t = gw.store.get_team(team_id)
    for k, v in changes.items():
        setattr(t, k, v)
    gw.store.put_team(t)
    return t


def chat(gw, t, text="hello", model="claude-sonnet", **kw):
    return gw.chat(t, model=model, messages=[{"role": "user", "content": text}], **kw)


def test_model_policy(gateway):
    support = gateway.store.get_team("customer-support")
    with pytest.raises(GatewayError) as e:
        chat(gateway, support, model="claude-sonnet")
    assert e.value.status == 403 and e.value.type == "model_not_allowed"
    with pytest.raises(GatewayError) as e:
        chat(gateway, support, model="gpt-9")
    assert e.value.status == 400


@pytest.mark.parametrize("bad", [[], [{"role": "assistant", "content": "x"}],
                                 [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
                                 [{"role": "user", "content": "  "}]])
def test_message_validation(gateway, bad):
    with pytest.raises(GatewayError) as e:
        gateway.chat(team(gateway), model="claude-sonnet", messages=bad)
    assert e.value.status == 400


@pytest.mark.parametrize("mode", ["pseudonymize", "redact"])
def test_raw_pii_never_reaches_the_model(make_gateway, mode):
    provider = MockProvider()
    gw = make_gateway(provider)
    t = team(gw, pii_mode=mode)
    out = gw.chat(t, model="claude-sonnet", system="Customer email on file: jane.doe@example.com",
                  messages=[{"role": "user", "content": PII_PROMPT}])
    sent = json.dumps(provider.received)
    assert not any(v in sent for v in RAW_VALUES)
    assert "[EMAIL_1]" in sent and out["pii"]["found"]["EMAIL"] == 2
    # pseudonymize restores the real value in the answer; redact does not
    assert ("jane.doe@example.com" in out["output"]) is (mode == "pseudonymize")


def test_block_mode_rejects_before_any_model_call(make_gateway):
    provider = MockProvider()
    gw = make_gateway(provider)
    with pytest.raises(GatewayError) as e:
        chat(gw, team(gw, "marketing"), PII_PROMPT, model="nova-lite")
    assert e.value.status == 422 and provider.calls == 0


def test_audit_log_has_no_content(make_gateway):
    gw = make_gateway()
    chat(gw, team(gw), PII_PROMPT)
    with pytest.raises(GatewayError):
        chat(gw, team(gw, "marketing"), PII_PROMPT, model="nova-lite")
    dumped = json.dumps(gw.store.audit_records())
    assert not any(v in dumped for v in RAW_VALUES + ["Reply to"])
    assert len(gw.store.audit_records()) == 2


def test_request_rate_limit_and_window_reset(gateway, clock):
    t = team(gateway, requests_per_minute=3, cache_ttl_s=0)
    for _ in range(3):
        chat(gateway, t)
    with pytest.raises(GatewayError) as e:
        chat(gateway, t)
    assert e.value.status == 429 and "Retry-After" in e.value.headers
    clock.t += 60
    assert chat(gateway, t)["output"]


def test_token_limits(gateway):
    t = team(gateway, tokens_per_minute=3000, max_output_tokens=1000, cache_ttl_s=0)
    with pytest.raises(GatewayError) as e:
        chat(gateway, t, "x" * 20_000)
    assert e.value.status == 413
    chat(gateway, t, max_tokens=1000)
    chat(gateway, t, max_tokens=1000)
    with pytest.raises(GatewayError) as e:
        chat(gateway, t, max_tokens=1000)
    assert e.value.type == "token_rate_limited"


def test_max_tokens_is_capped_by_policy(make_gateway):
    seen = {}

    class Spy(MockProvider):
        def converse(self, model_id, messages, system, max_tokens, temperature, guardrail=None):
            seen["max_tokens"] = max_tokens
            return super().converse(model_id, messages, system, max_tokens, temperature, guardrail)

    gw = make_gateway(Spy())
    chat(gw, team(gw, max_output_tokens=256), max_tokens=50_000)
    assert seen["max_tokens"] == 256


def test_cache_only_for_deterministic_requests(make_gateway):
    provider = MockProvider()
    gw = make_gateway(provider)
    t = team(gw)
    first, second = chat(gw, t, "Explain APR"), chat(gw, t, "Explain APR")
    assert (first["cache"], second["cache"]) == ("MISS", "HIT")
    assert second["usage"]["cost_usd"] == 0 and provider.calls == 1
    assert chat(gw, t, "Explain APR", temperature=0.7)["cache"] == "OFF" and provider.calls == 2


def test_cache_shared_across_customers_without_storing_pii(make_gateway):
    gw = make_gateway()
    t = team(gw)
    chat(gw, t, "Write to ann@example.com about the fee")
    hit = chat(gw, t, "Write to bob@example.com about the fee")
    assert hit["cache"] == "HIT" and "bob@example.com" in hit["output"] and "ann@example.com" not in hit["output"]
    cached = gw.store._db.execute("SELECT body FROM cache").fetchall()
    assert "example.com" not in json.dumps(cached)


def test_fallback_when_primary_throttles(make_gateway, settings):
    provider = MockProvider(down_models={settings.models["claude-sonnet"].model_id})
    gw = make_gateway(provider)
    out = chat(gw, team(gw), temperature=0.5)
    assert out["model"] == "nova-pro" and out["fallback_used"]
    assert [c["model_id"] for c in provider.received].count(settings.models["claude-sonnet"].model_id) == 2  # one retry


def test_fallback_respects_team_policy(make_gateway, settings):
    provider = MockProvider(down_models={settings.models["nova-lite"].model_id})
    gw = make_gateway(provider)
    with pytest.raises(GatewayError) as e:      # marketing may not use the fallback (claude-haiku)
        chat(gw, team(gw, "marketing"), "tagline", model="nova-lite")
    assert e.value.status == 503


def test_circuit_breaker_skips_a_failing_model(make_gateway, settings):
    sonnet = settings.models["claude-sonnet"].model_id
    provider = MockProvider(down_models={sonnet})
    gw = make_gateway(provider)
    t = team(gw, cache_ttl_s=0)
    for _ in range(3):
        chat(gw, t)
    calls_before = [c["model_id"] for c in provider.received].count(sonnet)
    chat(gw, t)
    assert [c["model_id"] for c in provider.received].count(sonnet) == calls_before   # breaker open: went straight to nova-pro


def test_non_retryable_errors_do_not_fall_back(make_gateway):
    class Bad(MockProvider):
        def converse(self, *a, **k):
            self.calls += 1
            raise ModelError("ValidationException", "bad input", retryable=False)

    gw = make_gateway(Bad())
    with pytest.raises(GatewayError) as e:
        chat(gw, team(gw))
    assert e.value.status == 400 and gw.provider.calls == 1


def test_failed_calls_release_the_budget_reservation(make_gateway, settings, clock):
    provider = MockProvider(down_models={m.model_id for m in settings.models.values()})
    gw = make_gateway(provider)
    with pytest.raises(GatewayError):
        chat(gw, team(gw))
    assert gw.store.get_spend("risk-analytics", month_of(clock())) == 0


def test_spend_is_settled_to_actual_cost(gateway, clock, settings):
    out = chat(gateway, team(gateway), "Explain APR", max_tokens=2000)
    spent = gateway.store.get_spend("risk-analytics", month_of(clock()))
    spec = settings.models["claude-sonnet"]
    assert spent == spec.cost_micro_usd(out["usage"]["input_tokens"], out["usage"]["output_tokens"])


def test_budget_refuses_before_calling_the_model(make_gateway, clock):
    provider = MockProvider()
    gw = make_gateway(provider)
    t = team(gw, monthly_budget_usd=0.01)
    with pytest.raises(GatewayError) as e:
        chat(gw, t, max_tokens=2048)                 # worst case ~$0.03 > $0.01
    assert e.value.status == 402 and provider.calls == 0


@pytest.mark.parametrize("mode,can_overspend", [("reserve", False), ("post_charge", True)])
def test_concurrent_requests_and_budget(make_gateway, clock, mode, can_overspend):
    """50 parallel requests against a budget that fits ~10. Reservation never overspends; charging after the call does."""
    provider = MockProvider(latency_ms=20)
    gw = make_gateway(provider, budget_mode=mode)
    t = team(gw, monthly_budget_usd=0.01, requests_per_minute=1000, cache_ttl_s=0, max_output_tokens=50)

    def worker():
        try:
            chat(gw, t, "Summarise the policy change in two lines.", max_tokens=50)
        except GatewayError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(50)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    spent = gw.store.get_spend(t.team_id, month_of(clock()))
    assert (spent > t.budget_micro_usd) is can_overspend


def test_usage_report(gateway):
    t = team(gateway)
    chat(gateway, t, "a")
    chat(gateway, t, "a")
    with pytest.raises(GatewayError):
        chat(gateway, t, model="gpt-9")
    report = gateway.usage("risk-analytics")["teams"][0]
    assert report["requests"] == 3 and report["outcomes"] == {"ok": 2, "unknown_model": 1}
    assert report["by_model"]["claude-sonnet"]["cache_hits"] == 1
    rows = gateway.usage_rows()
    assert sum(r["requests"] for r in rows) == 3


def test_policy_validation():
    with pytest.raises(ValueError):
        TeamPolicy(team_id="x", client_ids=[], allowed_models=[], monthly_budget_usd=1, pii_mode="maybe")
