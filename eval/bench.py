"""Load and resilience benchmark against the real HTTP server (uvicorn) with the mock model.

Scenarios
  overhead     500 sequential requests, then 2,000 requests, 16 concurrent clients, model latency 0: what the gateway itself adds
  failures     every model call fails 10% of the time: success rate with retry + fallback vs a direct call
  outage       Claude Sonnet fully down: success rate and model calls per request (circuit breaker)
  budget       200 parallel requests against a budget that fits ~20: reserve-then-settle vs charge-after
  cache        a workload where 40% of deterministic prompts repeat: hit rate and cost saved

    python eval/bench.py                 # writes eval/results/bench.json
Numbers are from one local process with SQLite state; on Lambda + DynamoDB the overhead is higher
(network round trips to DynamoDB), so treat overhead as the gateway's own CPU cost, not production latency.
"""
from __future__ import annotations

import json
import platform
import random
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import uvicorn

from llmgw.app import create_app
from llmgw.auth import INVOKE_SCOPE, TokenVerifier, mint_dev_token
from llmgw.config import get_settings
from llmgw.pii import RegexDetector
from llmgw.providers.base import ModelError
from llmgw.providers.mock import MockProvider
from llmgw.service import Gateway, GatewayError, month_of
from llmgw.store.sqlite import SqliteStore
from llmgw.teams import TeamPolicy

rng = random.Random(42)
PORT = 8765
PII_SNIPPETS = ["Customer email is ann.lee@example.com.", "Card 4111 1111 1111 1111 was declined.",
                "Call back on (415) 555-0132.", "SSN 123-45-6789 on file."]


def big_team(team_id, models, **kw):
    return TeamPolicy(team_id=team_id, client_ids=[team_id], allowed_models=models, monthly_budget_usd=10_000,
                      requests_per_minute=1_000_000, tokens_per_minute=10**9, **kw)


def make(settings, provider, teams, **kw):
    store = SqliteStore(settings.home / f"state-{time.time_ns()}.db")
    for t in teams:
        store.put_team(t)
    return Gateway(settings, store, provider, RegexDetector(), **kw)


def serve(gateway, settings):
    app = create_app(gateway, TokenVerifier.for_settings(settings))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    return server, thread


def prompt(i, pii_share=0.3):
    text = f"Request {i}: summarise the overdraft policy change for a customer in two sentences."
    if rng.random() < pii_share:
        text += " " + rng.choice(PII_SNIPPETS)
    return text


def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(p / 100 * len(values)))], 2)


def overhead(settings, n=2000, concurrency=16):
    teams = [big_team("risk", ["claude-sonnet", "nova-pro"]), big_team("support", ["claude-haiku", "nova-lite"])]
    gw = make(settings, MockProvider(), teams)
    server, thread = serve(gw, settings)
    headers = {t.team_id: {"Authorization": "Bearer " + mint_dev_token(settings.home, settings.issuer, t.team_id, [INVOKE_SCOPE])}
               for t in teams}
    jobs = [("risk", "claude-sonnet") if i % 2 else ("support", "claude-haiku") for i in range(n)]
    latencies, statuses = [], []
    with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as c:
        for _ in range(50):   # warm-up
            c.post("/v1/chat", headers=headers["risk"], json={"model": "claude-sonnet", "messages": [{"role": "user", "content": "warm"}]})

        def call(i):
            team, model = jobs[i]
            body = {"model": model, "temperature": 0.2, "messages": [{"role": "user", "content": prompt(i)}]}
            t0 = time.perf_counter()
            r = c.post("/v1/chat", headers=headers[team], json=body)
            return (time.perf_counter() - t0) * 1000, r.status_code

        t0 = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            for ms, status in pool.map(call, range(n)):
                latencies.append(ms)
                statuses.append(status)
        wall = time.perf_counter() - t0
    server.should_exit = True
    thread.join()
    gateway_ms = [r["gateway_ms"] for r in gw.store.audit_records() if r["outcome"] == "ok"]
    return {"requests": n, "concurrency": concurrency, "ok": statuses.count(200), "throughput_rps": round(n / wall, 1),
            "client_latency_ms": {"p50": pct(latencies, 50), "p95": pct(latencies, 95), "p99": pct(latencies, 99)},
            "gateway_processing_ms": {"p50": pct(gateway_ms, 50), "p95": pct(gateway_ms, 95), "p99": pct(gateway_ms, 99)}}


def failures(settings, n=2000, fail_rate=0.10):
    team = big_team("risk", ["claude-sonnet", "nova-pro"], cache_ttl_s=0)
    direct_provider = MockProvider(fail_rate=fail_rate, seed=1)
    direct_ok = 0
    for i in range(n):
        try:
            direct_provider.converse(settings.models["claude-sonnet"].model_id, [{"role": "user", "content": "x"}], None, 100, 0)
            direct_ok += 1
        except ModelError:
            pass
    provider = MockProvider(fail_rate=fail_rate, seed=1)
    gw = make(settings, provider, [team])
    ok = fallback = 0
    for i in range(n):
        try:
            out = gw.chat(team, model="claude-sonnet", messages=[{"role": "user", "content": prompt(i, 0)}])
            ok += 1
            fallback += out["fallback_used"]
        except GatewayError:
            pass
    return {"requests": n, "model_failure_rate": fail_rate, "direct_call_success": round(direct_ok / n, 4),
            "gateway_success": round(ok / n, 4), "served_by_fallback": round(fallback / n, 4),
            "model_calls_per_request": round(provider.calls / n, 3)}


def outage(settings, n=1000):
    team = big_team("risk", ["claude-sonnet", "nova-pro"], cache_ttl_s=0)
    provider = MockProvider(down_models={settings.models["claude-sonnet"].model_id})
    gw = make(settings, provider, [team])
    ok = sum(gw.chat(team, model="claude-sonnet", messages=[{"role": "user", "content": prompt(i, 0)}])["fallback_used"]
             for i in range(n))
    gw.breaker.threshold = 10**9   # same run with the circuit breaker disabled, for comparison
    provider2 = MockProvider(down_models={settings.models["claude-sonnet"].model_id})
    gw2 = make(settings, provider2, [team])
    gw2.breaker.threshold = 10**9
    for i in range(n):
        gw2.chat(team, model="claude-sonnet", messages=[{"role": "user", "content": prompt(i, 0)}])
    return {"requests": n, "direct_call_success": 0.0, "gateway_success": round(ok / n, 4),
            "model_calls_per_request_with_breaker": round(provider.calls / n, 3),
            "model_calls_per_request_without_breaker": round(provider2.calls / n, 3)}


def budget(settings, n=200):
    result = {}
    for mode in ("post_charge", "reserve"):
        team = TeamPolicy(team_id="sandbox", client_ids=["sandbox"], allowed_models=["claude-sonnet"], monthly_budget_usd=0.02,
                          requests_per_minute=10**6, tokens_per_minute=10**9, max_output_tokens=60, cache_ttl_s=0)
        gw = make(settings, MockProvider(latency_ms=50), [team], budget_mode=mode)
        served = []

        def call(i):
            try:
                gw.chat(team, model="claude-sonnet", messages=[{"role": "user", "content": prompt(i, 0)}], max_tokens=60)
                return 1
            except GatewayError:
                return 0

        with ThreadPoolExecutor(32) as pool:
            served = sum(pool.map(call, range(n)))
        spent = gw.store.get_spend("sandbox", month_of(time.time())) / 1e6
        result[mode] = {"budget_usd": 0.02, "requests_served": served, "spent_usd": round(spent, 6),
                        "overspend_pct": round(max(0.0, spent - 0.02) / 0.02 * 100, 1)}
    return result


def cache(settings, n=2000, repeat_share=0.4):
    team = big_team("support", ["claude-haiku", "nova-lite"])
    provider = MockProvider()
    gw = make(settings, provider, [team])
    pool_of_common = [f"How do I dispute a card charge? Variant {k}" for k in range(25)]
    hits = cost = 0
    for i in range(n):
        text = rng.choice(pool_of_common) if rng.random() < repeat_share else prompt(i, 0.3)
        out = gw.chat(team, model="claude-haiku", messages=[{"role": "user", "content": text}])
        hits += out["cache"] == "HIT"
        cost += out["usage"]["cost_usd"]
    spec = settings.models["claude-haiku"]
    uncached = sum(spec.cost_micro_usd(r.get("input_tokens", 0), r.get("output_tokens", 0))
                   for r in gw.store.audit_records() if r["cache"] == "MISS") / 1e6
    avg = uncached / (n - hits)
    return {"requests": n, "cache_hit_rate": round(hits / n, 4), "model_calls": provider.calls,
            "cost_usd": round(cost, 6), "cost_without_cache_usd_est": round(cost + hits * avg, 6),
            "cost_saved_pct": round(100 * hits * avg / (cost + hits * avg), 1)}


def main(out="eval/results/bench.json"):
    home = Path(tempfile.mkdtemp(prefix="llmgw-bench-"))
    settings = get_settings(home=home, mode="local", retry_base_s=0.0)
    results = {"environment": {"python": platform.python_version(), "machine": platform.machine(), "state": "sqlite (local)",
                               "model": "mock, 0 ms unless noted"}}
    for name, fn in [("overhead_single_client", lambda s: overhead(s, n=500, concurrency=1)), ("overhead", overhead), ("failures", failures), ("outage", outage), ("budget", budget), ("cache", cache)]:
        t0 = time.perf_counter()
        results[name] = fn(settings)
        print(f"{name:<9} {json.dumps(results[name])}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
