"""The request pipeline. Order matters and is the core design of the gateway:

 1. model allowed for this team?                      403
 2. PII pseudonymized (or request blocked)             422 in "block" mode
 3. requests-per-minute and tokens-per-minute limits   429 (+ Retry-After)
 4. response cache (deterministic requests only)       no model call, no cost
 5. budget: reserve the worst-case cost atomically     402
 6. call the model; retry, then fall back to the next model; circuit breaker skips a failing model
 7. settle the reservation to the actual cost, restore PII in the answer, write the audit record

The audit record never contains prompt or answer text: only a hash, counts, cost and outcome.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .config import ModelSpec, Settings
from .pii import Detector, pseudonymize, restore
from .providers.base import Completion, ModelError, ModelProvider
from .providers.mock import estimate_tokens
from .store.base import Store
from .teams import TeamPolicy

MAX_PROMPT_CHARS = 200_000


class GatewayError(Exception):
    def __init__(self, status: int, type_: str, message: str, headers: dict | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status, self.type, self.message = status, type_, message
        self.headers = headers or {}
        self.extra = extra or {}


def month_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def usd(micro: int) -> float:
    return round(micro / 1_000_000, 6)


class CircuitBreaker:
    """Skip a model for ``cooldown_s`` after ``threshold`` consecutive retryable failures (per container)."""

    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0, clock=time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self.failures: Counter = Counter()
        self.open_until: dict[str, float] = {}

    def is_open(self, alias: str) -> bool:
        return self.open_until.get(alias, 0) > self.clock()

    def record(self, alias: str, ok: bool) -> None:
        if ok:
            self.failures[alias] = 0
            return
        self.failures[alias] += 1
        if self.failures[alias] >= self.threshold:
            self.open_until[alias] = self.clock() + self.cooldown_s
            self.failures[alias] = 0


def validate_messages(messages: list[dict]) -> None:
    if not messages:
        raise GatewayError(400, "invalid_request", "messages must not be empty.")
    for i, m in enumerate(messages):
        if m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str) or not m["content"].strip():
            raise GatewayError(400, "invalid_request", f"messages[{i}] needs role user|assistant and non-empty text content.")
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise GatewayError(400, "invalid_request", "messages must alternate user/assistant, starting with user.")
    if messages[-1]["role"] != "user":
        raise GatewayError(400, "invalid_request", "The last message must be from the user.")


class Gateway:
    def __init__(self, settings: Settings, store: Store, provider: ModelProvider, detector: Detector,
                 budget_mode: str = "reserve", clock=time.time, sleep=time.sleep):
        if budget_mode not in ("reserve", "post_charge"):
            raise ValueError("budget_mode must be reserve or post_charge")
        self.settings, self.store, self.provider, self.detector = settings, store, provider, detector
        self.budget_mode = budget_mode
        self.clock, self.sleep = clock, sleep
        self.breaker = CircuitBreaker()
        self._rng = random.Random()

    # ------------------------------------------------------------------------------------
    def chat(self, team: TeamPolicy, *, model: str, messages: list[dict], system: str | None = None,
             max_tokens: int | None = None, temperature: float = 0.0, request_id: str | None = None) -> dict:
        request_id = request_id or uuid.uuid4().hex
        started = self.clock()
        t0 = time.perf_counter()
        audit = {"request_id": request_id, "ts": iso(started), "team_id": team.team_id, "model_requested": model,
                 "pii_mode": team.pii_mode}
        try:
            result = self._chat(team, audit, model, messages, system, max_tokens, temperature, started)
            audit.update(status=200, outcome="ok")
            return result
        except GatewayError as e:
            audit.update(status=e.status, outcome=e.type)
            raise
        finally:
            audit["gateway_ms"] = round((time.perf_counter() - t0) * 1000 - audit.get("model_ms", 0), 2)
            self.store.write_audit(audit)

    def _chat(self, team, audit, alias, messages, system, max_tokens, temperature, now) -> dict:
        spec = self.settings.models.get(alias)
        if spec is None:
            raise GatewayError(400, "unknown_model", f"Unknown model '{alias}'. Available: {sorted(self.settings.models)}.")
        if alias not in team.allowed_models:
            raise GatewayError(403, "model_not_allowed", f"Team '{team.team_id}' may use {team.allowed_models}, not '{alias}'.")
        validate_messages(messages)
        if not 0.0 <= temperature <= 1.0:
            raise GatewayError(400, "invalid_request", "temperature must be between 0 and 1.")
        max_tokens = min(max_tokens or team.max_output_tokens, team.max_output_tokens)
        if sum(len(m["content"]) for m in messages) + len(system or "") > MAX_PROMPT_CHARS:
            raise GatewayError(413, "request_too_large", f"Prompt exceeds {MAX_PROMPT_CHARS} characters.")

        # 2. PII --------------------------------------------------------------------------
        mapping: dict[str, str] = {}
        if team.pii_mode != "off":
            ps = pseudonymize([system or ""] + [m["content"] for m in messages], self.detector)
            audit["pii_found"] = dict(ps.counts)
            if ps.found and team.pii_mode == "block":
                raise GatewayError(422, "pii_blocked", f"Request contains personal data ({', '.join(sorted(ps.counts))}); "
                                   f"this team's policy blocks it.", extra={"pii_found": dict(ps.counts)})
            system = ps.texts[0] or None
            messages = [{"role": m["role"], "content": t} for m, t in zip(messages, ps.texts[1:])]
            mapping = ps.mapping

        # 3. Rate limits ------------------------------------------------------------------
        window = int(now // 60)
        retry_after = {"Retry-After": str(60 - int(now % 60))}
        ok, used = self.store.hit_window(f"rpm#{team.team_id}#{window}", 1, team.requests_per_minute, 120)
        if not ok:
            raise GatewayError(429, "rate_limited", f"Limit of {team.requests_per_minute} requests per minute reached.", retry_after)
        audit["rpm_remaining"] = team.requests_per_minute - used
        est_in = estimate_tokens((system or "") + "".join(m["content"] for m in messages))
        token_need = est_in + max_tokens
        if token_need > team.tokens_per_minute:
            raise GatewayError(413, "request_too_large", f"Request needs ~{token_need} tokens; the team limit is "
                               f"{team.tokens_per_minute} per minute. Lower max_tokens or split the input.")
        ok, _ = self.store.hit_window(f"tpm#{team.team_id}#{window}", token_need, team.tokens_per_minute, 120)
        if not ok:
            raise GatewayError(429, "token_rate_limited", f"Limit of {team.tokens_per_minute} tokens per minute reached.", retry_after)

        # 4. Cache ------------------------------------------------------------------------
        cache_key = None
        if team.cache_ttl_s and temperature == 0:
            cache_key = hashlib.sha256(json.dumps([team.team_id, alias, spec.model_id, system, messages, max_tokens],
                                                  sort_keys=True).encode()).hexdigest()
            hit = self.store.cache_get(cache_key)
            if hit:
                audit.update(cache="HIT", model_used=hit["model"], cost_micro_usd=0, input_tokens=0, output_tokens=0,
                             prompt_sha256=cache_key)
                return self._response(team, audit, hit["model"], hit["model_id"], hit["text"], hit["stop_reason"], 0, 0, 0,
                                      fallback=hit["model"] != alias, cache="HIT", mapping=mapping, now=now)

        # 5. Budget -----------------------------------------------------------------------
        chain = [spec] + [self.settings.models[f] for f in spec.fallback
                          if f in team.allowed_models and f in self.settings.models and f != alias]
        month = month_of(now)
        reserved = max(s.cost_micro_usd(est_in, max_tokens) for s in chain)
        if self.budget_mode == "reserve":
            ok, spent = self.store.reserve_spend(team.team_id, month, reserved, team.budget_micro_usd)
            if not ok:
                raise GatewayError(402, "budget_exceeded", f"Monthly budget of ${team.monthly_budget_usd:,.2f} would be exceeded "
                                   f"(spent ${usd(spent):,.4f}, this request may cost up to ${usd(reserved):,.4f}).")
        elif self.store.get_spend(team.team_id, month) >= team.budget_micro_usd:
            raise GatewayError(402, "budget_exceeded", f"Monthly budget of ${team.monthly_budget_usd:,.2f} is used up.")

        # 6. Model call with retry, fallback and circuit breaker ---------------------------
        try:
            used_spec, completion, attempts = self._call_chain(chain, messages, system, max_tokens, temperature, team)
        except GatewayError:
            if self.budget_mode == "reserve":
                self.store.adjust_spend(team.team_id, month, -reserved)
            raise
        cost = used_spec.cost_micro_usd(completion.input_tokens, completion.output_tokens)
        self.store.adjust_spend(team.team_id, month, cost - reserved if self.budget_mode == "reserve" else cost)
        audit.update(model_used=used_spec.alias, model_id=used_spec.model_id, attempts=attempts,
                     fallback_used=used_spec.alias != alias, input_tokens=completion.input_tokens,
                     output_tokens=completion.output_tokens, cost_micro_usd=cost, model_ms=round(completion.latency_ms, 2),
                     cache="MISS" if cache_key else "OFF", prompt_sha256=hashlib.sha256(
                         json.dumps([system, messages]).encode()).hexdigest())
        if cache_key and completion.stop_reason == "end_turn":
            self.store.cache_put(cache_key, {"model": used_spec.alias, "model_id": used_spec.model_id, "text": completion.text,
                                             "stop_reason": completion.stop_reason}, team.cache_ttl_s)
        return self._response(team, audit, used_spec.alias, used_spec.model_id, completion.text, completion.stop_reason,
                              completion.input_tokens, completion.output_tokens, cost, fallback=used_spec.alias != alias,
                              cache=audit["cache"], mapping=mapping, now=now)

    def _call_chain(self, chain: list[ModelSpec], messages, system, max_tokens, temperature, team):
        attempts, last_error = 0, None
        guardrail = {"id": team.guardrail_id, "version": team.guardrail_version} if team.guardrail_id else None
        usable = [s for s in chain if not self.breaker.is_open(s.alias)] or chain[-1:]
        for spec in usable:
            for retry in range(2):                         # one retry on the same model, then fall back
                if attempts >= self.settings.max_attempts:
                    break
                attempts += 1
                try:
                    completion = self.provider.converse(spec.model_id, messages, system, max_tokens, temperature, guardrail)
                    self.breaker.record(spec.alias, ok=True)
                    return spec, completion, attempts
                except ModelError as e:
                    last_error = e
                    if not e.retryable:
                        status = 400 if e.code == "ValidationException" else 502
                        raise GatewayError(status, "model_error", f"{spec.alias}: {e}") from e
                    self.breaker.record(spec.alias, ok=False)
                    if retry == 0 and spec is not usable[-1] and self.breaker.is_open(spec.alias):
                        break
                    if retry == 0:
                        self.sleep(self.settings.retry_base_s * (1 + self._rng.random()))
        raise GatewayError(503, "upstream_unavailable", f"All models failed after {attempts} attempts: {last_error}",
                           {"Retry-After": "5"})

    def _response(self, team, audit, alias, model_id, text, stop_reason, tin, tout, cost, *, fallback, cache, mapping, now):
        month = month_of(now)
        spent = self.store.get_spend(team.team_id, month)
        output = restore(text, mapping) if team.pii_mode == "pseudonymize" else text
        return {
            "id": audit["request_id"], "model": alias, "model_id": model_id, "fallback_used": fallback,
            "output": output, "stop_reason": stop_reason,
            "usage": {"input_tokens": tin, "output_tokens": tout, "cost_usd": usd(cost)},
            "cache": cache,
            "pii": {"mode": team.pii_mode, "found": audit.get("pii_found", {})},
            "budget": {"month": month, "spent_usd": usd(spent), "limit_usd": team.monthly_budget_usd,
                       "remaining_usd": round(team.monthly_budget_usd - usd(spent), 6)},
            "rate_limit": {"requests_per_minute": team.requests_per_minute, "remaining": audit.get("rpm_remaining")},
        }

    # ------------------------------------------------------------------------------------
    def usage(self, team_id: str | None = None, month: str | None = None) -> dict:
        month = month or month_of(self.clock())
        teams = [self.store.get_team(team_id)] if team_id else self.store.list_teams()
        out = []
        for team in filter(None, teams):
            records = [r for r in self.store.audit_records(team.team_id, since=f"{month}-01")
                       if r["ts"].startswith(month)]
            by_model: dict[str, Counter] = defaultdict(Counter)
            outcomes, pii = Counter(), Counter()
            for r in records:
                outcomes[r["outcome"]] += 1
                pii.update(r.get("pii_found", {}))
                if r["outcome"] == "ok":
                    c = by_model[r["model_used"]]
                    c["requests"] += 1
                    c["input_tokens"] += r.get("input_tokens", 0)
                    c["output_tokens"] += r.get("output_tokens", 0)
                    c["cost_micro_usd"] += r.get("cost_micro_usd", 0)
                    c["cache_hits"] += r.get("cache") == "HIT"
                    c["fallbacks"] += bool(r.get("fallback_used"))
            spent = self.store.get_spend(team.team_id, month)
            out.append({
                "team_id": team.team_id, "month": month, "budget_usd": team.monthly_budget_usd, "spent_usd": usd(spent),
                "budget_used_pct": round(100 * spent / team.budget_micro_usd, 1) if team.budget_micro_usd else None,
                "requests": len(records), "outcomes": dict(outcomes), "pii_detected": dict(pii),
                "by_model": {m: {**{k: v for k, v in c.items() if k != "cost_micro_usd"}, "cost_usd": usd(c["cost_micro_usd"])}
                             for m, c in sorted(by_model.items())},
            })
        return {"month": month, "teams": out}

    def usage_rows(self, month: str | None = None) -> list[dict]:
        """One row per day x team x model x outcome, for Tableau or a spreadsheet."""
        month = month or month_of(self.clock())
        rows: dict[tuple, Counter] = defaultdict(Counter)
        for r in self.store.audit_records(None, since=f"{month}-01"):
            if not r["ts"].startswith(month):
                continue
            key = (r["ts"][:10], r.get("team_id"), r.get("model_used") or r.get("model_requested"), r["outcome"])
            c = rows[key]
            c["requests"] += 1
            c["input_tokens"] += r.get("input_tokens", 0)
            c["output_tokens"] += r.get("output_tokens", 0)
            c["cost_micro_usd"] += r.get("cost_micro_usd", 0)
            c["cache_hits"] += r.get("cache") == "HIT"
            c["fallbacks"] += bool(r.get("fallback_used"))
            c["pii_entities"] += sum(r.get("pii_found", {}).values())
        return [{"date": k[0], "team_id": k[1], "model": k[2], "outcome": k[3], "requests": c["requests"],
                 "input_tokens": c["input_tokens"], "output_tokens": c["output_tokens"], "cost_usd": usd(c["cost_micro_usd"]),
                 "cache_hits": c["cache_hits"], "fallbacks": c["fallbacks"], "pii_entities": c["pii_entities"]}
                for k, c in sorted(rows.items())]
