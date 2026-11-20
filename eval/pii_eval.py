"""How well does PII pseudonymization work, and does any raw value reach the model?

Builds a seeded, labeled set of support-style prompts with PII in many formats, including formats the
regex detector is NOT designed to catch (bare 10-digit phones, unlabeled account numbers, SSNs with no
dashes or label, international numbers), plus look-alike values that are not PII. Every prompt is then
sent through the full gateway with the mock model, and every value the model received is checked.

    python eval/pii_eval.py            # writes eval/results/pii_eval.json
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from llmgw.config import get_settings
from llmgw.pii import RegexDetector, aba_ok
from llmgw.providers.mock import MockProvider
from llmgw.service import Gateway
from llmgw.store.sqlite import SqliteStore
from llmgw.teams import TeamPolicy

rng = random.Random(20261121)
FIRST = ["jane", "omar", "li", "priya", "carlos", "ana", "dev", "sam", "mei", "tom", "fatima", "luke"]
LAST = ["doe", "khan", "wei", "patel", "ruiz", "silva", "nair", "lee", "chen", "brown", "ali", "obi"]
DOMAINS = ["example.com", "mail.example.org", "corp-example.net", "example.co.uk", "bank.example.io"]


def luhn_complete(prefix: str, length: int) -> str:
    body = prefix + "".join(rng.choice("0123456789") for _ in range(length - len(prefix) - 1))
    for check in "0123456789":
        digits = body + check
        total, alt = 0, False
        for ch in reversed(digits):
            d = int(ch) * (2 if alt else 1)
            total += d - 9 if d > 9 else d
            alt = not alt
        if total % 10 == 0:
            return digits
    raise AssertionError


def group(d: str, sizes, sep: str) -> str:
    out, i = [], 0
    for s in sizes:
        out.append(d[i:i + s]); i += s
    return sep.join(out)


def email():
    f, l = rng.choice(FIRST), rng.choice(LAST)
    local = rng.choice([f"{f}.{l}", f"{f}{l}{rng.randint(1, 99)}", f"{f[0]}{l}", f"{f}_{l}+billing"])
    return "EMAIL", local + "@" + rng.choice(DOMAINS), "standard"


def phone():
    a, b, c = rng.randint(201, 989), rng.randint(200, 999), rng.randint(0, 9999)
    fmt = rng.choice(["paren", "dash", "dot", "intl_space", "intl_compact", "space", "bare10"])
    value = {"paren": f"({a}) {b}-{c:04d}", "dash": f"{a}-{b}-{c:04d}", "dot": f"{a}.{b}.{c:04d}",
             "intl_space": f"+1 {a} {b} {c:04d}", "intl_compact": f"+1{a}{b}{c:04d}", "space": f"{a} {b} {c:04d}",
             "bare10": f"{a}{b}{c:04d}"}[fmt]
    return "PHONE", value, fmt


def ssn():
    a = rng.choice([x for x in range(1, 900) if x != 666])
    d = f"{a:03d}{rng.randint(1, 99):02d}{rng.randint(1, 9999):04d}"
    fmt = rng.choice(["dashed", "dashed", "labeled_plain", "bare9"])
    return "SSN", (f"{d[:3]}-{d[3:5]}-{d[5:]}" if fmt == "dashed" else d), fmt


def card():
    kind = rng.choice(["visa", "mc", "amex", "discover"])
    d = {"visa": luhn_complete("4", 16), "mc": luhn_complete(str(rng.randint(51, 55)), 16),
         "amex": luhn_complete(rng.choice(["34", "37"]), 15), "discover": luhn_complete("6011", 16)}[kind]
    sizes = (4, 6, 5) if kind == "amex" else (4, 4, 4, 4)
    fmt = rng.choice(["space", "dash", "plain"])
    return "CREDIT_DEBIT_NUMBER", {"space": group(d, sizes, " "), "dash": group(d, sizes, "-"), "plain": d}[fmt], f"{kind}_{fmt}"


def account():
    d = "".join(rng.choice("0123456789") for _ in range(rng.randint(8, 12)))
    fmt = rng.choice(["labeled", "labeled", "ending_in", "unlabeled"])
    return "BANK_ACCOUNT_NUMBER", (d[-4:] if fmt == "ending_in" else d), fmt


def routing():
    while True:
        d = "".join(rng.choice("0123456789") for _ in range(9))
        if aba_ok(d):
            return "BANK_ROUTING", d, "labeled"


def ip():
    return "IP_ADDRESS", ".".join(str(rng.randint(1, 254)) for _ in range(4)), "ipv4"


def aws_key():
    return "AWS_ACCESS_KEY", "AKIA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)), "access_key"


def sentence(kind, value, fmt) -> str:
    if kind == "SSN":
        return rng.choice(["Their SSN is {v}.", "SSN on the application: {v}."]).format(v=value) if fmt != "bare9" \
            else rng.choice(["Tax id provided was {v} on the form.", "The applicant wrote {v} in box 4."]).format(v=value)
    if kind == "BANK_ACCOUNT_NUMBER":
        if fmt == "unlabeled":
            return rng.choice(["Please move the funds into {v} today.", "Transfer from {v} failed twice."]).format(v=value)
        if fmt == "ending_in":
            return "The customer's account ending in {v} was charged twice.".format(v=value)
        return rng.choice(["Account number {v} shows a hold.", "acct #{v} is overdrawn.", "A/C {v} needs review."]).format(v=value)
    if kind == "BANK_ROUTING":
        return rng.choice(["Routing number {v} was used.", "ABA {v} for the wire."]).format(v=value)
    templates = {
        "EMAIL": ["Send the statement to {v}.", "Customer emailed from {v} yesterday.", "Reply to {v} and copy the branch."],
        "PHONE": ["Call the customer back on {v}.", "Their mobile is {v}.", "They phoned from {v} about a dispute."],
        "CREDIT_DEBIT_NUMBER": ["Card {v} was declined at checkout.", "Dispute on card {v} for $82.10.", "Block {v} immediately."],
        "IP_ADDRESS": ["Login came from {v}.", "Flag sessions from {v}."],
        "AWS_ACCESS_KEY": ["Someone pasted {v} into the ticket."],
    }
    return rng.choice(templates[kind]).format(v=value)


NEGATIVES = [
    lambda: f"Order ORD-{rng.randint(10000, 99999)} shipped on 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}.",
    lambda: f"Total due is ${rng.randint(10, 9999):,}.{rng.randint(0, 99):02d} by Friday.",
    lambda: f"Branch zip code {rng.randint(10000, 99999)}.",
    lambda: f"Case {rng.randint(1000000, 9999999)} was escalated.",
    lambda: f"Reference {luhn_complete('4', 16)[:-1]}{rng.choice('0123456789')} is an internal batch id.",
    lambda: f"APR moved from {rng.randint(10, 29)}.{rng.randint(0, 99):02d}% to {rng.randint(10, 29)}.{rng.randint(0, 99):02d}%.",
    lambda: f"Meeting at {rng.randint(1, 12)}:{rng.randint(0, 59):02d} pm in room {rng.randint(100, 999)}.",
    lambda: f"Release {rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 50)} fixed the export.",
    lambda: f"Ticket #{rng.randint(10000, 99999)} mentions a fee of {rng.randint(1, 50)} dollars.",
]
GENERATORS = [email, phone, ssn, card, account, routing, ip, aws_key]


def build(n_prompts=600):
    prompts = []
    for i in range(n_prompts):
        gold, parts = [], []
        for _ in range(rng.randint(0, 3)):
            kind, value, fmt = rng.choice(GENERATORS)()
            gold.append({"type": kind, "value": value, "format": fmt})
            parts.append(sentence(kind, value, fmt))
        for _ in range(rng.randint(1, 2)):
            parts.append(rng.choice(NEGATIVES)())
        rng.shuffle(parts)
        prompts.append({"id": i, "text": "Draft a reply. " + " ".join(parts), "gold": gold})
    return prompts


def main(out_path="eval/results/pii_eval.json"):
    prompts = build()
    det = RegexDetector()
    per_type, per_fmt = defaultdict(Counter), defaultdict(Counter)
    fp, fp_examples = Counter(), []
    for p in prompts:
        found = [(e.type, p["text"][e.start:e.end]) for e in det.detect(p["text"])]
        matched = set()
        for g in p["gold"]:
            hit = any(v == g["value"] or g["value"] in v for _, v in found)
            per_type[g["type"]]["total"] += 1
            per_type[g["type"]]["found"] += hit
            per_fmt[f"{g['type']}:{g['format']}"]["total"] += 1
            per_fmt[f"{g['type']}:{g['format']}"]["found"] += hit
            matched |= {v for _, v in found if v == g["value"] or g["value"] in v}
        for t, v in found:
            if v not in matched:
                fp[t] += 1
                fp_examples.append({"type": t, "value": v, "text": p["text"]})

    # End to end: send every prompt through the gateway and inspect what the model received.
    home = Path(tempfile.mkdtemp())
    settings = get_settings(home=home, mode="local", retry_base_s=0.0)
    store = SqliteStore(":memory:")
    team = TeamPolicy(team_id="eval", client_ids=["eval"], allowed_models=["nova-lite"], monthly_budget_usd=1000,
                      requests_per_minute=100_000, tokens_per_minute=10**9, cache_ttl_s=0)
    store.put_team(team)
    provider = MockProvider()
    gw = Gateway(settings, store, provider, det)
    for p in prompts:
        gw.chat(team, model="nova-lite", messages=[{"role": "user", "content": p["text"]}])
    received = [r["messages"][-1]["content"] for r in provider.received]
    leaked = [g | {"id": p["id"]} for p, sent in zip(prompts, received) for g in p["gold"] if g["value"] in sent]

    total = sum(c["total"] for c in per_type.values())
    found = sum(c["found"] for c in per_type.values())
    detections = found + sum(fp.values())
    designed = {k: c for k, c in per_fmt.items() if k.split(":")[1] not in ("bare10", "bare9", "unlabeled")}
    d_total = sum(c["total"] for c in designed.values())
    d_found = sum(c["found"] for c in designed.values())
    result = {
        "prompts": len(prompts), "pii_values": total,
        "recall": round(found / total, 4), "precision": round(found / detections, 4) if detections else None,
        "recall_on_supported_formats": round(d_found / d_total, 4),
        "values_reaching_model": len(leaked), "leak_rate": round(len(leaked) / total, 4),
        "leaks_by_format": dict(Counter(f"{g['type']}:{g['format']}" for g in leaked)),
        "per_type": {k: {"total": c["total"], "recall": round(c["found"] / c["total"], 3)} for k, c in sorted(per_type.items())},
        "per_format": {k: {"total": c["total"], "recall": round(c["found"] / c["total"], 3)} for k, c in sorted(per_fmt.items())},
        "false_positives": dict(fp), "false_positive_examples": fp_examples[:10],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["prompts", "pii_values", "recall", "precision", "recall_on_supported_formats",
                                              "values_reaching_model", "leak_rate", "leaks_by_format", "false_positives"]}, indent=1))
    print("per format:")
    for k, v in result["per_format"].items():
        print(f"  {k:<38} n={v['total']:<4} recall={v['recall']}")
    return result


if __name__ == "__main__":
    main(*sys.argv[1:])
