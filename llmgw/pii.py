"""PII detection and reversible pseudonymization.

Before a prompt leaves the gateway, every detected value is replaced by a placeholder such as
``[EMAIL_1]``. The same value always gets the same placeholder within a request, so the model can
still reason about it ("send the statement to [EMAIL_1]"). In ``pseudonymize`` mode the gateway
swaps the real values back into the model's answer, so the user sees a normal reply while the
model provider never saw the data.

Detectors:
* ``RegexDetector`` (local and fallback): structured identifiers that can be validated
  (card numbers with the Luhn check, routing numbers with the ABA checksum, SSN area rules, ...).
  It does not find names or street addresses.
* ``ComprehendDetector`` (AWS): Amazon Comprehend ``DetectPiiEntities``, which also finds names
  and addresses.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Entity:
    type: str
    start: int
    end: int


class Detector(Protocol):
    def detect(self, text: str) -> list[Entity]: ...


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def aba_ok(digits: str) -> bool:
    d = [int(c) for c in digits]
    return len(d) == 9 and (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"
                    r"|(?<![\w+])\+1\d{10}\b")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_SSN_CTX = re.compile(r"\b(?:SSN|social security(?: number)?)\b[^\d\n]{0,30}?(\d{9})\b", re.I)   # label, then 9 digits nearby
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_ACCOUNT = re.compile(r"\b(?:account|acct|a/c)\.?\s*(?:number|no\.?|num|#)?\s*(?:ending in\s*)?[:#]?\s*(\d{4,17})\b", re.I)
_ROUTING = re.compile(r"\b(?:routing|ABA|RTN)\s*(?:number|no\.?|#)?\s*[:#]?\s*(\d{9})\b", re.I)
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


class RegexDetector:
    def detect(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        add = lambda t, s, e: found.append(Entity(t, s, e))  # noqa: E731
        for m in _EMAIL.finditer(text):
            add("EMAIL", m.start(), m.end())
        for m in _SSN.finditer(text):
            add("SSN", m.start(), m.end())
        for m in _SSN_CTX.finditer(text):
            add("SSN", m.start(1), m.end(1))
        for m in _CARD.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and luhn_ok(digits):
                add("CREDIT_DEBIT_NUMBER", m.start(), m.end())
        for m in _ROUTING.finditer(text):
            if aba_ok(m.group(1)):
                add("BANK_ROUTING", m.start(1), m.end(1))
        for m in _ACCOUNT.finditer(text):
            add("BANK_ACCOUNT_NUMBER", m.start(1), m.end(1))
        for m in _PHONE.finditer(text):
            add("PHONE", m.start(), m.end())
        for m in _IPV4.finditer(text):
            add("IP_ADDRESS", m.start(), m.end())
        for m in _AWS_KEY.finditer(text):
            add("AWS_ACCESS_KEY", m.start(), m.end())
        return _resolve_overlaps(found)


# Order of precedence when two detections overlap (most specific first).
_PRIORITY = {t: i for i, t in enumerate(["AWS_ACCESS_KEY", "EMAIL", "CREDIT_DEBIT_NUMBER", "SSN", "BANK_ROUTING",
                                         "BANK_ACCOUNT_NUMBER", "PHONE", "IP_ADDRESS"])}


def _resolve_overlaps(entities: list[Entity]) -> list[Entity]:
    ranked = sorted(entities, key=lambda e: (_PRIORITY.get(e.type, 99), -(e.end - e.start), e.start))
    kept: list[Entity] = []
    for e in ranked:
        if all(e.end <= k.start or e.start >= k.end for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)


class ComprehendDetector:
    """Amazon Comprehend DetectPiiEntities. Text is sent in chunks under the 100 KB request limit."""

    CHUNK = 40_000

    def __init__(self, client=None, region: str | None = None, min_score: float = 0.5, ignore: set[str] | None = None):
        import boto3

        self.client = client or boto3.client("comprehend", region_name=region)
        self.min_score = min_score
        self.ignore = ignore if ignore is not None else set(
            filter(None, os.getenv("PII_IGNORE_TYPES", "DATE_TIME,URL,AGE").split(",")))

    def detect(self, text: str) -> list[Entity]:
        out: list[Entity] = []
        offset = 0
        while offset < len(text):
            end = min(len(text), offset + self.CHUNK)
            if end < len(text):                       # do not cut a word in half
                space = text.rfind(" ", offset, end)
                end = space if space > offset else end
            chunk = text[offset:end]
            if chunk.strip():
                resp = self.client.detect_pii_entities(Text=chunk, LanguageCode="en")
                out += [Entity(e["Type"], offset + e["BeginOffset"], offset + e["EndOffset"])
                        for e in resp["Entities"] if e["Score"] >= self.min_score and e["Type"] not in self.ignore]
            offset = end
        return _resolve_overlaps(out)


@dataclass
class Pseudonymized:
    texts: list[str]
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> original value
    counts: Counter = field(default_factory=Counter)        # entity type -> occurrences

    @property
    def found(self) -> bool:
        return bool(self.mapping)


def pseudonymize(texts: list[str], detector: Detector) -> Pseudonymized:
    """Replace PII in every text with placeholders that are consistent across all the texts."""
    by_value: dict[tuple[str, str], str] = {}
    per_type: Counter = Counter()
    result = Pseudonymized(texts=[])
    for text in texts:
        pieces, last = [], 0
        for e in detector.detect(text):
            value = text[e.start:e.end]
            key = (e.type, value)
            if key not in by_value:
                per_type[e.type] += 1
                by_value[key] = f"[{e.type}_{per_type[e.type]}]"
                result.mapping[by_value[key]] = value
            result.counts[e.type] += 1
            pieces += [text[last:e.start], by_value[key]]
            last = e.end
        result.texts.append("".join(pieces) + text[last:])
    return result


def restore(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    pattern = re.compile("|".join(re.escape(p) for p in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group()], text)
