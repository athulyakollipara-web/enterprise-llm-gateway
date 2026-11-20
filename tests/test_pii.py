import pytest

from llmgw.pii import RegexDetector, aba_ok, luhn_ok, pseudonymize, restore

D = RegexDetector()


def types(text):
    return [(e.type, text[e.start:e.end]) for e in D.detect(text)]


def test_checksums():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert aba_ok("021000021") and not aba_ok("021000022")


@pytest.mark.parametrize("text,expected", [
    ("mail jane.doe+bank@sub.example.co.uk now", ("EMAIL", "jane.doe+bank@sub.example.co.uk")),
    ("SSN 123-45-6789", ("SSN", "123-45-6789")),
    ("social security number: 123456789", ("SSN", "123456789")),
    ("Their SSN is 123456789.", ("SSN", "123456789")),
    ("card 4111-1111-1111-1111 declined", ("CREDIT_DEBIT_NUMBER", "4111-1111-1111-1111")),
    ("amex 3782 822463 10005", ("CREDIT_DEBIT_NUMBER", "3782 822463 10005")),
    ("call (415) 555-0132 today", ("PHONE", "(415) 555-0132")),
    ("call +1 415 555 0132", ("PHONE", "+1 415 555 0132")),
    ("acct #00123456789", ("BANK_ACCOUNT_NUMBER", "00123456789")),
    ("routing number 021000021", ("BANK_ROUTING", "021000021")),
    ("from 10.20.30.40", ("IP_ADDRESS", "10.20.30.40")),
    ("key AKIAIOSFODNN7EXAMPLE", ("AWS_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")),
])
def test_detects(text, expected):
    assert expected in types(text)


@pytest.mark.parametrize("text", [
    "Order 2024-11-03 shipped", "Total $1,234.56 due", "zip 94105", "invoice 4111111111111112",
    "SSN 000-12-3456 is invalid", "routing 123456789", "version 1.2.3.4000", "ticket #48213",
])
def test_ignores_lookalikes(text):
    assert types(text) == []


def test_pseudonymize_is_consistent_and_reversible():
    p = pseudonymize(["Email a@x.com and b@x.com", "again a@x.com"], D)
    assert p.texts == ["Email [EMAIL_1] and [EMAIL_2]", "again [EMAIL_1]"]
    assert p.counts["EMAIL"] == 3
    assert restore("sent to [EMAIL_2] and [EMAIL_1]", p.mapping) == "sent to b@x.com and a@x.com"


def test_placeholder_prefixes_do_not_collide():
    mapping = {f"[EMAIL_{i}]": f"u{i}@x.com" for i in range(1, 12)}
    assert restore("[EMAIL_1] [EMAIL_11]", mapping) == "u1@x.com u11@x.com"
