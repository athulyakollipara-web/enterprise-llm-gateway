"""AWS code paths, checked with botocore's Stubber against the real service models (parameter names and shapes)."""
import json
import time

import boto3
import pytest
from botocore.stub import ANY, Stubber

from llmgw.onboard import onboard
from llmgw.pii import ComprehendDetector
from llmgw.providers.base import ModelError
from llmgw.providers.bedrock import BedrockProvider
from llmgw.store.dynamo import DynamoStore
from llmgw.teams import TeamPolicy


def client(name):
    return boto3.client(name, region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")


@pytest.fixture
def ddb():
    c = client("dynamodb")
    with Stubber(c) as stub:
        yield c, stub
        stub.assert_no_pending_responses()


def test_rate_window_is_one_conditional_update(ddb):
    c, stub = ddb
    store = DynamoStore("state", "audit", client=c)
    stub.add_response("update_item", {"Attributes": {"value": {"N": "3"}}}, {
        "TableName": "state", "Key": {"pk": {"S": "LIMIT#rpm#t#1"}, "sk": {"S": "W"}},
        "UpdateExpression": "ADD #v :a SET #t = if_not_exists(#t, :ttl)",
        "ConditionExpression": "attribute_not_exists(#v) OR #v <= :max",
        "ExpressionAttributeNames": {"#v": "value", "#t": "ttl"},
        "ExpressionAttributeValues": {":a": {"N": "1"}, ":max": {"N": "59"}, ":ttl": ANY},
        "ReturnValues": "UPDATED_NEW", "ReturnValuesOnConditionCheckFailure": "ALL_OLD"})
    assert store.hit_window("rpm#t#1", 1, 60, 120) == (True, 3)


def test_rate_window_denied(ddb):
    c, stub = ddb
    store = DynamoStore("state", "audit", client=c)
    stub.add_client_error("update_item", service_error_code="ConditionalCheckFailedException", http_status_code=400,
                          modeled_fields={"Item": {"value": {"N": "60"}}})
    assert store.hit_window("rpm#t#1", 1, 60, 120) == (False, 60)


def test_budget_reservation_has_no_ttl(ddb):
    c, stub = ddb
    store = DynamoStore("state", "audit", client=c)
    stub.add_response("update_item", {"Attributes": {"value": {"N": "500"}}}, {
        "TableName": "state", "Key": {"pk": {"S": "SPEND#risk"}, "sk": {"S": "2026-11"}},
        "UpdateExpression": "ADD #v :a", "ConditionExpression": "attribute_not_exists(#v) OR #v <= :max",
        "ExpressionAttributeNames": {"#v": "value"}, "ExpressionAttributeValues": {":a": {"N": "500"}, ":max": {"N": "999500"}},
        "ReturnValues": "UPDATED_NEW", "ReturnValuesOnConditionCheckFailure": "ALL_OLD"})
    assert store.reserve_spend("risk", "2026-11", 500, 1_000_000) == (True, 500)


def test_put_team_maps_clients_and_removes_stale_ones(ddb):
    c, stub = ddb
    store = DynamoStore("state", "audit", client=c)
    old = TeamPolicy(team_id="risk", client_ids=["old"], allowed_models=["nova-lite"], monthly_budget_usd=5)
    new = TeamPolicy(team_id="risk", client_ids=["new"], allowed_models=["nova-lite"], monthly_budget_usd=5)
    stub.add_response("get_item", {"Item": {"body": {"S": json.dumps(old.to_dict())}}})
    stub.add_response("put_item", {}, {"TableName": "state", "Item": {"pk": {"S": "TEAM#risk"}, "sk": {"S": "POLICY"}, "body": ANY}})
    stub.add_response("put_item", {}, {"TableName": "state", "Item": {"pk": {"S": "CLIENT#new"}, "sk": {"S": "TEAM"},
                                                                       "team_id": {"S": "risk"}}})
    stub.add_response("delete_item", {}, {"TableName": "state", "Key": {"pk": {"S": "CLIENT#old"}, "sk": {"S": "TEAM"}}})
    store.put_team(new)


def test_expired_cache_items_are_ignored(ddb):
    c, stub = ddb
    store = DynamoStore("state", "audit", client=c)
    stub.add_response("get_item", {"Item": {"body": {"S": "{}"}, "ttl": {"N": str(int(time.time()) - 5)}}})
    assert store.cache_get("k") is None


def test_audit_query_by_team(ddb):
    c, stub = ddb
    store = DynamoStore("state", "audit", client=c)
    rec = {"ts": "2026-11-02T00:00:00.000Z", "request_id": "r1", "team_id": "risk", "outcome": "ok"}
    stub.add_response("query", {"Items": [{"body": {"S": json.dumps(rec)}}]}, {
        "TableName": "audit", "KeyConditionExpression": "pk = :t AND sk >= :s",
        "ExpressionAttributeValues": {":t": {"S": "risk"}, ":s": {"S": "2026-11-01"}}})
    assert store.audit_records("risk", since="2026-11-01") == [rec]


def test_bedrock_converse_request_shape():
    c = client("bedrock-runtime")
    with Stubber(c) as stub:
        stub.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi [EMAIL_1]"}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 12, "outputTokens": 4, "totalTokens": 16},
            "metrics": {"latencyMs": 100}}, {
            "modelId": "us.amazon.nova-lite-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello [EMAIL_1]"}]}],
            "system": [{"text": "be brief"}],
            "inferenceConfig": {"maxTokens": 100, "temperature": 0.0},
            "guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "3"}})
        out = BedrockProvider(client=c).converse("us.amazon.nova-lite-v1:0", [{"role": "user", "content": "hello [EMAIL_1]"}],
                                                 "be brief", 100, 0.0, {"id": "gr-1", "version": "3"})
    assert (out.text, out.input_tokens, out.output_tokens) == ("Hi [EMAIL_1]", 12, 4)


@pytest.mark.parametrize("code,retryable", [("ThrottlingException", True), ("ServiceUnavailableException", True),
                                            ("ValidationException", False), ("AccessDeniedException", False)])
def test_bedrock_error_mapping(code, retryable):
    c = client("bedrock-runtime")
    with Stubber(c) as stub:
        stub.add_client_error("converse", service_error_code=code, http_status_code=400)
        with pytest.raises(ModelError) as e:
            BedrockProvider(client=c).converse("m", [{"role": "user", "content": "x"}], None, 10, 0.0)
    assert e.value.retryable is retryable


def test_comprehend_offsets_and_threshold():
    c = client("comprehend")
    text = "Write to jane@example.com for Maria Lopez on 2026-01-02"
    with Stubber(c) as stub:
        stub.add_response("detect_pii_entities", {"Entities": [
            {"Score": 0.99, "Type": "EMAIL", "BeginOffset": 9, "EndOffset": 25},
            {"Score": 0.97, "Type": "NAME", "BeginOffset": 30, "EndOffset": 41},
            {"Score": 0.30, "Type": "PHONE", "BeginOffset": 0, "EndOffset": 5},
            {"Score": 0.99, "Type": "DATE_TIME", "BeginOffset": 45, "EndOffset": 55}]},
            {"Text": text, "LanguageCode": "en"})
        found = ComprehendDetector(client=c, ignore={"DATE_TIME"}).detect(text)
    assert [(e.type, text[e.start:e.end]) for e in found] == [("EMAIL", "jane@example.com"), ("NAME", "Maria Lopez")]


def test_comprehend_chunks_long_text_and_shifts_offsets():
    c = client("comprehend")
    det = ComprehendDetector(client=c, ignore=set())
    det.CHUNK = 20
    text = "aaaa bbbb cccc dddd eeee ffff a@b.co"
    with Stubber(c) as stub:
        stub.add_response("detect_pii_entities", {"Entities": []}, {"Text": "aaaa bbbb cccc dddd", "LanguageCode": "en"})
        stub.add_response("detect_pii_entities", {"Entities": [{"Score": 0.9, "Type": "EMAIL", "BeginOffset": 11, "EndOffset": 17}]},
                          {"Text": " eeee ffff a@b.co", "LanguageCode": "en"})
        found = det.detect(text)
    assert text[found[0].start:found[0].end] == "a@b.co"


def test_onboard_creates_a_client_credentials_app_client():
    cog, ddb_client = client("cognito-idp"), client("dynamodb")
    store = DynamoStore("state", "audit", client=ddb_client)
    with Stubber(cog) as cstub, Stubber(ddb_client) as dstub:
        dstub.add_response("get_item", {})
        cstub.add_response("create_user_pool_client", {"UserPoolClient": {"ClientId": "abc123", "ClientSecret": "s" * 52}}, {
            "UserPoolId": "us-east-1_pool", "ClientName": "llmgw-fraud-ops", "GenerateSecret": True,
            "AllowedOAuthFlows": ["client_credentials"], "AllowedOAuthScopes": ["gateway/invoke"],
            "AllowedOAuthFlowsUserPoolClient": True, "AccessTokenValidity": 60, "TokenValidityUnits": {"AccessToken": "minutes"},
            "EnableTokenRevocation": True})
        dstub.add_response("get_item", {})
        dstub.add_response("put_item", {})
        dstub.add_response("put_item", {}, {"TableName": "state", "Item": {"pk": {"S": "CLIENT#abc123"}, "sk": {"S": "TEAM"},
                                                                            "team_id": {"S": "fraud-ops"}}})
        policy, secret = onboard(store, cog, "us-east-1_pool",
                                 TeamPolicy(team_id="fraud-ops", client_ids=[], allowed_models=["claude-haiku"], monthly_budget_usd=300))
    assert policy.client_ids == ["abc123"] and secret == "s" * 52
