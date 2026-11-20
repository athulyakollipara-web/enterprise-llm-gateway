"""CDK stack assertions (skipped when aws-cdk-lib is not installed)."""
import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk.assertions import Match, Template  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))
from stack import GatewayStack  # noqa: E402


@pytest.fixture(scope="module")
def template():
    return Template.from_stack(GatewayStack(cdk.App(), "Test", domain_prefix="llmgw-test"))


def test_tables_have_ttl_and_point_in_time_recovery(template):
    template.resource_count_is("AWS::DynamoDB::Table", 2)
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "TimeToLiveSpecification": {"AttributeName": "ttl", "Enabled": True},
        "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True}})


def test_machine_to_machine_auth(template):
    template.has_resource_properties("AWS::Cognito::UserPoolResourceServer", {"Identifier": "gateway"})
    template.has_resource_properties("AWS::Cognito::UserPoolClient", {"AllowedOAuthFlows": ["client_credentials"],
                                                                     "GenerateSecret": True})


def test_lambda_runs_in_aws_mode_with_least_privilege(template):
    template.resource_count_is("AWS::Lambda::Function", 1)
    template.has_resource_properties("AWS::Lambda::Function", {
        "PackageType": "Image", "Environment": {"Variables": Match.object_like({"GATEWAY_MODE": "aws", "PII_DETECTOR": "comprehend"})}})
    template.has_resource_properties("AWS::IAM::Policy", {"PolicyDocument": {"Statement": Match.array_with([
        Match.object_like({"Action": ["bedrock:InvokeModel", "bedrock:Converse"]})])}})


def test_api_has_an_outer_throttle(template):
    template.has_resource_properties("AWS::ApiGatewayV2::Stage", {
        "DefaultRouteSettings": {"ThrottlingBurstLimit": 200, "ThrottlingRateLimit": 100}})
