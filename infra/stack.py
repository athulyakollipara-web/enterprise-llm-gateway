"""AWS CDK stack for the gateway.

    Cognito user pool (machine-to-machine): resource server "gateway" with scopes invoke and admin,
        a hosted domain for the client-credentials token endpoint, and one admin client for the platform team.
    DynamoDB: state table (limits, spend, cache, policies) and audit table, both with TTL and point-in-time recovery.
    Lambda (container image, Lambda Web Adapter) running the FastAPI app.
    HTTP API in front, with a stage-wide throttle as an outer safety limit.

Team clients are NOT created here: they are added at runtime with `llmgw onboard`, so onboarding a team
needs no redeploy. That is also why the app verifies tokens itself rather than an API Gateway JWT authorizer
(whose audience list would need every client id).
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]


class GatewayStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, domain_prefix: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        table_args = dict(partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
                          sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
                          billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST, time_to_live_attribute="ttl",
                          point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                              point_in_time_recovery_enabled=True),
                          encryption=dynamodb.TableEncryption.AWS_MANAGED)
        state = dynamodb.Table(self, "State", removal_policy=RemovalPolicy.DESTROY, **table_args)
        audit = dynamodb.Table(self, "Audit", removal_policy=RemovalPolicy.RETAIN, **table_args)

        pool = cognito.UserPool(self, "Clients", self_sign_up_enabled=False, removal_policy=RemovalPolicy.DESTROY)
        invoke = cognito.ResourceServerScope(scope_name="invoke", scope_description="Call models through the gateway")
        admin = cognito.ResourceServerScope(scope_name="admin", scope_description="Manage team policies and read usage")
        server = pool.add_resource_server("Gateway", identifier="gateway", scopes=[invoke, admin])
        domain = pool.add_domain("Domain", cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix))
        admin_client = pool.add_client(
            "PlatformAdmin", user_pool_client_name="llmgw-platform-admin", generate_secret=True,
            o_auth=cognito.OAuthSettings(flows=cognito.OAuthFlows(client_credentials=True),
                                         scopes=[cognito.OAuthScope.resource_server(server, admin)]),
            access_token_validity=Duration.minutes(60))

        fn = lambda_.DockerImageFunction(
            self, "GatewayFn",
            code=lambda_.DockerImageCode.from_image_asset(str(ROOT), file="Dockerfile"),
            memory_size=1024, timeout=Duration.seconds(60),
            log_group=logs.LogGroup(self, "GatewayLogs", retention=logs.RetentionDays.ONE_MONTH,
                                    removal_policy=RemovalPolicy.DESTROY),
            environment={
                "GATEWAY_MODE": "aws", "COGNITO_USER_POOL_ID": pool.user_pool_id,
                "STATE_TABLE": state.table_name, "AUDIT_TABLE": audit.table_name, "PII_DETECTOR": "comprehend",
            },
        )
        state.grant_read_write_data(fn)
        audit.grant_read_write_data(fn)
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:Converse"],
            resources=["arn:aws:bedrock:*::foundation-model/*", f"arn:aws:bedrock:*:{self.account}:inference-profile/*"]))
        fn.add_to_role_policy(iam.PolicyStatement(actions=["bedrock:ApplyGuardrail"],
                                                  resources=[f"arn:aws:bedrock:*:{self.account}:guardrail/*"]))
        fn.add_to_role_policy(iam.PolicyStatement(actions=["comprehend:DetectPiiEntities"], resources=["*"]))

        api = apigw.HttpApi(self, "Api", api_name="llm-gateway",
                            default_integration=integrations.HttpLambdaIntegration("GatewayIntegration", fn))
        stage = api.default_stage.node.default_child
        stage.default_route_settings = apigw.CfnStage.RouteSettingsProperty(throttling_burst_limit=200, throttling_rate_limit=100)

        cloudwatch.Alarm(self, "GatewayErrors", metric=fn.metric_errors(period=Duration.minutes(5)), threshold=5,
                         evaluation_periods=1, alarm_description="Gateway Lambda errors (unhandled exceptions)")

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "TokenUrl", value=f"{domain.base_url()}/oauth2/token")
        CfnOutput(self, "UserPoolId", value=pool.user_pool_id)
        CfnOutput(self, "AdminClientId", value=admin_client.user_pool_client_id)
        CfnOutput(self, "StateTable", value=state.table_name)
        CfnOutput(self, "AuditTable", value=audit.table_name)
