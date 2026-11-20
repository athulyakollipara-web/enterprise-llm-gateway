#!/usr/bin/env python3
import aws_cdk as cdk

from stack import GatewayStack

app = cdk.App()
GatewayStack(app, "EnterpriseLlmGateway", domain_prefix=app.node.try_get_context("domain_prefix") or "llmgw-dev")
app.synth()
