"""Onboard a team in AWS: create its Cognito app client (client-credentials) and write its policy.

    llmgw onboard --team fraud-ops --models claude-haiku,nova-lite --budget 300 --rpm 60 --owner fraud-leads

The client secret is printed once. Hand it to the team through your secrets process; it is not stored here.
"""
from __future__ import annotations

from .auth import INVOKE_SCOPE
from .teams import TeamPolicy


def create_app_client(cognito, user_pool_id: str, team_id: str) -> tuple[str, str]:
    out = cognito.create_user_pool_client(
        UserPoolId=user_pool_id, ClientName=f"llmgw-{team_id}", GenerateSecret=True,
        AllowedOAuthFlows=["client_credentials"], AllowedOAuthScopes=[INVOKE_SCOPE], AllowedOAuthFlowsUserPoolClient=True,
        AccessTokenValidity=60, TokenValidityUnits={"AccessToken": "minutes"}, EnableTokenRevocation=True,
    )["UserPoolClient"]
    return out["ClientId"], out["ClientSecret"]


def onboard(store, cognito, user_pool_id: str, policy: TeamPolicy) -> tuple[TeamPolicy, str]:
    if store.get_team(policy.team_id):
        raise ValueError(f"Team {policy.team_id} already exists; update its policy instead.")
    client_id, secret = create_app_client(cognito, user_pool_id, policy.team_id)
    policy.client_ids = [client_id]
    store.put_team(policy)
    return policy, secret
