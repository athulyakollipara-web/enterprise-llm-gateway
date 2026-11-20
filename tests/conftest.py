import pytest
from fastapi.testclient import TestClient

from llmgw.app import create_app
from llmgw.auth import INVOKE_SCOPE, TokenVerifier, mint_dev_token
from llmgw.config import get_settings
from llmgw.pii import RegexDetector
from llmgw.providers.mock import MockProvider
from llmgw.service import Gateway
from llmgw.store.sqlite import SqliteStore
from llmgw.teams import DEMO_TEAMS, TeamPolicy


class Clock:
    def __init__(self, t=1_790_000_000.0):   # a fixed instant; tests move it forward explicitly
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def settings(tmp_path):
    return get_settings(home=tmp_path, mode="local", retry_base_s=0.0)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make_gateway(settings, clock):
    def make(provider=None, budget_mode="reserve", teams=DEMO_TEAMS, store=None):
        store = store or SqliteStore(":memory:")
        for t in teams:
            store.put_team(TeamPolicy.from_dict(t.to_dict()))
        return Gateway(settings, store, provider or MockProvider(), RegexDetector(), budget_mode=budget_mode,
                       clock=clock, sleep=lambda s: None)
    return make


@pytest.fixture
def gateway(make_gateway):
    return make_gateway()


@pytest.fixture
def client(gateway, settings):
    return TestClient(create_app(gateway, TokenVerifier.for_settings(settings)))


@pytest.fixture
def auth(settings):
    def headers(client_id="dev-risk-analytics", scopes=(INVOKE_SCOPE,), **kw):
        return {"Authorization": "Bearer " + mint_dev_token(settings.home, settings.issuer, client_id, list(scopes), **kw)}
    return headers


def ask(text, model="claude-sonnet", **kw):
    return {"model": model, "messages": [{"role": "user", "content": text}], **kw}
