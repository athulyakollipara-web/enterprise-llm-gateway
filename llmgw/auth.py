"""Caller authentication with OAuth2 client-credentials access tokens (Amazon Cognito).

Each team gets its own Cognito app client. The access token's ``client_id`` claim identifies the
team, and its ``scope`` claim says what it may do (``gateway/invoke``, ``gateway/admin``).
The gateway verifies tokens itself (signature, issuer, expiry, token_use), because teams are
onboarded at runtime and an API Gateway JWT authorizer would need a redeploy for every new client.

In local mode the gateway signs its own development tokens with a key kept in GATEWAY_HOME.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

INVOKE_SCOPE = "gateway/invoke"
ADMIN_SCOPE = "gateway/admin"
DEV_KID = "llmgw-dev"


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Caller:
    client_id: str
    scopes: frozenset[str]
    subject: str


def _dev_key(home: Path) -> rsa.RSAPrivateKey:
    path = home / "dev-signing-key.pem"
    if not path.exists():
        home.mkdir(parents=True, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        path.chmod(0o600)
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def mint_dev_token(home: Path, issuer: str, client_id: str, scopes: list[str], ttl_s: int = 3600) -> str:
    """A token shaped like a Cognito client-credentials access token. Local mode only."""
    now = int(time.time())
    claims = {"iss": issuer, "sub": client_id, "client_id": client_id, "token_use": "access",
              "scope": " ".join(scopes), "iat": now, "exp": now + ttl_s, "jti": f"{client_id}-{now}"}
    return jwt.encode(claims, _dev_key(home), algorithm="RS256", headers={"kid": DEV_KID})


class TokenVerifier:
    def __init__(self, issuer: str, public_key=None, jwks_url: str | None = None):
        self.issuer = issuer
        self.public_key = public_key
        self.jwks = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600) if jwks_url else None

    @classmethod
    def for_settings(cls, settings) -> "TokenVerifier":
        if settings.mode == "aws":
            return cls(settings.issuer, jwks_url=f"{settings.issuer}/.well-known/jwks.json")
        return cls(settings.issuer, public_key=_dev_key(settings.home).public_key())

    def verify(self, authorization: str | None) -> Caller:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("Missing bearer token.")
        token = authorization.split(" ", 1)[1].strip()
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key if self.jwks else self.public_key
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.issuer,
                                options={"require": ["exp", "iss", "client_id"], "verify_aud": False})
        except jwt.PyJWTError as e:
            raise AuthError(f"Invalid token: {e}") from e
        if claims.get("token_use") != "access":
            raise AuthError("Only access tokens are accepted.")
        return Caller(client_id=claims["client_id"], scopes=frozenset(claims.get("scope", "").split()),
                      subject=claims.get("sub", claims["client_id"]))
