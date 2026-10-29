from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

import jwt

from app.config import settings


AI_CONTENT_PROOF_TTL_MINUTES = 15
_PROOF_TYPE = "ai_bottle_content"
_PROOF_ISSUER = "the-player"
_PROOF_AUDIENCE = "bottle-write"


def _proof_secret() -> bytes:
    """从主密钥派生独立密钥域，防止 AI 凭证被当作访问令牌。"""
    return hmac.new(
        settings.JWT_SECRET.encode("utf-8"),
        b"the-player:ai-content-proof:v1",
        hashlib.sha256,
    ).digest()


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def create_ai_content_proof(user_id: int, content: str) -> str:
    """生成一个短期凭证，仅对指定用户的该条完整 AI 内容有效。"""
    now = datetime.now(timezone.utc)
    payload = {
        "type": _PROOF_TYPE,
        "uid": user_id,
        "content_sha256": _content_digest(content),
        "iss": _PROOF_ISSUER,
        "aud": _PROOF_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=AI_CONTENT_PROOF_TTL_MINUTES),
    }
    return jwt.encode(payload, _proof_secret(), algorithm="HS256")


def verify_ai_content_proof(proof: str, user_id: int, content: str) -> bool:
    """验证凭证的签名、用户、内容和有效期；任一不匹配均返回 False。"""
    if not proof:
        return False

    try:
        payload = jwt.decode(
            proof,
            _proof_secret(),
            algorithms=["HS256"],
            audience=_PROOF_AUDIENCE,
            issuer=_PROOF_ISSUER,
            options={"require": ["type", "uid", "content_sha256", "iat", "exp", "iss", "aud"]},
        )
        if payload.get("type") != _PROOF_TYPE:
            return False
        if payload.get("uid") != user_id:
            return False
        expected_digest = str(payload.get("content_sha256", ""))
        return hmac.compare_digest(expected_digest, _content_digest(content))
    except (jwt.InvalidTokenError, ValueError, TypeError):
        return False
