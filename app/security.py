from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from dataclasses import dataclass

from pwdlib import PasswordHash


USERNAME_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9_]{2,20}$")
_LEGACY_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_password_hash = PasswordHash.recommended()


@dataclass(frozen=True)
class PasswordVerification:
    valid: bool
    replacement_hash: str | None = None


def normalize_username(value: str) -> str:
    """归一化用户名，使注册、登录和唯一性检查使用相同语义。"""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def validate_new_username(value: str) -> str:
    username = normalize_username(value)
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("用户名需为2-20位，仅可使用中文、字母、数字和下划线")
    return username


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, stored_password: str) -> PasswordVerification:
    """验证 Argon2id 密码，并兼容旧明文和 SHA-256 数据。"""
    try:
        valid, replacement_hash = _password_hash.verify_and_update(password, stored_password)
        return PasswordVerification(valid=valid, replacement_hash=replacement_hash)
    except Exception:
        pass

    valid = hmac.compare_digest(password, stored_password)
    if not valid and _LEGACY_SHA256_PATTERN.fullmatch(stored_password):
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        valid = hmac.compare_digest(digest, stored_password.lower())

    return PasswordVerification(
        valid=valid,
        replacement_hash=hash_password(password) if valid else None,
    )
