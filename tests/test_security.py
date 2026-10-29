import pytest

from app.security import (
    hash_password,
    normalize_username,
    validate_new_username,
    verify_password,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Alice_01 ", "alice_01"),
        ("小明_2026", "小明_2026"),
        ("123456789123", "123456789123"),
        ("ＡＬＩＣＥ", "alice"),
    ],
)
def test_valid_usernames_are_normalized(raw, expected):
    assert validate_new_username(raw) == expected


@pytest.mark.parametrize("username", ["a", "a" * 21, "hello world", "a-b", "用户😀"])
def test_invalid_usernames_are_rejected(username):
    with pytest.raises(ValueError):
        validate_new_username(username)


def test_normalize_username_is_case_insensitive():
    assert normalize_username("Alice") == normalize_username("alice")


def test_argon2_password_hash_and_verify():
    first_hash = hash_password("abcdefgh")
    second_hash = hash_password("abcdefgh")

    assert first_hash != "abcdefgh"
    assert first_hash != second_hash
    assert verify_password("abcdefgh", first_hash).valid is True
    assert verify_password("wrong-password", first_hash).valid is False


def test_legacy_plaintext_password_requests_rehash():
    result = verify_password("legacy-pass", "legacy-pass")

    assert result.valid is True
    assert result.replacement_hash is not None
    assert result.replacement_hash.startswith("$argon2")
