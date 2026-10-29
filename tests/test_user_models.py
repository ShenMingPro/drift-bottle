import pytest
from pydantic import ValidationError

from app.models.user import LoginRequest, RegisterRequest


def register_payload(**overrides):
    payload = {
        "username": "小明_2026",
        "password": "abcdefgh",
        "gender": "男",
        "age": 20,
        "hobbies": ["编程"],
        "contactValue": "123456",
        "province": "上海市",
        "city": "上海市",
        "university": "测试大学",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("password", ["abcdefgh", "12345678", "!@#$%^&*"])
def test_registration_accepts_simple_eight_character_passwords(password):
    request = RegisterRequest(**register_payload(password=password))
    assert request.password == password


def test_registration_rejects_short_password():
    with pytest.raises(ValidationError):
        RegisterRequest(**register_payload(password="1234567"))


def test_legacy_account_request_field_is_accepted_temporarily():
    request = LoginRequest(account="123456789123", password="legacy")
    assert request.username == "123456789123"
