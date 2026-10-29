import jwt

import app.ai_content_proof as proof_module
from app.ai_content_proof import create_ai_content_proof, verify_ai_content_proof
from app.auth_jwt import create_access_token, decode_access_token
from app.routers.content_gen import _attach_proofs


def test_proof_is_bound_to_exact_content_and_user():
    proof = create_ai_content_proof(7, "今晚有人一起跑步吗")

    assert verify_ai_content_proof(proof, 7, "今晚有人一起跑步吗") is True
    assert verify_ai_content_proof(proof, 7, "今晚有人一起跑步吗！") is False
    assert verify_ai_content_proof(proof, 7, "今晚有人  一起跑步吗") is False
    assert verify_ai_content_proof(proof, 8, "今晚有人一起跑步吗") is False


def test_tampered_and_expired_proofs_are_rejected(monkeypatch):
    proof = create_ai_content_proof(7, "好无聊 谁来聊天")
    tampered = proof[: len(proof) // 2] + ("a" if proof[len(proof) // 2] != "a" else "b") + proof[len(proof) // 2 + 1 :]
    assert verify_ai_content_proof(tampered, 7, "好无聊 谁来聊天") is False

    monkeypatch.setattr(proof_module, "AI_CONTENT_PROOF_TTL_MINUTES", -1)
    expired = create_ai_content_proof(7, "好无聊 谁来聊天")
    assert verify_ai_content_proof(expired, 7, "好无聊 谁来聊天") is False


def test_access_tokens_and_ai_proofs_cannot_be_interchanged():
    access_token = create_access_token(7)
    proof = create_ai_content_proof(7, "求一个饭搭子")

    assert verify_ai_content_proof(access_token, 7, "求一个饭搭子") is False
    try:
        decode_access_token(proof)
    except jwt.InvalidTokenError:
        pass
    else:
        raise AssertionError("AI 内容凭证不得被当作登录令牌")


def test_only_deterministically_safe_suggestions_receive_proofs():
    suggestions = _attach_proofs(
        [
            {"content": "今晚有人一起跑步吗", "style": "找人玩"},
            {"content": "加我微信：abcdef123", "style": "想聊天"},
            {"content": "约炮吗", "style": "想聊天"},
        ],
        user_id=7,
    )

    assert len(suggestions) == 1
    assert suggestions[0]["content"] == "今晚有人一起跑步吗"
    assert verify_ai_content_proof(suggestions[0]["proof"], 7, suggestions[0]["content"])
