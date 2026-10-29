"""无效模型结果必须报错，不能填充固定建议并缓存为完成。"""

import json

import pytest
from langchain_core.messages import AIMessage

from app.agents.ice_break_agent import extract_result


@pytest.mark.parametrize("payload", [
    {"hints": ["聊篮球", "聊音乐", "聊电影"], "scenario": []},
    {"hints": ["聊篮球", "聊音乐", "聊电影"], "scenario": {}},
    {"hints": "字符串不是建议列表", "scenario": "reply"},
    {"hints": [None], "scenario": "reply"},
    ["根结果也不能是列表"],
])
def test_invalid_model_fields_fail_without_caching_result(payload):
    state = {
        "user_id": 7,
        "request_id": "failed-format-request",
        "messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))],
    }
    with pytest.raises(ValueError):
        extract_result(state)
    assert not state.get("completed_requests")


def test_non_text_model_response_is_not_recorded_as_completed():
    state = {
        "user_id": 7,
        "request_id": "non-text-request",
        "messages": [AIMessage(content=[{"type": "text", "text": "格式错误"}])],
    }
    with pytest.raises(ValueError):
        extract_result(state)
    assert not state.get("completed_requests")
