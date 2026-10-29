"""严格读取模型结果；无效输出作为调用失败向上层传播。"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage


def parse_json_response(messages: list, required: set[str]) -> dict[str, Any]:
    if not messages or not isinstance(messages[-1], AIMessage) or not isinstance(messages[-1].content, str):
        raise ValueError("模型未返回文本结果")
    text = messages[-1].content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("模型结果不是有效 JSON") from exc
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("模型结果缺少必要字段或根对象格式无效")
    return result


def string_list(result: dict[str, Any], key: str, *, min_items: int = 0) -> list[str]:
    values = result.get(key)
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{key} 必须是非空字符串组成的列表")
    values = [value.strip() for value in values]
    if len(values) < min_items:
        raise ValueError(f"{key} 没有返回足够的有效内容")
    return values
