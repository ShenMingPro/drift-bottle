from __future__ import annotations

import re


SENSITIVE_WORDS: list[str] = [
    "色情", "裸体", "做爱", "性交", "强奸", "猥亵", "淫秽",
    "赌博", "博彩", "下注", "押注", "庄家",
    "毒品", "大麻", "冰毒", "海洛因", "吸毒",
    "杀人", "砍人", "爆炸", "炸弹", "恐怖袭击",
    "法轮", "反华", "颠覆", "分裂国家",
    "傻逼", "操你", "妈的", "狗日的", "王八蛋", "去死",
    "代开发票", "刷单", "兼职赚钱", "日赚",
    "约炮", "一夜情", "卖淫", "嫖娼",
]

CONTACT_PATTERNS: list[dict[str, str]] = [
    {"name": "手机号", "pattern": r"1[3-9]\d{9}"},
    {"name": "微信号", "pattern": r"(微信|vx|weixin)[：:]\s*[a-zA-Z0-9_-]{6,20}"},
    {"name": "QQ号", "pattern": r"(QQ|qq)[：:]\s*\d{5,12}"},
    {"name": "邮箱", "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"},
    {"name": "网址链接", "pattern": r"https?://[^\s]+"},
]


def check_sensitive_words(content: str) -> list[str]:
    return [word for word in SENSITIVE_WORDS if word in content]


def check_contact_info(content: str) -> list[str]:
    return [
        rule["name"]
        for rule in CONTACT_PATTERNS
        if re.search(rule["pattern"], content)
    ]


def check_spam(content: str) -> str:
    stripped = content.strip()
    if len(stripped) == 0:
        return "内容质量异常：内容为空"
    if re.fullmatch(r"[\s\d]+", stripped):
        return "内容质量异常：内容仅包含数字和空格"
    if re.fullmatch(r"[\W_]+", stripped):
        return "内容质量异常：内容仅包含符号"
    if len(stripped) >= 4 and len(set(stripped)) == 1:
        return "内容质量异常：内容为单一重复字符"
    if len(stripped) >= 8:
        max_repeat = 1
        current_repeat = 1
        for index in range(1, len(stripped)):
            if stripped[index] == stripped[index - 1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
        if max_repeat >= 6:
            return f"内容质量异常：存在连续重复字符（重复{max_repeat}次）"
    return "内容质量正常"


def is_deterministically_safe(content: str) -> bool:
    """无 LLM 调用的毫秒级安全门，仅安全内容可获得 AI 来源凭证。"""
    return (
        not check_sensitive_words(content)
        and not check_contact_info(content)
        and "异常" not in check_spam(content)
    )
