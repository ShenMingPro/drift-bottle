from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用尚未定义的类型

import time  # 时间库，用于计算节点执行耗时
from typing import Annotated, Any, TypedDict  # 类型注解工具：Annotated添加元数据、Any任意类型、TypedDict定义带类型的字典

from langchain_core.messages import HumanMessage, SystemMessage  # LangChain消息类型
from langchain_openai import ChatOpenAI  # LangChain的OpenAI兼容LLM客户端，用于调用DeepSeek模型
from langgraph.graph import END, StateGraph  # LangGraph核心组件：END表示状态图终止节点、StateGraph用于构建有状态图
from langgraph.graph.message import add_messages  # LangGraph消息追加reducer，确保messages列表自动追加而非覆盖

from app.agent_logger import get_logger, global_metrics  # 结构化日志工具
from app.content_safety import check_contact_info, check_sensitive_words, check_spam
from app.agents.result_parsing import parse_json_response


# ============================================================
# ContentGuardAgent 状态图流程图（Pipeline模式）（内容审核）
# ============================================================
#
#   ┌────────────────────┐
#   │  run_deterministic │  确定性检测：强制执行敏感词+联系方式+垃圾信息检测
#   └────────┬───────────┘
#            │
#            ▼
#   ┌────────────────────┐     确定性检测命中违禁
#   │  check_deterministic│ ─────────────────────→ 直接返回不通过 → [END]
#   └────────┬───────────┘
#            │ 确定性检测全部通过
#            ▼
#   ┌────────────────────┐
#   │  llm_review        │  LLM语义审核：将确定性检测结果嵌入提示词，LLM做最终语义判断
#   └────────┬───────────┘
#            │
#            ▼
#   ┌────────────────────┐
#   │  extract_result    │  解析LLM返回的JSON，提取审核结论
#   └────────┬───────────┘
#            │
#            ▼
#          [END]
#
# 设计思路：
#   1. 先强制执行所有确定性检测（敏感词、联系方式、垃圾信息），不依赖LLM决定是否调用
#   2. 确定性检测命中违禁→直接返回不通过，无需LLM参与（快速、可靠）
#   3. 确定性检测全部通过→交给LLM做语义审核（理解隐含含义、擦边球等）
#   4. 相比ReAct模式，Pipeline模式确保确定性检测100%执行，不会因为LLM跳过工具而漏检
#   5. 解析失败向上报告技术故障，禁止将无效结果当成审核通过或内容违规
#
# ============================================================


class GuardState(TypedDict):
    """ContentGuardAgent的状态定义，贯穿整个审核流程图"""
    content: str  # 待审核的文本内容
    content_type: str  # 内容类型："bottle"纸团 或 "message"聊天消息
    user_id: int  # 发布内容的用户ID
    messages: Annotated[list, add_messages]  # LLM对话消息列表（使用add_messages reducer自动追加）
    passed: bool  # 最终审核结论：是否通过
    reason: str  # 最终审核原因描述
    risk_level: str  # 最终风险等级："low"/"medium"/"high"
    # 确定性检测结果（由run_deterministic节点填充）
    sensitive_hits: list[str]  # 命中的敏感词列表
    contact_hits: list[str]  # 命中的联系方式类型列表
    spam_result: str  # 垃圾信息检测结果描述


# ============================================================
# 状态图节点函数
# ============================================================

def run_deterministic(state: GuardState) -> dict[str, Any]:
    """状态图节点1：强制执行所有确定性检测

    不依赖LLM的function calling，直接执行敏感词、联系方式、垃圾信息检测。
    确保这些基础检测100%执行，不会因为LLM跳过工具而漏检。
    """
    log = get_logger("guard_agent")
    content = state.get("content", "")
    user_id = state.get("user_id", 0)

    start = time.time()
    sensitive_hits = check_sensitive_words(content)
    contact_hits = check_contact_info(content)
    spam_result = check_spam(content)
    duration_ms = round((time.time() - start) * 1000, 1)

    # 记录确定性检测结果日志
    log.info("deterministic_check", user_id=user_id, duration_ms=duration_ms,
             sensitive_count=len(sensitive_hits), contact_count=len(contact_hits),
             spam_ok="异常" not in spam_result)

    return {
        "sensitive_hits": sensitive_hits,
        "contact_hits": contact_hits,
        "spam_result": spam_result,
    }


def check_deterministic(state: GuardState) -> dict[str, Any]:
    """状态图节点2：检查确定性检测结果，命中违规则设置passed=False

    如果确定性检测发现违禁内容（敏感词/联系方式/垃圾信息），
    设置passed=False，后续条件边会直接跳到END。
    如果全部通过，设置passed=True，后续条件边会跳到LLM语义审核。
    """
    sensitive_hits = state.get("sensitive_hits", [])
    contact_hits = state.get("contact_hits", [])
    spam_result = state.get("spam_result", "内容质量正常")

    # 检查是否命中敏感词
    if sensitive_hits:
        return {
            "passed": False,
            "reason": f"内容包含违禁词汇：{'、'.join(sensitive_hits[:5])}",
            "risk_level": "high",
        }

    # 检查是否命中联系方式
    if contact_hits:
        return {
            "passed": False,
            "reason": f"内容包含联系方式：{'、'.join(contact_hits)}",
            "risk_level": "medium",
        }

    # 检查是否为垃圾信息
    if "异常" in spam_result:
        return {
            "passed": False,
            "reason": spam_result,
            "risk_level": "medium",
        }

    # 确定性检测全部通过，继续到LLM语义审核
    return {
        "passed": True,
        "reason": "确定性检测通过",
        "risk_level": "low",
    }


def should_skip_llm(state: GuardState) -> str:
    """条件边判断函数：确定性检测是否已经发现违禁内容

    返回值：
      - "reject"：确定性检测已发现违禁（passed=False），直接返回不通过
      - "llm_review"：确定性检测通过（passed=True），需要LLM做语义审核
    """
    if not state.get("passed", False):
        return "reject"
    return "llm_review"


# LLM语义审核系统提示词
LLM_REVIEW_PROMPT: str = """你是一个严格的内容审核助手，负责审核用户在社交平台发布的内容。

以下是确定性检测的结果（已通过，未发现明显违禁）：
- 敏感词检测：未命中
- 联系方式检测：未命中
- 垃圾信息检测：内容质量正常

你需要在此基础上进行语义审核，检测确定性工具无法识别的隐含违规内容。

审核维度：
1. 色情低俗：包含色情暗示、低俗擦边、性暗示、求约等隐晦表达
2. 暴力恐怖：包含暴力威胁、恐吓、血腥等隐晦表达
3. 违法犯罪：包含诈骗、赌博、毒品、传销等隐晦表达
4. 骚扰霸凌：包含人身攻击、侮辱谩骂、歧视性言论等
5. 不良引导：诱导他人添加联系方式、诱导转账等

审核原则：
- 宁可误拒，不可漏放
- 正常的日常分享、倾诉烦恼、讨论兴趣爱好是允许的
- "有没有一起打游戏"、"陪我一起玩"等邀请游戏的表达，不能判定违规
- 单纯的"想交朋友"、"想聊天"不能判定违规

请严格按照以下JSON格式返回审核结果（不要包含在代码块中）：
{"passed": true/false, "reason": "审核通过"/"违规原因描述", "risk_level": "low"/"medium"/"high"}"""


def _get_llm() -> ChatOpenAI:
    """获取DeepSeek LLM实例，配置模型名称、API密钥、基础URL和生成参数"""
    from app.config import settings  # 延迟导入配置模块，避免循环导入

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        max_retries=0,
        temperature=0.0,  # 温度设为0，确保审核结果确定性最高
        max_tokens=256,  # 最大生成token数，审核结果较短
    )


async def llm_review(state: GuardState) -> dict[str, Any]:
    """状态图节点3：LLM语义审核

    将确定性检测结果和待审核内容嵌入SystemMessage，
    让LLM做最终的语义判断（检测隐含违规、擦边球等确定性工具无法识别的内容）。
    """
    log = get_logger("guard_agent")
    content = state.get("content", "")
    content_type = state.get("content_type", "bottle")
    user_id = state.get("user_id", 0)
    type_label = "纸团" if content_type == "bottle" else "聊天消息"

    messages = [
        SystemMessage(content=LLM_REVIEW_PROMPT),
        HumanMessage(content=f"请审核以下{type_label}内容：\n\n{content}"),
    ]

    start = time.time()
    llm = _get_llm()
    response = await llm.ainvoke(messages)
    duration_ms = round((time.time() - start) * 1000, 1)

    # 记录LLM调用日志（含Token消耗和耗时）
    tokens = getattr(response, "usage_metadata", {}) or {}
    total_tokens = tokens.get("total_tokens", 0)
    global_metrics.record_llm_call(tokens=total_tokens, duration_ms=duration_ms)
    log.info("llm_review", user_id=user_id, duration_ms=duration_ms,
             total_tokens=total_tokens, content_type=content_type)

    return {"messages": [response]}


def extract_result(state: GuardState) -> dict[str, Any]:
    """严格校验审核结果；无效输出作为技术故障向上传播。"""
    result = parse_json_response(state.get("messages", []), {"passed", "reason", "risk_level"})
    passed, reason, risk_level = result["passed"], result["reason"], result["risk_level"]
    if type(passed) is not bool or not isinstance(reason, str) or risk_level not in ("low", "medium", "high"):
        raise ValueError("审核结果字段格式无效")
    global_metrics.record_guard_result(passed=passed)
    get_logger("guard_agent").info("guard_result", user_id=state.get("user_id", 0), passed=passed, risk_level=risk_level)
    return {"passed": passed, "reason": reason, "risk_level": risk_level}


def build_guard_graph() -> StateGraph:
    """构建ContentGuardAgent的LangGraph状态图（Pipeline模式）

    流程说明：
      run_deterministic → check_deterministic → (条件边)
        ├── 命中违禁 → 直接返回不通过 → END
        └── 全部通过 → llm_review → extract_result → END
    """
    graph = StateGraph(GuardState)

    # 添加节点
    graph.add_node("run_deterministic", run_deterministic)  # 节点1：强制执行确定性检测
    graph.add_node("check_deterministic", check_deterministic)  # 节点2：检查确定性检测结果
    graph.add_node("llm_review", llm_review)  # 节点3：LLM语义审核
    graph.add_node("extract_result", extract_result)  # 节点4：提取LLM审核结果

    # 设置入口节点
    graph.set_entry_point("run_deterministic")

    # 添加边
    graph.add_edge("run_deterministic", "check_deterministic")  # 确定性检测 → 检查结果
    graph.add_conditional_edges("check_deterministic", should_skip_llm, {
        "reject": END,  # 命中违禁 → 直接结束（不通过）
        "llm_review": "llm_review",  # 全部通过 → LLM语义审核
    })
    graph.add_edge("llm_review", "extract_result")  # LLM审核 → 提取结果
    graph.add_edge("extract_result", END)  # 提取结果 → 结束

    return graph


guard_graph = build_guard_graph()
guard_app = guard_graph.compile()
