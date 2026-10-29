from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用尚未定义的类型

import asyncio  # 异步IO库，用于判断工具函数是否为协程函数并执行异步工具
import json  # JSON序列化/反序列化库，用于工具返回值和LLM响应解析
import re  # 正则表达式库，用于清理LLM返回的Markdown代码块标记
import time  # 时间库，用于计算节点执行耗时
from typing import Annotated, Any, NotRequired, TypedDict  # 类型注解工具

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # LangChain消息类型：AIMessage(LLM回复)、HumanMessage(用户输入)、SystemMessage(系统指令)、ToolMessage(工具执行结果)
from langchain_core.tools import tool  # LangChain工具装饰器，将函数标记为LLM可调用的工具
from langchain_openai import ChatOpenAI  # LangChain的OpenAI兼容LLM客户端，用于调用DeepSeek模型
from langgraph.graph import END, StateGraph  # LangGraph核心组件：END表示状态图终止节点、StateGraph用于构建有状态图
from langgraph.graph.message import add_messages  # LangGraph消息追加reducer，确保messages列表自动追加而非覆盖
from httpx import TransportError
from openai import APIConnectionError, APIStatusError

from app.agent_logger import get_logger, get_trace_id, global_metrics, new_trace_id  # 结构化日志工具
from app.agents.workflow_contracts import AgentStatus, IceBreakAnalysisContext
from app.config import settings


# ============================================================
# SupervisorAgent 状态图流程图（Supervisor模式 + Tool/Function Calling）
# ============================================================
#
#                        ┌────────────────────┐
#                        │  init_conversation  │  初始化对话：构建SystemMessage + HumanMessage
#                        └────────┬───────────┘
#                                 │
#                                 ▼
#                        ┌────────────────────┐
#                   ┌──→ │     agent_node      │  Supervisor LLM推理，决定调用哪个子Agent
#                   │    │  (LLM+Tool Calling) │
#                   │    └────────┬───────────┘
#                   │             │
#                   │             ├── Tool Calls ──→ execute_tools ──┐
#                   │             │                                  │
#                   │             │                     ┌────────────┘
#                   │             │                     │
#                   │             │                     ▼
#                   │             │         ┌──────────────────────────────┐
#                   │             │         │   execute_tools 执行         │
#                   │             │         │   根据工具名调用子Agent：     │
#                   │             │         │   ├─ call_match_agent        │
#                   │             │         │   ├─ call_guard_agent        │
#                   │             │         │   ├─ call_ice_break_agent    │
#                   │             │         │   ├─ call_content_gen        │
#                   │             │         │   ├─ call_profile_analysis   │
#                   │             │         │   └─ call_conversation_guide │
#                   │             │         └────────────┬─────────────────┘
#                   │             │                      │
#                   │             └──────────────────────┘
#                   │                    tools → agent（ReAct循环）
#                   │
#                   │             ┌── 无Tool Calls ──→ extract_result → [END]
#                   │             │
#                   └─────────────┘
#
# 复杂编排场景（Supervisor LLM通过ReAct循环自主编排）：
#
#   场景A: pick_full（链式调用）
#     call_match_agent → call_profile_analysis_agent → call_conversation_guide_agent
#       → call_ice_break_agent → call_guard_agent
#     拾纸团 → 分析契合度 → 制定沟通策略 → 生成并审核破冰建议
#
#   场景B: write_smart（条件重试）
#     call_guard_agent → [不通过] → call_content_gen_agent
#     审核内容 → 审核不通过则生成替代内容
#
#   场景C: chat_rescue（组合调用）
#     call_conversation_guide_agent → call_ice_break_agent
#     分析对话状态 → 生成破冰/回复建议
#
#   场景D: onboarding（组合调用）
#     call_profile_analysis_agent → call_content_gen_agent → call_guard_agent
#     分析画像 → 生成内容 → 审核内容
#
# ============================================================


class SupervisorState(TypedDict):
    """SupervisorAgent的状态定义，贯穿整个监督编排流程图"""
    user_id: int  # 当前用户ID
    action: str  # 用户意图动作："pick"旧版单步拾纸团 / "pick_full"拾纸团全流程 / "write"写纸团 / "write_smart"写纸团智能流程 / "ice_break"破冰建议 / "content_gen"内容生成 / "send_message"发消息 / "chat_rescue"对话救场 / "onboarding"新用户引导
    content: str  # 用户提交的内容（写纸团/发消息时使用）
    target_user_id: int  # 对方用户ID（破冰建议/对话救场时使用）
    target_university: str  # 拾纸团时选择的目标学校
    target_city: str  # 拾纸团时选择的目标城市
    target_province: str  # 拾纸团时选择的目标省份
    content_type: str  # 内容类型："bottle"纸团 或 "message"聊天消息
    messages: Annotated[list, add_messages]  # LLM对话消息列表（使用add_messages reducer自动追加）
    final_result: str  # 最终返回给路由层的JSON字符串结果
    guard_passed: bool  # 内容审核是否通过的标志位
    guard_reason: str  # 内容审核的原因描述
    # 以下字段用于跨子Agent协作。旧路由可以不传入，初始化节点会补齐默认值。
    completed_steps: NotRequired[list[str]]  # 已完成的子Agent步骤
    artifacts: NotRequired[dict[str, dict[str, Any]]]  # 各子Agent的结构化结果
    step_count: NotRequired[int]  # Supervisor已经执行的工具步骤数
    retry_count: NotRequired[int]  # 当前任务已经重试的次数
    max_steps: NotRequired[int]  # 防止ReAct循环失控的最大步骤数
    status: NotRequired[str]  # running/success/no_result/failed/budget_exceeded
    error: NotRequired[dict[str, Any] | None]  # 最后一次结构化错误
    deadline_at: NotRequired[float]  # 整条任务的时间预算截止点（time.monotonic 秒）


SUPERVISOR_MAX_STEPS = 8
VALID_AGENT_STATUSES = {status.value for status in AgentStatus}
FAILURE_STATUSES = {"retryable_error", "failed", "fatal_error"}
RETRY_DELAY_SECONDS = 0.2
# Research 可能调用未知外部工具，不自动重放；其余子 Agent 只读取数据并生成文本。
RETRY_SAFE_TOOLS = {
    "call_match_agent", "call_guard_agent", "call_ice_break_agent",
    "call_content_gen_agent", "call_profile_analysis_agent", "call_conversation_guide_agent",
}

# action只允许调用与目标相关的子Agent。复杂流程仍由Supervisor根据中间结果决定顺序，
# 但不能调用完全无关的工具。
ACTION_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "pick": {"call_match_agent"},
    "write": {"call_guard_agent"},
    "send_message": {"call_guard_agent"},
    "ice_break": {"call_ice_break_agent"},
    "content_gen": {"call_content_gen_agent"},
    "pick_full": {
        "call_match_agent",
        "call_profile_analysis_agent",
        "call_conversation_guide_agent",
        "call_ice_break_agent",
        "call_guard_agent",
    },
    "write_smart": {"call_guard_agent", "call_content_gen_agent", "call_research_agent"},
    "chat_rescue": {
        "call_conversation_guide_agent",
        "call_ice_break_agent",
        "call_research_agent",
    },
    "onboarding": {
        "call_profile_analysis_agent",
        "call_content_gen_agent",
        "call_guard_agent",
        "call_research_agent",
    },
}

REQUIRED_ARTIFACTS: dict[str, set[str]] = {
    "pick": {"match"},
    "write": {"guard"},
    "send_message": {"guard"},
    "ice_break": {"ice_break"},
    "content_gen": {"content_gen"},
    "pick_full": {"match", "profile_analysis", "conversation_guide", "ice_break", "guard"},
    "write_smart": {"guard"},
    "chat_rescue": {"conversation_guide", "ice_break"},
    "onboarding": {"profile_analysis", "content_gen", "guard"},
}


def create_supervisor_state(
    *,
    user_id: int,
    action: str,
    content: str = "",
    target_user_id: int = 0,
    university: str = "",
    city: str = "",
    province: str = "",
    content_type: str = "bottle",
    max_steps: int = SUPERVISOR_MAX_STEPS,
) -> SupervisorState:
    """创建带完整协作字段的Supervisor初始状态。"""
    return {
        "user_id": user_id,
        "action": action,
        "content": content,
        "target_user_id": target_user_id,
        "target_university": university,
        "target_city": city,
        "target_province": province,
        "content_type": content_type,
        "messages": [],
        "final_result": "",
        "guard_passed": False,
        "guard_reason": "",
        "completed_steps": [],
        "artifacts": {},
        "step_count": 0,
        "retry_count": 0,
        "max_steps": max_steps,
        "status": "running",
        "error": None,
    }


# ============================================================
# Tool定义：将专业子Agent封装为Supervisor可调用的工具
# ============================================================

@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def call_match_agent(
    user_id: int,
    university: str,
    city: str,
    province: str,
    reuse_profile: bool = True,
) -> str:
    """拾纸团智能匹配工具。当用户想要拾取纸团时调用此工具。
    此工具会调用MatchAgent执行分层递进查询+多维度评分，返回最佳匹配的纸团。
    输入：用户ID、目标学校、目标城市、目标省份
    输出：匹配结果的JSON字符串（包含纸团信息、对方信息、评分和匹配原因）"""
    from app.agents.match_agent import match_app  # 延迟导入MatchAgent编译后的应用，避免循环导入

    # 构造MatchAgent所需的初始状态字典
    match_state = {
        "user_id": user_id,
        "target_university": university,
        "target_city": city,
        "target_province": province,
        "reuse_profile": reuse_profile,
        "messages": [],
        "best_match": None,
        "match_reason": "",
        "match_score": 0.0,
        "content_score": 0.0,
        "match_level": "",
    }

    # 异步调用MatchAgent的状态图，传入初始状态
    result = await match_app.ainvoke(match_state)

    # 通过显式status区分“没有候选”和“工具执行失败”。
    best_match = result.get("best_match")
    return json.dumps({
        "agent": "match",
        "status": "success" if best_match else "no_result",
        "result": "success" if best_match else "no_candidates",
        "best_match": best_match,
        "match_reason": result.get("match_reason", ""),
        "match_score": result.get("match_score", 0.0),
        "content_score": result.get("content_score", 0.0),
        "match_level": result.get("match_level", ""),
    }, ensure_ascii=False)


@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def call_guard_agent(content: str, content_type: str, user_id: int) -> str:
    """内容审核工具。当用户提交内容（写纸团或发消息）时调用此工具进行审核。
    此工具会调用ContentGuardAgent执行敏感词检测、联系方式检测、垃圾信息检测。
    输入：待审核内容、内容类型（bottle/message）、用户ID
    输出：审核结果的JSON字符串（包含是否通过、原因、风险等级）"""
    from app.agents.content_guard_agent import guard_app  # 延迟导入ContentGuardAgent编译后的应用

    # 构造ContentGuardAgent所需的初始状态字典
    guard_state = {
        "content": content,
        "content_type": content_type,
        "user_id": user_id,
        "messages": [],
        "passed": False,
        "reason": "",
        "risk_level": "",
        "sensitive_hits": [],  # 确定性检测：命中的敏感词列表
        "contact_hits": [],  # 确定性检测：命中的联系方式类型列表
        "spam_result": "",  # 确定性检测：垃圾信息检测结果
    }

    # 异步调用ContentGuardAgent的状态图
    result = await guard_app.ainvoke(guard_state)

    return json.dumps({
        "agent": "guard",
        "status": "success",
        "passed": result.get("passed", False),  # fail-closed：默认拒绝
        "reason": result.get("reason", "审核未通过"),
        "risk_level": result.get("risk_level", "medium"),
    }, ensure_ascii=False)


@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def call_ice_break_agent(
    user_id: int,
    target_user_id: int,
    analysis_context: IceBreakAnalysisContext | None = None,
) -> str:
    """破冰建议生成工具。当用户需要与匹配到的陌生人开启对话或回复消息时调用此工具。
    此工具会调用IceBreakAgent查询对方画像和聊天记录，生成3条破冰/回复建议。
    输入：当前用户ID、对方用户ID
    输出：破冰建议结果的JSON字符串（包含3条建议和场景类型）"""
    from app.agents.ice_break_agent import ice_break_app  # 延迟导入IceBreakAgent编译后的应用

    # 构造IceBreakAgent所需的初始状态字典
    ice_state = {
        "user_id": user_id,
        "target_user_id": target_user_id,
        "messages": [],
        "scenario": "",
        "hints": [],
        "analysis_context": analysis_context.model_dump(exclude_defaults=True) if analysis_context else None,
    }

    # 异步调用IceBreakAgent的状态图
    result = await ice_break_app.ainvoke(ice_state)

    return json.dumps({
        "agent": "ice_break",
        "status": "success",
        "user_id": user_id,
        "target_user_id": target_user_id,
        "hints": result.get("hints", []),
        "scenario": result.get("scenario", "ice_break"),
    }, ensure_ascii=False)


@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def call_content_gen_agent(user_id: int) -> str:
    """纸团内容生成工具。当用户需要写纸团但不知道写什么时调用此工具。
    此工具会调用ContentGenAgent查询用户画像和热门话题，生成3条个性化内容建议。
    输入：用户ID
    输出：内容建议结果的JSON字符串（包含3条建议，每条有内容和风格标签）"""
    from app.agents.content_gen_agent import content_gen_app  # 延迟导入ContentGenAgent编译后的应用

    # 构造ContentGenAgent所需的初始状态字典
    gen_state = {
        "user_id": user_id,
        "messages": [],
        "suggestions": [],
    }

    # 异步调用ContentGenAgent的状态图
    result = await content_gen_app.ainvoke(gen_state)

    return json.dumps({
        "agent": "content_gen",
        "status": "success",
        "user_id": user_id,
        "suggestions": result.get("suggestions", []),
    }, ensure_ascii=False)


@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def call_profile_analysis_agent(user_id: int, target_user_id: int = 0) -> str:
    """用户画像分析工具。分析用户画像完整度、生成个性标签、提供完善建议。
    如果提供target_user_id，还会额外分析两个用户的契合度。
    输入：用户ID、对方用户ID（可选，0表示不分析契合度）
    输出：画像分析结果的JSON字符串（包含个性标签、完整度评分、完善建议、契合度分析）"""
    from app.agents.profile_analysis_agent import profile_analysis_app  # 延迟导入ProfileAnalysisAgent编译后的应用

    # 构造ProfileAnalysisAgent所需的初始状态字典
    analysis_state = {
        "user_id": user_id,
        "target_user_id": target_user_id,
        "messages": [],
        "user_profile": "",
        "target_profile": "",
        "tags": [],
        "completeness_score": 0,
        "suggestions": [],
        "compatibility": "",
    }

    # 异步调用ProfileAnalysisAgent的状态图
    result = await profile_analysis_app.ainvoke(analysis_state)

    return json.dumps({
        "agent": "profile_analysis",
        "status": "success",
        "user_id": user_id,
        "target_user_id": target_user_id or None,
        "tags": result.get("tags", []),
        "completeness_score": result.get("completeness_score", 0),
        "suggestions": result.get("suggestions", []),
        "compatibility": result.get("compatibility", ""),
    }, ensure_ascii=False)


@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def call_conversation_guide_agent(user_id: int, target_user_id: int) -> str:
    """对话引导工具。分析对话状态（活跃/冷场/新对话），生成话题转换建议和回复策略。
    当用户在聊天中遇到冷场或不知道怎么回复时调用此工具。
    输入：当前用户ID、对方用户ID
    输出：对话引导结果的JSON字符串（包含对话状态、话题建议、回复策略、破冰话术）"""
    from app.agents.conversation_guide_agent import conversation_guide_app  # 延迟导入ConversationGuideAgent编译后的应用

    # 构造ConversationGuideAgent所需的初始状态字典
    guide_state = {
        "user_id": user_id,
        "target_user_id": target_user_id,
        "messages": [],
        "user_profile": "",
        "target_profile": "",
        "chat_history": "",
        "conversation_status": "new",
        "topic_suggestions": [],
        "reply_strategies": [],
        "ice_breaker": "",
    }

    # 异步调用ConversationGuideAgent的状态图
    result = await conversation_guide_app.ainvoke(guide_state)

    # 将对话引导结果序列化为JSON字符串返回
    return json.dumps({
        "agent": "conversation_guide",
        "status": "success",
        "user_id": user_id,
        "target_user_id": target_user_id,
        "conversation_status": result.get("conversation_status", "new"),
        "topic_suggestions": result.get("topic_suggestions", []),
        "reply_strategies": result.get("reply_strategies", []),
        "ice_breaker": result.get("ice_breaker", ""),
    }, ensure_ascii=False)


@tool
async def call_research_agent(user_id: int, query: str) -> str:
    """资料查询工具：通过MCP Research Agent获取热点或网页信息。"""
    from app.agents.ai_chat_agent import ai_chat_app

    research_state = {
        "user_id": user_id,
        "messages": [HumanMessage(content=query)],
        "final_response": "",
        "mcp_tool_calls": [],
        "mcp_results": [],
    }
    result = await ai_chat_app.ainvoke(research_state)
    if result.get("status") in FAILURE_STATUSES or result.get("error"):
        error = result.get("error") or {}
        if not isinstance(error, dict):
            error = {"error": str(error)}
        return json.dumps({
            "agent": "research",
            "status": "retryable_error" if error.get("retryable") else "fatal_error",
            "error": error.get("error", "资料查询失败，请稍后重试"),
            "error_code": error.get("error_code", "RESEARCH_FAILED"),
            "retryable": bool(error.get("retryable")),
        }, ensure_ascii=False)
    response = str(result.get("final_response", "")).strip()
    return json.dumps(
        {
            "agent": "research",
            "status": "success" if response else "no_result",
            "query": query,
            "evidence": response,
        },
        ensure_ascii=False,
    )


# Supervisor的工具列表：专业子Agent作为Supervisor可调用的工具
supervisor_tools = [
    call_match_agent,
    call_guard_agent,
    call_ice_break_agent,
    call_content_gen_agent,
    call_profile_analysis_agent,
    call_conversation_guide_agent,
    call_research_agent,
]

# 工具名到工具函数的映射字典，用于execute_tools中根据工具名查找对应函数
supervisor_tool_map = {t.name: t for t in supervisor_tools}


def _decode_tool_payload(value: Any) -> dict[str, Any]:
    """把子Agent结果转换为字典，避免后续节点反复猜字符串格式。"""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "agent": "unknown",
            "status": "fatal_error",
            "error": value,
            "error_code": "INVALID_TOOL_RESULT",
        }
    return {
        "agent": "unknown",
        "status": "fatal_error",
        "error": "子Agent返回了无法识别的结果",
        "error_code": "INVALID_TOOL_RESULT",
    }


def _coerce_payload_status(payload: dict[str, Any]) -> dict[str, Any]:
    """把子Agent状态限制在统一集合内，避免未知字符串被当成成功。"""
    normalized = dict(payload)
    status = str(normalized.get("status", "success"))
    if status not in VALID_AGENT_STATUSES:
        return {
            **normalized,
            "status": "fatal_error",
            "error": "子Agent返回了未知状态",
            "error_code": "INVALID_AGENT_STATUS",
        }
    if status in FAILURE_STATUSES and not normalized.get("error"):
        normalized["error"] = "子Agent执行失败"
    if normalized.get("error") and status not in FAILURE_STATUSES:
        normalized.update(status="fatal_error", error_code="INCONSISTENT_AGENT_RESULT", retryable=False)
    return normalized


def _artifact_key(tool_name: str, payload: dict[str, Any]) -> str:
    """得到稳定的协作结果名称，用于父状态中的artifacts索引。"""
    agent_name = str(payload.get("agent", ""))
    if agent_name:
        return agent_name
    return tool_name.removeprefix("call_").removesuffix("_agent")


def _tool_allowed(state: SupervisorState, tool_name: str) -> bool:
    """限制Supervisor只能调用当前任务允许的子Agent。"""
    action = state.get("action", "")
    allowed = ACTION_TOOL_ALLOWLIST.get(action)
    return allowed is None or tool_name in allowed


def _normalize_tool_args(
    state: SupervisorState,
    tool_name: str,
    raw_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """补齐并校正工具参数，身份和匹配目标以父状态为准。"""
    args = dict(raw_args or {})
    user_id = state.get("user_id", 0)
    if tool_name in {
        "call_match_agent",
        "call_guard_agent",
        "call_ice_break_agent",
        "call_content_gen_agent",
        "call_profile_analysis_agent",
        "call_conversation_guide_agent",
        "call_research_agent",
    }:
        args["user_id"] = user_id

    artifacts = state.get("artifacts") or {}
    match_payload = artifacts.get("match", {})
    best_match = match_payload.get("best_match") or {}
    target_user_id = (
        state.get("target_user_id", 0)
        or best_match.get("user_id", 0)
    )
    if tool_name in {"call_profile_analysis_agent", "call_ice_break_agent", "call_conversation_guide_agent"}:
        if target_user_id:
            args["target_user_id"] = target_user_id
        elif state.get("action") == "pick_full":
            args["target_user_id"] = 0
    if tool_name == "call_ice_break_agent":
        # 使用服务端父状态中的分析结论，模型不能自行伪造上游上下文。
        context = IceBreakAnalysisContext.from_artifacts(
            artifacts.get("profile_analysis"), artifacts.get("conversation_guide")
        )
        args.pop("analysis_context", None)
        if context is not None:
            args["analysis_context"] = context.model_dump(exclude_defaults=True)

    if tool_name == "call_match_agent":
        args.setdefault("university", state.get("target_university", ""))
        args.setdefault("city", state.get("target_city", ""))
        args.setdefault("province", state.get("target_province", ""))
        # 旧版pick要求find_best_match再次查询画像；完整流程沿用工具默认的复用行为。
        if state.get("action") == "pick":
            args["reuse_profile"] = False
        else:
            # 不让模型把完整流程误切到旧版路径，同时保持原有四参数调用形态。
            args.pop("reuse_profile", None)
    if tool_name == "call_guard_agent":
        args.setdefault("content", state.get("content", ""))
        args.setdefault("content_type", state.get("content_type", "bottle"))
        if not args.get("content"):
            hints = (artifacts.get("ice_break") or {}).get("hints") or []
            suggestions = (artifacts.get("content_gen") or {}).get("suggestions") or []
            if hints:
                args["content"] = hints[0]
            elif suggestions and isinstance(suggestions[0], dict):
                args["content"] = suggestions[0].get("content", "")
    if tool_name == "call_research_agent":
        args.setdefault("query", state.get("content", "请查询与当前任务相关的校园热点"))
    return args


def _record_artifact(
    state: SupervisorState,
    tool_name: str,
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    """将一次子Agent结果写入父状态，并返回新的结果索引。"""
    artifacts = dict(state.get("artifacts") or {})
    key = _artifact_key(tool_name, payload)
    artifacts[key] = payload
    completed_steps = list(state.get("completed_steps") or [])
    if payload.get("status") in {"success", "no_result"} and key not in completed_steps:
        completed_steps.append(key)
    return artifacts, completed_steps, str(payload.get("status", "success"))


def _error_payload(
    agent: str, message: str, error_code: str = "TOOL_FAILED", *, retryable: bool = False,
) -> dict[str, Any]:
    """生成不会被Supervisor误认为成功结果的结构化错误。"""
    agent_name = agent.removeprefix("call_").removesuffix("_agent") or agent
    return {
        "agent": agent_name,
        "status": "retryable_error" if retryable else "fatal_error",
        "error": message,
        "error_code": error_code,
        "retryable": retryable,
    }


def _exception_payload(agent: str, exc: Exception) -> dict[str, Any]:
    """只重试明确的临时故障；对外结果不携带供应商响应或数据库细节。"""
    timed_out = isinstance(exc, TimeoutError)
    retryable = isinstance(exc, (TimeoutError, ConnectionError, TransportError, APIConnectionError))
    if isinstance(exc, APIStatusError):
        retryable = exc.status_code in {408, 429} or exc.status_code >= 500
    return _error_payload(
        agent,
        "调用超时，请稍后重试" if timed_out else "服务调用失败，请稍后重试",
        "CALL_TIMEOUT" if timed_out else "TOOL_FAILED",
        retryable=retryable,
    )


async def _invoke_worker(tool: Any, args: dict[str, Any]) -> Any:
    coroutine = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if asyncio.iscoroutinefunction(coroutine):
        return await tool.ainvoke(args)
    return await asyncio.to_thread(tool.invoke, args)


def _remaining_budget(state: SupervisorState) -> float:
    """返回整条任务剩余的可用秒数；未设置截止点时视为不限时。"""
    deadline = state.get("deadline_at")
    if not deadline:
        return float("inf")
    return float(deadline) - time.monotonic()


# ============================================================
# 状态图节点定义
# ============================================================

async def execute_tools(state: SupervisorState) -> dict[str, Any]:
    """有界执行子 Agent；失败耗尽后直接结束，避免依赖模型决定是否重试。"""
    log = get_logger("supervisor_agent")
    tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
    tool_messages: list[ToolMessage] = []
    artifacts = dict(state.get("artifacts") or {})
    completed_steps = list(state.get("completed_steps") or [])
    step_count = int(state.get("step_count", 0))
    retry_count = int(state.get("retry_count", 0))
    max_steps = int(state.get("max_steps", SUPERVISOR_MAX_STEPS))
    last_error = None
    no_result = False
    stop_batch = False

    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", ""))
        # 整条任务超出时间预算后，不再发起任何新的子Agent调用。
        if _remaining_budget(state) <= 0:
            payload = _error_payload(tool_name, "任务已超过时间预算", "TIME_BUDGET_EXCEEDED")
            tool_messages.append(ToolMessage(
                content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id, status="error",
            ))
            last_error = payload
            stop_batch = True
            continue
        if stop_batch:
            payload = _error_payload(tool_name, "前置子Agent没有成功，已跳过当前调用", "DEPENDENCY_NOT_READY")
            tool_messages.append(ToolMessage(
                content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id, status="error",
            ))
            continue

        if step_count >= max_steps:
            payload = _error_payload(tool_name, "已达到Supervisor最大步骤数", "STEP_BUDGET_EXCEEDED")
        elif tool_name not in supervisor_tool_map:
            payload = _error_payload(tool_name or "unknown", "未知的子Agent工具", "UNKNOWN_TOOL")
            step_count += 1
        elif not _tool_allowed(state, tool_name):
            payload = _error_payload(tool_name, "当前任务不允许调用该子Agent", "TOOL_NOT_ALLOWED")
            step_count += 1
        elif (
            state.get("action") in {"pick_full", "chat_rescue"}
            and tool_name in {"call_profile_analysis_agent", "call_ice_break_agent", "call_conversation_guide_agent"}
            and not (
                state.get("target_user_id", 0)
                or (artifacts.get("match", {}).get("best_match") or {}).get("user_id")
            )
        ):
            payload = _error_payload(tool_name, "缺少目标用户，不能执行依赖匹配结果的Agent", "DEPENDENCY_NOT_READY")
        else:
            max_attempts = settings.AGENT_MAX_ATTEMPTS if tool_name in RETRY_SAFE_TOOLS else 1
            for attempt in range(max_attempts):
                # 每次尝试前重新计算剩余预算，并把单次超时压到剩余时间内。
                remaining = _remaining_budget(state)
                if remaining <= 0:
                    payload = _error_payload(tool_name, "任务已超过时间预算", "TIME_BUDGET_EXCEEDED")
                    break
                if attempt:
                    retry_count += 1
                    await asyncio.sleep(min(RETRY_DELAY_SECONDS * attempt, remaining))
                step_count += 1
                tool_start = time.monotonic()
                try:
                    args = _normalize_tool_args({**state, "artifacts": artifacts}, tool_name, tool_call.get("args"))
                    raw_result = await asyncio.wait_for(
                        _invoke_worker(supervisor_tool_map[tool_name], args),
                        timeout=min(settings.AGENT_CALL_TIMEOUT_SECONDS, remaining),
                    )
                    payload = _decode_tool_payload(raw_result)
                    payload.setdefault("agent", _artifact_key(tool_name, payload))
                    payload.setdefault("status", "success")
                    payload = _coerce_payload_status(payload)
                except Exception as exc:
                    payload = _exception_payload(tool_name, exc)
                    # exc_info=True 让结构化日志带上完整堆栈，便于线上定位根因。
                    log.error("sub_agent_error", tool_name=tool_name,
                              error_type=type(exc).__name__, attempt=attempt + 1,
                              trace_id=get_trace_id(), user_id=state.get("user_id", 0),
                              exc_info=True)
                else:
                    log.info("sub_agent_called", tool_name=tool_name,
                             duration_ms=round((time.monotonic() - tool_start) * 1000, 1),
                             attempt=attempt + 1, status=payload.get("status"),
                             trace_id=get_trace_id(), user_id=state.get("user_id", 0))
                if not (
                    payload.get("status") == "retryable_error"
                    and payload.get("retryable", True)
                    and attempt + 1 < max_attempts
                    and step_count < max_steps
                ):
                    break

        artifacts, completed_steps, result_status = _record_artifact(
            {**state, "artifacts": artifacts, "completed_steps": completed_steps}, tool_name, payload,
        )
        failed = result_status in FAILURE_STATUSES
        tool_messages.append(ToolMessage(
            content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id,
            status="error" if failed else "success",
        ))
        if failed:
            last_error = payload
            stop_batch = True
        elif tool_name == "call_match_agent" and result_status == "no_result":
            no_result = True
            stop_batch = True

    status = "running"
    if last_error:
        status = "budget_exceeded" if last_error.get("error_code") in {
            "STEP_BUDGET_EXCEEDED", "TIME_BUDGET_EXCEEDED",
        } else "failed"
    elif no_result:
        status = "no_result"
    return {
        "messages": tool_messages, "artifacts": artifacts, "completed_steps": completed_steps,
        "step_count": step_count, "retry_count": retry_count, "max_steps": max_steps,
        "status": status, "error": last_error,
    }


# Supervisor LLM系统提示词：定义全局指挥官的角色、可用工具、编排策略和输出格式
SUPERVISOR_SYSTEM_PROMPT: str = """你是The Player社交平台的全局智能调度助手（Supervisor），负责根据用户意图调度不同的专业子Agent来完成任务。

你可以调度以下7个子Agent（作为工具调用）：
1. call_match_agent：拾纸团智能匹配。当用户想拾取纸团时调用。
   参数：user_id（用户ID）、university（目标学校）、city（目标城市）、province（目标省份）
2. call_guard_agent：内容审核。当用户提交内容（写纸团/发消息）时调用，审核是否违规。
   参数：content（待审核内容）、content_type（"bottle"或"message"）、user_id（用户ID）
3. call_ice_break_agent：破冰建议生成。当用户需要与匹配到的陌生人开启对话或回复消息时调用。
   参数：user_id（当前用户ID）、target_user_id（对方用户ID）
4. call_content_gen_agent：纸团内容生成。当用户需要写纸团但不知道写什么时调用。
   参数：user_id（用户ID）
5. call_profile_analysis_agent：用户画像分析。分析用户画像完整度、个性标签、完善建议。
   参数：user_id（用户ID）、target_user_id（对方用户ID，0表示不分析契合度）
6. call_conversation_guide_agent：对话引导。分析对话状态，生成话题转换建议和回复策略。
   参数：user_id（当前用户ID）、target_user_id（对方用户ID）
7. call_research_agent：资料查询。通过MCP查询热点或网页信息，返回可引用的资料。
   参数：user_id（用户ID）、query（查询问题）

═══════════════════════════════════════════════════════════════
调度策略（重要！按action类型执行对应的编排流程）
═══════════════════════════════════════════════════════════════

【单步调用】
- 拾纸团（pick）：只调用call_match_agent匹配纸团
- 写纸团（write）：只调用call_guard_agent审核内容
- 发消息（send_message）：只调用call_guard_agent审核内容
- 破冰建议（ice_break）：只调用call_ice_break_agent
- 内容生成（content_gen）：只调用call_content_gen_agent

【链式调用】
- 拾纸团全流程（pick_full）：
  第1步：调用call_match_agent匹配纸团
  第2步：从匹配结果中获取对方user_id，调用call_profile_analysis_agent分析契合度（target_user_id=对方user_id）
  第3步：调用call_conversation_guide_agent判断沟通策略（target_user_id=对方user_id）
  第4步：调用call_ice_break_agent生成破冰建议（target_user_id=对方user_id）
  第5步：从破冰建议中选择需要展示的文本，调用call_guard_agent审核
  如果没有候选，立即结束，不要调用后续Agent；最终汇总所有结果。

【条件重试】
- 写纸团智能流程（write_smart）：
  第1步：调用call_guard_agent审核内容
  如果审核通过：直接输出结果
  如果审核不通过：可先调用call_research_agent获取灵感，再调用call_content_gen_agent生成替代内容建议

【组合调用】
- 对话救场（chat_rescue）：
  第1步：调用call_conversation_guide_agent分析对话状态并生成引导建议
  如果需要新鲜话题，调用call_research_agent查询资料；然后调用call_ice_break_agent生成破冰/回复建议
  最终：汇总对话分析+破冰建议，输出完整引导方案

- 新用户引导（onboarding）：
  第1步：调用call_profile_analysis_agent分析用户画像（target_user_id=0）
  第2步：调用call_content_gen_agent根据画像生成个性化纸团内容建议
  第3步：调用call_guard_agent审核生成的内容（content_type="bottle"）
  最终：汇总画像分析+内容建议+审核结果，输出完整引导方案

═══════════════════════════════════════════════════════════════

⚠️ 严格的输出格式要求 ⚠️
当你完成所有工具调用后，必须将最终结果转为纯JSON输出，不得包含任何其他内容。
禁止使用Markdown格式、禁止使用表格、禁止使用代码块标记、禁止添加任何解释性文字。

输出格式：
{"action": "你执行的动作类型", "result": 工具返回的完整JSON结果, "message": "给用户的提示信息"}

对于链式/组合调用，result应包含所有子Agent的结果：
{"action": "pick_full", "result": {"match": 匹配结果, "profile_analysis": 画像分析结果, "conversation_guide": 对话策略, "ice_break": 破冰建议, "guard": 审核结果}, "message": "..."}

再次强调：只输出JSON，不要输出任何其他内容。"""


def _get_llm() -> ChatOpenAI:
    """获取DeepSeek LLM实例，配置模型名称、API密钥、基础URL和生成参数"""
    from app.config import settings  # 延迟导入配置模块，避免循环导入

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.1,  # 温度设为0.1，Supervisor需要确定性高的路由决策（不需要创意性）
        max_tokens=1024,  # 最大生成token数，复杂编排场景输出较长，需要更多token
        timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        max_retries=0,  # 由节点统一控制总尝试次数，避免与 SDK 重试叠加。
    )


def init_conversation(state: SupervisorState) -> dict[str, Any]:
    """状态图节点1：初始化Supervisor LLM对话，构建SystemMessage和HumanMessage

    根据用户的action类型，构造对应的请求消息，
    Supervisor LLM将根据消息内容自主决定调用哪个子Agent。
    支持9种action：5种单步调用 + 4种复杂编排。
    """
    user_id = state.get("user_id", 0)  # 从状态中获取用户ID，默认为0
    action = state.get("action", "")  # 从状态中获取用户意图动作，默认为空字符串
    content = state.get("content", "")  # 从状态中获取用户提交的内容，默认为空字符串
    target_user_id = state.get("target_user_id", 0)  # 从状态中获取对方用户ID，默认为0
    university = state.get("target_university", "")  # 从状态中获取目标学校，默认为空字符串
    city = state.get("target_city", "")  # 从状态中获取目标城市，默认为空字符串
    province = state.get("target_province", "")  # 从状态中获取目标省份，默认为空字符串

    # 根据用户意图动作，构造不同的请求描述
    action_descriptions = {
        # ── 单步调用 ──
        "pick": (
            f"用户想要拾取纸团。\n"
            f"用户ID：{user_id}\n"
            f"目标学校：{university or '不限'}\n"
            f"目标城市：{city or '不限'}\n"
            f"目标省份：{province or '不限'}\n"
            f"请调用call_match_agent为用户匹配最佳纸团。"
        ),
        "write": (
            f"用户想要写纸团。\n"
            f"用户ID：{user_id}\n"
            f"纸团内容：{content}\n"
            f"内容类型：bottle\n"
            f"请调用call_guard_agent审核此内容是否合规。"
        ),
        "send_message": (
            f"用户想要发送聊天消息。\n"
            f"用户ID：{user_id}\n"
            f"消息内容：{content}\n"
            f"内容类型：message\n"
            f"请调用call_guard_agent审核此消息是否合规。"
        ),
        "ice_break": (
            f"用户需要破冰/回复建议。\n"
            f"用户ID：{user_id}\n"
            f"对方用户ID：{target_user_id}\n"
            f"请调用call_ice_break_agent生成3条建议。"
        ),
        "content_gen": (
            f"用户需要纸团内容建议。\n"
            f"用户ID：{user_id}\n"
            f"请调用call_content_gen_agent生成3条内容建议。"
        ),
        # ── 链式调用 ──
        "pick_full": (
            f"用户想要拾取纸团，并希望获得完整的匹配+分析+破冰建议。\n"
            f"用户ID：{user_id}\n"
            f"目标学校：{university or '不限'}\n"
            f"目标城市：{city or '不限'}\n"
            f"目标省份：{province or '不限'}\n"
            f"请根据当前协作记录推进任务：先调用call_match_agent；有候选后依次调用画像分析、对话引导和破冰Agent；没有候选时立即结束；产生可展示话术后调用call_guard_agent审核。每一步都要依据已有结果，不要重复调用已经完成的Agent。"
        ),
        # ── 条件重试 ──
        "write_smart": (
            f"用户想要写纸团，并希望获得智能审核+替代建议。\n"
            f"用户ID：{user_id}\n"
            f"纸团内容：{content}\n"
            f"内容类型：bottle\n"
            f"请执行：1.调用call_guard_agent审核内容 → 如果审核通过则直接输出 → 如果审核不通过，调用call_content_gen_agent生成替代内容建议"
        ),
        # ── 组合调用 ──
        "chat_rescue": (
            f"用户在聊天中遇到冷场，不知道怎么继续对话，需要救场帮助。\n"
            f"用户ID：{user_id}\n"
            f"对方用户ID：{target_user_id}\n"
            f"请按顺序执行：1.调用call_conversation_guide_agent分析对话状态并生成引导建议 → 2.调用call_ice_break_agent生成破冰/回复建议"
        ),
        "onboarding": (
            f"新用户刚注册，需要引导完善资料并生成第一条纸团内容。\n"
            f"用户ID：{user_id}\n"
            f"请按顺序执行：1.调用call_profile_analysis_agent分析用户画像（target_user_id=0） → 2.调用call_content_gen_agent生成个性化纸团内容 → 3.对生成的内容调用call_guard_agent审核（content_type=\"bottle\"）"
        ),
    }

    # 获取当前动作的描述，如果动作不在映射中则使用通用描述
    human_content = action_descriptions.get(action, (
        f"用户请求：{action}\n"
        f"用户ID：{user_id}\n"
        f"请根据用户意图调用合适的工具。"
    ))

    # 旧版单步pick的消息链只有初始请求，工具结果直接交给下一轮模型；
    # 其他复杂流程仍携带结构化协作记录以支持多Agent编排。
    if action != "pick":
        coordination_state = {
            "completed_steps": state.get("completed_steps", []),
            "artifacts": state.get("artifacts", {}),
            "step_count": state.get("step_count", 0),
            "retry_count": state.get("retry_count", 0),
        }
        human_content += (
            "\n\n当前协作记录（只把它当作任务状态，不要重新猜测用户身份）：\n"
            f"{json.dumps(coordination_state, ensure_ascii=False)}\n"
            "请根据已有结果决定下一步；如果上一步返回no_result或fatal_error，"
            "不要调用依赖该结果的Agent。"
        )

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),  # 系统提示词，定义Supervisor的角色和调度策略
        HumanMessage(content=human_content),  # 用户请求消息，包含动作类型和参数信息
    ]

    # 生成新的trace_id，用于关联本次Supervisor调用的所有日志
    trace_id = new_trace_id()
    log = get_logger("supervisor_agent")  # 获取结构化日志记录器
    log.info("supervisor_start", user_id=user_id, action=action, trace_id=trace_id)  # 记录Supervisor启动日志

    return {
        "messages": messages,
        "completed_steps": list(state.get("completed_steps", [])),
        "artifacts": dict(state.get("artifacts", {})),
        "step_count": int(state.get("step_count", 0)),
        "retry_count": int(state.get("retry_count", 0)),
        "max_steps": int(state.get("max_steps", SUPERVISOR_MAX_STEPS)),
        "status": "running",
        "error": state.get("error"),
        # 每次任务重新计时，避免复用旧状态时预算已被前一次任务耗尽。
        "deadline_at": time.monotonic() + settings.AGENT_TIME_BUDGET_SECONDS,
    }


async def agent_node(state: SupervisorState) -> dict[str, Any]:
    """状态图节点2：Supervisor LLM推理节点，使用Function Calling自主决定调用哪个子Agent

    Supervisor LLM接收当前对话历史，结合可用工具列表，决定：
      - 调用call_match_agent（拾纸团场景）
      - 调用call_guard_agent（内容审核场景）
      - 调用call_ice_break_agent（破冰建议场景）
      - 调用call_content_gen_agent（内容生成场景）
      - 调用call_profile_analysis_agent（画像分析场景）
      - 调用call_conversation_guide_agent（对话引导场景）
      - 调用call_research_agent（需要外部热点或网页资料时）
      - 直接输出最终结果（无需调用工具）

    在复杂编排场景中，Supervisor会通过ReAct循环依次调用多个子Agent：
      - pick_full：match → profile_analysis → conversation_guide → ice_break → guard
      - write_smart：guard → [不通过] content_gen
      - chat_rescue：conversation_guide → ice_break
      - onboarding：profile_analysis → content_gen → guard
    """
    log = get_logger("supervisor_agent")  # 获取结构化日志记录器
    user_id = state.get("user_id", 0)  # 从状态中获取用户ID，默认为0
    start = time.time()  # 记录当前时间，用于计算LLM推理耗时
    retry_count = int(state.get("retry_count", 0))
    for attempt in range(settings.AGENT_MAX_ATTEMPTS):
        # 总时间预算耗尽时直接熔断，不再发起新的模型推理。
        remaining = _remaining_budget(state)
        if remaining <= 0:
            error = _error_payload("supervisor", "任务已超过时间预算", "TIME_BUDGET_EXCEEDED")
            log.error("supervisor_time_budget_exceeded", trace_id=get_trace_id(), user_id=user_id)
            return {"status": "budget_exceeded", "error": error, "retry_count": retry_count}
        if attempt:
            retry_count += 1
            await asyncio.sleep(min(RETRY_DELAY_SECONDS * attempt, remaining))
        try:
            llm_with_tools = _get_llm().bind_tools(supervisor_tools)
            response = await asyncio.wait_for(
                llm_with_tools.ainvoke(state["messages"]),
                timeout=min(settings.AGENT_CALL_TIMEOUT_SECONDS, remaining),
            )
            if not isinstance(response, AIMessage):
                raise ValueError("Supervisor模型返回了无效消息")
            break
        except Exception as exc:
            error = _exception_payload("supervisor", exc)
            log.error("supervisor_llm_error", error_type=type(exc).__name__,
                      attempt=attempt + 1, trace_id=get_trace_id(), user_id=user_id,
                      exc_info=True)
            if not error["retryable"] or attempt + 1 >= settings.AGENT_MAX_ATTEMPTS:
                return {"status": "failed", "error": error, "retry_count": retry_count}
    duration_ms = round((time.time() - start) * 1000, 1)  # 计算LLM推理耗时（毫秒），保留1位小数

    # 记录Supervisor LLM推理日志（含Token消耗和是否发起Tool Calls）
    tokens = getattr(response, "usage_metadata", {}) or {}  # 从LLM响应中获取Token使用量元数据，若不存在则返回空字典
    total_tokens = tokens.get("total_tokens", 0)  # 获取本次LLM调用消耗的总Token数
    global_metrics.record_llm_call(tokens=total_tokens, duration_ms=duration_ms)  # 记录全局LLM调用指标（Token数+耗时）
    has_tool_calls = bool(response.tool_calls)  # 判断LLM是否发起了Tool Calls（即是否需要调用子Agent）
    log.info("supervisor_llm_step", user_id=user_id, duration_ms=duration_ms,
             total_tokens=total_tokens, has_tool_calls=has_tool_calls, trace_id=get_trace_id())  # 记录LLM推理步骤日志

    return {"messages": [response], "retry_count": retry_count, "error": None}


def should_use_tools(state: SupervisorState) -> str:
    """条件边判断函数：检查Supervisor LLM是否需要调用工具（子Agent）
 
    返回值：
      - "tools"：Supervisor LLM发出了Tool Calls，需要执行子Agent
      - "extract_result"：Supervisor LLM给出了最终结果，提取结果
    """
    if state.get("step_count", 0) >= state.get("max_steps", SUPERVISOR_MAX_STEPS):
        return "extract_result"
    if state.get("status") in {"failed", "budget_exceeded", "no_result"}:
        return "extract_result"
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "extract_result"  # 否则返回"extract_result"，表示LLM已给出最终结果


def _decode_last_llm_result(state: SupervisorState) -> dict[str, Any] | None:
    """只在没有可用artifacts时读取Supervisor最终文本。"""
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, AIMessage) or not isinstance(message.content, str):
            continue
        text = message.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and value:
            return value
    return None


def _aggregate_artifacts(state: SupervisorState) -> dict[str, Any] | None:
    """从父状态中的完整结果索引生成最终业务结果。"""
    artifacts = state.get("artifacts") or {}
    if not artifacts:
        return None
    action = state.get("action", "")
    if action == "pick":
        result = artifacts.get("match") or {}
    elif action == "pick_full":
        result = {
            key: artifacts[key]
            for key in ("match", "profile_analysis", "conversation_guide", "ice_break", "guard")
            if key in artifacts
        }
    elif action in {"ice_break", "content_gen", "write", "send_message"}:
        preferred = {
            "ice_break": "ice_break",
            "content_gen": "content_gen",
            "write": "guard",
            "send_message": "guard",
        }[action]
        result = artifacts.get(preferred, {})
    elif action == "write_smart":
        result = {
            key: artifacts[key]
            for key in ("guard", "research", "content_gen")
            if key in artifacts
        }
    elif action == "chat_rescue":
        result = {
            key: artifacts[key]
            for key in ("conversation_guide", "research", "ice_break")
            if key in artifacts
        }
    elif action == "onboarding":
        result = {
            key: artifacts[key]
            for key in ("profile_analysis", "research", "content_gen", "guard")
            if key in artifacts
        }
    else:
        result = dict(artifacts)

    if not result:
        return None
    message = "子Agent结果已由Supervisor汇总"
    if action == "pick":
        message = "已找到匹配纸团" if result.get("best_match") else "暂时没有纸团"
    return {
        "action": action,
        "result": result,
        "message": message,
    }


def extract_result(state: SupervisorState) -> dict[str, Any]:
    """优先从结构化协作结果汇总，最后才读取Supervisor文本。"""
    user_id = state.get("user_id", 0)
    action = state.get("action", "")
    aggregated = _aggregate_artifacts(state)
    # 已知业务必须依靠真正执行的工具结果，不能使用模型自行编造的最终 JSON。
    result = aggregated if action in REQUIRED_ARTIFACTS else aggregated or _decode_last_llm_result(state)
    error = state.get("error")
    status = state.get("status", "success")
    artifacts = state.get("artifacts") or {}
    error = error or next((
        payload for payload in artifacts.values() if payload.get("status") in FAILURE_STATUSES
    ), None)
    if error and status != "budget_exceeded":
        status = "failed"
    if status not in {"failed", "budget_exceeded"} and action in REQUIRED_ARTIFACTS:
        match = artifacts.get("match") or {}
        if action in {"pick", "pick_full"} and match.get("status") == "no_result":
            status = "no_result"
        else:
            required = set(REQUIRED_ARTIFACTS[action])
            if action == "write_smart" and (artifacts.get("guard") or {}).get("passed") is False:
                required.add("content_gen")
            missing = [key for key in sorted(required) if (artifacts.get(key) or {}).get("status") != "success"]
            if missing:
                status = "failed"
                error = _error_payload("supervisor", "必要的Agent步骤未完成，请重新尝试", "MISSING_AGENT_RESULT")

    if result is None:
        result = {
            "action": action,
            "result": None,
            "message": "没有得到有效的子Agent结果",
        }
        if status != "budget_exceeded":
            status = "failed"
        error = error or _error_payload("supervisor", "没有得到有效的子Agent结果", "INVALID_AGENT_RESULT")

    guard_payload = artifacts.get("guard") or {}
    guard_required = action in {
        "write",
        "send_message",
        "write_smart",
        "onboarding",
    } or (action == "pick_full" and "ice_break" in artifacts)
    if guard_required and not guard_payload and status not in {"failed", "budget_exceeded"}:
        status = "failed"
        error = _error_payload("guard", "审核结果缺失", "MISSING_AGENT_RESULT")
    if status == "running":
        status = "success"
    if guard_required:
        guard_passed = guard_payload.get("passed") is True and guard_payload.get("status") == "success"
        guard_reason = "" if guard_passed else str(guard_payload.get("reason", "审核结果缺失"))
    else:
        guard_passed = status not in {"failed", "budget_exceeded"}
        guard_reason = ""

    if status in {"failed", "budget_exceeded"}:
        guard_passed = False
        error = error or _error_payload("supervisor", "流程执行失败，请稍后重试", "WORKFLOW_FAILED")
        guard_reason = str(error.get("error", "子Agent执行失败"))
        result = {**result, "status": status, "error": error, "message": guard_reason}

    log = get_logger("supervisor_agent")
    log.info(
        "supervisor_complete",
        user_id=user_id,
        action=action,
        status=status,
        completed_steps=state.get("completed_steps", []),
        trace_id=get_trace_id(),
    )
    return {
        "final_result": json.dumps(result, ensure_ascii=False),
        "guard_passed": guard_passed,
        "guard_reason": guard_reason,
        "status": status,
        "error": error,
        "artifacts": artifacts,
        "completed_steps": list(state.get("completed_steps", [])),
        "step_count": int(state.get("step_count", 0)),
        "retry_count": int(state.get("retry_count", 0)),
    }


def should_continue_after_tools(state: SupervisorState) -> str:
    """工具节点完成后，终止状态不再额外调用一次Supervisor模型。"""
    if state.get("status") in {"failed", "budget_exceeded", "no_result"}:
        return "extract_result"
    if state.get("step_count", 0) >= state.get("max_steps", SUPERVISOR_MAX_STEPS):
        return "extract_result"
    return "agent"


def build_supervisor_graph() -> StateGraph:
    """构建SupervisorAgent的LangGraph状态图（Supervisor模式 + Tool/Function Calling）

    流程说明：
      init_conversation → agent_node → (条件边)
        ├── 有Tool Calls → execute_tools → agent_node（ReAct循环，直到Supervisor给出最终结果）
        └── 无Tool Calls → extract_result → END

    复杂编排通过ReAct循环实现：Supervisor LLM在每一轮推理中决定调用哪个子Agent，
    子Agent结果返回后，Supervisor LLM根据结果决定下一步操作（继续调用或输出最终结果）。
    """
    graph = StateGraph(SupervisorState)  # 创建状态图实例，指定状态类型为SupervisorState

    # 添加四个节点到状态图中
    graph.add_node("init_conversation", init_conversation)  # 节点1：初始化对话，构建SystemMessage+HumanMessage
    graph.add_node("agent", agent_node)  # 节点2：Supervisor LLM推理节点，决定调用哪个子Agent
    graph.add_node("tools", execute_tools)  # 节点3：工具执行节点，执行子Agent并返回ToolMessage
    graph.add_node("extract_result", extract_result)  # 节点4：结果提取节点，从LLM回复或ToolMessage中提取最终结果

    # 设置入口节点
    graph.set_entry_point("init_conversation")  # 设置状态图的入口节点为init_conversation

    # 添加边和条件边
    graph.add_edge("init_conversation", "agent")  # 固定边：初始化对话后进入LLM推理节点
    graph.add_conditional_edges("agent", should_use_tools, {  # 条件边：根据LLM是否发起Tool Calls决定下一步
        "tools": "tools",  # 如果需要调用工具，进入工具执行节点
        "extract_result": "extract_result",  # 如果LLM已给出最终结果，进入结果提取节点
    })
    graph.add_conditional_edges("tools", should_continue_after_tools, {
        "agent": "agent",
        "extract_result": "extract_result",
    })
    graph.add_edge("extract_result", END)  # 固定边：结果提取后进入终止节点

    return graph  # 返回构建好的状态图


# 构建并编译SupervisorAgent的状态图
supervisor_graph = build_supervisor_graph()  # 调用构建函数，创建Supervisor状态图
supervisor_app = supervisor_graph.compile()  # 编译状态图为可执行的应用，支持ainvoke调用
