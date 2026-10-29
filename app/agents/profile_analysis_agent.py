from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用尚未定义的类型

import json  # JSON序列化/反序列化库，用于LLM响应解析
import random  # 随机数库，用于生成随机种子确保每次LLM生成不同内容
import time  # 时间库，用于生成时间戳打破DeepSeek服务端缓存
from typing import Annotated, Any, TypedDict  # 类型注解工具：Annotated添加元数据、Any任意类型、TypedDict定义带类型的字典

from langchain_core.messages import HumanMessage, SystemMessage  # LangChain消息类型
from langchain_openai import ChatOpenAI  # LangChain的OpenAI兼容LLM客户端，用于调用DeepSeek模型
from langgraph.graph import END, StateGraph  # LangGraph核心组件：END表示状态图终止节点、StateGraph用于构建有状态图
from langgraph.graph.message import add_messages  # LangGraph消息追加reducer

from app.agent_logger import get_logger, global_metrics  # Agent结构化日志模块
from app.agents.result_parsing import parse_json_response, string_list


# ============================================================
# ProfileAnalysisAgent 状态图流程图（Pipeline模式）（用户画像分析）
# ============================================================
#
#   ┌────────────────────┐
#   │  fetch_profiles    │  查询数据库：获取用户画像（+对方画像，如有target_user_id）
#   └────────┬───────────┘
#            │
#            ▼
#   ┌────────────────────┐
#   │  analyze_profile   │  LLM分析：生成个性标签、完整度评分、完善建议、契合度分析
#   └────────┬───────────┘
#            │
#            ▼
#   ┌────────────────────┐
#   │  extract_result    │  解析LLM返回的JSON，提取分析结果
#   └────────┬───────────┘
#            │
#            ▼
#          [END]
#
# 设计思路：
#   1. 先从数据库查询用户画像（可选查询对方画像用于契合度分析）
#   2. 将画像数据嵌入SystemMessage，让LLM一次调用生成分析结果
#   3. 分析维度：个性标签、完整度评分、完善建议、契合度（双人）
#   4. 供Supervisor在拾纸团全流程、新用户引导等场景中调用
#
# ============================================================


class ProfileAnalysisState(TypedDict):
    """ProfileAnalysisAgent的状态定义，贯穿整个画像分析流程图"""
    user_id: int  # 当前用户ID
    target_user_id: int  # 对方用户ID（0表示不分析契合度）
    messages: Annotated[list, add_messages]  # LLM对话消息列表（使用add_messages reducer自动追加）
    user_profile: str  # 当前用户画像JSON字符串（由fetch_profiles节点填充）
    target_profile: str  # 对方用户画像JSON字符串（由fetch_profiles节点填充，无对方时为空）
    tags: list[str]  # LLM生成的个性标签列表
    completeness_score: int  # 画像完整度评分（0-100）
    suggestions: list[str]  # 完善建议列表
    compatibility: str  # 契合度分析描述（仅当有target_user_id时生成）


# ============================================================
# 数据库查询函数
# ============================================================

async def _query_user_profile(user_id: int) -> str:
    """查询用户画像信息，返回JSON字符串"""
    from app.database import pool  # 延迟导入数据库连接池，避免循环导入

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT nickname, gender, age, hobbies, province, city, university, "
                "(contact_value IS NOT NULL AND contact_value <> '') AS has_contact "
                "FROM users WHERE id = %s",
                [user_id],
            )
            row = await cur.fetchone()
            if row:
                profile = {
                    "昵称": row[0] or "",
                    "性别": row[1] or "",
                    "年龄": row[2] or 0,
                    "爱好": row[3] or "",
                    "省份": row[4] or "",
                    "城市": row[5] or "",
                    "学校": row[6] or "",
                    # 只告诉模型是否填写，不把真实联系方式发送到第三方模型。
                    "已填写联系方式": bool(row[7]),
                }
                return json.dumps(profile, ensure_ascii=False)
            return f"未找到用户ID为{user_id}的用户"


# ============================================================
# 状态图节点函数
# ============================================================

async def fetch_profiles(state: ProfileAnalysisState) -> dict[str, Any]:
    """状态图节点1：从数据库查询用户画像（和对方画像）

    根据是否传入target_user_id决定是否查询对方画像。
    有对方画像时，后续LLM会额外生成契合度分析。
    """
    log = get_logger("profile_analysis_agent")
    user_id = state.get("user_id", 0)
    target_user_id = state.get("target_user_id", 0)
    start = time.time()

    user_profile = await _query_user_profile(user_id)

    target_profile = ""
    if target_user_id > 0:
        target_profile = await _query_user_profile(target_user_id)

    duration_ms = round((time.time() - start) * 1000, 1)
    log.info("fetch_profiles", user_id=user_id, target_user_id=target_user_id, duration_ms=duration_ms)

    return {
        "user_profile": user_profile,
        "target_profile": target_profile,
    }


# LLM系统提示词模板
PROFILE_ANALYSIS_SYSTEM_TEMPLATE: str = """你是一个社交平台用户画像分析专家，负责分析用户信息并给出专业建议。

以下是用户的个人信息：
{user_profile}
{target_section}
分析任务：
1. 个性标签：根据用户信息生成3-5个简短的个性标签（如"游戏达人""文艺青年""运动健将"等）
2. 画像完整度：评估用户信息的完整程度，给出0-100的评分
   - 有昵称+10分，有性别+10分，有年龄+10分，有爱好+20分，有省份+10分，有城市+10分，有学校+20分，有联系方式+10分
3. 完善建议：如果完整度低于80分，给出1-3条具体的完善建议
{compatibility_task}
请严格按照以下JSON格式返回（不要包含在代码块中）：
{{{dual_output}}}"""


def _get_llm() -> ChatOpenAI:
    """获取DeepSeek LLM实例，配置模型名称、API密钥、基础URL和生成参数"""
    from app.config import settings  # 延迟导入配置模块，避免循环导入

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        max_retries=0,
        temperature=0.3,
        max_tokens=512,
    )


async def analyze_profile(state: ProfileAnalysisState) -> dict[str, Any]:
    """状态图节点2：调用LLM分析用户画像

    根据是否有对方画像，动态构建不同的提示词：
      - 无对方画像：只分析个性标签、完整度、完善建议
      - 有对方画像：额外生成契合度分析
    每次调用注入随机种子+时间戳，确保LLM生成不同内容。
    """
    seed = random.randint(1, 999999)
    ts = time.time()

    user_profile = state.get("user_profile", "未知")
    target_profile = state.get("target_profile", "")

    has_target = target_profile and "未找到" not in target_profile

    if has_target:
        target_section = f"\n以下是对方的个人信息：\n{target_profile}"
        compatibility_task = "4. 契合度分析：分析两个用户的契合度，包括共同点、互补点、推荐互动方式（50字以内）"
        dual_output = (
            '"tags": ["标签1", "标签2", "标签3"], '
            '"completeness_score": 85, '
            '"suggestions": ["建议1", "建议2"], '
            '"compatibility": "契合度分析描述"'
        )
    else:
        target_section = ""
        compatibility_task = ""
        dual_output = (
            '"tags": ["标签1", "标签2", "标签3"], '
            '"completeness_score": 85, '
            '"suggestions": ["建议1", "建议2"]'
        )

    sys_content = PROFILE_ANALYSIS_SYSTEM_TEMPLATE.format(
        user_profile=user_profile,
        target_section=target_section,
        compatibility_task=compatibility_task,
        dual_output=dual_output,
    )
    dynamic_sys = f"[调用指纹:{seed}_{ts}] " + sys_content

    messages = [
        SystemMessage(content=dynamic_sys),
        HumanMessage(content="请分析这个用户的画像信息。"),
    ]

    start = time.time()
    llm = _get_llm()
    response = await llm.ainvoke(messages)
    duration_ms = round((time.time() - start) * 1000, 1)
    tokens = getattr(response, "usage_metadata", {}) or {}
    total_tokens = tokens.get("total_tokens", 0)
    global_metrics.record_llm_call(tokens=total_tokens, duration_ms=duration_ms)
    log = get_logger("profile_analysis_agent")
    log.info("analyze_profile", user_id=state.get("user_id", 0), duration_ms=duration_ms, total_tokens=total_tokens, has_target=has_target)

    return {"messages": [response]}


def extract_result(state: ProfileAnalysisState) -> dict[str, Any]:
    """严格校验画像分析结果，失败时不返回固定默认值。"""
    result = parse_json_response(state.get("messages", []), {"tags", "completeness_score", "suggestions"})
    tags = string_list(result, "tags", min_items=1)
    suggestions = string_list(result, "suggestions")
    score = result["completeness_score"]
    compatibility = result.get("compatibility", "")
    if type(score) is not int or not 0 <= score <= 100 or not isinstance(compatibility, str):
        raise ValueError("画像分析字段格式无效")
    get_logger("profile_analysis_agent").info("extract_result", user_id=state.get("user_id", 0), completeness_score=score, tags_count=len(tags[:5]))
    return {"tags": tags[:5], "completeness_score": score, "suggestions": suggestions[:3], "compatibility": compatibility}


def build_profile_analysis_graph() -> StateGraph:
    """构建ProfileAnalysisAgent的LangGraph状态图（Pipeline模式）

    流程说明：
      fetch_profiles → analyze_profile → extract_result → END
    """
    graph = StateGraph(ProfileAnalysisState)

    graph.add_node("fetch_profiles", fetch_profiles)
    graph.add_node("analyze_profile", analyze_profile)
    graph.add_node("extract_result", extract_result)

    graph.set_entry_point("fetch_profiles")

    graph.add_edge("fetch_profiles", "analyze_profile")
    graph.add_edge("analyze_profile", "extract_result")
    graph.add_edge("extract_result", END)

    return graph


profile_analysis_graph = build_profile_analysis_graph()
profile_analysis_app = profile_analysis_graph.compile()
