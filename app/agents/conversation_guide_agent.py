from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用尚未定义的类型

import asyncio  # 异步IO库，用于并行查询数据库
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
# ConversationGuideAgent 状态图流程图（Pipeline模式）（对话引导）
# ============================================================
#
#   ┌────────────────────────┐
#   │  fetch_conversation    │  查询数据库：获取双方画像+聊天记录+对话状态判断
#   └────────┬───────────────┘
#            │
#            ▼
#   ┌────────────────────────┐
#   │  analyze_conversation  │  LLM分析：判断对话状态、生成话题转换建议和回复策略
#   └────────┬───────────────┘
#            │
#            ▼
#   ┌────────────────────────┐
#   │  extract_result        │  解析LLM返回的JSON，提取引导结果
#   └────────┬───────────────┘
#            │
#            ▼
#          [END]
#
# 设计思路：
#   1. 先从数据库查询双方画像和聊天记录（不依赖LLM的function calling）
#   2. 将上下文嵌入SystemMessage，让LLM一次调用生成对话引导
#   3. 分析维度：对话状态判断、冷场检测、话题转换建议、回复策略
#   4. 供Supervisor在对话救场场景中调用
#
# ============================================================


class ConversationGuideState(TypedDict):
    """ConversationGuideAgent的状态定义，贯穿整个对话引导流程图"""
    user_id: int  # 当前用户ID
    target_user_id: int  # 对方用户ID
    messages: Annotated[list, add_messages]  # LLM对话消息列表（使用add_messages reducer自动追加）
    user_profile: str  # 当前用户画像JSON字符串（由fetch_conversation节点填充）
    target_profile: str  # 对方用户画像JSON字符串（由fetch_conversation节点填充）
    chat_history: str  # 聊天记录文本（由fetch_conversation节点填充）
    conversation_status: str  # 对话状态："active"活跃 / "stalled"冷场 / "new"新对话
    topic_suggestions: list[str]  # 话题转换建议列表
    reply_strategies: list[str]  # 回复策略列表
    ice_breaker: str  # 推荐的破冰话术


# ============================================================
# 数据库查询函数
# ============================================================

async def _query_user_profile(user_id: int) -> str:
    """查询用户画像信息，返回JSON字符串"""
    from app.database import pool  # 延迟导入数据库连接池，避免循环导入

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT nickname, gender, age, hobbies, province, city, university FROM users WHERE id = %s",
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
                }
                return json.dumps(profile, ensure_ascii=False)
            return f"未找到用户ID为{user_id}的用户"


async def _query_chat_history(user_id: int, target_user_id: int) -> str:
    """查询两个用户之间的聊天记录，返回格式化文本"""
    from app.database import pool

    u1, u2 = min(user_id, target_user_id), max(user_id, target_user_id)

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM conversations WHERE user1_id = %s AND user2_id = %s",
                [u1, u2],
            )
            conv_row = await cur.fetchone()

            if not conv_row:
                return "两人还没有对话记录"

            conv_id = conv_row[0]
            await cur.execute(
                "SELECT sender_id, content, created_at FROM messages "
                "WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 10",
                [conv_id],
            )
            rows = await cur.fetchall()

            if not rows:
                return "对话存在但没有消息"

            messages_list = []
            for row in reversed(rows):
                sender = "我" if row[0] == user_id else "对方"
                messages_list.append(f"{sender}：{row[1]}")

            return "\n".join(messages_list)


# ============================================================
# 状态图节点函数
# ============================================================

async def fetch_conversation(state: ConversationGuideState) -> dict[str, Any]:
    """状态图节点1：从数据库查询双方画像和聊天记录

    并行查询三个数据源：当前用户画像、对方画像、聊天记录。
    将结果存入状态中，供后续LLM分析节点使用。
    """
    log = get_logger("conversation_guide_agent")
    user_id = state.get("user_id", 0)
    target_user_id = state.get("target_user_id", 0)
    start = time.time()

    user_profile, target_profile, chat_history = await asyncio.gather(
        _query_user_profile(user_id),
        _query_user_profile(target_user_id),
        _query_chat_history(user_id, target_user_id),
    )

    duration_ms = round((time.time() - start) * 1000, 1)
    log.info("fetch_conversation", user_id=user_id, target_user_id=target_user_id, duration_ms=duration_ms)

    return {
        "user_profile": user_profile,
        "target_profile": target_profile,
        "chat_history": chat_history,
    }


# LLM系统提示词模板
CONVERSATION_GUIDE_SYSTEM_TEMPLATE: str = """你是一个社交对话引导专家，帮助用户在社交平台上维持和推进对话。

以下是我的个人信息：
{user_profile}

以下是对方的个人信息：
{target_profile}

以下是我们的聊天记录：
{chat_history}

分析任务：
1. 对话状态判断：
   - "active"：对话活跃，双方互动频繁
   - "stalled"：对话冷场，需要话题转换（如连续几条都是单字回复、长时间无互动、话题已聊完）
   - "new"：还没有开始对话

2. 话题转换建议：生成3个新话题建议，要结合双方的共同爱好和背景
   - 每个建议15字以内，要具体不要泛泛而谈
   - 不要用"我们可以聊聊..."这种正式句式
   - 要像朋友之间随口提起话题的感觉

3. 回复策略：给出3条回复策略建议
   - 每条策略20字以内，简洁实用
   - 策略要针对当前对话状态（如冷场时建议主动提问，活跃时建议深入话题）

4. 破冰话术：推荐一句最自然的破冰/救场话术（30字以内）

请严格按照以下JSON格式返回（不要包含在代码块中）：
{{"conversation_status": "active/stalled/new", "topic_suggestions": ["话题1", "话题2", "话题3"], "reply_strategies": ["策略1", "策略2", "策略3"], "ice_breaker": "破冰话术"}}"""


def _get_llm() -> ChatOpenAI:
    """获取DeepSeek LLM实例，配置模型名称、API密钥、基础URL和生成参数"""
    from app.config import settings  # 延迟导入配置模块，避免循环导入

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        max_retries=0,
        temperature=0.7,
        max_tokens=512,
    )


async def analyze_conversation(state: ConversationGuideState) -> dict[str, Any]:
    """状态图节点2：调用LLM分析对话状态并生成引导建议

    将双方画像和聊天记录嵌入SystemMessage，一次LLM调用生成分析结果。
    每次调用注入随机种子+时间戳，确保LLM生成不同内容。
    """
    seed = random.randint(1, 999999)
    ts = time.time()

    user_profile = state.get("user_profile", "未知")
    target_profile = state.get("target_profile", "未知")
    chat_history = state.get("chat_history", "暂无聊天记录")

    sys_content = CONVERSATION_GUIDE_SYSTEM_TEMPLATE.format(
        user_profile=user_profile,
        target_profile=target_profile,
        chat_history=chat_history,
    )
    dynamic_sys = f"[调用指纹:{seed}_{ts}] " + sys_content

    messages = [
        SystemMessage(content=dynamic_sys),
        HumanMessage(content="请分析我们的对话状态，给我一些话题和回复建议。"),
    ]

    start = time.time()
    llm = _get_llm()
    response = await llm.ainvoke(messages)
    duration_ms = round((time.time() - start) * 1000, 1)
    tokens = getattr(response, "usage_metadata", {}) or {}
    total_tokens = tokens.get("total_tokens", 0)
    global_metrics.record_llm_call(tokens=total_tokens, duration_ms=duration_ms)
    log = get_logger("conversation_guide_agent")
    log.info("analyze_conversation", user_id=state.get("user_id", 0), duration_ms=duration_ms, total_tokens=total_tokens)

    return {"messages": [response]}


def extract_result(state: ConversationGuideState) -> dict[str, Any]:
    """严格校验对话引导结果，失败时不生成替代话术。"""
    result = parse_json_response(state.get("messages", []), {"conversation_status", "topic_suggestions", "reply_strategies", "ice_breaker"})
    topics = string_list(result, "topic_suggestions", min_items=1)
    strategies = string_list(result, "reply_strategies", min_items=1)
    status, opening = result["conversation_status"], result["ice_breaker"]
    if status not in ("active", "stalled", "new") or not isinstance(opening, str) or not opening.strip():
        raise ValueError("对话引导字段格式无效")
    get_logger("conversation_guide_agent").info("extract_result", user_id=state.get("user_id", 0), conversation_status=status)
    return {"conversation_status": status, "topic_suggestions": topics[:3], "reply_strategies": strategies[:3], "ice_breaker": opening.strip()}


def build_conversation_guide_graph() -> StateGraph:
    """构建ConversationGuideAgent的LangGraph状态图（Pipeline模式）

    流程说明：
      fetch_conversation → analyze_conversation → extract_result → END
    """
    graph = StateGraph(ConversationGuideState)

    graph.add_node("fetch_conversation", fetch_conversation)
    graph.add_node("analyze_conversation", analyze_conversation)
    graph.add_node("extract_result", extract_result)

    graph.set_entry_point("fetch_conversation")

    graph.add_edge("fetch_conversation", "analyze_conversation")
    graph.add_edge("analyze_conversation", "extract_result")
    graph.add_edge("extract_result", END)

    return graph


conversation_guide_graph = build_conversation_guide_graph()
conversation_guide_app = conversation_guide_graph.compile()
