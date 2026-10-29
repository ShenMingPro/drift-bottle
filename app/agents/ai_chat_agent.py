"""AI聊天Agent：集成MCP工具调用的对话式AI助手

本Agent实现了一个支持MCP工具调用的AI聊天助手，用户可以通过自然语言与AI对话，
AI会根据用户意图自动调用MCP Server提供的工具（如热搜趋势查询），并将结果整合到回复中。

Agent流程图（ReAct模式 + MCP工具调用）：
                        ┌────────────────────┐
                        │  init_conversation  │  初始化：构建SystemMessage + 历史消息
                        └────────┬───────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                   ┌──→ │     agent_node      │  LLM推理，决定是否调用MCP工具
                   │    │  (LLM+Tool Calling) │
                   │    └────────┬───────────┘
                   │             │
                   │             ├── Tool Calls ──→ execute_mcp_tools ──┐
                   │             │                                       │
                   │             │                          ┌────────────┘
                   │             │                          │
                   │             │                          ▼
                   │             │              ┌──────────────────────────┐
                   │             │              │  execute_mcp_tools       │
                   │             │              │  调用MCP Server工具：     │
                   │             │              │  ├─ Trends Hub (22+工具) │
                   │             │              │  │  ├─ get-weibo-trending│
                   │             │              │  │  ├─ get-zhihu-trending│
                   │             │              │  │  └─ ...更多趋势工具    │
                   │             │              │  ├─ Search1API (6工具)   │
                   │             │              │  │  ├─ search (搜索)     │
                   │             │              │  │  ├─ news (新闻)       │
                   │             │              │  │  ├─ crawl (爬取)      │
                   │             │              │  │  ├─ sitemap (站点地图) │
                   │             │              │  │  ├─ reasoning (推理)  │
                   │             │              │  │  └─ trending (趋势)   │
                   │             │              └────────────┬─────────────┘
                   │             │                           │
                   │             └───────────────────────────┘
                   │                    tools → agent（ReAct循环）
                   │
                   │             ┌── 无Tool Calls ──→ format_response → [END]
                   │             │
                   └─────────────┘
"""

from __future__ import annotations  # 启用延迟类型注解评估

import asyncio
import json  # JSON序列化/反序列化库
import re  # 正则表达式库，用于清理LLM返回的Markdown标记
import time  # 时间库，用于计算节点执行耗时
from typing import Annotated, Any, NotRequired, TypedDict  # 类型注解工具

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage  # LangChain消息类型
from langchain_openai import ChatOpenAI  # LangChain的OpenAI兼容LLM客户端
from langgraph.graph import END, StateGraph  # LangGraph核心组件
from langgraph.graph.message import add_messages  # LangGraph消息追加reducer

from app.agent_logger import get_logger, get_trace_id, global_metrics, new_trace_id  # 结构化日志工具


# AI聊天系统提示词：定义AI助手的角色、能力和交互规范
AI_CHAT_SYSTEM_PROMPT: str = """你是The Player社交平台的AI助手，你可以帮助用户查询全网热点趋势资讯，以及进行网页搜索和内容爬取。

你可以使用以下MCP工具来获取实时热点数据：
- get-weibo-trending：获取微博热搜榜
- get-zhihu-trending：获取知乎热榜
- get-bilibili-rank：获取哔哩哔哩视频排行榜
- get-douyin-trending：获取抖音热搜榜单
- get-toutiao-trending：获取今日头条热榜
- get-36kr-trending：获取36氪热榜
- get-douban-rank：获取豆瓣实时热门榜单
- get-netease-news-trending：获取网易新闻热点榜
- get-tencent-news-trending：获取腾讯新闻热点榜
- get-thepaper-trending：获取澎湃新闻热榜
- get-nytimes-news：获取纽约时报新闻
- get-bbc-news：获取BBC新闻
- get-theverge-news：获取The Verge新闻
- get-ifanr-news：获取爱范儿科技快讯
- get-infoq-news：获取InfoQ技术资讯
- get-juejin-article-rank：获取掘金文章榜
- get-sspai-rank：获取少数派热榜
- get-smzdm-rank：获取什么值得买热门
- get-weread-rank：获取微信读书排行榜
- get-9to5mac-news：获取9to5Mac苹果相关新闻
- get-gcores-new：获取机核网游戏资讯
- custom-rss：获取自定义RSS订阅源内容

你还可以使用以下Search1API工具进行网页搜索和内容获取：
- search：网页搜索，支持Google/Bing/DuckDuckGo等13个搜索引擎，可指定搜索范围和时间
- news：新闻搜索，支持Google/Bing/HackerNews等新闻源
- crawl：爬取指定URL的网页内容，获取完整文章或页面信息
- sitemap：获取指定网站的所有相关链接
- reasoning：深度推理，使用DeepSeek R1进行复杂问题分析和思考
- trending：获取GitHub和HackerNews的热门趋势话题

交互规范：
1. 当用户询问热点、趋势、新闻等话题时，优先使用热点趋势工具获取数据
2. 当用户需要搜索特定信息、查找资料时，使用search工具进行网页搜索
3. 当用户想了解某个网页的具体内容时，使用crawl工具爬取页面
4. 当用户提出复杂的分析、推理类问题时，可以使用reasoning工具进行深度思考
5. 如果用户没有指定具体平台，可以根据话题类型推荐合适的平台查询
6. 获取数据后，用简洁友好的语言为用户总结关键信息，不要直接返回原始数据
7. 每次最多展示5-10条热点，避免信息过载
8. 如果工具调用失败，友好地告知用户并建议稍后重试
9. 可以根据用户兴趣推荐相关的热点内容
10. 回复时使用中文，保持轻松活泼的语气"""


class AIChatState(TypedDict):
    """AI聊天Agent的状态定义，贯穿整个对话流程"""
    user_id: int  # 当前用户ID
    messages: Annotated[list, add_messages]  # LLM对话消息列表（使用add_messages reducer自动追加）
    final_response: str  # 最终返回给用户的回复文本
    mcp_tool_calls: list[dict[str, Any]]  # 本轮需要执行的MCP工具调用列表
    mcp_results: list[dict[str, Any]]  # MCP工具调用结果列表
    status: NotRequired[str]
    error: NotRequired[dict[str, Any] | None]
    tool_call_count: NotRequired[int]


AI_CHAT_MAX_TOOL_CALLS = 8


def _failure(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "agent": "research", "status": "retryable_error" if retryable else "fatal_error",
        "error_code": code, "error": message, "retryable": retryable,
    }


def _text_content(content: Any) -> str | None:
    """聊天接口只接受文本；兼容 LangChain 的文本内容块。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                return None
        return "".join(parts)
    return None


def _get_llm() -> ChatOpenAI:
    """获取DeepSeek LLM实例，配置模型名称、API密钥和生成参数"""
    from app.config import settings  # 延迟导入配置模块

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.7,  # 聊天场景温度较高，增加回复的多样性和趣味性
        max_tokens=2048,  # 聊天场景需要较长的回复，设置较大的token上限
        timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        max_retries=0,
    )


def init_conversation(state: AIChatState) -> dict[str, Any]:
    """状态图节点1：初始化AI聊天对话，构建SystemMessage和消息列表

    如果messages中已有历史消息（多轮对话），则保留历史；
    否则创建新的SystemMessage作为对话起始。
    """
    log = get_logger("ai_chat_agent")  # 获取结构化日志记录器
    user_id = state.get("user_id", 0)  # 获取用户ID
    messages = state.get("messages", [])  # 获取已有消息列表

    # 如果消息列表为空（新对话），添加SystemMessage
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AI_CHAT_SYSTEM_PROMPT)] + messages

    # 生成新的trace_id
    trace_id = new_trace_id()
    log.info("ai_chat_start", user_id=user_id, trace_id=trace_id)

    return {"messages": messages, "status": "running", "error": None,
            "tool_call_count": 0, "final_response": ""}


async def agent_node(state: AIChatState) -> dict[str, Any]:
    """状态图节点2：LLM推理节点，决定是否调用MCP工具

    LLM接收当前对话历史，结合MCP工具列表，决定：
      - 调用MCP工具（如get-weibo-trending获取微博热搜）
      - 直接回复用户（无需调用工具）
    """
    log = get_logger("ai_chat_agent")  # 获取结构化日志记录器
    user_id = state.get("user_id", 0)  # 获取用户ID
    start = time.time()  # 记录开始时间

    # 从MCP管理器获取工具列表并绑定到LLM
    from app.mcp_client import mcp_manager  # 延迟导入MCP管理器
    from app.config import settings

    try:
        llm = _get_llm()
        langchain_tools = mcp_manager.get_all_langchain_tools()
        if langchain_tools:
            llm = llm.bind_tools(langchain_tools)
        else:
            log.warning("ai_chat_no_mcp_tools", user_id=user_id, trace_id=get_trace_id())
        response = await asyncio.wait_for(
            llm.ainvoke(state["messages"]), timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        log.error("ai_chat_model_error", error_type=type(exc).__name__, trace_id=get_trace_id())
        timed_out = isinstance(exc, TimeoutError)
        return {"status": "failed", "error": _failure(
            "MODEL_TIMEOUT" if timed_out else "MODEL_FAILED",
            "AI生成超时，请稍后重试" if timed_out else "AI服务暂时不可用，请稍后重试",
            # 整个研究流程可能已调用有副作用的工具，不能由父工作流自动重放。
            retryable=timed_out and not state.get("tool_call_count", 0),
        )}

    if (
        not isinstance(response, AIMessage) or response.invalid_tool_calls
        or _text_content(response.content) is None
    ):
        return {"status": "failed", "error": _failure(
            "INVALID_MODEL_RESPONSE", "AI返回了无效结果，请稍后重试",
        )}

    duration_ms = round((time.time() - start) * 1000, 1)  # 计算耗时

    # 记录LLM推理日志
    tokens = getattr(response, "usage_metadata", {}) or {}
    total_tokens = tokens.get("total_tokens", 0)
    global_metrics.record_llm_call(tokens=total_tokens, duration_ms=duration_ms)
    has_tool_calls = bool(response.tool_calls)
    log.info("ai_chat_llm_step", user_id=user_id, duration_ms=duration_ms,
             total_tokens=total_tokens, has_tool_calls=has_tool_calls, trace_id=get_trace_id())

    return {"messages": [response], "status": "running", "error": None}


async def execute_mcp_tools(state: AIChatState) -> dict[str, Any]:
    """执行工具；任何失败都保留为错误消息并终止本次研究，避免伪造证据。"""
    from app.config import settings
    from app.mcp_client import MCPToolError, mcp_manager

    log = get_logger("ai_chat_agent")
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    count = state.get("tool_call_count", 0)
    failure = None

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        if failure is not None:
            tool_messages.append(ToolMessage(
                content="前置工具调用失败，已跳过本次调用", tool_call_id=tool_call_id,
                status="error",
            ))
            continue

        try:
            if count >= AI_CHAT_MAX_TOOL_CALLS:
                raise MCPToolError("已达到工具调用次数上限", "TOOL_BUDGET_EXCEEDED")
            count += 1
            client = next((client for client in mcp_manager.get_all_clients().values()
                           if tool_name in {t["name"] for t in client.get_tools()}), None)
            if client is None:
                raise MCPToolError("请求的工具不可用", "UNKNOWN_TOOL")

            start = time.monotonic()
            # 同时约束第三方客户端实现，工具本身的未知副作用禁止自动重试。
            result_text = await asyncio.wait_for(
                client.call_tool(tool_name, tool_call["args"]),
                timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
            )
            tool_messages.append(ToolMessage(content=result_text, tool_call_id=tool_call_id))
            log.info("ai_chat_tool_called", tool_name=tool_name,
                     duration_ms=round((time.monotonic() - start) * 1000, 1),
                     trace_id=get_trace_id(), user_id=state.get("user_id", 0))
        except Exception as exc:
            code = getattr(exc, "error_code", "MCP_TOOL_TIMEOUT" if isinstance(exc, TimeoutError)
                           else "MCP_TOOL_FAILED")
            failure = _failure(code, "工具调用失败，请稍后重试")
            tool_messages.append(ToolMessage(
                content=json.dumps(failure, ensure_ascii=False),
                tool_call_id=tool_call_id, status="error",
            ))
            log.error("ai_chat_tool_error", tool_name=tool_name,
                      error_code=code, error_type=type(exc).__name__, trace_id=get_trace_id())

    return {"messages": tool_messages, "tool_call_count": count,
            "status": "failed" if failure else "running", "error": failure}


def should_continue_after_tools(state: AIChatState) -> str:
    return "format_response" if state.get("status") == "failed" else "agent"


def should_use_tools(state: AIChatState) -> str:
    """条件边判断函数：检查LLM是否需要调用MCP工具

    返回值：
      - "tools"：LLM发起了Tool Calls，需要执行MCP工具
      - "format_response"：LLM给出了最终回复，格式化输出
    """
    if state.get("status") == "failed":
        return "format_response"
    last_message = state["messages"][-1]  # 获取最后一条消息
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"  # 需要调用工具
    return "format_response"  # 直接格式化回复


def format_response(state: AIChatState) -> dict[str, Any]:
    """状态图节点4：从LLM最终回复中提取格式化的响应文本

    提取AIMessage中的content作为最终回复，
    清理可能的Markdown代码块标记。
    """
    if state.get("status") == "failed":
        return {"final_response": "", "status": "failed", "error": state.get("error")}
    log = get_logger("ai_chat_agent")  # 获取结构化日志记录器
    last_message = state["messages"][-1]  # 获取最后一条消息

    text = _text_content(last_message.content) if isinstance(last_message, AIMessage) else None
    if text is None:
        return {"final_response": "", "status": "failed", "error": _failure(
            "INVALID_MODEL_RESPONSE", "AI返回了无效结果，请稍后重试",
        )}
    response_text = text.strip()

    # 清理Markdown代码块标记（如果LLM误用了代码块格式）
    if response_text.startswith("```"):
        response_text = re.sub(r"^```\w*\n?", "", response_text)
        response_text = re.sub(r"\n?```$", "", response_text)

    log.info("ai_chat_complete", user_id=state.get("user_id", 0),
             response_length=len(response_text), trace_id=get_trace_id())

    if not response_text:
        return {"final_response": "", "status": "failed", "error": _failure(
            "EMPTY_MODEL_RESPONSE", "AI没有生成有效回复，请稍后重试",
        )}
    return {"final_response": response_text, "status": "success", "error": None}


def build_ai_chat_graph() -> StateGraph:
    """构建AI聊天Agent的LangGraph状态图（ReAct模式 + MCP工具调用）

    流程说明：
      init_conversation → agent_node → (条件边)
        ├── 有Tool Calls → execute_mcp_tools → agent_node（ReAct循环）
        └── 无Tool Calls → format_response → END
    """
    graph = StateGraph(AIChatState)  # 创建状态图实例

    # 添加四个节点
    graph.add_node("init_conversation", init_conversation)  # 节点1：初始化对话
    graph.add_node("agent", agent_node)  # 节点2：LLM推理
    graph.add_node("tools", execute_mcp_tools)  # 节点3：执行MCP工具
    graph.add_node("format_response", format_response)  # 节点4：格式化回复

    # 设置入口节点
    graph.set_entry_point("init_conversation")

    # 添加边和条件边
    graph.add_edge("init_conversation", "agent")  # 初始化后进入LLM推理
    graph.add_conditional_edges("agent", should_use_tools, {  # 条件边
        "tools": "tools",  # 需要调用工具
        "format_response": "format_response",  # 直接格式化回复
    })
    graph.add_conditional_edges("tools", should_continue_after_tools, {
        "agent": "agent", "format_response": "format_response",
    })
    graph.add_edge("format_response", END)  # 格式化回复后结束

    return graph


# 构建并编译AI聊天Agent的状态图
ai_chat_graph = build_ai_chat_graph()  # 构建状态图
ai_chat_app = ai_chat_graph.compile()  # 编译为可执行应用
