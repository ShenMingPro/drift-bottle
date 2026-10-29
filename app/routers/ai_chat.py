"""AI聊天路由：提供AI对话接口，支持SSE流式输出和MCP工具调用

本路由处理用户与AI助手的对话请求，支持：
  - 普通对话：直接调用AI聊天Agent获取回复
  - SSE流式对话：逐步推送AI思考过程和最终回复
  - MCP工具列表查询：获取当前可用的MCP工具列表
"""

from __future__ import annotations  # 启用延迟类型注解评估

import json  # JSON序列化库
import logging  # 日志库
import uuid  # UUID生成，用于thread_id
from typing import Any  # 类型注解工具

from fastapi import APIRouter, Depends, HTTPException, Query  # FastAPI路由和依赖注入
from fastapi.responses import JSONResponse  # JSON响应类
from pydantic import BaseModel  # Pydantic数据模型基类

from app.agents.ai_chat_agent import ai_chat_app  # AI聊天Agent编译后的应用
from app.auth_jwt import get_current_user  # JWT认证依赖项
from app.mcp_client import mcp_manager  # MCP客户端管理器全局实例

router = APIRouter()  # 创建路由器实例
logger = logging.getLogger("ai_chat")  # 创建日志记录器


def _assert_success(result: dict[str, Any]) -> None:
    if result.get("status") in {"failed", "fatal_error", "retryable_error"}:
        code = (result.get("error") or {}).get("error_code", "AI_CHAT_FAILED")
        raise HTTPException(
            status_code=504 if "TIMEOUT" in code else 502,
            detail="AI服务暂时不可用，请稍后重试",
        )


class ChatRequest(BaseModel):
    """AI聊天请求模型：定义用户发送聊天消息时的请求体格式"""
    userId: int  # 用户ID
    message: str  # 用户消息内容
    history: list[dict[str, str]] | None = None  # 对话历史（可选），格式：[{"role": "user"/"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    """AI聊天响应模型：定义AI回复的响应体格式"""
    response: str  # AI回复文本
    toolUsed: list[str] | None = None  # 本次对话使用的MCP工具列表（可选）


@router.post("/ai-chat")
async def chat_with_ai(req: ChatRequest, current_user_id: int = Depends(get_current_user)) -> JSONResponse:
    """AI聊天接口（需JWT认证）：接收用户消息，调用AI聊天Agent获取回复

    流程：
      1. 验证用户身份（JWT认证 + userId匹配）
      2. 构建AI聊天Agent的初始状态（包含对话历史）
      3. 调用Agent执行ReAct循环（可能调用MCP工具）
      4. 返回AI回复和使用的工具列表

    参数：
      - req：ChatRequest请求体，包含userId、message和可选的history

    返回：
      - response：AI回复文本
      - toolUsed：使用的MCP工具名称列表
    """
    # 验证用户身份
    if current_user_id != req.userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    # 构建对话历史消息列表
    from langchain_core.messages import AIMessage, HumanMessage  # 延迟导入消息类型

    messages = []
    if req.history:
        for msg in req.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # 添加当前用户消息
    messages.append(HumanMessage(content=req.message))

    # 构建AI聊天Agent的初始状态
    chat_state = {
        "user_id": req.userId,
        "messages": messages,
        "final_response": "",
        "mcp_tool_calls": [],
        "mcp_results": [],
    }

    # 每次调用传入独立thread_id，确保LangGraph不会缓存
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        # 调用AI聊天Agent
        result = await ai_chat_app.ainvoke(chat_state, config)
        _assert_success(result)

        # 提取使用的工具名称
        tools_used = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc.get("name", ""))

        # 获取最终回复
        response_text = result.get("final_response", "")

        return JSONResponse(
            content={
                "response": response_text,
                "toolUsed": tools_used if tools_used else None,
            },
            headers={"Cache-Control": "no-store"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI聊天失败: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="AI服务暂时不可用，请稍后重试") from e


@router.get("/ai-chat/stream")
async def stream_chat_with_ai(
    userId: int = Query(...),
    message: str = Query(...),
    current_user_id: int = Depends(get_current_user),
) -> Any:
    """AI聊天流式接口（需JWT认证）：通过SSE逐步推送AI思考过程和回复

    相比普通 /ai-chat 接口，流式接口的优势：
      1. 用户无需等待完整LLM响应，先看到"正在思考..."等进度提示
      2. MCP工具调用过程可视化，用户知道AI正在查询什么数据
      3. 回复逐字/逐段推送，减少感知等待时间

    SSE事件流格式：
      event: progress  →  {"stage": "thinking"/"tool_calling", "message": "进度描述", "tool": "工具名"}
      event: response  →  {"content": "AI回复片段"}
      event: done      →  {"toolUsed": ["工具名列表"]}
      event: error     →  {"message": "错误描述"}
    """
    # 验证用户身份
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    from sse_starlette.sse import EventSourceResponse  # SSE响应类

    async def event_generator():
        """SSE事件生成器：逐步推送AI聊天的进度和回复"""
        from langchain_core.messages import HumanMessage  # 延迟导入消息类型

        try:
            # 阶段1：正在思考
            yield {
                "event": "progress",
                "data": json.dumps({"stage": "thinking", "message": "正在思考你的问题..."}, ensure_ascii=False),
            }

            # 构建消息列表
            messages = [HumanMessage(content=message)]

            # 构建Agent初始状态
            chat_state = {
                "user_id": userId,
                "messages": messages,
                "final_response": "",
                "mcp_tool_calls": [],
                "mcp_results": [],
            }
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            # 使用astream_events获取流式事件
            tools_used = []
            streamed_text = ""
            final_state = None
            root_run_id = None

            async for event in ai_chat_app.astream_events(chat_state, config, version="v2"):
                kind = event.get("event", "")

                if kind == "on_chain_start" and not event.get("parent_ids"):
                    root_run_id = event.get("run_id")

                if kind == "on_chain_end" and event.get("run_id") == root_run_id:
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        final_state = output

                # 捕获LLM开始生成的事件
                if kind == "on_chat_model_start":
                    yield {
                        "event": "progress",
                        "data": json.dumps({"stage": "thinking", "message": "正在思考..."}, ensure_ascii=False),
                    }

                # 捕获LLM生成token的事件（流式输出）
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # 只在非工具调用时推送文本片段
                        if not getattr(chunk, "tool_calls", None) and not getattr(chunk, "additional_kwargs", {}).get("tool_calls"):
                            streamed_text += chunk.content
                            yield {
                                "event": "response",
                                "data": json.dumps({"content": chunk.content}, ensure_ascii=False),
                            }

                # 捕获工具调用开始的事件
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tools_used.append(tool_name)
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "stage": "tool_calling",
                            "message": f"正在查询{tool_name}...",
                            "tool": tool_name,
                        }, ensure_ascii=False),
                    }

                # 捕获工具调用完成的事件
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "stage": "tool_complete",
                            "message": f"已获取{tool_name}数据",
                            "tool": tool_name,
                        }, ensure_ascii=False),
                    }

            # 同一次图执行已经产生了最终状态，不能再 ainvoke 一遍，
            # 否则会重复调用模型和 MCP 工具。done 事件让前端以最终状态校准文本。
            if not isinstance(final_state, dict):
                raise RuntimeError("AI聊天流未返回完整结果")
            final_response = ""
            if isinstance(final_state, dict):
                _assert_success(final_state)
                final_response = str(final_state.get("final_response", ""))
            if not final_response:
                final_response = streamed_text

            # 结束事件
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "toolUsed": tools_used if tools_used else None,
                        "response": final_response,
                    },
                    ensure_ascii=False,
                ),
            }

        except Exception as e:
            logger.error(f"AI聊天流式失败: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"message": "AI服务暂时不可用，请稍后重试"}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/ai-chat/tools")
async def get_available_tools(current_user_id: int = Depends(get_current_user)) -> JSONResponse:
    """获取当前可用的MCP工具列表（需JWT认证）

    返回所有已连接MCP Server提供的工具列表，
    供前端展示可用功能或用户选择。
    """
    tools = mcp_manager.get_all_tools()  # 获取所有工具列表
    return JSONResponse(
        content={"tools": tools},
        headers={"Cache-Control": "no-store"},
    )
