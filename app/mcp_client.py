"""MCP客户端模块：连接MCP Server（如@baranwang/mcp-trends-hub），获取可用工具列表并调用工具

本模块实现了MCP（Model Context Protocol）客户端，通过stdio方式连接MCP Server进程，
获取Server提供的工具列表，并将这些工具转换为LangChain Tool格式供AI聊天Agent使用。

MCP协议流程：
  1. 通过stdio启动MCP Server子进程（如 npx -y mcp-trends-hub@1.6.0）
  2. 建立ClientSession并初始化
  3. 调用list_tools()获取Server提供的工具列表
  4. 调用call_tool()执行具体工具并获取结果
"""

from __future__ import annotations  # 启用延迟类型注解评估

import asyncio
import logging  # 日志库
from contextlib import AsyncExitStack  # 异步上下文管理器栈，用于管理MCP连接的生命周期
from typing import Any  # 类型注解工具

from mcp import ClientSession, StdioServerParameters  # MCP Python SDK核心类
from mcp.client.stdio import stdio_client  # MCP stdio传输层客户端

logger = logging.getLogger("mcp_client")  # 创建日志记录器


class MCPToolError(RuntimeError):
    """保留工具失败类型；未知工具可能有副作用，默认禁止自动重放。"""

    def __init__(self, message: str, error_code: str, *, retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


class MCPClient:
    """MCP客户端：管理与MCP Server的连接，提供工具发现和调用能力

    通过stdio方式连接MCP Server子进程，支持：
      - 连接管理：启动/关闭MCP Server子进程
      - 工具发现：获取Server提供的工具列表
      - 工具调用：执行指定工具并返回结果
      - LangChain集成：将MCP工具转换为LangChain Tool格式
    """

    def __init__(self) -> None:
        """初始化MCP客户端，设置默认属性"""
        self.session: ClientSession | None = None  # MCP客户端会话，连接后非空
        self.exit_stack = AsyncExitStack()  # 异步上下文管理器栈，管理连接生命周期
        self._tools_cache: list[dict[str, Any]] = []  # 工具列表缓存，避免重复查询
        self._connected: bool = False  # 连接状态标志位

    @property
    def is_connected(self) -> bool:
        """返回当前是否已连接到MCP Server"""
        return self._connected and self.session is not None

    async def connect_to_server(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """连接到MCP Server子进程

        通过stdio方式启动MCP Server子进程，建立ClientSession并初始化。

        参数：
          - command：启动命令（如"npx"）
          - args：命令参数列表（如["-y", "mcp-trends-hub@1.6.0"]）
          - env：环境变量字典（如{"TRENDS_HUB_HIDDEN_FIELDS": "cover"}）
        """
        server_params = StdioServerParameters(
            command=command,  # 启动命令
            args=args or [],  # 命令参数，默认为空列表
            env=env,  # 环境变量，默认为None（继承当前进程环境变量）
        )

        logger.info(f"正在连接MCP Server: {command} {' '.join(args or [])}")

        try:
            # 通过stdio方式建立与MCP Server的传输通道
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport  # 获取读写流

            # 创建并初始化ClientSession
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()  # 初始化MCP会话，完成握手

            self._connected = True  # 标记为已连接

            # 获取并缓存工具列表
            await self.refresh_tools()

            logger.info(
                f"MCP Server连接成功，可用工具: {[t['name'] for t in self._tools_cache]}"
            )
        except Exception as e:
            logger.error(f"连接MCP Server失败: {e}")
            self._connected = False
            raise

    async def refresh_tools(self) -> list[dict[str, Any]]:
        """刷新并缓存MCP Server提供的工具列表

        返回：
          - 工具列表，每个工具包含name、description、inputSchema字段
        """
        if not self.session:
            return []

        response = await self.session.list_tools()  # 调用MCP协议获取工具列表
        self._tools_cache = [
            {
                "name": tool.name,  # 工具名称（如"get-weibo-trending"）
                "description": tool.description or "",  # 工具描述
                "inputSchema": tool.inputSchema or {},  # 工具输入参数的JSON Schema
            }
            for tool in response.tools
        ]
        return self._tools_cache

    def get_tools(self) -> list[dict[str, Any]]:
        """获取缓存的工具列表（不触发网络请求）

        返回：
          - 工具列表缓存
        """
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """调用MCP Server上的指定工具

        参数：
          - tool_name：工具名称（如"get-weibo-trending"）
          - arguments：工具参数字典（如{}或{"limit": 10}）

        返回：
          - 工具执行结果的字符串（JSON格式）
        """
        if not self.session:
            raise MCPToolError("工具服务尚未连接", "MCP_NOT_CONNECTED")

        from app.config import settings

        logger.info("调用MCP工具: %s", tool_name)

        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments or {}),
                timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
            )
            if getattr(result, "isError", False):
                raise MCPToolError("工具服务返回失败，请稍后重试", "MCP_TOOL_ERROR")

            # 将CallToolResult中的content列表拼接为字符串
            text_parts = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    text_parts.append(content_item.text)
                else:
                    text_parts.append(str(content_item))

            result_text = "\n".join(text_parts)
            logger.info(f"MCP工具 {tool_name} 返回结果长度: {len(result_text)}")
            return result_text

        except MCPToolError:
            logger.warning("MCP工具返回失败: %s", tool_name)
            raise
        except TimeoutError as exc:
            logger.warning("MCP工具调用超时: %s", tool_name)
            raise MCPToolError("工具调用超时，请稍后重试", "MCP_TOOL_TIMEOUT") from exc
        except Exception as exc:
            logger.exception("调用MCP工具 %s 失败", tool_name)
            raise MCPToolError("工具暂时无法使用，请稍后重试", "MCP_TOOL_FAILED") from exc

    def get_langchain_tools(self) -> list[dict[str, Any]]:
        """将MCP工具列表转换为LangChain Function Calling格式

        将MCP工具的inputSchema转换为LangChain Tool所需的格式，
        供ChatOpenAI.bind_tools()使用。

        返回：
          - LangChain格式的工具定义列表
        """
        langchain_tools = []
        for tool in self._tools_cache:
            langchain_tools.append({
                "type": "function",  # 工具类型为函数调用
                "function": {
                    "name": tool["name"],  # 函数名称
                    "description": tool["description"],  # 函数描述
                    "parameters": tool["inputSchema"],  # 参数JSON Schema
                },
            })
        return langchain_tools

    async def close(self) -> None:
        """关闭MCP客户端，释放所有资源（关闭子进程、断开连接）"""
        logger.info("正在关闭MCP客户端...")
        self._connected = False
        try:
            await self.exit_stack.aclose()  # 关闭所有异步上下文管理器（包括子进程）
        except Exception as e:
            logger.warning(f"关闭MCP客户端时出错: {e}")
        finally:
            self.session = None
            self._tools_cache = []


# ============================================================
# 全局MCP客户端管理器：单例模式，管理MCP Server连接的生命周期
# ============================================================

class MCPClientManager:
    """MCP客户端管理器：管理多个MCP Server的连接，提供统一的管理接口

    采用单例模式，在FastAPI应用启动时初始化，在应用关闭时清理。
    支持动态添加/移除MCP Server连接。
    """

    def __init__(self) -> None:
        """初始化管理器"""
        self._clients: dict[str, MCPClient] = {}  # 客户端字典：key为服务名，value为MCPClient实例

    async def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPClient:
        """添加并连接一个MCP Server

        参数：
          - name：服务名称（如"trends-hub"），用于后续查找
          - command：启动命令
          - args：命令参数列表
          - env：环境变量

        返回：
          - 已连接的MCPClient实例
        """
        # 如果已存在同名连接，先关闭旧连接
        if name in self._clients:
            await self._clients[name].close()

        client = MCPClient()  # 创建新的MCP客户端
        await client.connect_to_server(command, args, env)  # 连接到MCP Server
        self._clients[name] = client  # 存入字典
        logger.info(f"MCP Server '{name}' 已添加并连接")
        return client

    def get_client(self, name: str) -> MCPClient | None:
        """根据服务名称获取MCP客户端

        参数：
          - name：服务名称

        返回：
          - MCPClient实例，不存在则返回None
        """
        return self._clients.get(name)

    def get_all_clients(self) -> dict[str, MCPClient]:
        """获取所有已注册的MCP客户端

        返回：
          - 客户端字典
        """
        return self._clients

    async def remove_server(self, name: str) -> None:
        """移除并关闭指定名称的MCP Server连接

        参数：
          - name：服务名称
        """
        if name in self._clients:
            await self._clients[name].close()  # 关闭连接
            del self._clients[name]  # 从字典中移除
            logger.info(f"MCP Server '{name}' 已移除")

    async def close_all(self) -> None:
        """关闭所有MCP Server连接"""
        for name, client in self._clients.items():
            try:
                await client.close()
                logger.info(f"MCP Server '{name}' 已关闭")
            except Exception as e:
                logger.warning(f"关闭MCP Server '{name}' 时出错: {e}")
        self._clients.clear()  # 清空字典

    def get_all_tools(self) -> list[dict[str, Any]]:
        """获取所有MCP Server的工具列表（合并）

        返回：
          - 所有Server的工具列表
        """
        all_tools = []
        for name, client in self._clients.items():
            for tool in client.get_tools():
                all_tools.append({**tool, "server_name": name})  # 添加server_name标识
        return all_tools

    def get_all_langchain_tools(self) -> list[dict[str, Any]]:
        """获取所有MCP Server的LangChain格式工具列表（合并）

        返回：
          - 所有Server的LangChain格式工具列表
        """
        all_tools = []
        for name, client in self._clients.items():
            all_tools.extend(client.get_langchain_tools())
        return all_tools


# 全局单例实例
mcp_manager = MCPClientManager()
