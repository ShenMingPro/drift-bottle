from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent_logger import init_logging  # 结构化日志初始化
from app.ice_break_checkpoint import close_ice_break_checkpoint, init_ice_break_checkpoint
from app.pick_checkpoint import close_pick_workflow_checkpoint, init_pick_workflow_checkpoint
from app.config import settings
from app.database import close_pool, get_database, init_pool
from app.schema import ensure_backend_schema
from app.routers import (
    ai_chat,
    auth,
    bottle,
    chat,
    content_gen,
    history,
    ice_break,
    pick_runs,
    plaza,
    task,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化日志系统（structlog + LangSmith追踪）
    init_logging()
    from app.mcp_client import mcp_manager
    try:
        await init_pool()
        async with get_database().session() as cursor:
            await ensure_backend_schema(cursor)
        await init_ice_break_checkpoint()
        await init_pick_workflow_checkpoint()
        # 启动时：初始化MCP客户端管理器，连接配置的MCP Server
        await _init_mcp_servers()
        yield
    finally:
        # 启动中途失败或正常关闭时，都释放已经打开的资源。
        try:
            await mcp_manager.close_all()
        finally:
            try:
                await close_pick_workflow_checkpoint()
            finally:
                try:
                    await close_ice_break_checkpoint()
                finally:
                    await close_pool()


async def _init_mcp_servers() -> None:
    """初始化MCP Server连接：根据配置启动MCP Server子进程

    目前支持：
      - @baranwang/mcp-trends-hub：热点趋势聚合服务
      - search1api-mcp：网页搜索、新闻、爬取服务
    两者均通过 npx 启动子进程，使用stdio方式通信。
    """
    from app.mcp_client import mcp_manager
    from app.config import settings

    # 初始化 Trends Hub MCP Server（热搜趋势聚合）
    if settings.MCP_TRENDS_HUB_ENABLED:
        try:
            # 构建环境变量
            env_vars = {}
            if settings.MCP_TRENDS_HUB_HIDDEN_FIELDS:
                env_vars["TRENDS_HUB_HIDDEN_FIELDS"] = settings.MCP_TRENDS_HUB_HIDDEN_FIELDS
            if settings.MCP_TRENDS_HUB_CUSTOM_RSS_URL:
                env_vars["TRENDS_HUB_CUSTOM_RSS_URL"] = settings.MCP_TRENDS_HUB_CUSTOM_RSS_URL

            # 连接Trends Hub MCP Server
            await mcp_manager.add_server(
                name="trends-hub",
                command="npx",
                args=["-y", f"mcp-trends-hub@{settings.MCP_TRENDS_HUB_VERSION}"],
                env=env_vars if env_vars else None,
            )
        except BaseException as e:
            import logging
            logging.getLogger("main").error(f"连接MCP Trends Hub失败: {e}，AI聊天功能可能不可用")

    # 初始化 Search1API MCP Server（网页搜索、新闻、爬取）
    if settings.MCP_SEARCH1API_ENABLED and settings.MCP_SEARCH1API_KEY:
        try:
            await mcp_manager.add_server(
                name="search1api",
                command="npx",
                args=["-y", "search1api-mcp"],
                env={"SEARCH1API_KEY": settings.MCP_SEARCH1API_KEY},
            )
        except BaseException as e:
            import logging
            logging.getLogger("main").error(f"连接MCP Search1API失败: {e}，搜索功能可能不可用")


app = FastAPI(title="The Player API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(bottle.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(task.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(ice_break.router, prefix="/api")
app.include_router(content_gen.router, prefix="/api")
app.include_router(ai_chat.router, prefix="/api")
app.include_router(plaza.router, prefix="/api")
app.include_router(pick_runs.router, prefix="/api")

app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
