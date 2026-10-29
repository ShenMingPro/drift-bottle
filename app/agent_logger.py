"""Agent结构化日志模块

提供基于structlog的结构化日志功能，用于追踪Agent的推理过程、工具调用、耗时等关键信息。
同时支持LangSmith调用链追踪（配置LANGSMITH_API_KEY后自动启用）。

日志格式示例：
{
    "timestamp": "2026-06-07T12:00:00",
    "level": "info",
    "agent": "match_agent",
    "event": "agent_start",
    "user_id": 1,
    "action": "pick_full",
    "trace_id": "abc123",
    "duration_ms": 1523,
    "input": {...},
    "output": {...}
}
"""

from __future__ import annotations

import logging  # Python标准日志库，用于将structlog输出桥接到标准日志系统
import os  # 操作系统接口，用于设置环境变量启用LangSmith
import time  # 时间库，用于计算Agent执行耗时
import uuid  # UUID生成库，用于生成唯一的trace_id
from contextvars import ContextVar  # 上下文变量，用于在同一次请求中传递trace_id

import structlog  # 结构化日志库，提供JSON格式的日志输出


# ============================================================
# 上下文变量：在同一次请求的协程中传递trace_id
# ============================================================

# 当前请求的追踪ID，用于关联同一次Supervisor调用中的所有子Agent日志
current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")


# ============================================================
# structlog配置
# ============================================================

def setup_logging() -> None:
    """初始化structlog日志配置

    配置说明：
    - 使用JSON格式输出，方便日志采集系统（如ELK）解析
    - 添加timestamp、level、trace_id等标准字段
    - 将structlog输出桥接到Python标准logging系统
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # 合并上下文变量（如trace_id）
            structlog.processors.add_log_level,  # 添加日志级别字段
            structlog.processors.TimeStamper(fmt="iso"),  # 添加ISO格式时间戳
            structlog.processors.StackInfoRenderer(),  # 渲染异常堆栈信息
            structlog.processors.format_exc_info,  # 格式化异常信息
            structlog.processors.JSONRenderer(),  # 输出为JSON格式
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),  # 最低日志级别INFO
        context_class=dict,  # 上下文使用普通字典
        logger_factory=structlog.PrintLoggerFactory(),  # 输出到stdout
        cache_logger_on_first_use=True,  # 缓存logger实例，避免重复创建
    )


def setup_langsmith() -> None:
    """配置LangSmith调用链追踪

    当.env中配置了LANGSMITH_API_KEY后，LangChain/LangGraph会自动将
    Agent调用链数据上报到LangSmith平台，可在Web界面查看完整的调用链、
    Token消耗、耗时等信息。

    无需修改任何Agent代码，只需设置环境变量即可启用。
    """
    from app.config import settings  # 延迟导入，避免循环依赖

    if settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        # 未配置API_KEY时，确保不启用追踪
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)


def get_logger(agent_name: str) -> structlog.stdlib.BoundLogger:
    """获取绑定Agent名称的结构化日志器

    参数：
        agent_name: Agent名称，如"match_agent"、"guard_agent"等
    返回：
        绑定了agent字段的structlog日志器
    """
    return structlog.get_logger().bind(agent=agent_name)


def new_trace_id() -> str:
    """生成新的追踪ID并设置到上下文变量中

    每次Supervisor调用开始时调用，生成唯一的trace_id，
    后续所有子Agent的日志都会自动携带此trace_id，
    方便在同一次请求中关联所有Agent的日志。

    返回：
        生成的trace_id字符串
    """
    trace_id = uuid.uuid4().hex[:12]  # 取UUID前12位作为trace_id
    current_trace_id.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    """获取当前上下文中的trace_id"""
    return current_trace_id.get("")


# ============================================================
# Agent执行计时器：上下文管理器，自动记录Agent执行耗时
# ============================================================

class AgentTimer:
    """Agent执行计时器，用于记录Agent节点的执行耗时

    使用方式：
        async with AgentTimer(logger, "match_agent", "find_best_match", user_id=1) as timer:
            result = await some_function()
            timer.set_output(result)

    日志输出示例：
        {
            "agent": "match_agent",
            "event": "agent_node_complete",
            "node": "find_best_match",
            "duration_ms": 1523,
            "user_id": 1,
            "output_keys": ["best_match", "match_score"]
        }
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger, agent_name: str,
                 node_name: str, **context):
        """初始化计时器

        参数：
            logger: structlog日志器
            agent_name: Agent名称
            node_name: 节点名称
            **context: 额外的上下文信息（如user_id、action等）
        """
        self.logger = logger
        self.agent_name = agent_name
        self.node_name = node_name
        self.context = context
        self.start_time: float = 0.0
        self._output: dict = {}

    def set_output(self, output: dict) -> None:
        """设置节点输出结果（用于日志记录，不会修改实际输出）"""
        self._output = output

    async def __aenter__(self) -> "AgentTimer":
        """进入上下文：记录节点开始执行"""
        self.start_time = time.time()
        self.logger.info(
            "agent_node_start",
            node=self.node_name,
            trace_id=get_trace_id(),
            **self.context,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文：记录节点执行完成（含耗时和输出摘要）"""
        duration_ms = round((time.time() - self.start_time) * 1000, 1)

        if exc_type:
            # 节点执行异常
            self.logger.error(
                "agent_node_error",
                node=self.node_name,
                duration_ms=duration_ms,
                trace_id=get_trace_id(),
                error=str(exc_val),
                **self.context,
            )
        else:
            # 节点执行成功，记录耗时和输出字段摘要（不记录完整输出，避免日志过大）
            output_keys = list(self._output.keys()) if self._output else []
            self.logger.info(
                "agent_node_complete",
                node=self.node_name,
                duration_ms=duration_ms,
                trace_id=get_trace_id(),
                output_keys=output_keys,
                **self.context,
            )


# ============================================================
# Agent指标统计：累计记录关键业务指标
# ============================================================

class AgentMetrics:
    """Agent指标统计器，累计记录LLM调用次数、Token消耗、响应时间、审核通过率等

    使用方式：
        metrics = AgentMetrics()
        metrics.record_llm_call(tokens=256, duration_ms=1200)
        metrics.record_guard_result(passed=True)
        print(metrics.summary())
    """

    def __init__(self) -> None:
        """初始化指标统计器"""
        self.llm_call_count: int = 0  # LLM调用总次数
        self.total_tokens: int = 0  # Token消耗总量
        self.total_duration_ms: float = 0.0  # 总响应时间（毫秒）
        self.guard_total: int = 0  # 内容审核总次数
        self.guard_passed: int = 0  # 内容审核通过次数

    def record_llm_call(self, tokens: int = 0, duration_ms: float = 0.0) -> None:
        """记录一次LLM调用

        参数：
            tokens: 本次调用消耗的Token数
            duration_ms: 本次调用的响应时间（毫秒）
        """
        self.llm_call_count += 1
        self.total_tokens += tokens
        self.total_duration_ms += duration_ms

    def record_guard_result(self, passed: bool) -> None:
        """记录一次内容审核结果

        参数：
            passed: 审核是否通过
        """
        self.guard_total += 1
        if passed:
            self.guard_passed += 1

    def summary(self) -> dict:
        """返回指标摘要字典"""
        avg_duration = self.total_duration_ms / self.llm_call_count if self.llm_call_count else 0
        pass_rate = self.guard_passed / self.guard_total if self.guard_total else 0
        return {
            "llm_call_count": self.llm_call_count,
            "total_tokens": self.total_tokens,
            "avg_duration_ms": round(avg_duration, 1),
            "guard_pass_rate": round(pass_rate, 3),
        }


# 全局指标实例，用于跨请求累计统计
global_metrics = AgentMetrics()


# ============================================================
# 初始化函数：在应用启动时调用
# ============================================================

def init_logging() -> None:
    """初始化日志系统，在FastAPI lifespan中调用

    执行以下操作：
    1. 配置structlog结构化日志
    2. 配置LangSmith调用链追踪（如果配置了API_KEY）
    """
    setup_logging()
    setup_langsmith()
