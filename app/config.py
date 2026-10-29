from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "paper_bottle"
    PORT: int = 3000
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    # 每次 Agent/工具调用的等待上限；模型与已知只读 Agent 的总尝试次数。
    AGENT_CALL_TIMEOUT_SECONDS: float = Field(default=60, gt=0, le=300)
    AGENT_MAX_ATTEMPTS: int = Field(default=2, ge=1, le=3)
    # 单条任务的总时间预算；超出后不再发起新的模型或子 Agent 调用。
    AGENT_TIME_BUDGET_SECONDS: float = Field(default=120, gt=0, le=900)
    # 人与人聊天 AI 辅助检查点；相对路径按项目根解析，与业务 MySQL 独立。
    ICE_BREAK_CHECKPOINT_PATH: str = "data/ice_break_checkpoints.sqlite3"
    # 可恢复拾取工作流检查点；与破冰建议分开存储，避免清理单个功能时互相影响。
    PICK_WORKFLOW_CHECKPOINT_PATH: str = "data/pick_workflow_checkpoints.sqlite3"
    # LangSmith追踪配置（填写API_KEY后自动启用Agent调用链追踪）
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "piaoliuping"
    LANGCHAIN_TRACING_V2: bool = False
    # JWT认证配置
    JWT_SECRET: str = "change_me_in_production"  # JWT签名密钥，生产环境必须更换
    JWT_EXPIRE_HOURS: int = 72  # JWT过期时间（小时）
    # MCP服务配置
    MCP_TRENDS_HUB_ENABLED: bool = True  # 是否启用Trends Hub MCP服务
    MCP_TRENDS_HUB_VERSION: str = "1.6.0"  # mcp-trends-hub版本号
    MCP_TRENDS_HUB_HIDDEN_FIELDS: str = ""  # 隐藏字段列表（如"cover,get-nytimes-news:description"）
    MCP_TRENDS_HUB_CUSTOM_RSS_URL: str = ""  # 自定义RSS订阅源URL
    # Search1API MCP服务配置（网页搜索、新闻、爬取等）
    MCP_SEARCH1API_ENABLED: bool = True  # 是否启用Search1API MCP服务
    MCP_SEARCH1API_KEY: str = ""  # Search1API的API密钥

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
