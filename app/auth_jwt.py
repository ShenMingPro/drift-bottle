"""JWT认证模块

提供JWT令牌的生成、验证和解析功能，以及FastAPI依赖项用于API端点的权限校验。

认证流程：
1. 用户登录/注册 → 服务器返回JWT令牌
2. 前端将令牌存储在localStorage中
3. 后续请求在Authorization头中携带令牌：Authorization: Bearer <token>
4. 服务端通过get_current_user依赖项验证令牌并提取用户ID

令牌结构（Payload）：
{
    "sub": "<user_id>",       # 主题：用户ID
    "exp": "<expiration>",    # 过期时间
    "iat": "<issued_at>"      # 签发时间
}
"""

from __future__ import annotations

import jwt  # PyJWT库，用于JWT令牌的编码和解码
from datetime import datetime, timedelta, timezone  # 日期时间库，用于计算令牌过期时间
from fastapi import Depends, HTTPException, status  # FastAPI依赖注入和HTTP异常
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # FastAPI安全工具
from typing import Annotated  # 类型注解工具

from app.config import settings  # 应用配置，读取JWT密钥和过期时间


# ============================================================
# HTTP Bearer认证方案：从请求头中提取Authorization: Bearer <token>
# auto_error=False 表示未携带token时不自动抛403，而是返回None
# （这样可以在依赖项中自定义错误响应）
# ============================================================

_bearer_scheme = HTTPBearer(auto_error=False)


# ============================================================
# JWT令牌生成
# ============================================================

def create_access_token(user_id: int) -> str:
    """生成JWT访问令牌

    参数：
        user_id: 用户ID，将编码到令牌的sub字段中

    返回：
        编码后的JWT令牌字符串

    令牌包含以下字段：
        - sub: 用户ID（字符串形式）
        - exp: 过期时间（当前时间 + JWT_EXPIRE_HOURS）
        - iat: 签发时间
    """
    now = datetime.now(timezone.utc)  # 当前UTC时间
    payload = {
        "sub": str(user_id),  # 主题：用户ID（JWT标准要求sub为字符串）
        "exp": now + timedelta(hours=settings.JWT_EXPIRE_HOURS),  # 过期时间
        "iat": now,  # 签发时间
    }
    # 使用HS256算法和配置的密钥编码JWT
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


# ============================================================
# JWT令牌验证与解析
# ============================================================

def decode_access_token(token: str) -> dict:
    """解码并验证JWT令牌

    参数：
        token: JWT令牌字符串

    返回：
        令牌的payload字典，包含sub（用户ID）、exp（过期时间）、iat（签发时间）

    异常：
        jwt.ExpiredSignatureError: 令牌已过期
        jwt.InvalidTokenError: 令牌无效（格式错误、签名不匹配等）
    """
    return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])


# ============================================================
# FastAPI依赖项：从JWT令牌中提取当前用户ID
# ============================================================

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> int:
    """FastAPI依赖项：从请求头的JWT令牌中提取并验证当前用户ID

    使用方式：
        @router.get("/protected")
        async def protected_route(user_id: int = Depends(get_current_user)):
            # user_id 已通过JWT验证，可安全使用
            ...

    验证流程：
    1. 从请求头提取 Authorization: Bearer <token>
    2. 解码JWT令牌，验证签名和过期时间
    3. 从payload的sub字段提取用户ID
    4. 返回用户ID（整数）

    异常响应：
        - 401 Unauthorized: 未携带令牌、令牌无效或已过期
    """
    # 检查是否携带了Bearer令牌
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证令牌，请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials  # 提取令牌字符串

    try:
        # 解码并验证JWT令牌
        payload = decode_access_token(token)
        # 从payload的sub字段提取用户ID
        user_id = int(payload.get("sub", 0))
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌：用户ID异常",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id

    except jwt.ExpiredSignatureError:
        # 令牌已过期
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证令牌已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        # 令牌无效（格式错误、签名不匹配等）
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================
# 可选依赖项：尝试提取用户ID，未认证时返回0（用于兼容公开接口）
# ============================================================

async def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> int:
    """可选的FastAPI依赖项：尝试从JWT令牌中提取用户ID，未认证时返回0

    使用场景：某些接口既支持匿名访问又支持登录用户访问，
    登录用户可以获得个性化结果，匿名用户获得默认结果。

    使用方式：
        @router.get("/public-or-personalized")
        async def mixed_route(user_id: int = Depends(get_optional_user)):
            if user_id > 0:
                # 登录用户：返回个性化结果
            else:
                # 匿名用户：返回默认结果
    """
    if credentials is None:
        return 0  # 未携带令牌，返回0表示匿名用户

    try:
        payload = decode_access_token(credentials.credentials)
        return int(payload.get("sub", 0))
    except (jwt.InvalidTokenError, ValueError):
        return 0  # 令牌无效，返回0表示匿名用户
