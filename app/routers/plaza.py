"""广场模块路由：提供动态发布、查询、点赞、评论、文件上传等API

本路由处理广场模块的所有请求，包括：
  - 发布动态（文字/图片/视频）
  - 查询动态列表（支持分页）
  - 点赞/取消点赞
  - 发表评论 / 查询评论列表
  - 文件上传（图片/视频）
"""

from __future__ import annotations  # 启用延迟类型注解评估

import json  # JSON序列化库
import logging  # 日志库
import os  # 操作系统接口，用于文件路径操作
import time  # 时间库，用于生成唯一文件名
import uuid  # UUID库，用于生成唯一文件名

import aiomysql  # 异步MySQL驱动，用于DictCursor
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File  # FastAPI路由和依赖注入

from app.auth_jwt import get_current_user  # JWT认证依赖项
from app import database as db  # 数据库模块（通过模块引用获取最新的pool对象）
from app.models.plaza import (  # 广场模块数据模型
    CommentItem,
    CommentListResponse,
    CreateCommentRequest,
    CreateCommentResponse,
    CreatePostRequest,
    CreatePostResponse,
    LikeResponse,
    PostItem,
    PostListResponse,
)

router = APIRouter()  # 创建路由器实例
logger = logging.getLogger("plaza")  # 创建日志记录器

# 文件上传配置
UPLOAD_DIR = "public/uploads"  # 上传文件存储目录
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}  # 允许的图片类型
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}  # 允许的视频类型
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 图片最大10MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 视频最大50MB


@router.post("/plaza/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user_id: int = Depends(get_current_user),
) -> dict:
    """文件上传接口（需JWT认证）：上传图片或视频文件

    流程：
      1. 验证用户身份
      2. 检查文件类型和大小
      3. 保存文件到服务器
      4. 返回文件访问URL
    """
    # 确保上传目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 获取文件信息
    content_type = file.content_type or ""
    file_size = 0

    # 检查文件类型
    is_image = content_type in ALLOWED_IMAGE_TYPES
    is_video = content_type in ALLOWED_VIDEO_TYPES

    if not is_image and not is_video:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{content_type}，仅支持jpg/png/gif/webp图片和mp4/webm视频")

    # 检查文件大小（通过读取文件内容）
    content = await file.read()
    file_size = len(content)

    if is_image and file_size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"图片大小不能超过10MB，当前{round(file_size/1024/1024, 1)}MB")
    if is_video and file_size > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=400, detail=f"视频大小不能超过50MB，当前{round(file_size/1024/1024, 1)}MB")

    # 生成唯一文件名
    ext = os.path.splitext(file.filename or "")[1] or (".jpg" if is_image else ".mp4")
    unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # 写入文件
    with open(file_path, "wb") as f:
        f.write(content)

    # 返回文件访问URL（相对于静态文件服务路径）
    url = f"/uploads/{unique_name}"
    media_type = "image" if is_image else "video"

    logger.info(f"用户{current_user_id}上传文件：{unique_name}，类型={media_type}，大小={round(file_size/1024, 1)}KB")
    return {"success": True, "url": url, "mediaType": media_type, "message": "上传成功"}


@router.post("/plaza/posts", response_model=CreatePostResponse)
async def create_post(req: CreatePostRequest, current_user_id: int = Depends(get_current_user)) -> CreatePostResponse:
    """发布动态接口（需JWT认证）：用户发布一条新动态

    流程：
      1. 验证用户身份（JWT认证 + userId匹配）
      2. 插入动态记录到posts表
      3. 返回新动态ID

    修复说明：使用pool直接获取连接，避免get_db的cursor生命周期问题
    """
    # 安全校验：JWT中的用户ID必须与请求中的userId一致
    if current_user_id != req.userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    # 内容校验
    content = req.content.strip()
    if not content and not req.mediaUrls:
        raise HTTPException(status_code=400, detail="动态内容不能为空")

    # 自动推断媒体类型
    media_type = "text"
    if req.mediaUrls and len(req.mediaUrls) > 0:
        # 根据URL后缀判断是图片还是视频
        first_url = req.mediaUrls[0].lower()
        if any(first_url.endswith(ext) for ext in [".mp4", ".webm", ".mov"]):
            media_type = "video"
        else:
            media_type = "image"

    # 将媒体URL列表序列化为JSON字符串存储
    media_urls_json = json.dumps(req.mediaUrls, ensure_ascii=False) if req.mediaUrls else None

    # 使用连接池直接获取连接，确保事务正确提交
    async with db.pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO posts (user_id, content, media_urls, media_type) VALUES (%s, %s, %s, %s)",
                [req.userId, content, media_urls_json, media_type],
            )
            post_id = cur.lastrowid  # 获取新插入的动态ID

    logger.info(f"用户{req.userId}发布动态成功，post_id={post_id}，media_type={media_type}")
    return CreatePostResponse(success=True, postId=post_id, message="发布成功")


@router.get("/plaza/posts", response_model=PostListResponse)
async def get_posts(
    page: int = Query(1, ge=1),  # 页码，从1开始
    pageSize: int = Query(10, ge=1, le=50),  # 每页条数，默认10
    current_user_id: int = Depends(get_current_user),
) -> PostListResponse:
    """获取动态列表接口（需JWT认证）：分页查询广场动态

    按创建时间倒序排列，同时关联查询发布者信息和当前用户的点赞状态。
    """
    offset = (page - 1) * pageSize  # 计算偏移量

    # 查询动态列表，关联用户表获取昵称，关联点赞表判断当前用户是否点赞
    async with db.pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT p.id, p.user_id, p.content, p.media_urls, p.media_type,
                       p.like_count, p.comment_count, p.created_at,
                       u.nickname
                FROM posts p
                LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.created_at DESC
                LIMIT %s OFFSET %s
                """,
                [pageSize, offset],
            )
            rows = await cur.fetchall()

            # 查询当前用户对这批动态的点赞状态
            post_ids = [row["id"] for row in rows] if rows else []
            liked_post_ids = set()
            if post_ids:
                placeholders = ",".join(["%s"] * len(post_ids))
                await cur.execute(
                    f"SELECT post_id FROM post_likes WHERE user_id = %s AND post_id IN ({placeholders})",
                    [current_user_id] + post_ids,
                )
                liked_rows = await cur.fetchall()
                liked_post_ids = {r["post_id"] for r in liked_rows}

            # 查询总数
            await cur.execute("SELECT COUNT(*) as total FROM posts")
            total_row = await cur.fetchone()
            total = total_row["total"] if total_row else 0

    # 组装响应数据
    posts = []
    for row in rows:
        # 解析媒体URL列表
        media_urls = []
        if row.get("media_urls"):
            try:
                media_urls = json.loads(row["media_urls"])
            except (json.JSONDecodeError, TypeError):
                media_urls = []

        # 格式化时间
        created_at = row["created_at"]
        created_at_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else ""

        posts.append(PostItem(
            postId=row["id"],
            userId=row["user_id"],
            nickname=row.get("nickname", ""),
            content=row["content"],
            mediaUrls=media_urls,
            mediaType=row.get("media_type", "text"),
            likeCount=row.get("like_count", 0),
            commentCount=row.get("comment_count", 0),
            liked=row["id"] in liked_post_ids,  # 当前用户是否已点赞
            createdAt=created_at_str,
        ))

    return PostListResponse(posts=posts, total=total)


@router.post("/plaza/posts/{post_id}/like", response_model=LikeResponse)
async def toggle_like(post_id: int, current_user_id: int = Depends(get_current_user)) -> LikeResponse:
    """点赞/取消点赞接口（需JWT认证）：切换点赞状态

    如果当前用户未点赞 → 新增点赞记录，like_count+1
    如果当前用户已点赞 → 删除点赞记录，like_count-1
    """
    async with db.pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            # 检查是否已点赞
            await cur.execute(
                "SELECT id FROM post_likes WHERE post_id = %s AND user_id = %s",
                [post_id, current_user_id],
            )
            existing = await cur.fetchone()

            if existing:
                # 已点赞 → 取消点赞
                await cur.execute(
                    "DELETE FROM post_likes WHERE post_id = %s AND user_id = %s",
                    [post_id, current_user_id],
                )
                await cur.execute(
                    "UPDATE posts SET like_count = like_count - 1 WHERE id = %s AND like_count > 0",
                    [post_id],
                )
                liked = False
                logger.info(f"用户{current_user_id}取消点赞动态{post_id}")
            else:
                # 未点赞 → 新增点赞
                await cur.execute(
                    "INSERT INTO post_likes (post_id, user_id) VALUES (%s, %s)",
                    [post_id, current_user_id],
                )
                await cur.execute(
                    "UPDATE posts SET like_count = like_count + 1 WHERE id = %s",
                    [post_id],
                )
                liked = True
                logger.info(f"用户{current_user_id}点赞动态{post_id}")

            # 查询最新点赞数
            await cur.execute("SELECT like_count FROM posts WHERE id = %s", [post_id])
            row = await cur.fetchone()
            like_count = row["like_count"] if row else 0

    return LikeResponse(success=True, liked=liked, likeCount=like_count, message="操作成功")


@router.post("/plaza/posts/{post_id}/comments", response_model=CreateCommentResponse)
async def create_comment(
    post_id: int,
    req: CreateCommentRequest,
    current_user_id: int = Depends(get_current_user),
) -> CreateCommentResponse:
    """发表评论接口（需JWT认证）：用户对指定动态发表评论"""
    # 安全校验
    if current_user_id != req.userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")

    async with db.pool.acquire() as conn:
        async with conn.cursor() as cur:
            # 检查动态是否存在
            await cur.execute("SELECT id FROM posts WHERE id = %s", [post_id])
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="动态不存在")

            # 插入评论
            await cur.execute(
                "INSERT INTO post_comments (post_id, user_id, content) VALUES (%s, %s, %s)",
                [post_id, req.userId, req.content.strip()],
            )
            comment_id = cur.lastrowid

            # 更新评论计数
            await cur.execute(
                "UPDATE posts SET comment_count = comment_count + 1 WHERE id = %s",
                [post_id],
            )

    logger.info(f"用户{req.userId}评论动态{post_id}，comment_id={comment_id}")
    return CreateCommentResponse(success=True, commentId=comment_id, message="评论成功")


@router.get("/plaza/posts/{post_id}/comments", response_model=CommentListResponse)
async def get_comments(
    post_id: int,
    current_user_id: int = Depends(get_current_user),
) -> CommentListResponse:
    """获取评论列表接口（需JWT认证）：查询指定动态的所有评论"""
    async with db.pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT c.id, c.post_id, c.user_id, c.content, c.created_at,
                       u.nickname
                FROM post_comments c
                LEFT JOIN users u ON c.user_id = u.id
                WHERE c.post_id = %s
                ORDER BY c.created_at ASC
                """,
                [post_id],
            )
            rows = await cur.fetchall()

    # 组装评论列表
    comments = []
    for row in rows:
        created_at = row["created_at"]
        created_at_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else ""
        comments.append(CommentItem(
            commentId=row["id"],
            postId=row["post_id"],
            userId=row["user_id"],
            nickname=row.get("nickname", ""),
            content=row["content"],
            createdAt=created_at_str,
        ))

    return CommentListResponse(comments=comments, total=len(comments))
