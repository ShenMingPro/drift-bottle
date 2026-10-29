"""广场模块数据模型：定义动态、评论的请求和响应模型"""

from pydantic import BaseModel, Field


class CreatePostRequest(BaseModel):
    """发布动态请求模型"""
    userId: int  # 发布者用户ID
    content: str = Field(min_length=1, max_length=500)  # 动态文字内容（1-500字）
    mediaUrls: list[str] = Field(default_factory=list)  # 媒体URL列表（图片/视频链接）
    mediaType: str = "text"  # 媒体类型：text/image/video


class CreatePostResponse(BaseModel):
    """发布动态响应模型"""
    success: bool  # 是否成功
    postId: int = 0  # 新创建的动态ID
    message: str = ""  # 提示信息


class PostItem(BaseModel):
    """动态列表项模型（单条动态的完整信息）"""
    postId: int  # 动态ID
    userId: int  # 发布者ID
    nickname: str = ""  # 发布者昵称
    avatar: str = ""  # 发布者头像（预留）
    content: str  # 动态文字内容
    mediaUrls: list[str] = []  # 媒体URL列表
    mediaType: str = "text"  # 媒体类型
    likeCount: int = 0  # 点赞数
    commentCount: int = 0  # 评论数
    liked: bool = False  # 当前用户是否已点赞
    createdAt: str = ""  # 创建时间


class PostListResponse(BaseModel):
    """动态列表响应模型"""
    posts: list[PostItem] = []  # 动态列表
    total: int = 0  # 动态总数


class CommentItem(BaseModel):
    """评论项模型"""
    commentId: int  # 评论ID
    postId: int  # 动态ID
    userId: int  # 评论者ID
    nickname: str = ""  # 评论者昵称
    content: str  # 评论内容
    createdAt: str = ""  # 评论时间


class CommentListResponse(BaseModel):
    """评论列表响应模型"""
    comments: list[CommentItem] = []  # 评论列表
    total: int = 0  # 评论总数


class CreateCommentRequest(BaseModel):
    """发表评论请求模型"""
    userId: int  # 评论者ID
    content: str = Field(min_length=1, max_length=200)  # 评论内容（1-200字）


class CreateCommentResponse(BaseModel):
    """发表评论响应模型"""
    success: bool  # 是否成功
    commentId: int = 0  # 新创建的评论ID
    message: str = ""  # 提示信息


class LikeResponse(BaseModel):
    """点赞响应模型"""
    success: bool  # 是否成功
    liked: bool = False  # 当前状态（true=已点赞 false=已取消）
    likeCount: int = 0  # 最新点赞数
    message: str = ""  # 提示信息
