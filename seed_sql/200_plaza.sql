-- 广场模块数据库表结构
-- 包含：动态表、点赞表、评论表

-- 动态表：用户发布的动态（文字/图片/视频）
CREATE TABLE IF NOT EXISTS posts (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '动态ID',
    user_id INT NOT NULL COMMENT '发布者用户ID',
    content TEXT NOT NULL COMMENT '动态文字内容',
    media_urls TEXT DEFAULT NULL COMMENT '媒体文件URL列表，JSON数组格式（图片/视频）',
    media_type VARCHAR(20) DEFAULT 'text' COMMENT '媒体类型：text纯文字/image图片/video视频',
    like_count INT DEFAULT 0 COMMENT '点赞数（冗余字段，加速查询）',
    comment_count INT DEFAULT 0 COMMENT '评论数（冗余字段，加速查询）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_user_id (user_id) COMMENT '用户ID索引',
    INDEX idx_created_at (created_at DESC) COMMENT '创建时间倒序索引（按时间查询动态）'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='广场动态表';

-- 点赞表：记录用户对动态的点赞
CREATE TABLE IF NOT EXISTS post_likes (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '点赞记录ID',
    post_id INT NOT NULL COMMENT '动态ID',
    user_id INT NOT NULL COMMENT '点赞用户ID',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '点赞时间',
    UNIQUE KEY uk_post_user (post_id, user_id) COMMENT '联合唯一键：一个用户对一条动态只能点赞一次',
    INDEX idx_post_id (post_id) COMMENT '动态ID索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='动态点赞表';

-- 评论表：记录用户对动态的评论
CREATE TABLE IF NOT EXISTS post_comments (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '评论ID',
    post_id INT NOT NULL COMMENT '动态ID',
    user_id INT NOT NULL COMMENT '评论用户ID',
    content TEXT NOT NULL COMMENT '评论内容',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '评论时间',
    INDEX idx_post_id (post_id) COMMENT '动态ID索引',
    INDEX idx_created_at (created_at DESC) COMMENT '创建时间索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='动态评论表';
