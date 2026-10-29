from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用尚未定义的类型

import asyncio  # 异步IO库，用于执行异步工具函数
import json  # JSON序列化/反序列化库，用于工具返回值和LLM响应解析
import random  # 随机数库，用于同用户多纸团时随机选择一条
import re  # 正则表达式库，用于清理LLM返回的Markdown代码块标记
import time  # 时间库，用于计算LLM调用和工具执行耗时
from typing import Annotated, Any, Optional, TypedDict  # 类型注解工具：Annotated添加元数据、Any任意类型、Optional可选类型、TypedDict定义带类型的字典

from app.agent_logger import get_logger, global_metrics  # 结构化日志工具
from app.services.retrieval import text_similarity

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # LangChain消息类型：AIMessage(LLM回复)、HumanMessage(用户输入)、SystemMessage(系统指令)、ToolMessage(工具执行结果)
from langchain_core.tools import tool  # LangChain工具装饰器，将函数标记为LLM可调用的工具
from langchain_openai import ChatOpenAI  # LangChain的OpenAI兼容LLM客户端，用于调用DeepSeek模型
from langgraph.graph import END, StateGraph  # LangGraph核心组件：END表示状态图终止节点、StateGraph用于构建有状态图
from langgraph.graph.message import add_messages  # LangGraph消息追加reducer，确保messages列表自动追加而非覆盖


# ============================================================
# MatchAgent 状态图流程图（ReAct模式 + Tool/Function Calling）（用户匹配）
# ============================================================
#
#   ┌────────────────────┐
#   │  init_conversation  │  初始化对话：构建SystemMessage + HumanMessage
#   └────────┬───────────┘
#            │
#            ▼
#   ┌────────────────────┐     Tool Calls     ┌──────────────────┐
#   │     agent_node      │ ─────────────────→ │  execute_tools   │
#   │  (LLM+Tool Calling) │ ←───────────────── │  (手动执行工具)  │
#   └────────┬───────────┘   Tool Results      └──────────────────┘
#            │
#            ├── 有Tool Calls ──→ execute_tools（执行工具后返回agent_node）
#            │
#            └── 无Tool Calls ──→ extract_result（解析匹配结果）
#                                     │
#                                     ▼
#                                   [END]
#
# 设计思路：
#   1. LLM作为"匹配顾问"，通过Tool Calling自主编排匹配流程
#   2. 两个核心工具：查询用户画像、查找最佳匹配（内含分层递进查询+评分+原因生成）
#   3. LLM决定调用顺序：先查画像→再查最佳匹配→返回结果
#   4. find_best_match工具采用分层递进查询策略（核心优化）：
#      L1同校→L2同城→L3同省→L4全国，逐层查询数据库
#      找到第一个非空层级即停止，在该层级内评分
#      相比一次性查所有纸团再筛选，确保：
#      - 地理位置优先级绝对高于其他条件
#      - 老纸团也能被匹配到（不会被LIMIT截断）
#      - 每层查询量小，性能更优
#   5. LLM利用语义理解能力对匹配结果做最终确认和润色
#
# ============================================================


class MatchState(TypedDict):
    """MatchAgent的状态定义，贯穿整个匹配流程图"""
    user_id: int  # 当前用户ID
    target_university: str  # 拾纸团时选择的目标学校
    target_city: str  # 拾纸团时选择的目标城市
    target_province: str  # 拾纸团时选择的目标省份
    reuse_profile: bool  # 是否将query_user_profile结果传给find_best_match
    messages: Annotated[list, add_messages]  # LLM对话消息列表（使用add_messages reducer自动追加）
    user_profile: Optional[dict]  # query_user_profile查询到的画像，供后续匹配工具直接复用
    profile_status: str  # 画像加载状态：pending/success/user_not_found/tool_error
    match_status: str  # 匹配状态：pending/success/no_candidates/user_not_found/profile_required/tool_error
    best_match: Optional[dict]  # 最佳匹配结果（包含候选者信息和纸团内容）
    match_reason: str  # 匹配原因描述
    match_score: float  # 匹配分数
    content_score: float  # 纸团文本与用户兴趣的相关性
    match_level: str  # 匹配层级（university/city/province/nationwide）


# 评分权重常量
WEIGHT_GENDER = 0.30  # 性别匹配权重：30%
WEIGHT_HOBBY = 0.30  # 爱好匹配权重：30%，Jaccard相似度
WEIGHT_AGE = 0.20  # 年龄相似度权重：20%
WEIGHT_CONTENT = 0.20  # 纸团内容与用户兴趣的文本相关性：20%


def _parse_hobbies(hobbies_str: str) -> list[str]:
    """将逗号分隔的爱好字符串解析为爱好列表"""
    if not hobbies_str:  # 如果爱好字符串为空
        return []  # 返回空列表
    return [h.strip() for h in hobbies_str.split(",") if h.strip()]  # 按逗号分割，去除空白，过滤空串


def _calc_gender_score(user_gender: str, candidate_gender: str) -> float:
    """计算性别匹配分数：异性=1.0，同性=0.5，未设置=0.3"""
    if not user_gender or not candidate_gender:  # 任一方性别未设置
        return 0.3  # 返回低分，表示性别信息不完整
    if user_gender != candidate_gender:  # 性别不同（异性）
        return 1.0  # 异性匹配得最高分
    return 0.5  # 同性匹配得中等分


def _calc_hobby_score(user_hobbies: list[str], candidate_hobbies: list[str]) -> float:
    """计算爱好匹配分数（Jaccard相似度）"""
    if not user_hobbies or not candidate_hobbies:  # 任一方爱好列表为空
        return 0.0  # 无法计算相似度，返回0分
    user_set = set(user_hobbies)  # 将用户爱好列表转为集合
    cand_set = set(candidate_hobbies)  # 将候选者爱好列表转为集合
    intersection = user_set & cand_set  # 计算交集（共同爱好）
    union = user_set | cand_set  # 计算并集（所有爱好）
    if not union:  # 如果并集为空（理论上不会发生，因为上面已检查）
        return 0.0  # 返回0分
    return len(intersection) / len(union)  # Jaccard相似度 = 交集大小 / 并集大小


def _calc_age_score(user_age: int, candidate_age: int) -> float:
    """计算年龄相似度分数"""
    if user_age <= 0 or candidate_age <= 0:  # 任一方年龄无效
        return 0.5  # 返回中等分数，表示年龄信息不完整
    diff = abs(user_age - candidate_age)  # 计算年龄差的绝对值
    return max(0.0, 1.0 - min(1.0, diff / 8.0))  # 年龄差越大分数越低，最大有效年龄差8岁，最低0分


def _calc_content_score(
    user_hobbies: str,
    candidate_content: str,
    candidate_hobbies: str,
) -> float:
    """用可替换的文本检索基线衡量内容相关性。"""
    query = user_hobbies
    candidate = f"{candidate_content} {candidate_hobbies}"
    return text_similarity(query, candidate)


def _build_match_reason(
    match_level: str,
    gender_score: float,
    hobby_score: float,
    age_score: float,
    content_score: float,
    user_hobbies: str,
    candidate_hobbies: str,
) -> str:
    """根据匹配评分详情生成人类可读的匹配原因描述"""
    reasons = []  # 收集所有匹配原因的列表
    level_labels = {  # 匹配层级到中文标签的映射
        "university": "同校匹配",
        "city": "同城匹配",
        "province": "同省匹配",
        "nationwide": "全国匹配",
    }
    if match_level in level_labels:  # 如果匹配层级在映射中
        reasons.append(level_labels[match_level])  # 添加层级标签作为第一个原因
    if gender_score >= 1.0:  # 性别得分为1.0（异性匹配）
        reasons.append("异性匹配")  # 添加异性匹配原因
    elif gender_score >= 0.5:  # 性别得分为0.5（同性匹配）
        reasons.append("同性匹配")  # 添加同性匹配原因
    user_h = set(_parse_hobbies(user_hobbies))  # 解析当前用户爱好为集合
    cand_h = set(_parse_hobbies(candidate_hobbies))  # 解析候选者爱好为集合
    common = user_h & cand_h  # 计算共同爱好
    if common:  # 如果有共同爱好
        reasons.append("共同爱好：" + "、".join(list(common)[:3]))  # 添加共同爱好（最多显示3个）
    if age_score >= 0.75:  # 年龄得分较高（年龄差<=2岁）
        reasons.append("年龄相近")  # 添加年龄相近原因
    if content_score >= 0.20:
        reasons.append("纸团内容与兴趣相关")
    if not reasons:  # 如果没有任何匹配原因
        reasons.append("缘分推荐")  # 使用默认的缘分推荐描述
    return "；".join(reasons)  # 用中文分号连接所有原因


# ============================================================
# Tool定义：将匹配能力封装为LLM可调用的工具
# ============================================================


def _normalise_user_profile(profile: Any, expected_user_id: int) -> Optional[dict[str, Any]]:
    """校验并统一画像字段，避免工具之间传递不完整或属于其他用户的数据。"""
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except json.JSONDecodeError:
            return None
    if not isinstance(profile, dict):
        return None

    try:
        profile_user_id = int(profile.get("user_id", 0))
    except (TypeError, ValueError):
        return None
    if profile_user_id != expected_user_id:
        return None

    try:
        age = int(profile.get("age") or 0)
    except (TypeError, ValueError):
        age = 0

    return {
        "user_id": profile_user_id,
        "nickname": str(profile.get("nickname") or ""),
        "gender": str(profile.get("gender") or ""),
        "age": age,
        "hobbies": str(profile.get("hobbies") or ""),
        "province": str(profile.get("province") or ""),
        "city": str(profile.get("city") or ""),
        "university": str(profile.get("university") or ""),
    }


async def _load_user_profile(user_id: int) -> Optional[dict[str, Any]]:
    """从数据库加载一次用户画像；图内后续节点复用返回值，不再重复查询。"""
    from app.database import pool

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT nickname, gender, age, hobbies, province, city, university FROM users WHERE id = %s",
                [user_id],
            )
            row = await cur.fetchone()

    if not row:
        return None
    return {
        "user_id": user_id,
        "nickname": row[0] or "",
        "gender": row[1] or "",
        "age": row[2] or 0,
        "hobbies": row[3] or "",
        "province": row[4] or "",
        "city": row[5] or "",
        "university": row[6] or "",
    }


def _empty_match_payload(result: str, reason: str = "") -> dict[str, Any]:
    """为所有未匹配结果提供一致的字段，方便上层稳定读取。"""
    return {
        "result": result,
        "best_match": None,
        "match_reason": reason,
        "match_score": 0.0,
        "match_level": "",
    }

@tool  # LangChain工具装饰器，将函数注册为LLM可通过Function Calling调用的工具
async def query_user_profile(user_id: int) -> str:
    """查询用户画像信息（昵称、性别、年龄、爱好、学校、城市、省份等）。
    在匹配流程开始时调用，获取当前用户的详细信息用于后续匹配计算。
    输入：用户ID
    输出：用户画像的JSON字符串"""
    profile = await _load_user_profile(user_id)
    if profile is None:
        # 保留原有的未找到提示格式，兼容已有调用方；图状态会单独记录失败状态。
        return "User not found: " + str(user_id)
    return json.dumps(profile, ensure_ascii=False)


# 分层递进查询的基础SQL模板（不包含层级条件，由_query_layer动态拼接）
_BASE_MATCH_SQL = (
    "SELECT b.id AS bottle_id, b.user_id, b.content, b.created_at, "  # 查询纸团ID、用户ID、内容、创建时间
    "u.nickname, u.gender, u.age, u.hobbies, u.contact_type, u.contact_value, "  # 查询用户昵称、性别、年龄、爱好、联系方式类型、联系方式值
    "u.province, u.city, u.university "  # 查询用户省份、城市、学校
    "FROM bottles b JOIN users u ON b.user_id = u.id "  # 纸团表与用户表内连接
    "WHERE b.user_id != %s "  # 排除自己写的纸团
    "AND b.user_id NOT IN ("  # 排除已拾取过的用户的所有纸团
    "  SELECT b2.user_id FROM pick_records pr JOIN bottles b2 ON pr.bottle_id = b2.id WHERE pr.user_id = %s"  # 子查询：查找当前用户已拾取过的纸团的作者
    ") "
)

# 每层查询的最大候选数量（足够评分使用，同时控制查询性能）
_LAYER_LIMIT = 200


async def _query_layer(cur, user_id: int, extra_where: str, params: list) -> list[dict]:
    """执行单层级的数据库查询，返回该层级的候选纸团列表

    参数：
        cur: 数据库游标
        user_id: 当前用户ID
        extra_where: 额外的WHERE条件（如 u.university = %s）
        params: 额外WHERE条件对应的参数值
    返回：
        该层级的候选纸团字典列表
    """
    sql = _BASE_MATCH_SQL  # 以基础SQL模板为起点
    if extra_where:  # 如果有额外的WHERE条件
        sql += " AND " + extra_where  # 拼接额外条件（如 u.university = %s）
    sql += " ORDER BY b.created_at DESC LIMIT " + str(_LAYER_LIMIT)  # 按创建时间降序排列，限制返回数量
    all_params = [user_id, user_id] + params  # 合并所有SQL参数：[当前用户ID, 当前用户ID, 额外参数...]
    await cur.execute(sql, all_params)  # 执行SQL查询
    rows = await cur.fetchall()  # 获取所有查询结果行

    candidates = []  # 候选纸团列表
    for row in rows:  # 遍历每一行查询结果
        candidates.append({  # 将每行数据转换为字典
            "bottle_id": row[0],  # 纸团ID
            "user_id": row[1],  # 纸团作者ID
            "content": row[2],  # 纸团内容
            "created_at": str(row[3]),  # 创建时间（转为字符串）
            "nickname": row[4] or "",  # 昵称
            "gender": row[5] or "",  # 性别
            "age": row[6] or 0,  # 年龄
            "hobbies": row[7] or "",  # 爱好
            "contact_type": row[8] or "",  # 联系方式类型
            "contact_value": row[9] or "",  # 联系方式值
            "province": row[10] or "",  # 省份
            "city": row[11] or "",  # 城市
            "university": row[12] or "",  # 学校
        })
    return candidates  # 返回该层级的候选纸团列表


@tool  # LangChain工具装饰器
async def find_best_match(
    user_id: int,
    university: str,
    city: str,
    province: str,
    user_profile: Optional[dict[str, Any]] = None,
) -> str:
    """查找最佳匹配纸团。此工具采用分层递进查询策略：
    L1同校→L2同城→L3同省→L4全国，逐层查询数据库，
    找到第一个非空层级即停止，在该层级内进行多维度评分。
    相比一次性查询所有纸团再筛选，分层递进查询确保：
    1. 地理位置优先级绝对高于其他条件
    2. 老纸团也能被匹配到（不会被LIMIT截断）
    3. 每层查询量小，性能更优
    输入：当前用户ID、目标学校、目标城市、目标省份；可选的user_profile为前一步
    query_user_profile返回的画像。图内调用应优先复用该画像，避免重复查询数据库。
    输出：最佳匹配结果的JSON（包含纸团信息、对方信息、评分和匹配原因）"""
    from app.database import pool  # 延迟导入数据库连接池

    # 正常图内流程由execute_tools注入画像；没有注入时保留一次兼容性查询。
    profile_reused = user_profile is not None
    profile = _normalise_user_profile(user_profile, user_id)
    if user_profile is not None and profile is None:
        return json.dumps(_empty_match_payload("user_not_found", "用户画像无效"), ensure_ascii=False)
    if profile is None:
        profile = await _load_user_profile(user_id)
    if profile is None:
        return json.dumps(_empty_match_payload("user_not_found", "用户不存在"), ensure_ascii=False)

    # 定义四个层级的查询条件（按优先级递进）
    layer_configs = [
        {"level": "university", "where": "u.university = %s", "params": [university], "label": university},  # L1同校：按学校名筛选
        {"level": "city", "where": "u.city = %s", "params": [city], "label": city},  # L2同城：按城市名筛选
        {"level": "province", "where": "u.province = %s", "params": [province], "label": province},  # L3同省：按省份名筛选
        {"level": "nationwide", "where": "", "params": [], "label": "全国"},  # L4全国：无额外筛选条件
    ]

    candidates = []  # 当前层级的候选纸团列表
    match_level = ""  # 匹配到的层级标识

    # 分层递进查询：逐层查询，找到第一个非空层级即停止
    async with pool.acquire() as conn:  # 从连接池获取数据库连接
        async with conn.cursor() as cur:  # 创建异步游标
            for config in layer_configs:  # 按优先级遍历四个层级配置
                # 跳过空条件的层级（如用户没选学校就跳过同校层）
                if config["level"] != "nationwide" and not config["label"]:  # 非全国层级且条件为空
                    continue  # 跳过该层级

                candidates = await _query_layer(cur, user_id, config["where"], config["params"])  # 执行该层级查询

                if candidates:  # 如果该层级有候选纸团
                    match_level = config["level"]  # 记录匹配层级
                    break  # 找到第一个非空层级，停止查询

    if not candidates:  # 如果所有层级都没有候选纸团
        return json.dumps(_empty_match_payload("no_candidates"), ensure_ascii=False)

    # 同用户去重：同一用户可能有多条纸团，随机选一条
    user_bottles: dict[int, list] = {}  # 用户ID到其纸团列表的映射
    for cand in candidates:  # 遍历所有候选纸团
        uid = cand["user_id"]  # 获取纸团作者ID
        if uid not in user_bottles:  # 如果该作者还没有记录
            user_bottles[uid] = []  # 初始化该作者的纸团列表
        user_bottles[uid].append(cand)  # 将纸团添加到该作者的列表中

    deduped = []  # 去重后的候选列表（每个用户只保留一条纸团）
    for uid, bottles in user_bottles.items():  # 遍历每个用户及其纸团列表
        deduped.append(random.choice(bottles))  # 随机选择该用户的一条纸团

    # 使用前一步查询得到的画像评分，不再重复查询users表。
    user_gender = profile.get("gender", "")
    user_age = profile.get("age", 0)
    user_hobbies_text = profile.get("hobbies", "")
    user_hobbies = _parse_hobbies(user_hobbies_text)

    # 多维度评分并排序
    scored = []  # 评分结果列表，每个元素为(候选者, 总分, 性别分, 爱好分, 年龄分)
    for cand in deduped:  # 遍历去重后的每个候选者
        cand_hobbies = _parse_hobbies(cand.get("hobbies", ""))  # 解析候选者爱好列表
        gender_score = _calc_gender_score(user_gender, cand.get("gender", ""))  # 计算性别匹配分
        hobby_score = _calc_hobby_score(user_hobbies, cand_hobbies)  # 计算爱好匹配分
        age_score = _calc_age_score(user_age, cand.get("age", 0))  # 计算年龄相似度分
        content_score = _calc_content_score(
            user_hobbies_text,
            str(cand.get("content", "")),
            str(cand.get("hobbies", "")),
        )
        total_score = (
            WEIGHT_GENDER * gender_score
            + WEIGHT_HOBBY * hobby_score
            + WEIGHT_AGE * age_score
            + WEIGHT_CONTENT * content_score
        )
        scored.append(
            (cand, total_score, gender_score, hobby_score, age_score, content_score)
        )

    scored.sort(key=lambda x: x[1], reverse=True)  # 按总分降序排序

    # 只取最佳匹配
    best = scored[0]  # 取得分最高的候选者
    best_cand, best_score, g_score, h_score, a_score, c_score = best

    # 生成匹配原因
    match_reason = _build_match_reason(
        match_level, g_score, h_score, a_score, c_score,
        user_hobbies_text, best_cand.get("hobbies", "")  # 当前用户爱好和候选者爱好（用于查找共同爱好）
    )

    # 返回精简结果
    result = {
        "result": "success",  # 匹配结果状态
        "best_match": {  # 最佳匹配的详细信息
            "bottle_id": best_cand["bottle_id"],  # 纸团ID
            "user_id": best_cand["user_id"],  # 纸团作者ID
            "content": best_cand["content"],  # 纸团内容
            "created_at": best_cand["created_at"],  # 创建时间
            "nickname": best_cand["nickname"],  # 昵称
            "gender": best_cand["gender"],  # 性别
            "age": best_cand["age"],  # 年龄
            "hobbies": best_cand["hobbies"],  # 爱好
            "contact_type": best_cand["contact_type"],  # 联系方式类型
            "contact_value": best_cand["contact_value"],  # 联系方式值
            "province": best_cand["province"],  # 省份
            "city": best_cand["city"],  # 城市
            "university": best_cand["university"],  # 学校
        },
        "match_score": round(best_score, 4),  # 匹配总分（保留4位小数）
        "content_score": round(c_score, 4),
        "match_level": match_level,  # 匹配层级
        "match_reason": match_reason,  # 匹配原因描述
        "profile_reused": profile_reused,  # 标记是否复用了前一步画像，便于观察工具协作
    }
    return json.dumps(result, ensure_ascii=False)  # 序列化为JSON字符串返回给LLM


# 工具列表和工具映射
match_tools = [query_user_profile, find_best_match]  # MatchAgent可用的工具列表
match_tool_map = {t.name: t for t in match_tools}  # 工具名到工具函数的映射字典，用于execute_tools中查找


async def execute_tools(state: MatchState) -> dict[str, Any]:
    """手动实现的Tool执行节点：执行LLM发出的Tool Calls并返回ToolMessage"""
    log = get_logger("match_agent")  # 获取结构化日志器
    last_message = state["messages"][-1]  # 获取消息列表中的最后一条消息（应为AIMessage）
    tool_calls = last_message.tool_calls  # 从AIMessage中提取Tool Calls列表
    tool_messages = []  # 收集工具执行结果的ToolMessage列表
    state_updates: dict[str, Any] = {}
    reuse_profile = state.get("reuse_profile", True)
    local_profile = _normalise_user_profile(state.get("user_profile"), state.get("user_id", 0))

    # 画像是匹配的前置资料；即使模型一次发出多个调用，也先执行它。
    ordered_calls = sorted(
        enumerate(tool_calls),
        key=lambda item: (item[1]["name"] != "query_user_profile", item[0]),
    )
    results_by_call_id: dict[str, str] = {}

    for _, tool_call in ordered_calls:  # 遍历每个Tool Call
        tool_name = tool_call["name"]  # 获取工具名称
        tool_args = dict(tool_call.get("args") or {})  # 获取工具参数字典
        tool_call_id = tool_call["id"]  # 获取Tool Call的唯一标识ID（用于关联ToolMessage）

        if tool_name in match_tool_map:  # 如果工具名在映射中
            try:
                # 画像来自当前图状态，优先级高于模型自行生成的同名参数。
                if tool_name == "find_best_match":
                    if reuse_profile and local_profile is not None:
                        tool_args["user_profile"] = local_profile
                    else:
                        # 旧版pick不复用画像；其他流程缺失画像时也由工具兼容性查询。
                        tool_args.pop("user_profile", None)
                t = match_tool_map[tool_name]  # 获取工具函数对象
                tool_start = time.time()  # 记录工具执行开始时间
                # 判断工具是否为异步函数：优先检查.coroutine属性（@tool装饰后的异步函数），否则检查.func
                if asyncio.iscoroutinefunction(t.coroutine) if hasattr(t, "coroutine") and t.coroutine else asyncio.iscoroutinefunction(t.func):
                    result = await t.ainvoke(tool_args)  # 异步调用工具函数
                else:
                    result = t.invoke(tool_args)  # 同步调用工具函数
                duration_ms = round((time.time() - tool_start) * 1000, 1)  # 计算工具执行耗时（毫秒）
                log.info("tool_executed", tool_name=tool_name, duration_ms=duration_ms)  # 记录工具执行成功日志
                if isinstance(result, (dict, list)):
                    payload = result
                    result_text = json.dumps(result, ensure_ascii=False)
                else:
                    result_text = str(result)
                    try:
                        payload = json.loads(result_text)
                    except (TypeError, json.JSONDecodeError):
                        payload = None
                results_by_call_id[tool_call_id] = result_text

                if tool_name == "query_user_profile":
                    profile = _normalise_user_profile(payload, state.get("user_id", 0))
                    if profile is not None:
                        local_profile = profile
                        state_updates.update({"user_profile": profile, "profile_status": "success"})
                    else:
                        raise ValueError("无法获取有效的匹配用户画像")
                elif tool_name == "find_best_match" and isinstance(payload, dict):
                    if payload.get("result") not in {"success", "no_candidates"}:
                        raise ValueError("匹配工具未返回有效结果")
                    if payload.get("result") == "success" and not isinstance(payload.get("best_match"), dict):
                        raise ValueError("成功的匹配结果缺少候选纸团")
                    try:
                        score = float(payload.get("match_score", 0) or 0)
                    except (TypeError, ValueError):
                        score = 0.0
                    state_updates.update({
                        "best_match": payload.get("best_match"),
                        "match_reason": str(payload.get("match_reason", "")),
                        "match_score": score,
                        "content_score": float(payload.get("content_score", 0) or 0),
                        "match_level": str(payload.get("match_level", "")),
                        "match_status": str(payload.get("result", "tool_error")),
                    })
                elif tool_name == "find_best_match":
                    raise ValueError("匹配工具返回格式无效")
            except Exception as e:  # 工具执行出错
                log.error("tool_error", tool_name=tool_name, error=str(e))  # 记录工具执行错误日志
                raise
        else:  # 工具名不在映射中（未知工具）
            raise ValueError("匹配 Agent 调用了未知工具")

    # 虽然内部按依赖顺序执行，仍按模型原始调用顺序返回ToolMessage。
    tool_messages = [
        ToolMessage(content=results_by_call_id.get(call["id"], ""), tool_call_id=call["id"])
        for call in tool_calls
    ]
    return {"messages": tool_messages, **state_updates}


# LLM系统提示词（中文版）
MATCH_SYSTEM_PROMPT: str = """你是一个社交匹配助手。你的唯一任务是调用工具查找匹配，然后将工具返回的结果原样转为JSON输出。

可用工具：
1. query_user_profile：查询用户画像信息
2. find_best_match：查找最佳匹配纸团

工作流程：
1. 调用query_user_profile获取用户信息
2. 调用find_best_match查找最佳匹配
3. 将find_best_match返回的结果直接转为JSON输出

⚠️ 严格的输出格式要求 ⚠️
你的最终回复必须且只能是一个纯JSON对象，不得包含任何其他内容。
禁止使用Markdown格式、禁止使用表格、禁止使用代码块标记、禁止添加任何解释性文字。

有匹配时，直接输出以下JSON（字段值从find_best_match工具的返回值中复制）：
{"best_match":{"bottle_id":数字,"user_id":数字,"content":"字符串","created_at":"字符串","nickname":"字符串","gender":"字符串","age":数字,"hobbies":"字符串","contact_type":"字符串","contact_value":"字符串","province":"字符串","city":"字符串","university":"字符串"},"match_reason":"字符串","match_score":数字,"match_level":"字符串"}

没有候选纸团时：
{"best_match":null,"match_reason":"","match_score":0,"match_level":""}

再次强调：只输出JSON，不要输出任何其他内容。"""


def _get_llm() -> ChatOpenAI:
    """获取DeepSeek LLM实例"""
    from app.config import settings  # 延迟导入配置模块，避免循环导入

    return ChatOpenAI(  # 创建ChatOpenAI实例
        model="deepseek-chat",  # 使用DeepSeek-Chat模型
        api_key=settings.DEEPSEEK_API_KEY,  # DeepSeek API密钥
        base_url=settings.DEEPSEEK_BASE_URL,  # DeepSeek API基础URL
        timeout=settings.AGENT_CALL_TIMEOUT_SECONDS,
        max_retries=0,
        temperature=0.1,  # 温度参数：低温度确保输出稳定、确定性高
        max_tokens=512,  # 最大生成token数：限制输出长度，节省成本
    )


def init_conversation(state: MatchState) -> dict[str, Any]:
    """状态图节点1：初始化LLM对话，构建SystemMessage和HumanMessage"""
    user_id = state.get("user_id", 0)  # 从状态中获取当前用户ID，默认0
    university = state.get("target_university", "")  # 从状态中获取目标学校，默认空
    city = state.get("target_city", "")  # 从状态中获取目标城市，默认空
    province = state.get("target_province", "")  # 从状态中获取目标省份，默认空

    messages = [  # 构建初始消息列表
        SystemMessage(content=MATCH_SYSTEM_PROMPT),  # 系统消息：定义LLM角色和可用工具
        HumanMessage(content=(  # 用户消息：描述匹配请求
            f"请帮我找一个匹配的纸团。\n"  # 请求LLM执行匹配
            f"我的用户ID：{user_id}\n"  # 提供用户ID
            f"目标学校：{university or '不限'}\n"  # 提供目标学校，空则显示"不限"
            f"目标城市：{city or '不限'}\n"  # 提供目标城市
            f"目标省份：{province or '不限'}\n\n"  # 提供目标省份
            f"请先查询我的画像信息，然后查找最佳匹配。"  # 指导LLM的工作流程
        )),
    ]
    return {
        "messages": messages,
        "reuse_profile": state.get("reuse_profile", True),
        "user_profile": state.get("user_profile"),
        "profile_status": state.get("profile_status", "pending"),
        "match_status": state.get("match_status", "pending"),
    }  # 返回初始消息列表，add_messages reducer会追加到状态中


async def agent_node(state: MatchState) -> dict[str, Any]:
    """状态图节点2：LLM推理节点，使用Function Calling自主编排匹配流程"""
    log = get_logger("match_agent")  # 获取结构化日志器
    if "tool_error" in (state.get("profile_status"), state.get("match_status")):
        raise ValueError("匹配所需的工具执行失败")
    user_id = state.get("user_id", 0)  # 从状态中获取当前用户ID
    start = time.time()  # 记录LLM调用开始时间
    llm = _get_llm()  # 获取LLM实例
    llm_with_tools = llm.bind_tools(match_tools)  # 将工具列表绑定到LLM，启用Function Calling
    response = await llm_with_tools.ainvoke(state["messages"])  # 异步调用LLM，传入当前对话历史
    duration_ms = round((time.time() - start) * 1000, 1)  # 计算LLM调用耗时（毫秒）
    tokens = getattr(response, "usage_metadata", {}) or {}  # 从响应中提取Token使用量元数据
    total_tokens = tokens.get("total_tokens", 0)  # 获取总Token消耗数
    global_metrics.record_llm_call(tokens=total_tokens, duration_ms=duration_ms)  # 记录到全局指标统计
    has_tool_calls = bool(response.tool_calls)  # 判断LLM是否发起了工具调用
    log.info("agent_node", user_id=user_id, duration_ms=duration_ms, total_tokens=total_tokens, has_tool_calls=has_tool_calls)  # 记录LLM调用日志
    return {"messages": [response]}  # 返回LLM的回复消息，add_messages reducer会追加到消息列表


def should_use_tools(state: MatchState) -> str:
    """条件边判断函数：检查LLM是否需要调用工具"""
    last_message = state["messages"][-1]  # 获取消息列表中的最后一条消息
    if isinstance(last_message, AIMessage) and last_message.tool_calls:  # 如果是AIMessage且包含Tool Calls
        return "tools"  # 路由到工具执行节点
    return "extract_result"  # 否则路由到结果提取节点


def extract_result(state: MatchState) -> dict[str, Any]:
    """状态图节点3：从LLM最终回复或ToolMessage中提取匹配结果

    优先从LLM的JSON回复中提取；如果LLM返回了非标准格式（如Markdown表格），
    则回退到从find_best_match的ToolMessage中提取匹配结果。
    """
    log = get_logger("match_agent")  # 获取结构化日志器
    if "tool_error" in (state.get("profile_status"), state.get("match_status")):
        raise ValueError("匹配所需的工具执行失败")
    # 工具节点已经把结构化结果写入共享状态时，优先使用状态，避免再次从模型文本猜字段。
    if state.get("match_status") in {"user_not_found", "profile_required"}:
        raise ValueError("匹配所需的用户画像不可用")
    if state.get("match_status") in {"success", "no_candidates"}:
        return {
            "best_match": state.get("best_match"),
            "match_reason": state.get("match_reason", ""),
            "match_score": float(state.get("match_score", 0) or 0),
            "content_score": float(state.get("content_score", 0) or 0),
            "match_level": state.get("match_level", ""),
        }
    last_message = state["messages"][-1]  # 获取消息列表中的最后一条消息
    if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
        response_text = last_message.content.strip()  # 获取消息文本内容并去除首尾空白
        if response_text.startswith("```"):  # 如果LLM返回了Markdown代码块格式
            response_text = re.sub(r"^```\w*\n?", "", response_text)  # 移除开头的```和语言标识
            response_text = re.sub(r"\n?```$", "", response_text)  # 移除结尾的```
        try:
            result = json.loads(response_text)  # 尝试解析JSON
            if not isinstance(result, dict):
                raise ValueError("匹配结果必须是 JSON 对象")
            best_match = result.get("best_match")  # 提取最佳匹配信息
            if best_match is not None:  # 如果LLM返回了有效的best_match
                match_score = float(result.get("match_score", 0))  # 提取匹配分数
                match_level = str(result.get("match_level", ""))  # 提取匹配层级
                log.info("extract_result", user_id=state.get("user_id", 0), match_score=match_score, match_level=match_level)  # 记录成功解析日志
                return {  # 返回提取的结构化结果
                    "best_match": best_match,  # 最佳匹配详情
                    "match_reason": str(result.get("match_reason", "")),  # 匹配原因
                    "match_score": match_score,  # 匹配分数
                    "content_score": float(result.get("content_score", 0) or 0),
                    "match_level": match_level,  # 匹配层级
                }
        except (json.JSONDecodeError, KeyError, ValueError):  # JSON解析失败
            pass  # 跳过，尝试从ToolMessage中提取

    # 回退策略：从find_best_match的ToolMessage中直接提取匹配结果
    # 当LLM返回非JSON格式（如Markdown表格）时，find_best_match工具已经返回了正确的JSON
    for msg in reversed(state["messages"]):  # 从后往前遍历消息列表
        if isinstance(msg, ToolMessage):  # 如果是ToolMessage
            try:
                tool_data = json.loads(msg.content)  # 尝试解析ToolMessage的内容为JSON
                if tool_data.get("result") == "success" and tool_data.get("best_match"):  # 如果是find_best_match的成功结果
                    best_match = tool_data["best_match"]  # 提取最佳匹配信息
                    match_score = float(tool_data.get("match_score", 0))  # 提取匹配分数
                    match_level = str(tool_data.get("match_level", ""))  # 提取匹配层级
                    log.info("extract_result", user_id=state.get("user_id", 0), match_score=match_score, match_level=match_level)  # 记录成功解析日志（回退策略）
                    return {  # 返回从ToolMessage提取的结果
                        "best_match": best_match,  # 最佳匹配详情
                        "match_reason": str(tool_data.get("match_reason", "")),  # 匹配原因
                        "match_score": match_score,  # 匹配分数
                        "content_score": float(tool_data.get("content_score", 0) or 0),
                        "match_level": match_level,  # 匹配层级
                    }
            except (json.JSONDecodeError, KeyError, ValueError):  # 解析失败
                continue  # 继续查找下一个ToolMessage

    # 检查是否有no_candidates结果
    for msg in reversed(state["messages"]):  # 从后往前遍历消息列表
        if isinstance(msg, ToolMessage):  # 如果是ToolMessage
            try:
                tool_data = json.loads(msg.content)  # 尝试解析ToolMessage的内容为JSON
                if tool_data.get("result") == "no_candidates":  # 如果find_best_match返回无候选
                    return {  # 返回无匹配结果
                        "best_match": None,  # 无匹配
                        "match_reason": "暂时没有纸团",  # 原因描述
                        "match_score": 0.0,  # 零分
                        "content_score": 0.0,
                        "match_level": "",  # 空层级
                    }
            except (json.JSONDecodeError, KeyError, ValueError):  # 解析失败
                continue  # 继续查找

    log.warning("extract_result_parse_failed", user_id=state.get("user_id", 0))  # 记录解析失败警告日志
    raise ValueError("匹配流程没有返回有效的工具结果")


def build_match_graph() -> StateGraph:
    """构建MatchAgent的LangGraph状态图（ReAct模式 + Tool/Function Calling）"""
    graph = StateGraph(MatchState)  # 创建状态图实例，指定状态类型为MatchState

    graph.add_node("init_conversation", init_conversation)  # 添加节点1：初始化对话
    graph.add_node("agent", agent_node)  # 添加节点2：LLM推理
    graph.add_node("tools", execute_tools)  # 添加节点3：工具执行
    graph.add_node("extract_result", extract_result)  # 添加节点4：结果提取

    graph.set_entry_point("init_conversation")  # 设置入口节点为init_conversation
    graph.add_edge("init_conversation", "agent")  # 添加边：初始化对话 → LLM推理
    graph.add_conditional_edges("agent", should_use_tools, {  # 添加条件边：LLM推理后根据条件路由
        "tools": "tools",  # 如果需要调用工具 → 工具执行节点
        "extract_result": "extract_result",  # 如果不需要调用工具 → 结果提取节点
    })
    graph.add_edge("tools", "agent")  # 添加边：工具执行 → LLM推理（形成ReAct循环）
    graph.add_edge("extract_result", END)  # 添加边：结果提取 → 终止节点

    return graph  # 返回构建完成的状态图


match_graph = build_match_graph()  # 构建MatchAgent的状态图
match_app = match_graph.compile()  # 编译状态图为可执行的应用
