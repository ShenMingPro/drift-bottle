"""轻量级文本检索与相关性排序基线。

当前实现使用确定性的字符 n-gram 特征哈希和余弦相似度，不依赖外部
embedding 服务。接口保持简单，后续可以替换为 pgvector/Qdrant 或真实
embedding provider，而不改变匹配业务调用。
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Sequence

VECTOR_DIMENSION = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """切分英文词和中文二元组，作为无外部依赖的检索特征。"""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        value = match.group(0)
        if value.isascii():
            tokens.append(value)
        elif len(value) == 1:
            tokens.append(value)
        else:
            tokens.extend(value[index:index + 2] for index in range(len(value) - 1))
    return tokens


def _bucket(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSION
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def embed_text(text: str) -> list[float]:
    """把文本映射到固定维度向量，输出 L2 归一化结果。"""
    counts = Counter(tokenize(text))
    vector = [0.0] * VECTOR_DIMENSION
    for token, count in counts.items():
        index, sign = _bucket(token)
        vector[index] += sign * (1.0 + math.log1p(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def text_similarity(left: str, right: str) -> float:
    """返回 0~1 的文本相关性；空文本视为不相关。"""
    if not (left or "").strip() or not (right or "").strip():
        return 0.0
    return max(0.0, cosine_similarity(embed_text(left), embed_text(right)))


def candidate_text(candidate: Any) -> str:
    """从字符串或字典候选里提取用于相关性比较的文本。"""
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return " ".join(
            str(candidate.get(key, "") or "")
            for key in ("text", "content", "hobbies")
        )
    return str(candidate)


def rank_candidates(
    query: str,
    candidates: Sequence[Any],
    *,
    top_k: int = 10,
) -> list[tuple[float, Any]]:
    """按文本相关性排序候选，返回 (相似度, 原始候选) 列表。"""
    if top_k <= 0:
        return []
    query_vector = embed_text(query)
    ranked = [
        (max(0.0, cosine_similarity(query_vector, embed_text(candidate_text(item)))), item)
        for item in candidates
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]
