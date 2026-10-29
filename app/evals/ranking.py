"""排序评测指标：NDCG@K、MRR 和 Recall@K。"""

from __future__ import annotations

import math
from typing import Sequence


def ndcg_at_k(relevance: Sequence[float], k: int) -> float:
    """按二进制或分级相关性计算 NDCG@K。"""
    if k <= 0:
        return 0.0
    actual = list(relevance[:k])
    ideal = sorted(relevance, reverse=True)[:k]

    def dcg(values: Sequence[float]) -> float:
        return sum(
            (2 ** value - 1) / math.log2(index + 2)
            for index, value in enumerate(values)
        )

    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 0.0


def mean_reciprocal_rank(relevance: Sequence[Sequence[float]]) -> float:
    """计算多个查询的 MRR，相关性大于 0 视为相关。"""
    if not relevance:
        return 0.0
    reciprocal_ranks = []
    for row in relevance:
        rank = next((index + 1 for index, value in enumerate(row) if value > 0), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_k(relevance: Sequence[float], k: int, relevant_count: int | None = None) -> float:
    """计算 Recall@K；未提供相关总数时按整条序列中的正例数计算。"""
    if k <= 0:
        return 0.0
    total = relevant_count if relevant_count is not None else sum(
        1 for value in relevance if value > 0
    )
    if total <= 0:
        return 0.0
    return sum(1 for value in relevance[:k] if value > 0) / total
