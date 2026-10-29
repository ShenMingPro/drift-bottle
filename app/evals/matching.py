"""离线匹配评测：把检索服务输出转成排序指标。"""

from __future__ import annotations

from typing import Any, Sequence

from app.evals.ranking import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from app.services.retrieval import rank_candidates


def evaluate_matching_cases(
    cases: Sequence[dict[str, Any]],
    *,
    k: int = 10,
) -> dict[str, float]:
    """评估 query → 候选排序是否把 relevant_ids 排在前面。

    case 示例：
    {
      "query": "摄影 跑步",
      "relevant_ids": [2],
      "candidates": [{"id": 1, "content": "篮球"}, {"id": 2, "content": "摄影"}]
    }
    """
    relevance_rows: list[list[float]] = []
    for case in cases:
        relevant_ids = set(case.get("relevant_ids", []))
        ranked = rank_candidates(
            str(case.get("query", "")),
            list(case.get("candidates", [])),
            top_k=k,
        )
        relevance_rows.append([
            1.0 if item.get("id") in relevant_ids else 0.0
            for _score, item in ranked
        ])

    if not relevance_rows:
        return {"ndcg_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    ndcg_scores = []
    recall_scores = []
    for case, row in zip(cases, relevance_rows):
        relevant_count = len(set(case.get("relevant_ids", [])))
        ndcg_scores.append(ndcg_at_k(row, k))
        recall_scores.append(recall_at_k(row, k, relevant_count=relevant_count))

    return {
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores),
        "mrr": mean_reciprocal_rank(relevance_rows),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
    }
