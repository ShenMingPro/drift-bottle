"""轻量级文本检索和排序指标测试。"""

import pytest

from app.evals.ranking import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from app.evals.matching import evaluate_matching_cases
from app.services.retrieval import rank_candidates, text_similarity


def test_related_text_ranks_above_unrelated_text():
    query = "摄影 跑步"
    related = "最近喜欢摄影，周末会去跑步"
    unrelated = "今天篮球比赛输了"

    assert text_similarity(query, related) > text_similarity(query, unrelated)

    ranked = rank_candidates(
        query,
        [
            {"content": unrelated, "hobbies": "篮球"},
            {"content": related, "hobbies": "摄影,跑步"},
        ],
    )
    assert ranked[0][1]["hobbies"] == "摄影,跑步"


def test_retrieval_is_deterministic_and_respects_top_k():
    candidates = [f"候选内容 {index}" for index in range(5)]
    first = rank_candidates("候选内容", candidates, top_k=3)
    second = rank_candidates("候选内容", candidates, top_k=3)

    assert [item[1] for item in first] == [item[1] for item in second]
    assert len(first) == 3


def test_ranking_metrics():
    assert ndcg_at_k([1, 0, 1], 3) == pytest.approx(1.5 / (1 + 1 / 1.5849625), rel=1e-6)
    assert mean_reciprocal_rank([[0, 1, 0], [1, 0]]) == pytest.approx(0.75)
    assert recall_at_k([1, 0, 1, 0], 2, relevant_count=2) == pytest.approx(0.5)


def test_matching_eval_uses_retrieval_ranking():
    result = evaluate_matching_cases([
        {
            "query": "摄影 跑步",
            "relevant_ids": [2],
            "candidates": [
                {"id": 1, "content": "篮球比赛", "hobbies": "篮球"},
                {"id": 2, "content": "最近喜欢摄影", "hobbies": "摄影,跑步"},
            ],
        }
    ], k=2)

    assert result["ndcg_at_k"] > 0
    assert result["mrr"] == pytest.approx(1.0)
    assert result["recall_at_k"] == pytest.approx(1.0)
