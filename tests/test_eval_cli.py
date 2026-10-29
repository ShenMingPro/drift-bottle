"""匹配评测 CLI 和阈值门禁测试。"""

import json

from app.evals.cli import main


def test_eval_cli_outputs_metrics(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({
            "cases": [
                {
                    "query": "摄影",
                    "relevant_ids": [2],
                    "candidates": [
                        {"id": 1, "content": "篮球"},
                        {"id": 2, "content": "喜欢摄影"},
                    ],
                }
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(["--file", str(path), "--k", "2"])
    output = capsys.readouterr().out
    report = json.loads(output)

    assert code == 0
    assert report["ndcg_at_k"] > 0
    assert report["mrr"] == 1.0


def test_eval_cli_returns_nonzero_when_threshold_fails(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({
            "cases": [
                {
                    "query": "摄影",
                    "relevant_ids": [99],
                    "candidates": [{"id": 1, "content": "摄影"}],
                }
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(["--file", str(path), "--min-mrr", "0.5"])
    assert code == 1
    assert "EVAL_FAILED" in capsys.readouterr().out
