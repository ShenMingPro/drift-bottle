"""离线匹配评测 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evals.matching import evaluate_matching_cases


def _default_cases_path() -> Path:
    return Path(__file__).resolve().parents[2] / "evals" / "matching_cases.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行匹配排序离线评测")
    parser.add_argument("--file", type=Path, default=_default_cases_path())
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-ndcg", type=float, default=None)
    parser.add_argument("--min-mrr", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.file.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    report = evaluate_matching_cases(cases, k=args.k)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    thresholds = [
        ("ndcg_at_k", args.min_ndcg),
        ("mrr", args.min_mrr),
        ("recall_at_k", args.min_recall),
    ]
    failed = [
        f"{metric}={report[metric]:.4f} < {minimum:.4f}"
        for metric, minimum in thresholds
        if minimum is not None and report[metric] < minimum
    ]
    if failed:
        print("EVAL_FAILED: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
