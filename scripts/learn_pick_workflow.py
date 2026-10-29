r"""在项目根目录运行（无需启动服务器、MySQL 或模型）：
.\.venv\Scripts\python.exe scripts\learn_pick_workflow.py --case approve
.\.venv\Scripts\python.exe scripts\learn_pick_workflow.py --case cancel
.\.venv\Scripts\python.exe scripts\learn_pick_workflow.py --case fail
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# 直接运行 scripts 下的文件时，也能导入项目中的 app 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 学习演示也关闭可选的远程追踪，保证只在本机执行。
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import app.workflows.pick_workflow as workflow


def show(label, value):
    print(f"{label}：{json.dumps(value, ensure_ascii=False)}")


def simulated_agents(case):
    """只替换领域 Agent 的数据来源，图节点、条件边和审批逻辑仍用项目实现。"""
    async def match(user_id, university, city, province):
        return {
            "agent": "match", "status": "success",
            "best_match": {"bottle_id": 12, "user_id": 23, "nickname": "小林"},
            "match_reason": "同校且喜欢摄影", "match_level": "university",
        }

    async def profile(user_id, target_user_id):
        if case == "fail":
            return {
                "agent": "profile_analysis", "status": "failed",
                "error": "演示用画像服务错误", "error_code": "DEMO_FAILURE",
            }
        return {
            "agent": "profile_analysis", "status": "success",
            "tags": ["摄影"], "compatibility": "双方都喜欢校园摄影",
        }

    async def guide(user_id, target_user_id):
        return {
            "agent": "conversation_guide", "status": "success",
            "conversation_status": "new", "topic_suggestions": ["摄影"],
            "reply_strategies": ["先询问最近拍摄的校园景色"],
        }

    async def ice(user_id, target_user_id, analysis_context=None):
        show("模拟 IceBreak 收到的上游分析", analysis_context.model_dump(exclude_defaults=True))
        return {
            "agent": "ice_break", "status": "success", "scenario": "ice_break",
            "hints": ["最近拍过哪些校园景色？", "你更喜欢拍人像还是风景？", "联系方式：示例账号"],
        }

    async def guard(content, user_id):
        passed = "联系方式" not in content
        return {
            "agent": "guard", "status": "success", "passed": passed,
            "reason": "审核通过" if passed else "包含联系方式", "risk_level": "low",
        }

    return {
        "run_match": match, "run_profile_analysis": profile,
        "run_conversation_guide": guide, "run_ice_break": ice, "run_guard": guard,
    }


async def show_updates(graph, incoming, config):
    async for mode, update in graph.astream(incoming, config, stream_mode=["updates", "values"]):
        if mode == "values":
            show("合并后 State 中已有的字段", [key for key in update if key != "__interrupt__"])
            continue
        for node, values in update.items():
            if node == "__interrupt__":
                print("approval 调用 interrupt()：暂停等待用户确认；这不是执行失败。")
            else:
                show(f"节点 {node} 返回的状态更新", {k: v for k, v in values.items() if k != "trace"})


async def show_snapshot(graph, config):
    snapshot = await graph.aget_state(config)
    print(f"snapshot.next = {snapshot.next!r}")
    show("当前 status", snapshot.values["status"])
    show("trace 中已经完成的节点", [item["node"] for item in snapshot.values.get("trace", [])])
    return snapshot


async def demo(case):
    print("本演示运行项目的真实 LangGraph 图，领域 Agent 使用固定模拟数据。")
    print("不会调用外部模型或 MySQL；SQLite 检查点使用临时文件，退出后自动删除。")
    print("这里演示图执行，不调用 API，因此不会写拾取记录或扣次数。")
    config = {"configurable": {"thread_id": "learning-pick-run"}}
    initial = {
        "run_id": "learning-run", "user_id": 7, "university": "演示大学",
        "city": "演示市", "province": "演示省", "status": "running", "error": None, "trace": [],
    }
    show("输入状态", initial)
    show("检查点定位配置", config)
    with TemporaryDirectory(prefix="learn-pick-") as folder, ExitStack() as patches:
        for name, implementation in simulated_agents(case).items():
            patches.enter_context(patch.object(workflow, name, implementation))
        path = str(Path(folder) / "learning.sqlite")
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = workflow.build_pick_workflow().compile(checkpointer=saver)
            print("\n开始执行：每个节点返回部分状态，LangGraph 将它合并到共享状态。")
            await show_updates(graph, initial, config)
            snapshot = await show_snapshot(graph, config)
            if case == "fail":
                show("保留的错误", snapshot.values["error"])
                print("profile 失败 → 条件边选择 finish → END；后续 Agent 和 approval 均未执行。")
                return
            print("next 中有 approval，说明图正在等审批；图内 status 此时仍为 running。")
            print("项目 API 会将这个暂停快照展示为 awaiting_approval。")
            hint = snapshot.values["approved_hints"][0]
        print("\n已关闭 SQLite 连接；现在重开同一临时文件，并重新编译图，模拟服务重启。")
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = workflow.build_pick_workflow().compile(checkpointer=saver)
            await show_snapshot(graph, config)
            decision = {"approved": case == "approve", "hint": hint if case == "approve" else None}
            show("传给 Command(resume=...) 的人工决定", decision)
            await show_updates(graph, Command(resume=decision), config)
            final = await show_snapshot(graph, config)
            show("最终选择", final.values.get("selected_hint", ""))
            print("next 为空表示执行结束；恢复只继续 approval，没有重跑前面的 Agent。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="离线学习项目真实拾取状态图")
    parser.add_argument("--case", choices=["approve", "cancel", "fail"], default="approve")
    asyncio.run(demo(parser.parse_args().case))
