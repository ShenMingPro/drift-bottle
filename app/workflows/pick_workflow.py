"""可持久化、可人工审批的拾取工作流。

这里不再让 Supervisor 通过 Prompt 决定下一步，而是把固定的业务步骤画成
显式 LangGraph。Match、Profile、ConversationGuide、IceBreak 和 Guard 仍然是
原有的领域 Agent，只由工作流负责编排、暂停和恢复。
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import time
from functools import wraps
from typing import Annotated, Any, Callable, TypedDict

from pydantic import ValidationError
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents.supervisor_agent import (
    call_conversation_guide_agent,
    call_guard_agent,
    call_ice_break_agent,
    call_match_agent,
    call_profile_analysis_agent,
)
from app.agents.workflow_contracts import (
    ConversationArtifact,
    IceBreakArtifact,
    IceBreakAnalysisContext,
    MatchArtifact,
    ProfileArtifact,
)
from app.config import settings


logger = logging.getLogger(__name__)


class PickWorkflowState(TypedDict, total=False):
    """一次拾取任务的完整状态。

    status 的取值：
      - running: 正在执行
      - awaiting_approval: 已生成建议，等待用户确认
      - approved: 用户已确认
      - canceled: 用户放弃本次结果
      - rejected: 审核未通过
      - no_result: 没有可用纸团
      - failed: 工作流内部失败
    """

    run_id: str
    user_id: int
    university: str
    city: str
    province: str
    status: str
    error: str | None

    match: dict[str, Any]
    profile_analysis: dict[str, Any]
    conversation_guide: dict[str, Any]
    ice_break: dict[str, Any]
    guard: dict[str, Any]

    approved_hints: list[str]
    selected_hint: str
    approval: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], operator.add]


def traced(node_name: str):
    """记录节点状态和耗时，方便 API/前端展示 Agent 运行轨迹。"""
    def decorator(node: Callable):
        @wraps(node)
        async def wrapper(state: PickWorkflowState) -> dict[str, Any]:
            started = time.perf_counter()
            result = await node(state)
            result.setdefault("trace", [])
            result["trace"].append(
                {
                    "node": node_name,
                    "status": result.get("status", "unknown"),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            return result
        return wrapper
    return decorator


def _decode_payload(value: Any) -> dict[str, Any]:
    """把工具返回的 JSON 字符串转成字典，并拒绝非结构化结果。"""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("领域 Agent 返回了无效的 JSON") from exc
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("领域 Agent 返回了无法识别的结果")


def _validate_artifact(payload: dict[str, Any], model: type) -> dict[str, Any]:
    """在状态边界执行运行时 Pydantic 校验，而不是只依赖 Prompt。"""
    try:
        # match 工具为了兼容旧调用方多返回了一个 result 字段；
        # 它不属于共享契约，校验前显式移除。
        clean_payload = {
            key: value for key, value in payload.items() if key != "result"
        }
        return model.model_validate(clean_payload).model_dump(
            mode="json",
            exclude_defaults=True,
        )
    except ValidationError as exc:
        raise ValueError(f"{model.__name__} 校验失败：{exc.errors()[0].get('msg', '字段无效')}") from exc


async def _invoke_tool(tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """统一调用 LangChain Tool，兼容同步和异步实现。"""
    async def invoke():
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(arguments)
        return await asyncio.to_thread(tool.invoke, arguments)

    value = await asyncio.wait_for(invoke(), timeout=settings.AGENT_CALL_TIMEOUT_SECONDS)
    return _decode_payload(value)


async def run_match(
    user_id: int,
    university: str,
    city: str,
    province: str,
) -> dict[str, Any]:
    """封装匹配 Agent，方便单元测试替换真实模型。"""
    return await _invoke_tool(
        call_match_agent,
        {
            "user_id": user_id,
            "university": university,
            "city": city,
            "province": province,
        },
    )


async def run_profile_analysis(user_id: int, target_user_id: int) -> dict[str, Any]:
    return await _invoke_tool(
        call_profile_analysis_agent,
        {"user_id": user_id, "target_user_id": target_user_id},
    )


async def run_conversation_guide(user_id: int, target_user_id: int) -> dict[str, Any]:
    return await _invoke_tool(
        call_conversation_guide_agent,
        {"user_id": user_id, "target_user_id": target_user_id},
    )


async def run_ice_break(
    user_id: int,
    target_user_id: int,
    analysis_context: IceBreakAnalysisContext | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"user_id": user_id, "target_user_id": target_user_id}
    if analysis_context is not None:
        arguments["analysis_context"] = analysis_context.model_dump(exclude_defaults=True)
    return await _invoke_tool(
        call_ice_break_agent,
        arguments,
    )


async def run_guard(content: str, user_id: int) -> dict[str, Any]:
    return await _invoke_tool(
        call_guard_agent,
        {
            "content": content,
            "content_type": "message",
            "user_id": user_id,
        },
    )


def _public_match(payload: dict[str, Any]) -> dict[str, Any]:
    """从工作流快照中移除不必要的联系方式。

    联系方式的展示属于单独的授权产品逻辑，不应默认进入 Agent 状态，
    也不应该被写入审批页面的响应。
    """
    public = dict(payload)
    best_match = public.get("best_match")
    if isinstance(best_match, dict):
        safe_match = dict(best_match)
        safe_match.pop("contact_type", None)
        safe_match.pop("contact_value", None)
        public["best_match"] = safe_match
    return public


def _target_user_id(state: PickWorkflowState) -> int:
    match = state.get("match") or {}
    best_match = match.get("best_match") or {}
    try:
        return int(best_match.get("user_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


async def match_node(state: PickWorkflowState) -> dict[str, Any]:
    try:
        payload = await run_match(
            user_id=state["user_id"],
            university=state.get("university", ""),
            city=state.get("city", ""),
            province=state.get("province", ""),
        )
    except Exception:
        logger.exception("pick_match_failed run_id=%s", state.get("run_id"))
        return {"status": "failed", "error": "匹配服务暂时不可用，请稍后重试"}

    payload = _public_match(payload)
    try:
        payload = _validate_artifact(payload, MatchArtifact)
    except ValueError as exc:
        return {"status": "failed", "error": f"匹配结果契约无效：{exc}"}
    if payload.get("status") == "no_result":
        return {"match": payload, "status": "no_result", "error": None}
    if payload.get("status") != "success":
        return {
            "match": payload, "status": "failed",
            "error": f"匹配失败：{payload.get('error') or payload['status']}",
        }
    if _target_user_id({"match": payload}) <= 0:
        return {
            "match": payload,
            "status": "failed",
            "error": "匹配结果缺少目标用户",
        }
    return {"match": payload, "status": "running", "error": None}


async def profile_node(state: PickWorkflowState) -> dict[str, Any]:
    target_user_id = _target_user_id(state)
    try:
        payload = await run_profile_analysis(state["user_id"], target_user_id)
    except Exception:
        logger.exception("pick_profile_failed run_id=%s", state.get("run_id"))
        return {"status": "failed", "error": "画像分析服务暂时不可用，请稍后重试"}
    try:
        payload = _validate_artifact(payload, ProfileArtifact)
    except ValueError as exc:
        return {"status": "failed", "error": f"画像分析契约无效：{exc}"}
    if payload.get("status") != "success":
        return {
            "profile_analysis": payload, "status": "failed",
            "error": f"画像分析失败：{payload.get('error') or payload['status']}",
        }
    return {"profile_analysis": payload, "status": "running"}


async def guide_node(state: PickWorkflowState) -> dict[str, Any]:
    target_user_id = _target_user_id(state)
    try:
        payload = await run_conversation_guide(state["user_id"], target_user_id)
    except Exception:
        logger.exception("pick_guide_failed run_id=%s", state.get("run_id"))
        return {"status": "failed", "error": "对话分析服务暂时不可用，请稍后重试"}
    try:
        payload = _validate_artifact(payload, ConversationArtifact)
    except ValueError as exc:
        return {"status": "failed", "error": f"对话分析契约无效：{exc}"}
    if payload.get("status") != "success":
        return {
            "conversation_guide": payload, "status": "failed",
            "error": f"对话分析失败：{payload.get('error') or payload['status']}",
        }
    return {"conversation_guide": payload, "status": "running"}


async def ice_break_node(state: PickWorkflowState) -> dict[str, Any]:
    target_user_id = _target_user_id(state)
    try:
        context = IceBreakAnalysisContext.from_artifacts(
            state.get("profile_analysis"), state.get("conversation_guide")
        )
        payload = await run_ice_break(state["user_id"], target_user_id, analysis_context=context)
    except Exception:
        logger.exception("pick_ice_break_failed run_id=%s", state.get("run_id"))
        return {"status": "failed", "error": "破冰建议服务暂时不可用，请稍后重试"}
    try:
        payload = _validate_artifact(payload, IceBreakArtifact)
    except ValueError as exc:
        return {"status": "failed", "error": f"破冰建议契约无效：{exc}"}
    if payload.get("status") != "success":
        return {
            "ice_break": payload, "status": "failed",
            "error": f"破冰建议生成失败：{payload.get('error') or payload['status']}",
        }

    hints = payload.get("hints")
    if not isinstance(hints, list) or not hints:
        return {"ice_break": payload, "status": "failed", "error": "没有生成可展示的破冰建议"}
    return {"ice_break": payload, "status": "running"}


async def guard_node(state: PickWorkflowState) -> dict[str, Any]:
    """逐条审核破冰建议，只允许审核通过的内容进入审批。"""
    hints = [
        str(hint).strip()
        for hint in (state.get("ice_break") or {}).get("hints", [])
        if str(hint).strip()
    ][:3]
    if not hints:
        return {"status": "failed", "error": "没有可审核的破冰建议"}

    approved_items: list[dict[str, Any]] = []
    approved_hints: list[str] = []
    rejected_count = 0
    for hint in hints:
        try:
            result = await run_guard(hint, state["user_id"])
            if (
                not isinstance(result, dict)
                or result.get("status") != "success"
                or result.get("error")
                or not isinstance(result.get("passed"), bool)
            ):
                raise ValueError("审核服务没有返回有效的成功结果")
            passed = result["passed"]
            if passed:
                approved_hints.append(hint)
                approved_items.append(
                    {
                        "hint": hint,
                        "passed": True,
                        "reason": str(result.get("reason", "")),
                        "risk_level": str(result.get("risk_level", "")),
                    }
                )
            else:
                rejected_count += 1
        except Exception:
            logger.exception("pick_guard_failed run_id=%s", state.get("run_id"))
            # 部分审核成功也不继续审批，避免一次技术故障被解释为业务拒绝。
            safe_ice_break = dict(state.get("ice_break") or {})
            safe_ice_break["hints"] = []
            return {
                "ice_break": safe_ice_break,
                "guard": {
                    "agent": "guard", "status": "failed", "passed": False,
                    "error": "审核服务暂时不可用，请稍后重试",
                    "error_code": "GUARD_SERVICE_FAILED",
                },
                "approved_hints": [],
                "status": "failed",
                "error": "审核服务暂时不可用，请稍后重试",
            }

    passed = bool(approved_hints)
    safe_ice_break = dict(state.get("ice_break") or {})
    safe_ice_break["hints"] = approved_hints
    guard_payload = {
        "agent": "guard",
        "status": "success" if passed else "rejected",
        "passed": passed,
        "items": approved_items,
        "rejected_count": rejected_count,
    }
    return {
        "ice_break": safe_ice_break,
        "guard": guard_payload,
        "approved_hints": approved_hints,
        "status": "running" if passed else "rejected",
        "error": None if passed else "没有通过内容审核的破冰建议",
    }


async def approval_node(state: PickWorkflowState) -> dict[str, Any]:
    """暂停图执行，等待用户选择一条已经审核通过的破冰建议。

    注意：审批节点只读取图状态，不执行外部写操作。真正写入
    pick_records 由路由层在确认后完成，避免节点重放产生副作用。
    """
    approved_hints = [str(hint) for hint in state.get("approved_hints", [])]
    if not approved_hints:
        return {"status": "rejected", "error": "没有可审批的破冰建议"}

    decision = interrupt(
        {
            "type": "pick_approval",
            "runId": state.get("run_id", ""),
            "match": state.get("match"),
            "profileAnalysis": state.get("profile_analysis"),
            "conversationGuide": state.get("conversation_guide"),
            "iceBreak": state.get("ice_break"),
            "guard": state.get("guard"),
            "approvedHints": approved_hints,
        }
    )

    if not isinstance(decision, dict):
        return {"status": "failed", "error": "审批结果格式不正确"}

    if decision.get("approved") is not True:
        return {
            "approval": decision,
            "status": "canceled",
            "error": None,
        }

    selected_hint = str(decision.get("hint", "")).strip()
    if selected_hint not in approved_hints:
        return {
            "approval": decision,
            "status": "failed",
            "error": "只能选择已审核通过的破冰建议",
        }

    return {
        "approval": decision,
        "selected_hint": selected_hint,
        "status": "approved",
        "error": None,
    }


def finish_node(state: PickWorkflowState) -> dict[str, Any]:
    """终止节点只保留已有状态，不重新解释模型文本。"""
    return {"status": state.get("status", "failed")}


def route_after_match(state: PickWorkflowState) -> str:
    return "finish" if state.get("status") in {"no_result", "failed"} else "profile"


def route_after_profile(state: PickWorkflowState) -> str:
    return "finish" if state.get("status") == "failed" else "guide"


def route_after_guide(state: PickWorkflowState) -> str:
    return "finish" if state.get("status") == "failed" else "ice_break"


def route_after_ice_break(state: PickWorkflowState) -> str:
    return "finish" if state.get("status") == "failed" else "guard"


def route_after_guard(state: PickWorkflowState) -> str:
    return "finish" if state.get("status") in {"failed", "rejected"} else "approval"


def build_pick_workflow() -> StateGraph:
    """构建显式拾取工作流；检查点由运行时编译时注入。"""
    graph = StateGraph(PickWorkflowState)
    graph.add_node("match", traced("match")(match_node))
    graph.add_node("profile", traced("profile")(profile_node))
    graph.add_node("guide", traced("guide")(guide_node))
    graph.add_node("ice_break", traced("ice_break")(ice_break_node))
    graph.add_node("guard", traced("guard")(guard_node))
    graph.add_node("approval", traced("approval")(approval_node))
    graph.add_node("finish", finish_node)

    graph.set_entry_point("match")
    graph.add_conditional_edges(
        "match",
        route_after_match,
        {"profile": "profile", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "profile",
        route_after_profile,
        {"guide": "guide", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "guide",
        route_after_guide,
        {"ice_break": "ice_break", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "ice_break",
        route_after_ice_break,
        {"guard": "guard", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"approval": "approval", "finish": "finish"},
    )
    graph.add_edge("approval", END)
    graph.add_edge("finish", END)
    return graph


pick_workflow_graph = build_pick_workflow()
