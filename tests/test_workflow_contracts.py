import json

import pytest
from pydantic import ValidationError

from app.agents.workflow_contracts import (
    AgentStatus,
    IceBreakArtifact,
    MatchArtifact,
    MatchCandidate,
    ProfileArtifact,
    SupervisorDecision,
    WorkflowArtifact,
)


def test_match_artifact_serializes_nested_candidate_and_status():
    artifact = MatchArtifact(
        status=AgentStatus.SUCCESS,
        summary="Found a same-school candidate",
        best_match=MatchCandidate(
            bottle_id=12,
            user_id=23,
            content="Looking for a photography partner",
            university="Test University",
        ),
        match_reason="same university and shared hobby",
        match_score=0.82,
        match_level="university",
    )

    payload = json.loads(artifact.model_dump_json())

    assert payload["agent"] == "match"
    assert payload["status"] == "success"
    assert payload["best_match"]["user_id"] == 23
    assert artifact.succeeded is True
    assert artifact.terminal is True


def test_no_result_is_a_normal_terminal_state():
    artifact = MatchArtifact(
        status=AgentStatus.NO_RESULT,
        summary="No candidate matched the current constraints",
    )

    assert artifact.best_match is None
    assert artifact.succeeded is False
    assert artifact.retryable is False
    assert artifact.terminal is True


def test_failed_artifact_requires_an_error_message():
    with pytest.raises(ValidationError, match="error is required"):
        WorkflowArtifact(agent="research", status=AgentStatus.RETRYABLE_ERROR)

    artifact = WorkflowArtifact(
        agent="research",
        status=AgentStatus.RETRYABLE_ERROR,
        error="MCP server timed out",
        error_code="mcp_timeout",
        attempt=2,
    )

    assert artifact.retryable is True
    assert artifact.terminal is False

    fatal = WorkflowArtifact(
        agent="research",
        status="failed",
        error="Search provider is unavailable",
    )
    assert fatal.status is AgentStatus.FAILED
    assert fatal.terminal is True


def test_profile_and_ice_break_artifacts_keep_agent_specific_results():
    profile = ProfileArtifact(
        status=AgentStatus.SUCCESS,
        user_id=7,
        target_user_id=8,
        tags=["photography", "outdoors"],
        completeness_score=90,
        compatibility="Both users enjoy outdoor photography",
    )
    ice_break = IceBreakArtifact(
        status=AgentStatus.SUCCESS,
        user_id=7,
        target_user_id=8,
        scenario="ice_break",
        hints=["Do you prefer street or landscape photography?"],
    )

    assert profile.agent == "profile_analysis"
    assert profile.model_dump()["completeness_score"] == 90
    assert ice_break.agent == "ice_break"
    assert ice_break.model_dump()["hints"] == [
        "Do you prefer street or landscape photography?"
    ]


def test_supervisor_decision_distinguishes_routing_from_completion():
    route = SupervisorDecision(
        next_agent="profile_analysis",
        arguments={"user_id": 7, "target_user_id": 8},
        reason="A match is available and needs compatibility analysis",
    )
    complete = SupervisorDecision(
        is_complete=True,
        reason="All required artifacts are ready",
    )

    assert route.is_complete is False
    assert route.next_agent == "profile_analysis"
    assert complete.next_agent is None

    with pytest.raises(ValidationError, match="cannot have a next_agent"):
        SupervisorDecision(
            next_agent="ice_break",
            reason="Invalid completed decision",
            is_complete=True,
        )
