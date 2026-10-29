from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentStatus(str, Enum):
    """Shared outcome states returned by worker agents."""

    SUCCESS = "success"
    NO_RESULT = "no_result"
    RETRYABLE_ERROR = "retryable_error"
    FAILED = "failed"
    FATAL_ERROR = "fatal_error"


class WorkflowArtifact(BaseModel):
    """Base contract for one worker agent result."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    status: AgentStatus
    summary: str = ""
    error: str | None = None
    error_code: str | None = None
    attempt: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_error_state(self) -> Self:
        failure_states = {
            AgentStatus.RETRYABLE_ERROR,
            AgentStatus.FAILED,
            AgentStatus.FATAL_ERROR,
        }
        if self.status in failure_states and not self.error:
            raise ValueError("error is required for a failed artifact")
        if self.status not in failure_states and self.error is not None:
            raise ValueError("error is only allowed for a failed artifact")
        return self

    @property
    def succeeded(self) -> bool:
        return self.status is AgentStatus.SUCCESS

    @property
    def retryable(self) -> bool:
        return self.status is AgentStatus.RETRYABLE_ERROR

    @property
    def terminal(self) -> bool:
        return self.status is not AgentStatus.RETRYABLE_ERROR


class MatchCandidate(BaseModel):
    """Candidate selected by MatchAgent."""

    model_config = ConfigDict(extra="forbid")

    bottle_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    content: str = ""
    created_at: str = ""
    nickname: str = ""
    gender: str = ""
    age: int = Field(default=0, ge=0)
    hobbies: str = ""
    contact_type: str = ""
    contact_value: str = ""
    province: str = ""
    city: str = ""
    university: str = ""


class MatchArtifact(WorkflowArtifact):
    """Structured result returned by MatchAgent."""

    agent: Literal["match"] = "match"
    best_match: MatchCandidate | None = None
    match_reason: str = ""
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    content_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_level: Literal[
        "",
        "university",
        "city",
        "province",
        "nationwide",
    ] = ""

    @model_validator(mode="after")
    def validate_match_result(self) -> Self:
        if self.status is AgentStatus.SUCCESS and self.best_match is None:
            raise ValueError("best_match is required when matching succeeds")
        if self.status is not AgentStatus.SUCCESS and self.best_match is not None:
            raise ValueError("best_match is only allowed when matching succeeds")
        return self


class ProfileArtifact(WorkflowArtifact):
    """Structured result returned by ProfileAnalysisAgent."""

    agent: Literal["profile_analysis"] = "profile_analysis"
    user_id: int | None = Field(default=None, gt=0)
    target_user_id: int | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    completeness_score: int = Field(default=0, ge=0, le=100)
    suggestions: list[str] = Field(default_factory=list)
    compatibility: str = ""


class ConversationArtifact(WorkflowArtifact):
    """Conversation strategy returned by ConversationGuideAgent."""

    agent: Literal["conversation_guide"] = "conversation_guide"
    user_id: int | None = Field(default=None, gt=0)
    target_user_id: int | None = Field(default=None, gt=0)
    conversation_status: Literal["new", "active", "stalled"] = "new"
    topic_suggestions: list[str] = Field(default_factory=list)
    reply_strategies: list[str] = Field(default_factory=list)
    ice_breaker: str = ""


class IceBreakArtifact(WorkflowArtifact):
    """Concrete messages returned by IceBreakAgent."""

    agent: Literal["ice_break"] = "ice_break"
    user_id: int | None = Field(default=None, gt=0)
    target_user_id: int | None = Field(default=None, gt=0)
    scenario: Literal["ice_break", "reply"] = "ice_break"
    hints: list[str] = Field(default_factory=list)


class IceBreakAnalysisContext(BaseModel):
    """仅供本次破冰生成使用的上游结论，不传递身份或任意元数据。"""

    model_config = ConfigDict(extra="forbid")

    profile_tags: list[str] = Field(default_factory=list)
    compatibility: str = ""
    conversation_status: Literal["", "new", "active", "stalled"] = ""
    topic_suggestions: list[str] = Field(default_factory=list)
    reply_strategies: list[str] = Field(default_factory=list)
    suggested_opening: str = ""

    @classmethod
    def from_artifacts(
        cls,
        profile_analysis: dict[str, Any] | None,
        conversation_guide: dict[str, Any] | None,
    ) -> IceBreakAnalysisContext | None:
        profile = profile_analysis or {}
        guide = conversation_guide or {}
        profile = profile if profile.get("status") == "success" else {}
        guide = guide if guide.get("status") == "success" else {}
        context = cls(
            profile_tags=profile.get("tags", [])[:3],
            compatibility=profile.get("compatibility", "")[:300],
            conversation_status=guide.get("conversation_status", ""),
            topic_suggestions=guide.get("topic_suggestions", [])[:3],
            reply_strategies=guide.get("reply_strategies", [])[:3],
            suggested_opening=guide.get("ice_breaker", "")[:120],
        )
        return context if context.model_dump(exclude_defaults=True) else None


class SupervisorDecision(BaseModel):
    """One routing decision made by SupervisorAgent."""

    model_config = ConfigDict(extra="forbid")

    next_agent: str | None = Field(default=None, min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    is_complete: bool = False

    @model_validator(mode="after")
    def validate_routing_decision(self) -> Self:
        if self.is_complete and self.next_agent is not None:
            raise ValueError("a completed workflow cannot have a next_agent")
        if not self.is_complete and self.next_agent is None:
            raise ValueError("next_agent is required while a workflow is active")
        if self.is_complete and self.arguments:
            raise ValueError("a completed workflow cannot have agent arguments")
        return self
