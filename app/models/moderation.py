from pydantic import BaseModel


class ModerationResult(BaseModel):
    """内容审核结果模型，包含审核是否通过、原因和风险等级"""
    passed: bool
    reason: str
    riskLevel: str
