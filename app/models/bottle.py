from pydantic import BaseModel, Field


class WriteRequest(BaseModel):
    userId: int
    content: str = Field(min_length=1, max_length=200)
    aiProof: str | None = Field(default=None, max_length=2048)
    useBonus: bool = False


class WriteResponse(BaseModel):
    success: bool
    message: str
    writeBonusUsed: int = 0
    writeBonusRemaining: int = 3
    pickChances: int | None = None


class PickResponse(BaseModel):
    """旧版单步拾纸团接口的扁平响应。"""

    hasBottle: bool
    bottleUserId: int = 0
    content: str = ""
    createdAt: str = ""
    nickname: str = ""
    gender: str = ""
    age: int = 0
    hobbies: str = ""
    contact_type: str = ""
    contact_value: str = ""
    province: str = ""
    city: str = ""
    university: str = ""
    matchLevel: str = ""
    matchReason: str = ""
    matchScore: float = 0.0
