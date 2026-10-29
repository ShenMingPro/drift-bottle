from pydantic import BaseModel


class CheckinStatusResponse(BaseModel):
    canCheckin: bool
    inviteUsed: int
    inviteRemaining: int
    writeBonusUsed: int
    writeBonusRemaining: int


class CheckinResponse(BaseModel):
    success: bool
    message: str


class InviteResponse(BaseModel):
    success: bool
    message: str
    inviteRemaining: int


class WriteBonusResponse(BaseModel):
    success: bool
    message: str
    writeBonusUsed: int
    writeBonusRemaining: int


class UpdateChancesRequest(BaseModel):
    userId: int
    writeChances: int
    pickChances: int


class UserChancesResponse(BaseModel):
    writeChances: int
    pickChances: int
