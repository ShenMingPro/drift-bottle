from pydantic import BaseModel


class WriteRecord(BaseModel):
    id: int
    content: str
    createdAt: str


class PickRecord(BaseModel):
    content: str
    pickedAt: str
    bottleCreatedAt: str = ""
    bottleUserId: int = 0
    nickname: str
    gender: str
    age: int
    hobbies: str
    contact_type: str
    contact_value: str
    province: str
    city: str
    university: str


class HistoryResponse(BaseModel):
    writeRecords: list[WriteRecord]
    pickRecords: list[PickRecord]
