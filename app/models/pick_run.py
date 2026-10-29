from pydantic import BaseModel, Field


class PickRunRequest(BaseModel):
    userId: int = Field(gt=0)
    university: str = Field(default="", max_length=100)
    city: str = Field(default="", max_length=50)
    province: str = Field(default="", max_length=50)


class PickRunResumeRequest(BaseModel):
    userId: int = Field(gt=0)
    approved: bool
    hint: str | None = Field(default=None, max_length=200)
