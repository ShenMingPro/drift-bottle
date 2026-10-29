from pydantic import AliasChoices, BaseModel, Field, field_validator

from app.security import normalize_username, validate_new_username


class LoginRequest(BaseModel):
    username: str = Field(
        min_length=1,
        max_length=64,
        validation_alias=AliasChoices("username", "account"),
    )
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_login_username(cls, value: str) -> str:
        username = normalize_username(value)
        if not username:
            raise ValueError("用户名不能为空")
        return username


class LoginResponse(BaseModel):
    userId: int
    nickname: str
    gender: str
    age: int
    hobbies: str
    contactType: str
    contactValue: str
    province: str
    city: str
    university: str
    writeChances: int
    pickChances: int
    token: str  # JWT访问令牌，前端需存储并在后续请求中携带


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=2,
        max_length=20,
        validation_alias=AliasChoices("username", "account"),
        description="2-20位用户名，可使用中文、字母、数字和下划线",
    )
    password: str = Field(min_length=8, max_length=128, description="密码长度8-128位")
    gender: str
    age: int = Field(ge=10, le=100)
    hobbies: list[str] = Field(min_length=1)
    contactType: str = "QQ号"
    contactValue: str = Field(min_length=1)
    province: str = Field(min_length=1)
    city: str = Field(min_length=1)
    university: str = Field(min_length=1)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        return validate_new_username(value)


class RegisterResponse(BaseModel):
    userId: int
    nickname: str
    gender: str
    age: int
    hobbies: str
    contactType: str
    contactValue: str
    province: str
    city: str
    university: str
    token: str  # JWT访问令牌，注册成功后自动登录
