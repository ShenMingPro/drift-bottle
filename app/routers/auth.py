from fastapi import APIRouter, Depends, HTTPException
from aiomysql import IntegrityError

from app.auth_jwt import create_access_token  # JWT令牌生成函数
from app.database import get_db
from app.models.user import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from app.security import hash_password, verify_password

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db=Depends(get_db)):
    """用户登录接口：验证用户名密码，返回用户信息和JWT令牌

    认证流程：
    1. 根据username查询users表
    2. 在Python中验证密码哈希
    3. 验证成功 → 生成JWT令牌，随用户信息一起返回
    """
    await db.execute(
        "SELECT id, password, nickname, gender, age, hobbies, contact_type, contact_value, "
        "province, city, university, write_chances, pick_chances "
        "FROM users WHERE username = %s",
        [req.username],
    )
    row = await db.fetchone()
    verification = verify_password(req.password, row["password"]) if row else None
    if not row or not verification or not verification.valid:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if verification.replacement_hash:
        await db.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            [verification.replacement_hash, row["id"]],
        )

    # 生成JWT令牌，有效期由JWT_EXPIRE_HOURS配置控制
    token = create_access_token(user_id=row["id"])

    return LoginResponse(
        userId=row["id"],
        nickname=row["nickname"],
        gender=row["gender"] or "",
        age=row["age"] or 0,
        hobbies=row["hobbies"] or "",
        contactType=row["contact_type"] or "",
        contactValue=row["contact_value"] or "",
        province=row["province"] or "",
        city=row["city"] or "",
        university=row["university"] or "",
        writeChances=row["write_chances"],
        pickChances=row["pick_chances"],
        token=token,  # JWT令牌
    )


@router.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest, db=Depends(get_db)):
    """用户注册接口：创建新用户，返回用户信息和JWT令牌（自动登录）

    认证流程：
    1. 检查用户名是否已存在
    2. 插入新用户记录
    3. 生成JWT令牌，随用户信息一起返回（注册即登录）
    """
    await db.execute("SELECT id FROM users WHERE username = %s", [req.username])
    if await db.fetchone():
        raise HTTPException(status_code=409, detail="该用户名已被使用")

    hobbies_str = ",".join(req.hobbies)
    try:
        await db.execute(
            "INSERT INTO users (username, password, nickname, gender, age, hobbies, "
            "contact_type, contact_value, province, city, university) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                req.username,
                hash_password(req.password),
                req.username,
                req.gender,
                req.age,
                hobbies_str,
                req.contactType,
                req.contactValue.strip(),
                req.province,
                req.city,
                req.university,
            ],
        )
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="该用户名已被使用") from exc
    user_id = db.lastrowid

    # 注册成功后自动生成JWT令牌（注册即登录）
    token = create_access_token(user_id=user_id)

    return RegisterResponse(
        userId=user_id,
        nickname=req.username,
        gender=req.gender,
        age=req.age,
        hobbies=hobbies_str,
        contactType=req.contactType,
        contactValue=req.contactValue.strip(),
        province=req.province,
        city=req.city,
        university=req.university,
        token=token,  # JWT令牌
    )
