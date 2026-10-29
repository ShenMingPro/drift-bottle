from fastapi import APIRouter, Depends, HTTPException, Query

from app.agents.supervisor_agent import supervisor_app
from app.auth_jwt import get_current_user  # JWT认证依赖项
from app.database import get_database
from app.services.agent_responses import invoke_agent
from app.models.chat import (
    ConversationItem,
    ConversationListResponse,
    CreateConversationRequest,
    MessageItem,
    MessagesResponse,
    SendMessageRequest,
)

router = APIRouter()


async def _conversation_participants(db, conversation_id: int) -> tuple[int, int]:
    """读取会话成员；不存在时返回统一 404，避免暴露内部数据。"""
    await db.execute(
        "SELECT user1_id, user2_id FROM conversations WHERE id = %s",
        [conversation_id],
    )
    row = await db.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="对话不存在")
    return int(row["user1_id"]), int(row["user2_id"])


async def _assert_conversation_member(db, conversation_id: int, user_id: int) -> None:
    user1_id, user2_id = await _conversation_participants(db, conversation_id)
    if user_id not in {user1_id, user2_id}:
        raise HTTPException(status_code=403, detail="无权访问该对话")


@router.post("/conversations", response_model=ConversationItem)
async def create_conversation(req: CreateConversationRequest, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """创建对话接口（需JWT认证）：校验token中的用户ID与请求中的userId一致"""
    if current_user_id != req.userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    if req.userId == req.targetUserId:
        raise HTTPException(status_code=400, detail="不能和自己创建对话")

    async with db.session() as cursor:
        await cursor.execute("SELECT id FROM users WHERE id = %s", [req.targetUserId])
        if not await cursor.fetchone():
            raise HTTPException(status_code=400, detail="目标用户不存在")

        u1, u2 = min(req.userId, req.targetUserId), max(req.userId, req.targetUserId)

        await cursor.execute(
            "SELECT id FROM conversations WHERE user1_id = %s AND user2_id = %s", [u1, u2]
        )
        row = await cursor.fetchone()

        if row:
            conv_id = row["id"]
        else:
            await cursor.execute(
                "INSERT INTO conversations (user1_id, user2_id) VALUES (%s, %s)", [u1, u2]
            )
            conv_id = cursor.lastrowid

        await cursor.execute(
            "SELECT id, nickname, gender, age, hobbies, province, city, university, contact_type, contact_value FROM users WHERE id = %s",
            [req.targetUserId],
        )
        target = await cursor.fetchone()

        await cursor.execute(
            "SELECT content FROM messages WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 1",
            [conv_id],
        )
        last_msg_row = await cursor.fetchone()
        last_message = last_msg_row["content"] if last_msg_row else ""

        await cursor.execute(
            "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = %s AND sender_id != %s AND is_read = 0",
            [conv_id, req.userId],
        )
        unread_row = await cursor.fetchone()

    return ConversationItem(
        id=conv_id,
        targetUserId=req.targetUserId,
        targetNickname=target["nickname"] or "",
        targetGender=target["gender"] or "",
        targetAge=target["age"] or 0,
        targetHobbies=target["hobbies"] or "",
        targetProvince=target["province"] or "",
        targetCity=target["city"] or "",
        targetUniversity=target["university"] or "",
        targetContactType=target["contact_type"] or "",
        targetContactValue=target["contact_value"] or "",
        lastMessage=last_message[:50] if last_message else "",
        lastMessageAt="",
        unreadCount=unread_row["cnt"] if unread_row else 0,
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def get_conversations(userId: int = Query(...), db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """获取对话列表接口（需JWT认证）"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    async with db.session() as cursor:
        await cursor.execute(
            "SELECT c.id, c.user1_id, c.user2_id, c.last_message_at, "
            "u.id AS target_id, u.nickname, u.gender, u.age, u.hobbies, "
            "u.province, u.city, u.university, u.contact_type, u.contact_value "
            "FROM conversations c "
            "JOIN users u ON u.id = CASE WHEN c.user1_id = %s THEN c.user2_id ELSE c.user1_id END "
            "WHERE c.user1_id = %s OR c.user2_id = %s "
            "ORDER BY c.last_message_at DESC",
            [userId, userId, userId],
        )
        rows = await cursor.fetchall()

        conversations = []
        for row in rows:
            conv_id = row["id"]

            await cursor.execute(
                "SELECT content FROM messages WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 1",
                [conv_id],
            )
            last_msg_row = await cursor.fetchone()
            last_message = last_msg_row["content"] if last_msg_row else ""

            await cursor.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = %s AND sender_id != %s AND is_read = 0",
                [conv_id, userId],
            )
            unread_row = await cursor.fetchone()

            conversations.append(
                ConversationItem(
                    id=conv_id,
                    targetUserId=row["target_id"],
                    targetNickname=row["nickname"] or "",
                    targetGender=row["gender"] or "",
                    targetAge=row["age"] or 0,
                    targetHobbies=row["hobbies"] or "",
                    targetProvince=row["province"] or "",
                    targetCity=row["city"] or "",
                    targetUniversity=row["university"] or "",
                    targetContactType=row["contact_type"] or "",
                    targetContactValue=row["contact_value"] or "",
                    lastMessage=last_message[:50] if last_message else "",
                    lastMessageAt=str(row["last_message_at"]) if row["last_message_at"] else "",
                    unreadCount=unread_row["cnt"] if unread_row else 0,
                )
            )

    return ConversationListResponse(conversations=conversations)


@router.post("/messages", response_model=MessageItem)
async def send_message(req: SendMessageRequest, db=Depends(get_database), current_user_id: int = Depends(get_current_user)):
    """发送消息接口（需JWT认证）。所有消息都必须经过 ContentGuard。"""
    if current_user_id != req.senderId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 先做低成本授权检查，避免为越权请求调用审核模型。
    async with db.session() as cursor:
        await _assert_conversation_member(cursor, req.conversationId, req.senderId)

    supervisor_state = {
        "user_id": req.senderId,
        "action": "send_message",
        "content": req.content.strip(),
        "target_user_id": 0,
        "target_university": "",
        "target_city": "",
        "target_province": "",
        "content_type": "message",
        "messages": [],
        "final_result": "",
        "guard_passed": False,
        "guard_reason": "",
    }
    supervisor_result = await invoke_agent(supervisor_app, supervisor_state)

    if not supervisor_result.get("guard_passed", False):  # fail-closed：默认拒绝
        raise HTTPException(status_code=400, detail=supervisor_result.get("guard_reason", "内容审核未通过"))

    # 审核等待期间不持有连接；写入前重新检查权限，消息与会话时间一起提交。
    async with db.session(transaction=True) as cursor:
        await _assert_conversation_member(cursor, req.conversationId, req.senderId)
        await cursor.execute(
            "INSERT INTO messages (conversation_id, sender_id, content) VALUES (%s, %s, %s)",
            [req.conversationId, req.senderId, req.content.strip()],
        )
        msg_id = cursor.lastrowid

        await cursor.execute(
            "UPDATE conversations SET last_message_at = NOW() WHERE id = %s",
            [req.conversationId],
        )

        await cursor.execute(
            "SELECT id, sender_id, content, is_read, created_at FROM messages WHERE id = %s",
            [msg_id],
        )
        msg = await cursor.fetchone()

    return MessageItem(
        id=msg["id"],
        senderId=msg["sender_id"],
        content=msg["content"],
        isRead=bool(msg["is_read"]),
        createdAt=str(msg["created_at"]),
    )


@router.get("/messages", response_model=MessagesResponse)
async def get_messages(
    conversationId: int = Query(...),
    userId: int = Query(...),
    db=Depends(get_database),
    current_user_id: int = Depends(get_current_user),
):
    """获取消息列表接口（需JWT认证）"""
    if current_user_id != userId:
        raise HTTPException(status_code=403, detail="无权操作：用户身份不匹配")
    async with db.session() as cursor:
        await _assert_conversation_member(cursor, conversationId, current_user_id)
        await cursor.execute(
            "SELECT id, sender_id, content, is_read, created_at "
            "FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
            [conversationId],
        )
        rows = await cursor.fetchall()

        await cursor.execute(
            "UPDATE messages SET is_read = 1 WHERE conversation_id = %s AND sender_id != %s AND is_read = 0",
            [conversationId, userId],
        )

    messages = [
        MessageItem(
            id=r["id"],
            senderId=r["sender_id"],
            content=r["content"],
            isRead=bool(r["is_read"]),
            createdAt=str(r["created_at"]),
        )
        for r in rows
    ]

    return MessagesResponse(messages=messages)
