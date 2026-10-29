from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    userId: int
    targetUserId: int


class SendMessageRequest(BaseModel):
    conversationId: int = Field(gt=0)
    senderId: int = Field(gt=0)
    content: str = Field(min_length=1, max_length=2000)


class MessageItem(BaseModel):
    id: int
    senderId: int
    content: str
    isRead: bool
    createdAt: str


class ConversationItem(BaseModel):
    id: int
    targetUserId: int
    targetNickname: str
    targetGender: str = ""
    targetAge: int = 0
    targetHobbies: str = ""
    targetProvince: str = ""
    targetCity: str = ""
    targetUniversity: str = ""
    targetContactType: str = ""
    targetContactValue: str = ""
    lastMessage: str = ""
    lastMessageAt: str = ""
    unreadCount: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationItem]


class MessagesResponse(BaseModel):
    messages: list[MessageItem]
