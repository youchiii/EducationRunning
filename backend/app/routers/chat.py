from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.database import get_db_connection

from ..security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


def _to_iso(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


@dataclass
class _MessageRecord:
    id: int
    conversation_id: int
    sender_id: int
    receiver_id: int
    text: str
    reply_to_id: Optional[int]
    created_at: str
    is_read: bool

    def to_response(self) -> "ChatMessage":
        return ChatMessage(
            id=self.id,
            conversation_id=self.conversation_id,
            sender_id=self.sender_id,
            receiver_id=self.receiver_id,
            text=self.text,
            reply_to_id=self.reply_to_id,
            timestamp=_to_iso(self.created_at) or self.created_at,
            is_read=self.is_read,
        )


class ConversationLatestMessage(BaseModel):
    id: int
    text: str
    timestamp: str


class ConversationSummary(BaseModel):
    conversation_id: int
    partner_id: int
    partner_name: str
    latest_message: Optional[ConversationLatestMessage]
    unread_count: int


class ChatMessage(BaseModel):
    id: int
    conversation_id: int
    sender_id: int
    receiver_id: int
    text: str
    reply_to_id: Optional[int]
    timestamp: str
    is_read: bool


class SendMessageRequest(BaseModel):
    sender_id: int
    receiver_id: int
    text: str = Field(..., min_length=1, max_length=2000)
    reply_to_id: Optional[int] = None


class SendMessageResponse(BaseModel):
    conversation_id: int
    message: ChatMessage


class UpdateReadResponse(BaseModel):
    conversation_id: int
    updated_count: int


def _serialize_message(row: Any) -> _MessageRecord:
    return _MessageRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        sender_id=row["sender_id"],
        receiver_id=row["receiver_id"],
        text=row["text"],
        reply_to_id=row["reply_to_id"],
        created_at=row["created_at"],
        is_read=bool(row["is_read"]),
    )


def _normalize_participants(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _get_or_create_conversation(conn, participant_a: int, participant_b: int) -> int:
    first, second = _normalize_participants(participant_a, participant_b)
    existing = conn.execute(
        "SELECT id FROM chat_conversations WHERE participant_one_id = ? AND participant_two_id = ?",
        (first, second),
    ).fetchone()
    if existing:
        return existing["id"]

    cursor = conn.execute(
        "INSERT INTO chat_conversations (participant_one_id, participant_two_id) VALUES (?, ?)",
        (first, second),
    )
    return cursor.lastrowid


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[ConversationSummary]:
    user_id = current_user["id"]
    conn = get_db_connection()
    try:
        conversation_rows = conn.execute(
            """
            SELECT id, participant_one_id, participant_two_id, updated_at
            FROM chat_conversations
            WHERE participant_one_id = ? OR participant_two_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id, user_id),
        ).fetchall()

        if not conversation_rows:
            return []

        conversation_ids = [row["id"] for row in conversation_rows]
        partner_ids = {
            row["participant_two_id"] if row["participant_one_id"] == user_id else row["participant_one_id"]
            for row in conversation_rows
        }

        partner_lookup: Dict[int, str] = {}
        if partner_ids:
            placeholders = ",".join(["?"] * len(partner_ids))
            partner_rows = conn.execute(
                f"SELECT id, username FROM users WHERE id IN ({placeholders})",
                tuple(partner_ids),
            ).fetchall()
            partner_lookup = {row["id"]: row["username"] for row in partner_rows}

        latest_map: Dict[int, _MessageRecord] = {}
        for conversation_id in conversation_ids:
            row = conn.execute(
                """
                SELECT id, conversation_id, sender_id, receiver_id, text, reply_to_id, created_at, is_read
                FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if row:
                latest_map[conversation_id] = _serialize_message(row)

        unread_map: Dict[int, int] = {}
        placeholders = ",".join(["?"] * len(conversation_ids))
        unread_rows = conn.execute(
            f"""
            SELECT conversation_id, COUNT(*) AS unread
            FROM chat_messages
            WHERE conversation_id IN ({placeholders})
              AND receiver_id = ?
              AND is_read = 0
            GROUP BY conversation_id
            """,
            (*conversation_ids, user_id),
        ).fetchall()
        unread_map = {row["conversation_id"]: row["unread"] for row in unread_rows}

        summaries: List[ConversationSummary] = []
        for row in conversation_rows:
            partner_id = row["participant_two_id"] if row["participant_one_id"] == user_id else row["participant_one_id"]
            latest = latest_map.get(row["id"])
            summaries.append(
                ConversationSummary(
                    conversation_id=row["id"],
                    partner_id=partner_id,
                    partner_name=partner_lookup.get(partner_id, "Unknown"),
                    latest_message=
                    ConversationLatestMessage(
                        id=latest.id,
                        text=latest.text,
                        timestamp=_to_iso(latest.created_at) or latest.created_at,
                    )
                    if latest
                    else None,
                    unread_count=unread_map.get(row["id"], 0),
                )
            )

        return summaries
    finally:
        conn.close()


@router.get("/messages/{conversation_id}", response_model=List[ChatMessage])
async def list_messages(
    conversation_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[ChatMessage]:
    user_id = current_user["id"]
    conn = get_db_connection()
    try:
        conversation = conn.execute(
            "SELECT participant_one_id, participant_two_id FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if user_id not in (conversation["participant_one_id"], conversation["participant_two_id"]):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for conversation")

        rows = conn.execute(
            """
            SELECT id, conversation_id, sender_id, receiver_id, text, reply_to_id, created_at, is_read
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id,),
        ).fetchall()
        return [_serialize_message(row).to_response() for row in rows]
    finally:
        conn.close()


@router.post("/send", response_model=SendMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    payload: SendMessageRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> SendMessageResponse:
    user_id = current_user["id"]
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
    if payload.sender_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Sender must match authenticated user")
    if payload.sender_id == payload.receiver_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot send messages to yourself")

    conn = get_db_connection()
    try:
        receiver_exists = conn.execute(
            "SELECT id FROM users WHERE id = ?",
            (payload.receiver_id,),
        ).fetchone()
        if receiver_exists is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receiver not found")

        conversation_id = _get_or_create_conversation(conn, payload.sender_id, payload.receiver_id)

        if payload.reply_to_id is not None:
            reply_row = conn.execute(
                "SELECT id FROM chat_messages WHERE id = ? AND conversation_id = ?",
                (payload.reply_to_id, conversation_id),
            ).fetchone()
            if reply_row is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reply target not in conversation")

        cursor = conn.execute(
            """
            INSERT INTO chat_messages (conversation_id, sender_id, receiver_id, text, reply_to_id, is_read)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (conversation_id, payload.sender_id, payload.receiver_id, text, payload.reply_to_id),
        )
        message_id = cursor.lastrowid

        conn.execute(
            "UPDATE chat_conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()

        message_row = conn.execute(
            """
            SELECT id, conversation_id, sender_id, receiver_id, text, reply_to_id, created_at, is_read
            FROM chat_messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
        return SendMessageResponse(
            conversation_id=conversation_id,
            message=_serialize_message(message_row).to_response(),
        )
    finally:
        conn.close()


@router.patch("/read/{conversation_id}", response_model=UpdateReadResponse)
async def mark_conversation_read(
    conversation_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> UpdateReadResponse:
    user_id = current_user["id"]
    conn = get_db_connection()
    try:
        conversation = conn.execute(
            "SELECT participant_one_id, participant_two_id FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if user_id not in (conversation["participant_one_id"], conversation["participant_two_id"]):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for conversation")

        cursor = conn.execute(
            """
            UPDATE chat_messages
               SET is_read = 1
             WHERE conversation_id = ?
               AND receiver_id = ?
               AND is_read = 0
            """,
            (conversation_id, user_id),
        )
        conn.commit()
        return UpdateReadResponse(conversation_id=conversation_id, updated_count=cursor.rowcount)
    finally:
        conn.close()
