from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.auth_utils import (
    delete_user,
    get_all_users,
    get_pending_users,
    reset_password,
    update_user_status,
)

from ..security import get_current_teacher_user

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_teacher_user)])


class UserSummary(BaseModel):
    id: int
    username: str
    role: str
    status: str
    created_at: str


class UpdateStatusRequest(BaseModel):
    status: str = Field(..., pattern=r"^(active|rejected)$")


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)


@router.get("/pending", response_model=List[UserSummary])
async def list_pending_users() -> List[UserSummary]:
    return [
        UserSummary(
            id=row["id"],
            username=row["username"],
            role="student",
            status="pending",
            created_at=row["created_at"],
        )
        for row in get_pending_users()
    ]


@router.get("/users", response_model=List[UserSummary])
async def list_all_users() -> List[UserSummary]:
    return [
        UserSummary(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            status=row["status"],
            created_at=row["created_at"],
        )
        for row in get_all_users()
    ]


@router.post("/users/{user_id}/status")
async def update_status(user_id: int, payload: UpdateStatusRequest) -> Dict[str, str]:
    try:
        update_user_status(user_id, payload.status)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"status": payload.status}


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(user_id: int, payload: ResetPasswordRequest) -> Dict[str, str]:
    reset_password(user_id, payload.new_password)
    return {"status": "updated"}


@router.delete("/users/{user_id}")
async def remove_user(user_id: int) -> Dict[str, str]:
    delete_user(user_id)
    return {"status": "deleted"}
