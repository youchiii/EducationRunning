from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.auth_utils import create_user, get_user, verify_password

from ..security import create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=150)
    password: str = Field(..., min_length=1, max_length=200)


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=150)
    password: str = Field(..., min_length=6, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, object]


def _serialize_user(user_row) -> Dict[str, object]:
    return {
        "id": user_row["id"],
        "username": user_row["username"],
        "role": user_row["role"],
        "status": user_row["status"],
        "created_at": user_row["created_at"],
    }


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    user = get_user(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="ユーザー名またはパスワードが正しくありません。")

    if user["status"] != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="このアカウントは現在ログインできません。管理者の承認状況を確認してください。")

    token = create_access_token({"sub": str(user["id"])})
    return TokenResponse(access_token=token, user=_serialize_user(user))


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest) -> Dict[str, str]:
    success = create_user(payload.username, payload.password)
    if not success:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="このユーザー名は既に使用されています。別のユーザー名を指定してください。")
    return {"message": "ユーザー登録を受け付けました。管理者の承認をお待ちください。"}


@router.get("/me")
async def me(current_user: Dict[str, object] = Depends(get_current_user)) -> Dict[str, object]:
    return current_user
