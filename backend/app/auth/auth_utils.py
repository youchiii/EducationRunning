"""Utility helpers for authentication-related database operations."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Dict, List, Optional

import bcrypt

from .database import get_connection


def _row_to_dict(row: sqlite3.Row) -> Dict[str, object]:
    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "role": row["role"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def create_user(username: str, password: str, *, role: str = "student") -> bool:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    try:
        with closing(get_connection()) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, role, "pending", created_at),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(username: str) -> Optional[Dict[str, object]]:
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, status, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def get_user_by_id(user_id: int) -> Optional[Dict[str, object]]:
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, status, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_pending_users() -> List[Dict[str, object]]:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT id, username, password_hash, role, status, created_at FROM users WHERE status = ? ORDER BY created_at",
            ("pending",),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_all_users() -> List[Dict[str, object]]:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT id, username, password_hash, role, status, created_at FROM users ORDER BY created_at DESC",
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def update_user_status(user_id: int, status: str) -> None:
    if status not in {"active", "rejected"}:
        raise ValueError("Invalid status")

    with closing(get_connection()) as conn:
        cursor = conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("User not found")


def reset_password(user_id: int, new_password: str) -> None:
    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with closing(get_connection()) as conn:
        cursor = conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("User not found")


def delete_user(user_id: int) -> None:
    with closing(get_connection()) as conn:
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("User not found")
