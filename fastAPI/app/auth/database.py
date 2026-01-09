"""Database helpers for authentication components."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import bcrypt


DATABASE_PATH = Path(__file__).resolve().parent / "auth.sqlite3"

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"


def _ensure_default_admin_user(conn: sqlite3.Connection) -> None:
    """Insert the default admin user if it does not exist yet."""

    row = conn.execute("SELECT 1 FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)).fetchone()
    if row:
        return

    password_hash = bcrypt.hashpw(DEFAULT_ADMIN_PASSWORD.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO users (username, password_hash, role, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (DEFAULT_ADMIN_USERNAME, password_hash, "teacher", "active", created_at),
    )


def get_connection() -> sqlite3.Connection:
    """Return a connection to the authentication database."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialise the authentication-related persistence layer."""
    with closing(get_connection()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_default_admin_user(conn)
        conn.commit()
from contextlib import contextmanager

@contextmanager
def get_db_connection(readonly: bool = False):
    """
    互換用のラッパー。既存コードが `with get_db_connection() as conn:` 
    の形で使えるようにする。
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
