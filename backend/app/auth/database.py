"""Database helpers for authentication components."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "auth.sqlite3"


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