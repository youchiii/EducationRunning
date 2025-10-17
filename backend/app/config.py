from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class Settings(BaseModel):
    secret_key: str = Field(default_factory=lambda: os.getenv("APP_SECRET_KEY", "CHANGE_ME"))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(
        default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))
    )
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()]
    )
    dataset_retention_minutes: int = Field(
        default_factory=lambda: int(os.getenv("DATASET_RETENTION_MINUTES", "180"))
    )
    pose_tmp_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "tmp"
    )

    class Config:
        arbitrary_types_allowed = True


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.pose_tmp_dir.mkdir(parents=True, exist_ok=True)
    return settings
