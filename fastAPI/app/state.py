from __future__ import annotations

from datetime import timedelta
from typing import Optional

from .config import Settings
from .datasets import DatasetStore


dataset_store: Optional[DatasetStore] = None


def init_state(settings: Settings) -> None:
    global dataset_store
    dataset_store = DatasetStore(retention=timedelta(minutes=settings.dataset_retention_minutes))


def get_dataset_store() -> DatasetStore:
    if dataset_store is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Dataset store has not been initialised")
    return dataset_store
