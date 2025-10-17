from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, Optional
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetEntry:
    dataset_id: str
    df: pd.DataFrame
    original_name: str
    uploaded_at: datetime


class DatasetStore:
    def __init__(self, retention: timedelta) -> None:
        self._retention = retention
        self._datasets: Dict[str, DatasetEntry] = {}
        self._lock = RLock()

    def _prune_locked(self) -> None:
        if self._retention <= timedelta(0):
            return
        cutoff = datetime.utcnow() - self._retention
        stale_keys = [key for key, entry in self._datasets.items() if entry.uploaded_at < cutoff]
        for key in stale_keys:
            self._datasets.pop(key, None)

    def add(self, df: pd.DataFrame, original_name: str) -> DatasetEntry:
        with self._lock:
            self._prune_locked()
            dataset_id = uuid4().hex
            entry = DatasetEntry(
                dataset_id=dataset_id,
                df=df,
                original_name=original_name,
                uploaded_at=datetime.utcnow(),
            )
            self._datasets[dataset_id] = entry
            return entry

    def get(self, dataset_id: str) -> DatasetEntry:
        with self._lock:
            entry = self._datasets.get(dataset_id)
            if not entry:
                raise KeyError(f"Dataset '{dataset_id}' not found")
            return entry

    def remove(self, dataset_id: str) -> None:
        with self._lock:
            self._datasets.pop(dataset_id, None)

    def info(self, dataset_id: str) -> Optional[DatasetEntry]:
        with self._lock:
            return self._datasets.get(dataset_id)
