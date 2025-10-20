from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..security import get_current_user


@dataclass
class FactorSession:
    session_id: str
    df: pd.DataFrame
    columns: List[str]
    created_at: datetime


class FactorSessionStore:
    def __init__(self, retention_minutes: int = 120) -> None:
        self._retention = timedelta(minutes=retention_minutes)
        self._sessions: Dict[str, FactorSession] = {}
        self._lock = RLock()

    def _prune_locked(self) -> None:
        if self._retention <= timedelta(0):
            return
        cutoff = datetime.utcnow() - self._retention
        stale_keys = [key for key, entry in self._sessions.items() if entry.created_at < cutoff]
        for key in stale_keys:
            self._sessions.pop(key, None)

    def create(self, df: pd.DataFrame, columns: List[str]) -> FactorSession:
        with self._lock:
            self._prune_locked()
            session_id = uuid4().hex
            session = FactorSession(session_id=session_id, df=df, columns=columns, created_at=datetime.utcnow())
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> FactorSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise KeyError(session_id)
            return session


session_store = FactorSessionStore()

router = APIRouter(prefix="/fa", tags=["factor-analysis"], dependencies=[Depends(get_current_user)])


class UploadResponse(BaseModel):
    session_id: str
    columns: List[str]
    n_rows: int


class ScreeResponse(BaseModel):
    eigenvalues: List[float]
    explained_variance_ratio: List[float]


class FactorRunRequest(BaseModel):
    session_id: str = Field(..., description="The session identifier obtained from the upload endpoint")
    n_factors: int = Field(..., ge=1, description="Number of factors to extract")
    columns: Optional[List[str]] = Field(default=None, description="Optional subset of columns to include")


class FactorRunResponse(BaseModel):
    loadings: Dict[str, Dict[str, float]]
    communalities: Dict[str, float]
    uniqueness: Dict[str, float]
    factor_scores: List[Dict[str, float]]
    n_rows: int


SUPPORTED_ENCODINGS = ("utf-8", "utf-8-sig", "shift_jis")


def _read_csv(upload: UploadFile) -> pd.DataFrame:
    upload.file.seek(0)
    raw_bytes = upload.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSVファイルが空です。")

    last_error: Optional[Exception] = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            buffer = io.BytesIO(raw_bytes)
            return pd.read_csv(buffer, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSVの内容が不正です。") from exc
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"CSVのエンコードを判別できません: {last_error}")


def _prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_df = dataframe.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="数値列が不足しています。")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(how="any")
    if numeric_df.empty:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="欠損値が多すぎるため分析できません。")
    # Drop constant columns to avoid singular matrices
    variances = numeric_df.var(axis=0)
    non_constant_columns = variances[variances > 0].index.tolist()
    if len(non_constant_columns) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="数値列が不足しています。")
    return numeric_df[non_constant_columns]


def _get_session(session_id: str) -> FactorSession:
    try:
        return session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="セッションが見つかりません。") from exc


@router.post("/upload", response_model=UploadResponse)
async def upload_factor_dataset(file: UploadFile) -> UploadResponse:
    try:
        dataframe = _read_csv(file)
    finally:
        file.file.close()

    numeric_df = _prepare_dataframe(dataframe)
    session = session_store.create(numeric_df, numeric_df.columns.tolist())
    return UploadResponse(session_id=session.session_id, columns=session.columns, n_rows=len(session.df))


@router.get("/scree", response_model=ScreeResponse)
async def fetch_scree(session_id: str) -> ScreeResponse:
    session = _get_session(session_id)
    if len(session.columns) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="数値列が不足しています。")

    scaler = StandardScaler()
    try:
        standardized = scaler.fit_transform(session.df[session.columns])
    except Exception as exc:  # pragma: no cover - propagate numeric issues
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="データの前処理に失敗しました。") from exc

    pca = PCA()
    try:
        pca.fit(standardized)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="固有値の計算に失敗しました。") from exc

    eigenvalues = pca.explained_variance_.tolist()
    explained_ratio = pca.explained_variance_ratio_.tolist()

    return ScreeResponse(eigenvalues=[float(value) for value in eigenvalues], explained_variance_ratio=[float(value) for value in explained_ratio])


@router.post("/run", response_model=FactorRunResponse)
async def run_factor_analysis(request: FactorRunRequest) -> FactorRunResponse:
    session = _get_session(request.session_id)

    columns = request.columns or session.columns
    if not columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="数値列が不足しています。")

    missing = [column for column in columns if column not in session.columns]
    if missing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"存在しない列が指定されました: {', '.join(missing)}")

    unique_columns = list(dict.fromkeys(columns))
    if len(unique_columns) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="数値列が不足しています。")

    if request.n_factors < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="因子数は1以上にしてください。")

    if request.n_factors > len(unique_columns):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="因子数が大きすぎます。")

    subset = session.df[unique_columns]
    if subset.empty:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="データの前処理に失敗しました。")

    scaler = StandardScaler()
    try:
        standardized = scaler.fit_transform(subset)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="データの前処理に失敗しました。") from exc

    analyzer = FactorAnalyzer(n_factors=request.n_factors, rotation="varimax")
    try:
        analyzer.fit(standardized)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="因子分析の計算に失敗しました。データを確認してください。") from exc

    loadings_matrix = analyzer.loadings_
    loadings: Dict[str, Dict[str, float]] = {}
    for idx, column in enumerate(unique_columns):
        loadings[column] = {f"F{factor_idx + 1}": float(value) for factor_idx, value in enumerate(loadings_matrix[idx])}

    communalities_array = analyzer.get_communalities()
    uniqueness_array = analyzer.get_uniquenesses()
    communalities = {column: float(value) for column, value in zip(unique_columns, communalities_array)}
    uniqueness = {column: float(value) for column, value in zip(unique_columns, uniqueness_array)}

    try:
        scores_array = analyzer.transform(standardized)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="因子得点の算出に失敗しました。") from exc

    factor_scores: List[Dict[str, float]] = []
    for row in scores_array:
        factor_scores.append({f"F{idx + 1}": float(value) for idx, value in enumerate(row)})

    return FactorRunResponse(
        loadings=loadings,
        communalities=communalities,
        uniqueness=uniqueness,
        factor_scores=factor_scores,
        n_rows=len(subset),
    )
