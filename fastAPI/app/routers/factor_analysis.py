from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Annotated, Dict, List, Optional, Union, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..security import get_current_user
from ..state import get_dataset_store


@dataclass
class FactorSession:
    session_id: str
    df: pd.DataFrame
    columns: List[str]
    created_at: datetime
    factor_scores: Optional[pd.DataFrame] = None
    factor_names: List[str] = field(default_factory=list)


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


class ColumnTarget(BaseModel):
    type: Literal["column"]
    name: str


class ArrayTarget(BaseModel):
    type: Literal["array"]
    values: List[float]


RegressionTarget = Annotated[Union[ColumnTarget, ArrayTarget], Field(discriminator="type")]


class FactorRegressionRequest(BaseModel):
    session_id: str
    target: RegressionTarget
    factors: Optional[List[str]] = Field(default=None)
    standardize_target: bool = Field(default=False)
    sample_limit: int = Field(default=1000, ge=10, le=10000)


class FactorRegressionResponse(BaseModel):
    coefficients: Dict[str, float]
    std_coefficients: Dict[str, float]
    pvalues: Dict[str, float]
    r2: float
    adj_r2: float
    dw: float
    vif: Dict[str, float]
    fitted: List[float]
    residuals: List[float]
    indices: List[int]
    qq_theoretical: List[float]
    qq_sample: List[float]
    used_factors: List[str]
    n: int


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


class DatasetSessionRequest(BaseModel):
    dataset_id: str


@router.post("/upload", response_model=UploadResponse)
async def upload_factor_dataset(file: UploadFile) -> UploadResponse:
    try:
        dataframe = _read_csv(file)
    finally:
        file.file.close()

    numeric_df = _prepare_dataframe(dataframe)
    session = session_store.create(numeric_df, numeric_df.columns.tolist())
    return UploadResponse(session_id=session.session_id, columns=session.columns, n_rows=len(session.df))


@router.post("/from-dataset", response_model=UploadResponse)
async def create_session_from_dataset(request: DatasetSessionRequest) -> UploadResponse:
    dataset_store = get_dataset_store()
    try:
        dataset_entry = dataset_store.get(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="データセットが見つかりません。") from exc

    numeric_df = _prepare_dataframe(dataset_entry.df)
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
    factor_names = [f"F{idx + 1}" for idx in range(request.n_factors)]
    loadings: Dict[str, Dict[str, float]] = {}
    for idx, column in enumerate(unique_columns):
        loadings[column] = {factor_names[factor_idx]: float(value) for factor_idx, value in enumerate(loadings_matrix[idx])}

    communalities_array = analyzer.get_communalities()
    uniqueness_array = analyzer.get_uniquenesses()
    communalities = {column: float(value) for column, value in zip(unique_columns, communalities_array)}
    uniqueness = {column: float(value) for column, value in zip(unique_columns, uniqueness_array)}

    try:
        scores_array = analyzer.transform(standardized)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="因子得点の算出に失敗しました。") from exc

    factor_scores_df = pd.DataFrame(scores_array, index=subset.index, columns=factor_names)
    session.factor_scores = factor_scores_df
    session.factor_names = factor_names

    factor_scores: List[Dict[str, float]] = []
    for row in factor_scores_df.itertuples(index=False, name=None):
        factor_scores.append({factor_names[idx]: float(value) for idx, value in enumerate(row)})

    return FactorRunResponse(
        loadings=loadings,
        communalities=communalities,
        uniqueness=uniqueness,
        factor_scores=factor_scores,
        n_rows=len(subset),
    )


def _ensure_factor_scores(session: FactorSession) -> pd.DataFrame:
    if session.factor_scores is None or session.factor_scores.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="まず因子分析を実行してください。")
    return session.factor_scores


def _coerce_index_to_int(index_value: object, fallback: int) -> int:
    try:
        return int(index_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - non-numeric indices
        return fallback


def _compute_vif(matrix: pd.DataFrame, factors: List[str]) -> Dict[str, float]:
    if matrix.empty:
        return {}
    if len(factors) == 1:
        return {factors[0]: 1.0}
    values = matrix.values
    vif_results: Dict[str, float] = {}
    for idx, factor in enumerate(factors):
        try:
            vif_value = float(variance_inflation_factor(values, idx))
        except Exception:  # pragma: no cover - numerical edge cases
            vif_value = float("nan")
        vif_results[factor] = vif_value
    return vif_results


@router.post("/regression", response_model=FactorRegressionResponse)
async def run_factor_regression(request: FactorRegressionRequest) -> FactorRegressionResponse:
    session = _get_session(request.session_id)
    factor_scores = _ensure_factor_scores(session)

    available_factors = session.factor_names or factor_scores.columns.tolist()
    if not available_factors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="因子得点が利用できません。")

    factors = request.factors or available_factors
    missing_factors = [factor for factor in factors if factor not in factor_scores.columns]
    if missing_factors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"存在しない因子が指定されました: {', '.join(missing_factors)}")

    X = factor_scores[factors].copy()

    if isinstance(request.target, ColumnTarget):
        if request.target.name not in session.df.columns:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="目的変数の列が見つかりません。")
        y_series = session.df[request.target.name]
    elif isinstance(request.target, ArrayTarget):
        if len(request.target.values) != len(session.df):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="目的変数の配列長が因子得点と一致しません。")
        y_series = pd.Series(request.target.values, index=session.df.index, name="target")
    else:  # pragma: no cover - guarded by discriminator
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="目的変数の指定が不正です。")

    aligned = pd.concat([X, y_series], axis=1).dropna()
    if aligned.empty:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="欠損が多すぎるため回帰分析ができません。")

    X_aligned = aligned[factors]
    y_aligned = aligned[y_series.name]

    if not np.issubdtype(y_aligned.dtype, np.number):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="目的変数が数値ではありません。")

    if request.standardize_target:
        scaler_y = StandardScaler()
        y_model = pd.Series(scaler_y.fit_transform(y_aligned.to_frame()).ravel(), index=y_aligned.index)
    else:
        y_model = y_aligned

    X_with_const = sm.add_constant(X_aligned, has_constant="add")
    model = sm.OLS(y_model, X_with_const)
    results = model.fit()

    coefficients = {key: float(value) for key, value in results.params.items()}
    pvalues = {key: float(value) for key, value in results.pvalues.items()}

    scaler_X = StandardScaler()
    X_std = scaler_X.fit_transform(X_aligned)
    y_std = StandardScaler().fit_transform(y_aligned.to_frame()).ravel()
    std_model = sm.OLS(y_std, X_std).fit()
    std_coefficients = {factor: float(value) for factor, value in zip(factors, std_model.params)}

    vif_values = _compute_vif(X_aligned, factors)

    residuals = results.resid
    fitted = results.fittedvalues
    n_obs = int(results.nobs)
    sample_limit = min(request.sample_limit, len(fitted))
    indices = list(range(len(fitted)))[:sample_limit]
    sample_fitted = [float(fitted.iloc[idx]) for idx in indices]
    sample_residuals = [float(residuals.iloc[idx]) for idx in indices]
    sample_indices = [_coerce_index_to_int(fitted.index[idx], idx) for idx in indices]

    qq = stats.probplot(residuals, dist="norm")
    qq_theoretical = qq[0][0]
    qq_sample_raw = qq[0][1]
    if len(qq_theoretical) > sample_limit:
        step = int(np.ceil(len(qq_theoretical) / sample_limit))
        qq_indices = list(range(0, len(qq_theoretical), step))[:sample_limit]
        qq_theoretical = qq_theoretical[qq_indices]
        qq_sample_raw = qq_sample_raw[qq_indices]
    qq_theoretical_list = [float(value) for value in qq_theoretical]
    qq_sample_list = [float(value) for value in qq_sample_raw]

    return FactorRegressionResponse(
        coefficients=coefficients,
        std_coefficients=std_coefficients,
        pvalues=pvalues,
        r2=float(results.rsquared),
        adj_r2=float(results.rsquared_adj),
        dw=float(durbin_watson(residuals)),
        vif=vif_values,
        fitted=sample_fitted,
        residuals=sample_residuals,
        indices=sample_indices,
        qq_theoretical=qq_theoretical_list,
        qq_sample=qq_sample_list,
        used_factors=factors,
        n=n_obs,
    )
