from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Annotated, Dict, List, Optional, Union, Literal, Any
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
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None

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


class AutoNFactorsRequest(BaseModel):
    session_id: str
    target_cumvar: float = Field(default=0.7, description="Desired cumulative variance threshold (0.60-0.80)")
    pa_iter: int = Field(default=500, ge=50, le=5000, description="Number of iterations for parallel analysis")
    pa_percentile: float = Field(
        default=95.0,
        ge=50.0,
        le=99.5,
        description="Percentile used for parallel analysis threshold",
    )
    max_factors: Optional[int] = Field(default=None, ge=1, le=50, description="Optional upper bound for factor count")


class AutoNFactorsResponse(BaseModel):
    recommended_n: int
    by_rule: Dict[str, int]
    cumvar: List[float]
    eigenvalues: List[float]
    pa_threshold: List[float]
    kaiser: float
    rationale: str
    n_samples: int
    n_vars: int
    target_cumvar: float
    pa_percentile: float


class AutoNFactorsExplainResponse(BaseModel):
    explanation: str


GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def explain_with_gemini(payload: Dict[str, Any]) -> str:
    fallback_primary = (
        f"PA, 肘法, 累積説明率の結果を統合し、{payload['recommended_n']}因子を推奨。"
        "以降の固有値はランダム水準に近く、説明力の伸びが小さいためです。"
    )
    fallback_secondary = (
        f"{payload['recommended_n']}因子を推奨。PAしきい値を上回る因子のみ採用し、"
        "肘法と累積説明率（目安70％）も整合したためです。"
    )

    if not GEMINI_KEY or genai is None:
        return fallback_primary[:200]

    try:
        genai.configure(api_key=GEMINI_KEY)
        prompt = (
            "あなたは統計アシスタントです。以下に基づいて、"
            "なぜ推奨因子数 n が選ばれたのかを日本語で200字以内に説明してください。"
            "専門語は最小限、簡潔に。\n\n"
            f"変数数:{payload['n_vars']} サンプル数:{payload['n_samples']}\n"
            f"固有値:{payload['eigenvalues']}\n"
            f"PAしきい値:{payload['pa_threshold']}\n"
            f"累積説明率(％):{[round(c * 100, 1) for c in payload['cumvar']]}\n"
            f"各基準: PA={payload['by_rule']['pa']}, Kaiser={payload['by_rule']['kaiser']}, "
            f"肘={payload['by_rule']['elbow']}, 累積={payload['by_rule']['cum']}\n"
            f"推奨: n={payload['recommended_n']}（優先度: PA>肘>累積>Kaiser）"
        )
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        text = text.strip()
        if not text:
            return fallback_secondary[:200]
        return text[:220]
    except Exception:  # pragma: no cover - network/SDK failures
        return fallback_secondary[:200]


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


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _parallel_analysis_thresholds(
    n_samples: int,
    n_vars: int,
    iterations: int,
    percentile: float,
) -> np.ndarray:
    rng = default_rng()
    eigen_samples = np.empty((iterations, n_vars), dtype=np.float64)
    for idx in range(iterations):
        random_data = rng.standard_normal((n_samples, n_vars))
        corr = np.corrcoef(random_data, rowvar=False)
        corr = np.nan_to_num((corr + corr.T) / 2.0, nan=0.0)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.sort(np.real(eigvals))[::-1]
        eigen_samples[idx, :] = eigvals
    thresholds = np.percentile(eigen_samples, percentile, axis=0)
    return thresholds


def _detect_elbow(eigenvalues: np.ndarray) -> int:
    if eigenvalues.size <= 1:
        return 1
    x = np.arange(1, eigenvalues.size + 1, dtype=np.float64)
    y = eigenvalues.astype(np.float64)
    start = np.array([1.0, y[0]])
    end = np.array([float(eigenvalues.size), y[-1]])
    line_vec = end - start
    line_length = np.hypot(line_vec[0], line_vec[1])
    if line_length == 0:
        distances = np.abs(y - y.mean())
    else:
        distances = np.abs((line_vec[1] * (x - start[0]) - line_vec[0] * (y - start[1])) / line_length)
    elbow_index = int(np.argmax(distances))
    return elbow_index + 1


def _format_rule_label(rule: str, target_cumvar: float) -> str:
    mapping = {
        "pa": "PA",
        "kaiser": "Kaiser",
        "elbow": "肘法",
        "cum": f"累積説明率(≥{target_cumvar:.0%})",
    }
    return mapping.get(rule, rule)


def _join_rule_labels(labels: List[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]}と{labels[1]}"
    return "、".join(labels[:-1]) + f"、{labels[-1]}"


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


def _compute_auto_n_factors(session: FactorSession, request: AutoNFactorsRequest) -> AutoNFactorsResponse:
    columns = session.columns
    if len(columns) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="数値列が不足しています。")

    subset = session.df[columns]
    scaler = StandardScaler()
    try:
        standardized = scaler.fit_transform(subset)
    except Exception as exc:  # pragma: no cover - propagate numeric issues
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="データの前処理に失敗しました。") from exc

    n_samples, n_vars = standardized.shape
    if n_samples < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="サンプル数が少なすぎます。")

    corr = np.corrcoef(standardized, rowvar=False)
    corr = np.nan_to_num((corr + corr.T) / 2.0, nan=0.0)

    try:
        eigenvalues = np.linalg.eigvalsh(corr)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical issues
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="固有値の計算に失敗しました。") from exc

    eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

    total_variance = float(np.sum(eigenvalues))
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="固有値の計算結果が不正です。")

    target_cumvar = _clamp_float(request.target_cumvar, 0.60, 0.80)
    max_allowed = request.max_factors or n_vars
    max_allowed = _clamp_int(max_allowed, 1, n_vars)

    pa_thresholds = _parallel_analysis_thresholds(n_samples, n_vars, request.pa_iter, request.pa_percentile)
    pa_thresholds = np.asarray(pa_thresholds, dtype=np.float64)

    comparisons = np.where(eigenvalues > pa_thresholds)[0]
    n_pa = int(comparisons[-1] + 1) if comparisons.size else 1
    n_pa = _clamp_int(n_pa, 1, max_allowed)

    n_kaiser = int(np.count_nonzero(eigenvalues > 1.0))
    n_kaiser = _clamp_int(max(1, n_kaiser), 1, max_allowed)

    n_elbow = _detect_elbow(eigenvalues)
    if eigenvalues[n_elbow - 1] <= 1e-6:
        n_elbow = 1
    n_elbow = _clamp_int(n_elbow, 1, max_allowed)

    cumulative = np.cumsum(eigenvalues) / total_variance
    cum_indices = np.where(cumulative >= target_cumvar)[0]
    n_cum = int(cum_indices[0] + 1) if cum_indices.size else n_vars
    n_cum = _clamp_int(n_cum, 1, max_allowed)

    by_rule = {"pa": n_pa, "kaiser": n_kaiser, "elbow": n_elbow, "cum": n_cum}

    vote_counter: Dict[int, int] = {}
    for value in by_rule.values():
        vote_counter[value] = vote_counter.get(value, 0) + 1

    best_count = max(vote_counter.values())
    candidates = [value for value, count in vote_counter.items() if count == best_count]
    if len(candidates) == 1:
        recommended = candidates[0]
    else:
        priority = [("pa", n_pa), ("elbow", n_elbow), ("cum", n_cum), ("kaiser", n_kaiser)]
        recommended = candidates[0]
        for rule, value in priority:
            if value in candidates:
                recommended = value
                break

    supporting_rules = [rule for rule, value in by_rule.items() if value == recommended]
    supporting_labels = [_format_rule_label(rule, target_cumvar) for rule in supporting_rules]
    support_text = _join_rule_labels(supporting_labels)

    ordered_rules = ["pa", "elbow", "cum", "kaiser"]
    other_components = [
        f"{_format_rule_label(rule, target_cumvar)}={by_rule[rule]}"
        for rule in ordered_rules
        if rule not in supporting_rules
    ]

    if support_text:
        rationale = f"{support_text}が一致したため n={recommended} を推奨。"
    else:
        primary_rule = next(rule for rule in ordered_rules if by_rule[rule] == recommended)
        rationale = f"{_format_rule_label(primary_rule, target_cumvar)}を優先し n={recommended} を推奨。"

    if other_components:
        rationale += f" 他: {', '.join(other_components)}"

    if n_samples < 60:
        rationale += f" サンプル数が{n_samples}件と少ないため判断が不安定になりやすい点に注意してください。"

    response = AutoNFactorsResponse(
        recommended_n=int(recommended),
        by_rule={key: int(value) for key, value in by_rule.items()},
        cumvar=[float(value) for value in cumulative.tolist()],
        eigenvalues=[float(value) for value in eigenvalues.tolist()],
        pa_threshold=[float(value) for value in pa_thresholds.tolist()],
        kaiser=1.0,
        rationale=rationale,
        n_samples=int(n_samples),
        n_vars=int(n_vars),
        target_cumvar=float(target_cumvar),
        pa_percentile=float(request.pa_percentile),
    )

    return response


@router.post("/auto_n_factors", response_model=AutoNFactorsResponse)
async def auto_select_n_factors(request: AutoNFactorsRequest) -> AutoNFactorsResponse:
    session = _get_session(request.session_id)
    return _compute_auto_n_factors(session, request)


@router.post("/auto_n_factors_explain", response_model=AutoNFactorsExplainResponse)
async def auto_select_n_factors_explain(request: AutoNFactorsRequest) -> AutoNFactorsExplainResponse:
    session = _get_session(request.session_id)
    result = _compute_auto_n_factors(session, request)
    explanation = explain_with_gemini(result.model_dump())
    return AutoNFactorsExplainResponse(explanation=explanation)


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
