from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from ..security import get_current_user
from ..state import get_dataset_store

router = APIRouter(prefix="/analysis", tags=["analysis"], dependencies=[Depends(get_current_user)])


class RegressionRequest(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ensure_features(self) -> "RegressionRequest":
        if not self.features:
            raise ValueError("At least one feature must be selected")
        if self.target in self.features:
            raise ValueError("Target column must not be included in features")
        return self


class RegressionResponse(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    coefficients: Dict[str, float | None]
    std_coefficients: Dict[str, float | None]
    standard_errors: Dict[str, float | None]
    pvalues: Dict[str, float | None]
    intercept: float | None
    r_squared: float | None
    adjusted_r_squared: float | None
    mae: float | None
    mape: float | None
    dw: float | None
    y_true: List[float | None]
    y_pred: List[float | None]
    residuals: List[float | None]
    std_residuals: List[float | None]
    qq_theoretical: List[float | None]
    qq_sample: List[float | None]
    vif: Dict[str, float | None]
    n: int
    index: List[str]


class FactorAnalysisRequest(BaseModel):
    dataset_id: str
    columns: List[str]
    n_components: int

    @model_validator(mode="after")
    def _validate(self) -> "FactorAnalysisRequest":
        if len(self.columns) < 2:
            raise ValueError("Select at least two columns for factor analysis")
        if self.n_components < 1:
            raise ValueError("Number of factors must be at least 1")
        if self.n_components >= len(self.columns):
            raise ValueError("Number of factors must be less than the number of columns")
        return self


class FactorAnalysisResponse(BaseModel):
    dataset_id: str
    columns: List[str]
    n_components: int
    factor_loadings: List[Dict[str, float]]
    factor_scores_preview: List[Dict[str, float]]
    factor_scores: List[Dict[str, float]]
    explained_variance_ratio: List[float]
    cumulative_variance_ratio: List[float]


class FactorRegressionRequest(BaseModel):
    factor_scores: List[Dict[str, float]]
    target_column: str

    @model_validator(mode="after")
    def _validate(self) -> "FactorRegressionRequest":
        if not self.factor_scores:
            raise ValueError("Factor scores must be provided")
        if not self.target_column:
            raise ValueError("Target column is required")
        return self


class FactorRegressionResponse(BaseModel):
    coefficients: Dict[str, float]
    pvalues: Dict[str, float]
    r2: float
    adj_r2: float
    f_pvalue: float | None
    residuals: List[float]
    fitted_values: List[float]


def _resolve_dataset(dataset_id: str) -> pd.DataFrame:
    dataset_store = get_dataset_store()
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found") from exc
    return entry.df


def _sanitize_number(value: float | int | np.generic | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def _sanitize_sequence(values: Iterable[float | int | np.generic | None]) -> List[float | None]:
    return [_sanitize_number(value) for value in values]


def _sanitize_mapping(mapping: Mapping[str, float | int | np.generic | None]) -> Dict[str, float | None]:
    return {key: _sanitize_number(value) for key, value in mapping.items()}


def _compute_vif(matrix: pd.DataFrame) -> Dict[str, float | None]:
    if matrix.empty:
        return {}
    if matrix.shape[1] == 1:
        column = matrix.columns[0]
        return {column: 1.0}
    values = matrix.values
    result: Dict[str, float | None] = {}
    for idx, column in enumerate(matrix.columns):
        vif_value = variance_inflation_factor(values, idx)
        result[column] = _sanitize_number(vif_value)
    return result


@router.post("/regression", response_model=RegressionResponse)
async def run_regression(payload: RegressionRequest) -> RegressionResponse:
    df = _resolve_dataset(payload.dataset_id)

    missing_columns = [col for col in [payload.target, *payload.features] if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Columns not found: {', '.join(missing_columns)}")

    data = df[[payload.target, *payload.features]].dropna()
    if data.empty or len(data) < 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not enough data after dropping missing values")

    X = data[payload.features]
    y = data[payload.target]

    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    residuals = results.resid
    fitted = results.fittedvalues
    n_obs = int(results.nobs)

    mae_value = mean_absolute_error(y, fitted)

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_array = np.where(y != 0, np.abs((y - fitted) / y), np.nan)
    mape_value = float(np.nanmean(mape_array) * 100) if np.isfinite(np.nanmean(mape_array)) else None

    influence = results.get_influence()
    std_residuals = influence.resid_studentized_internal

    qq_theoretical, qq_sample = stats.probplot(residuals, dist="norm")[0]

    scaler_X = StandardScaler()
    X_std = scaler_X.fit_transform(X)
    y_std = StandardScaler().fit_transform(y.to_frame()).ravel()
    std_model = sm.OLS(y_std, sm.add_constant(X_std, has_constant="add")).fit()
    std_coefficients_raw = {
        feature: float(value)
        for feature, value in zip(["const", *payload.features], std_model.params)
    }

    coefficients_raw = {name: float(value) for name, value in results.params.items()}
    standard_errors_raw = {name: float(value) for name, value in results.bse.items()}
    pvalues_raw = {name: float(value) for name, value in results.pvalues.items()}

    coefficients = _sanitize_mapping(coefficients_raw)
    standard_errors = _sanitize_mapping(standard_errors_raw)
    pvalues = _sanitize_mapping(pvalues_raw)
    std_coefficients = _sanitize_mapping(std_coefficients_raw)

    intercept_value = _sanitize_number(coefficients_raw.get("const", results.params.get("const", 0.0)))
    r_squared_value = _sanitize_number(results.rsquared)
    adjusted_r_squared_value = _sanitize_number(results.rsquared_adj)
    mae_sanitized = _sanitize_number(mae_value)
    mape_sanitized = _sanitize_number(mape_value)
    dw_value = _sanitize_number(durbin_watson(residuals))

    y_true_list = _sanitize_sequence(y.tolist())
    y_pred_list = _sanitize_sequence(fitted)
    residuals_list = _sanitize_sequence(residuals)
    std_residuals_list = _sanitize_sequence(std_residuals)
    qq_theoretical_list = _sanitize_sequence(qq_theoretical)
    qq_sample_list = _sanitize_sequence(qq_sample)

    vif_values = _compute_vif(X)

    return RegressionResponse(
        dataset_id=payload.dataset_id,
        target=payload.target,
        features=payload.features,
        coefficients=coefficients,
        std_coefficients={key: value for key, value in std_coefficients.items() if key != "const"},
        standard_errors={key: value for key, value in standard_errors.items() if key != "const"},
        pvalues=pvalues,
        intercept=intercept_value,
        r_squared=r_squared_value,
        adjusted_r_squared=adjusted_r_squared_value,
        mae=mae_sanitized,
        mape=mape_sanitized,
        dw=dw_value,
        y_true=y_true_list,
        y_pred=y_pred_list,
        residuals=residuals_list,
        std_residuals=std_residuals_list,
        qq_theoretical=qq_theoretical_list,
        qq_sample=qq_sample_list,
        vif=vif_values,
        n=n_obs,
        index=[str(idx) for idx in y.index],
    )


@router.post("/factor", response_model=FactorAnalysisResponse)
async def run_factor_analysis(payload: FactorAnalysisRequest) -> FactorAnalysisResponse:
    df = _resolve_dataset(payload.dataset_id)

    missing_columns = [col for col in payload.columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Columns not found: {', '.join(missing_columns)}")

    data = df[payload.columns].dropna()
    if data.empty or len(data) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient rows for factor analysis")

    fa = FactorAnalysis(n_components=payload.n_components, random_state=42)
    try:
        transformed = fa.fit_transform(data)
    except Exception as exc:  # pragma: no cover - propagate numeric issues
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    loadings_matrix = fa.components_.T
    factor_loadings = []
    for column, row in zip(payload.columns, loadings_matrix):
        factor_loadings.append({
            "variable": column,
            **{f"factor_{i + 1}": float(value) for i, value in enumerate(row)},
        })

    factor_scores_preview = []
    preview_limit = min(5, len(transformed))
    for preview_idx in range(preview_limit):
        vector = transformed[preview_idx]
        factor_scores_preview.append(
            {
                "index": int(preview_idx),
                **{f"factor_{i + 1}": float(value) for i, value in enumerate(vector)},
            }
        )

    factor_scores: List[Dict[str, float]] = []
    for position, (row_index, vector) in enumerate(zip(data.index, transformed)):
        try:
            numeric_index = int(row_index)
        except (TypeError, ValueError):
            numeric_index = position
        factor_scores.append(
            {
                "row_index": int(numeric_index),
                **{f"factor_{i + 1}": float(value) for i, value in enumerate(vector)},
            }
        )

    eigenvalues = np.sum(fa.components_ ** 2, axis=1)
    total_variance = float(np.sum(eigenvalues))
    if total_variance <= 0:
        explained = [0.0 for _ in eigenvalues]
    else:
        explained = [float(value / total_variance) for value in eigenvalues]
    cumulative = list(np.cumsum(explained))

    return FactorAnalysisResponse(
        dataset_id=payload.dataset_id,
        columns=payload.columns,
        n_components=payload.n_components,
        factor_loadings=factor_loadings,
        factor_scores_preview=factor_scores_preview,
        factor_scores=factor_scores,
        explained_variance_ratio=explained,
        cumulative_variance_ratio=[float(value) for value in cumulative],
    )


@router.post("/factor/regression", response_model=FactorRegressionResponse)
async def run_factor_regression(payload: FactorRegressionRequest) -> FactorRegressionResponse:
    try:
        df = pd.DataFrame(payload.factor_scores)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Factor scores are empty")

    if payload.target_column not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Target column not found in factor scores")

    if "row_index" in df.columns:
        df = df.drop(columns=["row_index"])

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    filtered_df = numeric_df.dropna()
    if filtered_df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient data after cleaning")

    if payload.target_column not in filtered_df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Target column is not numeric")

    feature_columns = [col for col in filtered_df.columns if col != payload.target_column]
    if not feature_columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No factor columns available for regression")

    X = filtered_df[feature_columns]
    y = filtered_df[payload.target_column]

    if len(X) <= len(feature_columns):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not enough observations for regression")

    X = sm.add_constant(X, has_constant="add")
    try:
        model = sm.OLS(y, X).fit()
    except Exception as exc:  # pragma: no cover - propagate numeric issues
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    f_pvalue = None
    if model.f_pvalue is not None and not math.isnan(model.f_pvalue):
        f_pvalue = float(model.f_pvalue)

    return FactorRegressionResponse(
        coefficients={key: float(value) for key, value in model.params.items()},
        pvalues={key: float(value) for key, value in model.pvalues.items()},
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        f_pvalue=f_pvalue,
        residuals=[float(value) for value in model.resid.tolist()],
        fitted_values=[float(value) for value in model.fittedvalues.tolist()],
    )
