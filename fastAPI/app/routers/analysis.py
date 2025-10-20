from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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


class Coefficient(BaseModel):
    feature: str
    coefficient: float


class PredictionPair(BaseModel):
    actual: float
    predicted: float


class RegressionResponse(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    r_squared: float
    mse: float
    intercept: float
    coefficients: List[Coefficient]
    equation: str
    predictions: List[PredictionPair]


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


@router.post("/regression", response_model=RegressionResponse)
async def run_regression(payload: RegressionRequest) -> RegressionResponse:
    df = _resolve_dataset(payload.dataset_id)

    missing_columns = [col for col in [payload.target, *payload.features] if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Columns not found: {', '.join(missing_columns)}")

    data = df[[payload.target, *payload.features]].dropna()
    if data.empty or len(data) < 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not enough data after dropping missing values")

    X = data[payload.features]
    y = data[payload.target]

    test_size = payload.test_size
    if len(X) * test_size < 1:
        test_size = min(0.5, max(0.2, 1 / len(X)))

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if X_train.empty or X_test.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient data for regression split")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    mse_value = mean_squared_error(y_test, y_pred)

    coefficients = [Coefficient(feature=feature, coefficient=float(coef)) for feature, coef in zip(payload.features, model.coef_)]
    intercept = float(model.intercept_)
    equation_terms = [f"{coef.coefficient:.3f}×{coef.feature}" for coef in coefficients]
    equation = f"{payload.target} = {intercept:.3f} + " + " + ".join(equation_terms)

    predictions = [
        PredictionPair(actual=float(actual), predicted=float(pred))
        for actual, pred in zip(y_test, y_pred)
    ]

    return RegressionResponse(
        dataset_id=payload.dataset_id,
        target=payload.target,
        features=payload.features,
        r_squared=float(r_squared),
        mse=float(mse_value),
        intercept=intercept,
        coefficients=coefficients,
        equation=equation,
        predictions=predictions,
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
