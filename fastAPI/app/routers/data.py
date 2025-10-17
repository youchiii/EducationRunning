from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency, ttest_ind

from ..security import get_current_user
from ..state import get_dataset_store

router = APIRouter(prefix="/data", tags=["data"], dependencies=[Depends(get_current_user)])


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    original_name: str
    row_count: int
    column_count: int
    preview: List[Dict[str, object]]
    numeric_columns: List[str]
    categorical_columns: List[str]


class DatasetStatsResponse(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    numeric_columns: List[str]
    basic_statistics: Dict[str, Dict[str, Optional[float]]]
    correlation_matrix: Dict[str, Dict[str, Optional[float]]]


class SeriesResponse(BaseModel):
    dataset_id: str
    column: str
    values: List[Optional[float]]


class ChiSquareRequest(BaseModel):
    column_a: str = Field(..., description="First categorical column")
    column_b: str = Field(..., description="Second categorical column")


class ChiSquareResponse(BaseModel):
    chi2: float
    p_value: float
    dof: int
    significant: bool
    contingency_table: List[Dict[str, object]]


class TTestRequest(BaseModel):
    numeric_column: str
    group_column: str
    group_values: Optional[List[str]] = Field(
        default=None, description="If provided, specifies the two group values to compare"
    )


class TTestResponse(BaseModel):
    t_statistic: float
    p_value: float
    group_a: str
    group_b: str
    significant: bool
    mean_a: float
    mean_b: float


def _read_csv_content(upload: UploadFile) -> pd.DataFrame:
    upload.file.seek(0)
    raw_bytes = upload.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    for encoding in ("utf-8", "utf-8-sig", "shift_jis"):
        try:
            buffer = io.BytesIO(raw_bytes)
            df = pd.read_csv(buffer, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid CSV content") from exc
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported CSV encoding")


def _infer_categorical_columns(df: pd.DataFrame) -> List[str]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in df.columns:
        if column in cat_cols:
            continue
        unique_count = df[column].nunique(dropna=True)
        if 1 < unique_count <= 10:
            cat_cols.append(column)
    return cat_cols


def _safe_preview(df: pd.DataFrame) -> List[Dict[str, object]]:
    preview_df = df.head(50).copy()
    preview_df = preview_df.fillna("")
    return preview_df.to_dict(orient="records")


def _basic_stats(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            stats[column] = {
                "mean": None,
                "median": None,
                "mode": None,
                "variance": None,
                "std_dev": None,
            }
            continue
        mode_series = series.mode()
        mode_value: Optional[float] = None
        if not mode_series.empty:
            mode_value = float(mode_series.iloc[0])
        stats[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "mode": mode_value,
            "variance": float(series.var()) if len(series) > 1 else 0.0,
            "std_dev": float(series.std()) if len(series) > 1 else 0.0,
        }
    return stats


def _correlation_matrix(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if not numeric_columns:
        return {}
    corr_df = df[numeric_columns].corr()
    # Ensure NaNs become None for JSON serialisation
    corr_df = corr_df.replace({np.nan: None})
    return {row: {col: (None if pd.isna(value) else float(value)) for col, value in series.items()} for row, series in corr_df.iterrows()}


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile) -> DatasetUploadResponse:
    try:
        dataframe = _read_csv_content(file)
    finally:
        file.file.close()

    dataset_store = get_dataset_store()
    entry = dataset_store.add(dataframe, file.filename or "dataset.csv")

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = _infer_categorical_columns(dataframe)

    return DatasetUploadResponse(
        dataset_id=entry.dataset_id,
        original_name=entry.original_name,
        row_count=int(len(dataframe)),
        column_count=int(len(dataframe.columns)),
        preview=_safe_preview(dataframe),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


@router.get("/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def dataset_stats(dataset_id: str) -> DatasetStatsResponse:
    dataset_store = get_dataset_store()
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found") from exc

    df = entry.df
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return DatasetStatsResponse(
        dataset_id=dataset_id,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        numeric_columns=numeric_columns,
        basic_statistics=_basic_stats(df, numeric_columns),
        correlation_matrix=_correlation_matrix(df, numeric_columns),
    )


@router.get("/{dataset_id}/column/{column}/series", response_model=SeriesResponse)
async def column_series(dataset_id: str, column: str) -> SeriesResponse:
    dataset_store = get_dataset_store()
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found") from exc

    df = entry.df
    if column not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Column not found")

    series = df[column]
    values: List[Optional[float]] = []
    for value in series.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            values.append(None)
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(None)

    return SeriesResponse(dataset_id=dataset_id, column=column, values=values)


@router.post("/{dataset_id}/tests/chi2", response_model=ChiSquareResponse)
async def chi_square(dataset_id: str, payload: ChiSquareRequest) -> ChiSquareResponse:
    dataset_store = get_dataset_store()
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found") from exc

    df = entry.df
    if payload.column_a not in df.columns or payload.column_b not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid column selection")

    contingency = pd.crosstab(df[payload.column_a], df[payload.column_b])
    if contingency.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient data for chi-square test")

    chi2, p_value, dof, _ = chi2_contingency(contingency)
    significant = p_value < 0.05

    return ChiSquareResponse(
        chi2=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        significant=significant,
        contingency_table=[{"index": idx, **{str(col): value for col, value in row.items()}} for idx, row in contingency.iterrows()],
    )


@router.post("/{dataset_id}/tests/t", response_model=TTestResponse)
async def t_test(dataset_id: str, payload: TTestRequest) -> TTestResponse:
    dataset_store = get_dataset_store()
    try:
        entry = dataset_store.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found") from exc

    df = entry.df
    if payload.numeric_column not in df.columns or payload.group_column not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid column selection")

    groups = df[payload.group_column].dropna().unique().tolist()
    if len(groups) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Group column must contain at least two unique values")

    selected_groups: List[str]
    if payload.group_values:
        if len(payload.group_values) != 2:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Specify exactly two group values")
        selected_groups = payload.group_values
    else:
        selected_groups = [str(groups[0]), str(groups[1])]

    data_a = df[df[payload.group_column] == selected_groups[0]][payload.numeric_column].dropna()
    data_b = df[df[payload.group_column] == selected_groups[1]][payload.numeric_column].dropna()

    if data_a.empty or data_b.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Selected groups have insufficient data")

    t_stat, p_value = ttest_ind(data_a, data_b, equal_var=False)
    significant = p_value < 0.05

    return TTestResponse(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        group_a=str(selected_groups[0]),
        group_b=str(selected_groups[1]),
        significant=significant,
        mean_a=float(data_a.mean()),
        mean_b=float(data_b.mean()),
    )
