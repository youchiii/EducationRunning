from __future__ import annotations

import math
import time
from threading import Lock
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, model_validator

from ..security import get_current_user
from ..services.regression_advice import generate_advice


def _normalise_mapping(data: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not data:
        return {}
    normalised: Dict[str, float] = {}
    for key, value in data.items():
        if key is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        normalised[key_str] = float(numeric)
    return normalised


class TokenUsage(BaseModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)


class RegressionMetrics(BaseModel):
    r2: Optional[float] = Field(default=None)
    adj_r2: Optional[float] = Field(default=None)
    mae: Optional[float] = Field(default=None)
    mape: Optional[float] = Field(default=None)
    dw: Optional[float] = Field(default=None)
    n: int = Field(..., ge=1, le=1000000)


class ResidualsSummary(BaseModel):
    mean: Optional[float] = None
    std: Optional[float] = None
    skew: Optional[float] = None
    kurt: Optional[float] = None
    outliers_gt2: Optional[int] = Field(default=None, ge=0)


class RegressionAdviceResponse(BaseModel):
    advice: str
    model_used: str
    tokens: Optional[TokenUsage] = None


class RegressionAdviceRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    metrics: RegressionMetrics
    coefficients: Dict[str, float] = Field(default_factory=dict)
    std_coefficients: Optional[Dict[str, float]] = Field(default_factory=dict)
    pvalues: Optional[Dict[str, float]] = Field(default_factory=dict)
    vif: Optional[Dict[str, float]] = Field(default_factory=dict)
    residuals_summary: Optional[ResidualsSummary] = None
    notes: Optional[str] = Field(default=None, max_length=1000)
    target_name: str = Field(..., min_length=1, max_length=128)
    feature_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean(self) -> "RegressionAdviceRequest":
        self.session_id = self.session_id.strip()
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        self.target_name = self.target_name.strip()
        if not self.target_name:
            raise ValueError("target_name must not be empty")
        if self.notes is not None:
            self.notes = self.notes.strip()
            if not self.notes:
                self.notes = None
        self.coefficients = _normalise_mapping(self.coefficients)
        self.std_coefficients = _normalise_mapping(self.std_coefficients)
        self.pvalues = _normalise_mapping(self.pvalues)
        self.vif = _normalise_mapping(self.vif)
        if len(self.coefficients) > 128:
            raise ValueError("Too many coefficients supplied")
        cleaned_features = [name.strip() for name in self.feature_names if isinstance(name, str) and name.strip()]
        if len(cleaned_features) > 128:
            raise ValueError("Too many feature names supplied")
        self.feature_names = cleaned_features
        return self


class _MinuteRateLimiter:
    def __init__(self, limit: int = 5) -> None:
        self._limit = limit
        self._counts: Dict[tuple[str, int], int] = {}
        self._lock = Lock()

    def hit(self, key: str) -> None:
        current_minute = int(time.time() // 60)
        token = (key, current_minute)
        with self._lock:
            self._prune_locked(current_minute)
            count = self._counts.get(token, 0)
            if count >= self._limit:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="AIリクエストが多すぎます。しばらく待ってから再度お試しください。")
            self._counts[token] = count + 1

    def _prune_locked(self, current_minute: int) -> None:
        obsolete = [identifier for identifier in self._counts if identifier[1] < current_minute]
        for identifier in obsolete:
            self._counts.pop(identifier, None)


router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(get_current_user)])
_rate_limiter = _MinuteRateLimiter(limit=5)


@router.post("/regression-advice", response_model=RegressionAdviceResponse, response_class=ORJSONResponse)
async def fetch_regression_advice(request: Request, payload: RegressionAdviceRequest) -> RegressionAdviceResponse:
    client_host = request.client.host if request.client else "unknown"
    _rate_limiter.hit(f"ip:{client_host}")
    _rate_limiter.hit(f"session:{payload.session_id}")

    result = await generate_advice(payload)
    raw_tokens = result.get("tokens") or {}
    token_payload = {
        "input": int(raw_tokens.get("input") or raw_tokens.get("prompt") or 0),
        "output": int(raw_tokens.get("output") or raw_tokens.get("completion") or raw_tokens.get("candidates") or 0),
    }
    return RegressionAdviceResponse(
        advice=result.get("advice", ""),
        model_used=result.get("model_used", "unknown"),
        tokens=TokenUsage(**token_payload),
    )
