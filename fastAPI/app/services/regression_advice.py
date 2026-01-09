from __future__ import annotations

import asyncio
import math
import os
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

try:  # pragma: no cover - optional dependency guard
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency guard
    genai = None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..routers.ai import RegressionAdviceRequest

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
PROMPT_MAX_CHARS = 32000
REQUEST_TIMEOUT = 7.0
MAX_RETRIES = 1
GENERATION_CONFIG: Dict[str, Any] = {
    "temperature": 0.45,
    "top_p": 0.8,
    "top_k": 32,
    "max_output_tokens": 1024,
}
_SAFETY_SETTINGS = None
_CONFIGURED = False

PROMPT_TEMPLATE = """あなたは統計のプロダクトアナリストです。以下の重回帰結果と利用者の感想に基づき、\n「この先こういう可能性がある」「次にこう試すと良い」という**展望と具体的アクション**を日本語で提案してください。\n- 200〜400字程度の**要約**、3〜5個の**主要洞察**、2〜3個の**リスク/限界**、3〜5個の**次に試すこと**をMarkdownで返す\n- 専門用語は必要最小限。検証可能な提案にする（データ分割、交差検証、変換、交互作用、非線形、変数追加/削除、異常値処理、モデル比較など）\n- 誇張禁止。データの限界（n、外れ値、自己相関、VIF）に触れる\n- 出力はJSONの形にしない。以下の見出しでMarkdown整形のみ：\n  ## 要約\n  …本文…\n  ## 主要洞察\n  - …\n  ## リスク/限界\n  - …\n  ## 次に試すこと\n  - …\n\n入力（要約して使ってよい）:\n- 目的変数: {target_name}\n- サンプル数 n: {metrics_line}\n- 回帰係数: {coefficients_line}\n- 係数（標準化β）: {std_coefficients_line}\n- p値: {pvalues_line}\n- VIF: {vif_line}\n- 説明変数候補: {feature_names_line}\n- 残差サマリ: {residuals_summary_line}\n- ユーザーの感想: 「{notes_line}」\n"""

FALLBACK_TEMPLATE = """## 要約\nR²約{r2}で説明力は中程度。歩数や速度と心拍の関係を確認しつつ、外れ値や自己相関の影響を見直すと改善余地があります。\n## 主要洞察\n- 高いVIFがないか再確認する\n- 外れ値の振る舞いを可視化する\n- モデルの残差分布を点検する\n## リスク/限界\n- サンプル数{n}では複雑な仮説検証は難しい\n- 高心拍のデータ品質が分析を歪める可能性\n## 次に試すこと\n- クロスバリデーションで汎化性能を測る\n- 外れ値トリミングとロバスト回帰を比較\n- 心拍指標の変換や交互作用を追加検証\n"""


def _ensure_configured() -> None:
    global _CONFIGURED
    if _CONFIGURED or genai is None or not GEMINI_API_KEY:  # pragma: no branch - simple guard
        return
    genai.configure(api_key=GEMINI_API_KEY)
    _CONFIGURED = True


def _format_number(value: Optional[float]) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "NA"
    if isinstance(value, bool):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_mapping(mapping: Optional[Mapping[str, Any]], max_items: int = 12) -> str:
    if hasattr(mapping, "model_dump"):
        try:
            mapping = mapping.model_dump(exclude_none=True)
        except TypeError:
            mapping = mapping.model_dump()
    if not mapping:
        return "情報なし"
    formatted_items = []
    for key in mapping:
        if len(formatted_items) >= max_items:
            break
        value = mapping[key]
        formatted_items.append(f"{key}:{_format_number(_to_float(value))}")
    remainder = len(mapping) - len(formatted_items)
    suffix = f" …(+{remainder})" if remainder > 0 else ""
    return ", ".join(formatted_items) + suffix if formatted_items else "情報なし"


def _format_feature_names(names: Optional[list[str]], max_items: int = 16) -> str:
    if not names:
        return "情報なし"
    cleaned = [name.strip() for name in names if isinstance(name, str) and name.strip()]
    if not cleaned:
        return "情報なし"
    limited = cleaned[:max_items]
    suffix = f" …(+{len(cleaned) - len(limited)})" if len(cleaned) > len(limited) else ""
    return ", ".join(limited) + suffix


def _to_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_residuals_summary(summary: Optional[Mapping[str, Any]]) -> str:
    if hasattr(summary, "model_dump"):
        try:
            summary = summary.model_dump(exclude_none=True)
        except TypeError:
            summary = summary.model_dump()
    if not summary:
        return "情報なし"
    parts = []
    for key in ("mean", "std", "skew", "kurt", "outliers_gt2"):
        if key not in summary:
            continue
        label = {
            "mean": "平均",
            "std": "標準偏差",
            "skew": "歪度",
            "kurt": "尖度",
            "outliers_gt2": "|z|>2件数",
        }[key]
        value = summary[key]
        if key == "outliers_gt2":
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            parts.append(f"{label}:{count}")
        else:
            parts.append(f"{label}:{_format_number(_to_float(value))}")
    return ", ".join(parts) if parts else "情報なし"


def _sanitise_notes(notes: Optional[str]) -> str:
    if not notes:
        return "感想なし"
    cleaned = " ".join(notes.strip().split())
    return cleaned[:1000] if len(cleaned) > 1000 else cleaned


def _render_prompt(payload: "RegressionAdviceRequest") -> str:
    metrics = payload.metrics
    metrics_line = (
        f"{metrics.n}, R²={_format_number(metrics.r2)}（調整R²={_format_number(metrics.adj_r2)}）, "
        f"MAE={_format_number(metrics.mae)}, MAPE={_format_number(metrics.mape)}, "
        f"Durbin–Watson={_format_number(metrics.dw)}"
    )
    coefficients_line = _format_mapping(payload.coefficients)
    std_line = _format_mapping(payload.std_coefficients)
    pvalues_line = _format_mapping(payload.pvalues)
    vif_line = _format_mapping(payload.vif)
    feature_names_line = _format_feature_names(payload.feature_names)
    residuals_line = _format_residuals_summary(payload.residuals_summary)
    notes_line = _sanitise_notes(payload.notes)
    prompt = PROMPT_TEMPLATE.format(
        target_name=payload.target_name,
        metrics_line=metrics_line,
        coefficients_line=coefficients_line,
        std_coefficients_line=std_line,
        pvalues_line=pvalues_line,
        vif_line=vif_line,
        feature_names_line=feature_names_line,
        residuals_summary_line=residuals_line,
        notes_line=notes_line,
    )
    if len(prompt) > PROMPT_MAX_CHARS:
        raise ValueError("Prompt is too long")
    return prompt


def _build_fallback(payload: "RegressionAdviceRequest") -> Dict[str, Any]:
    metrics = payload.metrics
    r2 = _format_number(metrics.r2)
    rendered = FALLBACK_TEMPLATE.format(r2=r2, n=metrics.n)
    return {
        "advice": rendered.strip(),
        "model_used": "fallback",
        "tokens": {"input": 0, "output": 0},
    }


def _extract_tokens(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {"input": 0, "output": 0}
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    response_tokens = getattr(usage, "candidates_token_count", 0) or 0
    return {"input": int(prompt_tokens), "output": int(response_tokens)}


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    # google-generativeai may return candidates list.
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        combined: list[str] = []
        for part in parts:
            text_part = getattr(part, "text", None)
            if isinstance(text_part, str):
                combined.append(text_part)
        if combined:
            return "\n".join(segment.strip() for segment in combined if segment.strip()).strip()
    return ""


def _needs_retry(error: Exception) -> bool:
    if isinstance(error, asyncio.TimeoutError):
        return True
    retriable = (
        TimeoutError,
        OSError,
        RuntimeError,
        ValueError,
    )
    return isinstance(error, retriable)


async def _run_with_timeout(prompt: str) -> Any:
    if genai is None or not GEMINI_API_KEY:
        raise RuntimeError("Gemini SDK not available")
    _ensure_configured()
    model = genai.GenerativeModel(GEMINI_MODEL)
    call = model.generate_content
    return await asyncio.wait_for(
        asyncio.to_thread(call, prompt, generation_config=GENERATION_CONFIG, safety_settings=_SAFETY_SETTINGS),
        timeout=REQUEST_TIMEOUT,
    )


async def generate_advice(payload: "RegressionAdviceRequest") -> Dict[str, Any]:
    prompt: str
    try:
        prompt = _render_prompt(payload)
    except ValueError:
        return _build_fallback(payload)

    if genai is None or not GEMINI_API_KEY:
        return _build_fallback(payload)

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await _run_with_timeout(prompt)
            text = _extract_text(response)
            if not text:
                raise RuntimeError("Empty response from Gemini")
            return {
                "advice": text.strip(),
                "model_used": GEMINI_MODEL,
                "tokens": _extract_tokens(response),
            }
        except Exception as exc:  # pragma: no cover - depends on network
            last_error = exc
            if attempt >= MAX_RETRIES or not _needs_retry(exc):
                break
    return _build_fallback(payload)
