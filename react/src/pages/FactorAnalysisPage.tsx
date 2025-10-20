import { useEffect, useMemo, useState } from "react";
import { AlertCircle, UploadCloud, X } from "lucide-react";
import type { AxiosError } from "axios";

import Header from "../components/Header";
import { Button } from "../components/ui/button";
import PlotlyChart from "../components/PlotlyChart";
import {
  fetchFactorScree,
  runFactorAnalysisSession,
  uploadFactorDataset,
  type FactorRunResponse,
  type FactorScreeResponse,
  type FactorUploadResponse,
} from "../services/api";

const PRIMARY_COLOR = "#2563eb";
const SECONDARY_COLOR = "#ea580c";
const FACTOR_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR, "#14b8a6", "#a855f7", "#f97316", "#0ea5e9"];
const MAX_FACTORS = 6;

const formatError = (error: unknown, fallback: string) => {
  const axiosError = error as AxiosError<{ detail?: string }>;
  if (axiosError?.response?.data?.detail) {
    return axiosError.response.data.detail;
  }
  if (axiosError?.message) {
    return axiosError.message;
  }
  return fallback;
};

const sampleScores = (scores: Array<Record<string, number>>, limit: number) => {
  if (scores.length <= limit) {
    return scores;
  }
  const step = Math.ceil(scores.length / limit);
  return scores.filter((_, index) => index % step === 0);
};

const FactorAnalysisPage = () => {
  const [session, setSession] = useState<FactorUploadResponse | null>(null);
  const [scree, setScree] = useState<FactorScreeResponse | null>(null);
  const [nFactors, setNFactors] = useState(2);
  const [result, setResult] = useState<FactorRunResponse | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isUploading, setIsUploading] = useState(false);
  const [isFetchingScree, setIsFetchingScree] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeColumns = useMemo(() => {
    if (!session) {
      return [] as string[];
    }
    if (!showAdvanced || selectedColumns.length === 0) {
      return session.columns;
    }
    return selectedColumns;
  }, [session, showAdvanced, selectedColumns]);

  const maxSelectableFactors = useMemo(() => {
    if (!activeColumns.length) {
      return 1;
    }
    return Math.max(1, Math.min(MAX_FACTORS, activeColumns.length));
  }, [activeColumns]);

  useEffect(() => {
    if (nFactors > maxSelectableFactors) {
      setNFactors(maxSelectableFactors);
    }
    if (nFactors < 1) {
      setNFactors(1);
    }
  }, [nFactors, maxSelectableFactors]);

  const handleUpload = async (file: File | null) => {
    if (!file) {
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const uploadResponse = await uploadFactorDataset(file);
      setSession(uploadResponse);
      setSelectedColumns(uploadResponse.columns);
      setShowAdvanced(false);
      const initialFactors = Math.min(MAX_FACTORS, Math.max(1, Math.min(3, uploadResponse.columns.length)));
      setNFactors(initialFactors);
      setResult(null);
      setIsFetchingScree(true);
      const screeResponse = await fetchFactorScree(uploadResponse.session_id);
      setScree(screeResponse);
    } catch (requestError) {
      setSession(null);
      setScree(null);
      setResult(null);
      setError(formatError(requestError, "CSVの読み込みに失敗しました。"));
    } finally {
      setIsUploading(false);
      setIsFetchingScree(false);
    }
  };

  const toggleColumn = (column: string) => {
    setSelectedColumns((prev) => (prev.includes(column) ? prev.filter((item) => item !== column) : [...prev, column]));
  };

  const handleSelectAll = () => {
    if (!session) {
      return;
    }
    setSelectedColumns(session.columns);
  };

  const handleRun = async () => {
    if (!session) {
      setError("まずはCSVファイルをアップロードしてください。");
      return;
    }
    const columnsForRequest = showAdvanced ? selectedColumns : session.columns;
    if (columnsForRequest.length < 2) {
      setError("因子分析には2つ以上の数値列が必要です。");
      return;
    }
    if (nFactors > columnsForRequest.length) {
      setError("因子数が大きすぎます。列数以下に設定してください。");
      return;
    }
    setError(null);
    setIsRunning(true);
    try {
      const response = await runFactorAnalysisSession(
        session.session_id,
        nFactors,
        showAdvanced ? columnsForRequest : undefined,
      );
      setResult(response);
    } catch (requestError) {
      setResult(null);
      setError(formatError(requestError, "因子分析の実行に失敗しました。"));
    } finally {
      setIsRunning(false);
    }
  };

  const factorKeys = useMemo(() => {
    if (!result || !Object.keys(result.loadings).length) {
      return [] as string[];
    }
    const firstVariable = Object.keys(result.loadings)[0];
    return Object.keys(result.loadings[firstVariable]);
  }, [result]);

  const heatMapCells = (value: number) => {
    const magnitude = Math.min(1, Math.abs(value));
    const positiveColor = `rgba(37, 99, 235, ${0.2 + 0.6 * magnitude})`;
    const negativeColor = `rgba(234, 88, 12, ${0.2 + 0.6 * magnitude})`;
    return {
      backgroundColor: value >= 0 ? positiveColor : negativeColor,
      color: "#ffffff",
      fontWeight: 600,
    } as const;
  };

  const scatterData = useMemo(() => {
    if (!result || factorKeys.length < 2) {
      return null;
    }
    const sample = sampleScores(result.factor_scores, 2000);
    return {
      data: [
        {
          x: sample.map((row) => row[factorKeys[0]] ?? 0),
          y: sample.map((row) => row[factorKeys[1]] ?? 0),
          mode: "markers",
          marker: {
            color: PRIMARY_COLOR,
            opacity: sample.length > 500 ? 0.45 : 0.65,
            size: 6,
          },
          type: "scatter",
        } as Record<string, unknown>,
      ],
      layout: {
        height: 360,
        margin: { l: 48, r: 24, t: 24, b: 48 },
        xaxis: { title: factorKeys[0] },
        yaxis: { title: factorKeys[1] },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
      } as Record<string, unknown>,
    };
  }, [result, factorKeys]);

  const timeSeriesData = useMemo(() => {
    if (!result || factorKeys.length === 0) {
      return null;
    }
    const sample = sampleScores(result.factor_scores, 1500);
    const indices = sample.map((_, index) => index);
    const traces = factorKeys.map((factorKey, index) => ({
      x: indices,
      y: sample.map((row) => row[factorKey] ?? 0),
      type: "scatter",
      mode: "lines",
      line: { color: FACTOR_PALETTE[index % FACTOR_PALETTE.length], width: 2 },
      name: factorKey,
    }));

    return {
      data: traces.map((trace) => trace as Record<string, unknown>),
      layout: {
        height: 360,
        margin: { l: 48, r: 24, t: 24, b: 48 },
        xaxis: { title: "インデックス" },
        yaxis: { title: "因子スコア" },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        legend: { orientation: "h" },
      } as Record<string, unknown>,
    };
  }, [result, factorKeys]);

  return (
    <div className="space-y-6">
      <Header
        title="因子分析"
        subtitle="CSVをアップロードし、スクリープロットを参考に因子数を選んでください。標準化とVarimax回転は自動で行われます。"
      />

      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <p>{error}</p>
        </div>
      )}

      <section className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold text-foreground">CSVをアップロード</p>
            <p className="text-xs text-muted-foreground">1分ごとの行と数値列（移動距離、歩数、速度、心拍など）を含むCSVに対応しています。</p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              disabled={isUploading}
              onClick={() => {
                const input = document.getElementById("factor-upload-input") as HTMLInputElement | null;
                input?.click();
              }}
            >
              <UploadCloud className="mr-2 h-4 w-4" />
              {isUploading ? "アップロード中..." : "ファイルを選択"}
            </Button>
            {session && (
              <Button variant="ghost" size="sm" onClick={() => setSession(null)}>
                <X className="mr-2 h-4 w-4" />
                クリア
              </Button>
            )}
          </div>
        </div>
        <input
          id="factor-upload-input"
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(event) => {
            handleUpload(event.target.files?.[0] ?? null);
            event.target.value = "";
          }}
        />
        {session && (
          <div className="mt-4 grid gap-3 text-sm text-muted-foreground md:grid-cols-3">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">セッションID</p>
              <p className="font-medium text-foreground">{session.session_id}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">データ行数</p>
              <p className="font-medium text-foreground">{session.n_rows.toLocaleString()} 行</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">利用可能な列</p>
              <p className="font-medium text-foreground">{session.columns.join(", ")}</p>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-foreground">スクリープロット</p>
          {session && isFetchingScree && <span className="text-xs text-muted-foreground">計算中...</span>}
        </div>
        {scree && scree.eigenvalues.length > 0 ? (
          <PlotlyChart
            data={[
              {
                x: scree.eigenvalues.map((_, index) => index + 1),
                y: scree.eigenvalues,
                type: "scatter",
                mode: "lines+markers",
                line: { color: PRIMARY_COLOR },
                marker: { color: PRIMARY_COLOR },
                name: "固有値",
              } as Record<string, unknown>,
              {
                x: scree.eigenvalues.map((_, index) => index + 1),
                y: scree.eigenvalues.map(() => 1),
                type: "scatter",
                mode: "lines",
                line: { color: "#9ca3af", dash: "dash" },
                name: "固有値=1",
              } as Record<string, unknown>,
            ]}
            layout={{
              height: 360,
              margin: { l: 48, r: 24, t: 24, b: 48 },
              xaxis: { title: "因子", dtick: 1 },
              yaxis: { title: "固有値" },
              paper_bgcolor: "#ffffff",
              plot_bgcolor: "#ffffff",
              legend: { orientation: "h" },
            }}
          />
        ) : (
          <div className="mt-4 rounded-lg border border-dashed border-border/60 bg-muted/20 p-6 text-sm text-muted-foreground">
            CSVをアップロードすると固有値と寄与率を表示します。
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <div className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm font-semibold text-foreground">因子数</p>
              <p className="text-xs text-muted-foreground">1〜{maxSelectableFactors} の範囲で選択できます。</p>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={1}
                max={maxSelectableFactors}
                value={nFactors}
                onChange={(event) => setNFactors(Number(event.target.value))}
                className="w-full md:w-64"
                disabled={!session}
              />
              <span className="w-12 text-right text-sm font-semibold text-foreground">{nFactors}</span>
            </div>
          </div>

          {session && (
            <details
              className="rounded-lg border border-border/60 bg-background p-4 text-sm text-muted-foreground"
              onToggle={(event) => setShowAdvanced((event.target as HTMLDetailsElement).open)}
            >
              <summary className="cursor-pointer select-none text-foreground">
                詳細設定（列の選択）
              </summary>
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs text-muted-foreground">必要に応じて分析対象の列を絞り込めます。選択しない場合は全ての列を利用します。</p>
                  <Button variant="ghost" size="sm" onClick={handleSelectAll}>
                    すべて選択
                  </Button>
                </div>
                <div className="grid gap-2 md:grid-cols-2">
                  {session.columns.map((column) => (
                    <label key={column} className="flex items-center gap-3 rounded-lg border border-border/60 bg-background px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selectedColumns.includes(column)}
                        onChange={() => toggleColumn(column)}
                        className="h-4 w-4 rounded border-border accent-primary"
                      />
                      <span className="text-sm text-foreground">{column}</span>
                    </label>
                  ))}
                </div>
              </div>
            </details>
          )}

          <Button onClick={handleRun} disabled={!session || isRunning}>
            {isRunning ? "計算中..." : "因子分析を実行"}
          </Button>
        </div>
      </section>

      {result && (
        <section className="space-y-6">
          <div className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
            <p className="text-sm font-semibold text-foreground">因子負荷量</p>
            <div className="mt-3 overflow-auto rounded-lg border border-border/60">
              <table className="min-w-full divide-y divide-border/60 text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    <th className="px-3 py-2 text-left font-semibold text-muted-foreground">変数</th>
                    {factorKeys.map((factor) => (
                      <th key={factor} className="px-3 py-2 text-right font-semibold text-muted-foreground">
                        {factor}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.loadings).map(([variable, loadings]) => (
                    <tr key={variable} className="border-t border-border/40">
                      <td className="px-3 py-2 text-foreground">{variable}</td>
                      {factorKeys.map((factor) => (
                        <td key={factor} className="px-3 py-2 text-right" style={heatMapCells(loadings[factor])}>
                          {loadings[factor].toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
              <p className="text-sm font-semibold text-foreground">共同性</p>
              <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                {Object.entries(result.communalities).map(([variable, value]) => (
                  <li key={variable} className="flex items-center justify-between rounded-lg border border-border/40 bg-background px-3 py-2">
                    <span>{variable}</span>
                    <span className="font-semibold text-foreground">{value.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
              <p className="text-sm font-semibold text-foreground">独自性</p>
              <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                {Object.entries(result.uniqueness).map(([variable, value]) => (
                  <li key={variable} className="flex items-center justify-between rounded-lg border border-border/40 bg-background px-3 py-2">
                    <span>{variable}</span>
                    <span className="font-semibold text-foreground">{value.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {scatterData && (
            <div className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
              <p className="text-sm font-semibold text-foreground">因子得点の散布図（{factorKeys[0]} × {factorKeys[1]}）</p>
              <PlotlyChart data={scatterData.data} layout={scatterData.layout} />
            </div>
          )}

          {timeSeriesData && (
            <div className="rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
              <p className="text-sm font-semibold text-foreground">因子得点の時系列</p>
              <PlotlyChart data={timeSeriesData.data} layout={timeSeriesData.layout} />
            </div>
          )}
        </section>
      )}
    </div>
  );
};

export default FactorAnalysisPage;
