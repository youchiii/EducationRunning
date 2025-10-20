import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, UploadCloud, X } from "lucide-react";
import type { AxiosError } from "axios";

import Header from "../components/Header";
import { Button } from "../components/ui/button";
import PlotlyChart from "../components/PlotlyChart";
import InfoTooltip from "../components/InfoTooltip";
import { FACTOR_GLOSSARY } from "../constants/factorGlossary";
import { FACTOR_REGRESSION_HELP } from "../constants/factorRegressionHelp";
import {
  fetchFactorScree,
  runFactorAnalysisSession,
  uploadFactorDataset,
  type FactorRunResponse,
  type FactorScreeResponse,
  type FactorUploadResponse,
  runFactorRegression,
  type FactorRegressionResponse,
} from "../services/api";

const PRIMARY_COLOR = "#2563eb";
const SECONDARY_COLOR = "#ea580c";
const FACTOR_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR, "#14b8a6", "#a855f7", "#f97316", "#0ea5e9"];
const MAX_FACTORS = 6;

const HELP_TEXT = `
**おすすめの因子数の選び方**
- ① スクリープロットの“肘（エルボー）”で折れ曲がる手前までを採用
- ② 固有値>1 を満たす因子を目安に（Kaiser基準）
- ③ 累積説明率が概ね 60–80% になる範囲なら妥当
- ④ 解釈可能性：各因子に高い負荷（例 |loading|≥0.4）をもつ変数のまとまりがあるか
- ⑤ 交差負荷が多いなら因子数や回転を見直す
- 補足：上級者は Parallel Analysis も検討可（本アプリでは簡易基準を提示）

**標準化（自動）**
- 変数の単位・スケール差をなくすため、各列を 平均0・標準偏差1 に変換します。
- 標準化により、負荷量の大小比較が素直になります。

**Varimax回転（自動）**
- 因子負荷量の分散を最大化して単純構造を得る直交回転。
- 因子同士は無相関のまま、どの変数がどの因子を表すかが明確になります。
- 因子間の相関を許すなら Promax などもありますが、初期設定は Varimax です。
`;

type HelpSection = {
  title: string;
  items: string[];
};

const parseHelpText = (text: string): HelpSection[] => {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const sections: HelpSection[] = [];
  let current: HelpSection | null = null;

  lines.forEach((line) => {
    if (line.startsWith("**") && line.endsWith("**")) {
      const title = line.replace(/\*\*/g, "").trim();
      current = { title, items: [] };
      sections.push(current);
      return;
    }

    if (line.startsWith("-")) {
      if (!current) {
        current = { title: "", items: [] };
        sections.push(current);
      }
      current.items.push(line.replace(/^-\s*/u, "").trim());
      return;
    }

    if (!current) {
      current = { title: "", items: [] };
      sections.push(current);
    }
    current.items.push(line);
  });

  return sections;
};

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
  const [regressionTarget, setRegressionTarget] = useState<string>("");
  const [selectedRegressionFactors, setSelectedRegressionFactors] = useState<string[]>([]);
  const [standardizeTarget, setStandardizeTarget] = useState(false);
  const [isRunningRegression, setIsRunningRegression] = useState(false);
  const [regressionResult, setRegressionResult] = useState<FactorRegressionResponse | null>(null);
  const [regressionError, setRegressionError] = useState<string | null>(null);

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

  const renderHelpContent = useCallback((text: string) => {
    const sections = parseHelpText(text);
    return (
      <div className="space-y-4">
        {sections.map((section, sectionIndex) => (
          <div key={`${section.title}-${sectionIndex}`} className="space-y-2">
            {section.title && <p className="text-sm font-semibold text-foreground">{section.title}</p>}
            {section.items.length > 0 && (
              <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-foreground/90">
                {section.items.map((item, itemIndex) => (
                  <li key={`${section.title}-${itemIndex}`}>{item}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    );
  }, []);

  const helpContent = useMemo(() => renderHelpContent(HELP_TEXT), [renderHelpContent]);

  useEffect(() => {
    if (factorKeys.length > 0) {
      setSelectedRegressionFactors((prev) => (prev.length ? prev : factorKeys));
    } else {
      setSelectedRegressionFactors([]);
    }
  }, [factorKeys]);

  useEffect(() => {
    if (!session) {
      setRegressionTarget("");
      setRegressionResult(null);
      setStandardizeTarget(false);
    }
  }, [session]);

  const handleToggleRegressionFactor = (factor: string) => {
    setSelectedRegressionFactors((prev) =>
      prev.includes(factor) ? prev.filter((item) => item !== factor) : [...prev, factor],
    );
  };

  const handleRunRegression = async () => {
    if (!session || factorKeys.length === 0) {
      return;
    }
    if (!regressionTarget) {
      setRegressionError("目的変数を選択してください。");
      return;
    }
    const factorsToUse = selectedRegressionFactors.length ? selectedRegressionFactors : factorKeys;
    setRegressionError(null);
    setIsRunningRegression(true);
    try {
      const response = await runFactorRegression({
        session_id: session.session_id,
        target: { type: "column", name: regressionTarget },
        factors: factorsToUse,
        standardize_target: standardizeTarget,
      });
      setRegressionResult(response);
    } catch (requestError) {
      setRegressionResult(null);
      setRegressionError(formatError(requestError, "重回帰分析の実行に失敗しました。"));
    } finally {
      setIsRunningRegression(false);
    }
  };

  const regressionScatter = useMemo(() => {
    if (!regressionResult || regressionResult.fitted.length === 0) {
      return null;
    }
    return {
      data: [
        {
          x: regressionResult.fitted,
          y: regressionResult.residuals,
          mode: "markers",
          type: "scatter",
          marker: { color: PRIMARY_COLOR, opacity: regressionResult.fitted.length > 500 ? 0.45 : 0.65, size: 6 },
        } as Record<string, unknown>,
      ],
      layout: {
        height: 360,
        margin: { l: 48, r: 24, t: 24, b: 48 },
        xaxis: { title: "予測値" },
        yaxis: { title: "残差" },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
      } as Record<string, unknown>,
    };
  }, [regressionResult]);

  const regressionQQPlot = useMemo(() => {
    if (!regressionResult || regressionResult.qq_sample.length === 0) {
      return null;
    }
    return {
      data: [
        {
          x: regressionResult.qq_theoretical,
          y: regressionResult.qq_sample,
          mode: "markers",
          type: "scatter",
          marker: { color: SECONDARY_COLOR, size: 6, opacity: 0.7 },
          name: "残差",
        } as Record<string, unknown>,
        {
          x: regressionResult.qq_theoretical,
          y: regressionResult.qq_theoretical,
          mode: "lines",
          type: "scatter",
          line: { color: "#9ca3af", dash: "dot" },
          name: "45°ライン",
        } as Record<string, unknown>,
      ],
      layout: {
        height: 360,
        margin: { l: 48, r: 24, t: 24, b: 48 },
        xaxis: { title: "理論分位" },
        yaxis: { title: "標本分位" },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        legend: { orientation: "h" },
      } as Record<string, unknown>,
    };
  }, [regressionResult]);

  return (
    <div className="space-y-6">
      <Header
        title={
          <span className="flex items-center gap-2">
            因子分析
            <InfoTooltip
              asInline
              placement="right"
              iconSize={18}
              ariaLabel="因子分析ヘルプ"
              content={helpContent}
            />
          </span>
        }
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
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm font-semibold text-foreground">因子負荷量</p>
              <InfoTooltip
                asInline
                placement="top"
                ariaLabel="用語解説: 因子負荷量"
                content={FACTOR_GLOSSARY.loadings}
              />
            </div>
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
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm font-semibold text-foreground">共同性</p>
                <InfoTooltip
                  asInline
                  placement="top"
                  ariaLabel="用語解説: 共同性"
                  content={FACTOR_GLOSSARY.communalities}
                />
              </div>
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
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm font-semibold text-foreground">独自性</p>
                <InfoTooltip
                  asInline
                  placement="top"
                  ariaLabel="用語解説: 独自性"
                  content={FACTOR_GLOSSARY.uniqueness}
                />
              </div>
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
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm font-semibold text-foreground">因子得点の散布図（{factorKeys[0]} × {factorKeys[1]}）</p>
                <InfoTooltip
                  asInline
                  placement="top"
                  ariaLabel="用語解説: 因子得点の散布図"
                  content={FACTOR_GLOSSARY.scoreScatter}
                />
              </div>
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

      {result && (
        <section className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-2">
              <p className="text-lg font-semibold text-foreground">因子得点を使った重回帰</p>
              <InfoTooltip
                asInline
                ariaLabel="因子得点を使った重回帰のヘルプ"
                placement="right"
                content={renderHelpContent(FACTOR_REGRESSION_HELP.overview)}
              />
            </div>
            {regressionError && (
              <span className="text-sm text-destructive">{regressionError}</span>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">目的変数</label>
              <select
                value={regressionTarget}
                onChange={(event) => setRegressionTarget(event.target.value)}
                className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              >
                <option value="">選択してください</option>
                {session?.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">使用する因子</label>
              <div className="rounded-lg border border-border/60 bg-background p-3 text-xs text-muted-foreground">
                <div className="flex flex-wrap gap-3">
                  {factorKeys.map((factor) => (
                    <label key={factor} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={selectedRegressionFactors.includes(factor)}
                        onChange={() => handleToggleRegressionFactor(factor)}
                        className="h-4 w-4 rounded border-border accent-primary"
                      />
                      <span className="text-sm text-foreground">{factor}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">オプション</label>
              <label className="flex items-center gap-2 rounded-lg border border-border/60 bg-background px-3 py-2 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  checked={standardizeTarget}
                  onChange={(event) => setStandardizeTarget(event.target.checked)}
                  className="h-4 w-4 rounded border-border accent-primary"
                />
                <span>目的変数を標準化してフィット</span>
              </label>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={handleRunRegression}
              disabled={!regressionTarget || isRunningRegression}
            >
              {isRunningRegression ? "計算中..." : "重回帰を実行"}
            </Button>
            {selectedRegressionFactors.length !== factorKeys.length && (
              <span className="text-xs text-muted-foreground">
                使用中の因子: {selectedRegressionFactors.join(", ") || "なし"}
              </span>
            )}
            <span className="text-xs text-muted-foreground">
              サンプル数: {session?.n_rows.toLocaleString() ?? "-"}
            </span>
          </div>

          {regressionResult && (
            <div className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="space-y-3 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm font-semibold text-foreground">係数表</p>
                    <InfoTooltip
                      asInline
                      placement="top"
                      ariaLabel="回帰係数の解説"
                      content={renderHelpContent(FACTOR_REGRESSION_HELP.coefficients)}
                    />
                  </div>
                  <div className="overflow-auto rounded-lg border border-border/60">
                    <table className="min-w-full divide-y divide-border/60 text-sm">
                      <thead className="bg-muted/40">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold text-muted-foreground">項</th>
                          <th className="px-3 py-2 text-right font-semibold text-muted-foreground">β</th>
                          <th className="px-3 py-2 text-right font-semibold text-muted-foreground">標準化β</th>
                          <th className="px-3 py-2 text-right font-semibold text-muted-foreground">p値</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(regressionResult.coefficients).map(([term, coef]) => {
                          const standardized = term === "const" ? null : regressionResult.std_coefficients[term];
                          const pvalue = regressionResult.pvalues[term];
                          const isSignificant = term !== "const" && typeof pvalue === "number" && pvalue < 0.05;
                          return (
                            <tr key={term} className="border-t border-border/40">
                              <td className="px-3 py-2 text-foreground">{term}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">{coef.toFixed(4)}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">
                                {standardized !== null && standardized !== undefined ? standardized.toFixed(4) : "-"}
                              </td>
                              <td
                                className={`px-3 py-2 text-right font-semibold ${
                                  isSignificant ? "text-emerald-500" : "text-muted-foreground"
                                }`}
                              >
                                {pvalue.toFixed(4)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="space-y-3 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm font-semibold text-foreground">モデル指標</p>
                    <InfoTooltip
                      asInline
                      placement="top"
                      ariaLabel="モデル指標の解説"
                      content={renderHelpContent(FACTOR_REGRESSION_HELP.metrics)}
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="rounded-lg border border-border/60 bg-background p-3 text-sm text-muted-foreground">
                      <p>R²</p>
                      <p className="text-lg font-semibold text-foreground">{regressionResult.r2.toFixed(4)}</p>
                    </div>
                    <div className="rounded-lg border border-border/60 bg-background p-3 text-sm text-muted-foreground">
                      <p>調整R²</p>
                      <p className="text-lg font-semibold text-foreground">{regressionResult.adj_r2.toFixed(4)}</p>
                    </div>
                    <div className="rounded-lg border border-border/60 bg-background p-3 text-sm text-muted-foreground">
                      <p>Durbin–Watson</p>
                      <p className="text-lg font-semibold text-foreground">{regressionResult.dw.toFixed(3)}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-3 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm font-semibold text-foreground">VIF</p>
                    <InfoTooltip
                      asInline
                      placement="top"
                      ariaLabel="VIFの解説"
                      content={renderHelpContent(FACTOR_REGRESSION_HELP.vif)}
                    />
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    {Object.entries(regressionResult.vif).map(([factor, value]) => (
                      <li
                        key={factor}
                        className={`flex items-center justify-between rounded-lg border border-border/40 bg-background px-3 py-2 ${
                          value >= 5 ? "text-destructive" : "text-muted-foreground"
                        }`}
                      >
                        <span className="text-foreground">{factor}</span>
                        <span className="font-semibold text-foreground">{value.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-3 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm font-semibold text-foreground">残差診断</p>
                    <InfoTooltip
                      asInline
                      placement="top"
                      ariaLabel="残差診断の解説"
                      content={renderHelpContent(FACTOR_REGRESSION_HELP.residual)}
                    />
                  </div>
                  <div className="space-y-4">
                    {regressionScatter && <PlotlyChart data={regressionScatter.data} layout={regressionScatter.layout} />}
                    {regressionQQPlot && <PlotlyChart data={regressionQQPlot.data} layout={regressionQQPlot.layout} />}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
};

export default FactorAnalysisPage;
