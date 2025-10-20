import { useCallback, useMemo, useRef, useState } from "react";
import { AlertCircle, UploadCloud, X } from "lucide-react";

import Header from "../components/Header";
import NumericComparisonCard, { type ComparisonMetric } from "../components/NumericComparisonCard";
import { Button } from "../components/ui/button";
import { useDataset } from "../context/DatasetContext";
import {
  fetchColumnSeries,
  fetchDatasetStats,
  uploadDataset as uploadDatasetApi,
  type DatasetPreview,
  type DatasetStats,
} from "../services/api";

const METRIC_LABELS: Record<string, string> = {
  mean: "平均",
  median: "中央値",
  mode: "最頻値",
  variance: "分散",
  std_dev: "標準偏差",
};

const metricKeys = Object.keys(METRIC_LABELS) as Array<keyof typeof METRIC_LABELS>;

const PRIMARY_COLOR = "#2563eb";
const SECONDARY_COLOR = "#ea580c";

const formatNumber = (value: number | null, digits = 2) => {
  if (value === null || Number.isNaN(value)) {
    return null;
  }
  return Number.parseFloat(value.toFixed(digits));
};

const DataComparisonPage = () => {
  const { dataset: primaryDataset, stats: primaryStats, getSeries } = useDataset();
  const [secondaryDataset, setSecondaryDataset] = useState<DatasetPreview | null>(null);
  const [secondaryStats, setSecondaryStats] = useState<DatasetStats | null>(null);
  const secondarySeriesCacheRef = useRef<Record<string, Array<number | null>>>({});
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleSecondUpload = async (file: File | null) => {
    if (!file) {
      return;
    }
    setIsUploading(true);
    setUploadError(null);
    try {
      const preview = await uploadDatasetApi(file);
      setSecondaryDataset(preview);
      const statsResponse = await fetchDatasetStats(preview.dataset_id);
      setSecondaryStats(statsResponse);
      secondarySeriesCacheRef.current = {};
    } catch (error) {
      console.error(error);
      setUploadError("2つ目のCSVの読み込みに失敗しました。ファイル形式を確認してください。");
      setSecondaryDataset(null);
      setSecondaryStats(null);
      secondarySeriesCacheRef.current = {};
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const clearSecondaryDataset = () => {
    setSecondaryDataset(null);
    setSecondaryStats(null);
    secondarySeriesCacheRef.current = {};
    setUploadError(null);
  };

  const commonNumericColumns = useMemo(() => {
    if (!primaryStats || !secondaryStats) {
      return [] as string[];
    }
    const secondarySet = new Set(secondaryStats.numeric_columns);
    return primaryStats.numeric_columns.filter((column) => secondarySet.has(column));
  }, [primaryStats, secondaryStats]);

  const metricsByColumn = useMemo(() => {
    if (!primaryStats || !secondaryStats) {
      return new Map<string, ComparisonMetric[]>();
    }
    const map = new Map<string, ComparisonMetric[]>();
    for (const column of commonNumericColumns) {
      const baseStats = primaryStats.basic_statistics[column];
      const targetStats = secondaryStats.basic_statistics[column];
      const metrics: ComparisonMetric[] = metricKeys.map((key) => {
        const valueA = baseStats?.[key] ?? null;
        const valueB = targetStats?.[key] ?? null;
        const formattedA = formatNumber(valueA);
        const formattedB = formatNumber(valueB);
        const diff = formattedA !== null && formattedB !== null ? formatNumber(formattedB - formattedA) : null;
        return {
          key,
          label: METRIC_LABELS[key],
          valueA: formattedA,
          valueB: formattedB,
          diff,
        };
      });
      map.set(column, metrics);
    }
    return map;
  }, [primaryStats, secondaryStats, commonNumericColumns]);

  const getSecondarySeries = useCallback(
    async (column: string) => {
      if (!secondaryDataset) {
        throw new Error("二つ目のデータセットが読み込まれていません。");
      }
      const cached = secondarySeriesCacheRef.current[column];
      if (cached) {
        return cached;
      }
      const response = await fetchColumnSeries(secondaryDataset.dataset_id, column);
      secondarySeriesCacheRef.current = { ...secondarySeriesCacheRef.current, [column]: response.values };
      return response.values;
    },
    [secondaryDataset],
  );

  const triggerFileDialog = () => {
    fileInputRef.current?.click();
  };

  const renderUploadCard = () => {
    return (
      <div className="flex h-full flex-col justify-between rounded-2xl border border-dashed border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
        <div className="space-y-3">
          <p className="text-base font-semibold text-foreground">2つ目のCSVをアップロード</p>
          <p>
            基本統計量や分布グラフを比較するために、同じ構造のCSVファイルをアップロードしてください。1列目にヘッダーが必要です。
          </p>
        </div>
        <div className="mt-4 flex flex-col gap-3">
          <Button variant="outline" onClick={triggerFileDialog} disabled={isUploading}>
            <UploadCloud className="mr-2 h-4 w-4" />
            {isUploading ? "アップロード中..." : "ファイルを選択"}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(event) => handleSecondUpload(event.target.files?.[0] ?? null)}
          />
          <p className="text-xs text-muted-foreground">UTF-8 / Shift-JIS に対応しています。</p>
        </div>
      </div>
    );
  };

  if (!primaryDataset || !primaryStats) {
    return (
      <div className="space-y-6">
        <Header
          title="データ比較"
          subtitle="まずはヘッダーの「データをアップロード」から1つ目のCSVを読み込みます。"
        />
        <div className="flex items-start gap-3 rounded-xl border border-dashed border-border/60 bg-background/70 p-6 text-sm text-muted-foreground">
          <AlertCircle className="mt-0.5 h-4 w-4 text-primary" />
          <p>
            データ比較機能を利用するには、最初にベースとなるCSVデータセットをアップロードしてください。アップロード後にこちらのページで2つ目のCSVを指定できます。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Header
        title="データ比較"
        subtitle="同じ指標を持つ2つのCSVを比較し、基本統計量と分布の違いを素早く確認します。"
      />

      {uploadError && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <p>{uploadError}</p>
        </div>
      )}

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-border/60 bg-background/80 p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">基準データセット</p>
              <p className="mt-1 flex items-center gap-2 text-lg font-semibold" style={{ color: PRIMARY_COLOR }}>
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: PRIMARY_COLOR }} />
                {primaryDataset.original_name}
              </p>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm text-muted-foreground">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">行数</p>
              <p className="text-base font-medium text-foreground">{primaryDataset.row_count.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">列数</p>
              <p className="text-base font-medium text-foreground">{primaryDataset.column_count}</p>
            </div>
          </div>
        </div>
        {secondaryDataset && secondaryStats ? (
          <div className="flex h-full flex-col justify-between rounded-2xl border border-border/60 bg-background/80 p-6 shadow-sm">
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">比較データセット</p>
              <p className="flex items-center gap-2 text-lg font-semibold" style={{ color: SECONDARY_COLOR }}>
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: SECONDARY_COLOR }} />
                {secondaryDataset.original_name}
              </p>
              <p className="text-xs text-muted-foreground">
                {secondaryDataset.row_count.toLocaleString()} 行 / {secondaryDataset.column_count} 列
              </p>
            </div>
            <Button variant="ghost" size="sm" className="self-end text-muted-foreground" onClick={clearSecondaryDataset}>
              <X className="mr-2 h-4 w-4" />
              クリア
            </Button>
          </div>
        ) : (
          renderUploadCard()
        )}
      </section>

      {secondaryStats ? (
        <>
          <section className="rounded-2xl border border-border/60 bg-background/80 p-6 shadow-sm">
            <p className="text-sm text-muted-foreground">
              共通して比較できる数値列: <span className="font-semibold text-foreground">{commonNumericColumns.length}</span> / {primaryStats.numeric_columns.length}
            </p>
            {commonNumericColumns.length === 0 && (
              <p className="mt-3 text-sm text-destructive">
                数値列の名称が一致するカラムが見つかりませんでした。ヘッダー名が一致しているか確認してください。
              </p>
            )}
          </section>

          {commonNumericColumns.length > 0 && (
            <section className="space-y-4">
              <h2 className="text-lg font-semibold text-foreground">数値列の比較</h2>
              <div className="grid gap-4">
                {commonNumericColumns.map((column) => {
                  const metrics = metricsByColumn.get(column) ?? [];
                  return (
                    <NumericComparisonCard
                      key={column}
                      column={column}
                      datasetAName={primaryDataset.original_name}
                      datasetBName={secondaryDataset?.original_name ?? "2つ目のデータ"}
                      colorA={PRIMARY_COLOR}
                      colorB={SECONDARY_COLOR}
                      metrics={metrics}
                      getSeriesA={getSeries}
                      getSeriesB={getSecondarySeries}
                    />
                  );
                })}
              </div>
            </section>
          )}
        </>
      ) : (
        <section className="rounded-2xl border border-dashed border-border/60 bg-background/60 p-6 text-sm text-muted-foreground">
          <p>2つ目のCSVをアップロードすると、基本統計量の比較と重ね合わせグラフが表示されます。</p>
        </section>
      )}
    </div>
  );
};

export default DataComparisonPage;
