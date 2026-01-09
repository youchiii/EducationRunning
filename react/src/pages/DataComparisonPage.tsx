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

export type ColumnStat = {
  name: string;
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  p25: number | null;
  median: number | null;
  p75: number | null;
  max: number | null;
  dtype: "number" | "string" | "datetime" | "boolean" | "unknown";
};

export type DatasetSummary = {
  id: "A" | "B";
  filename: string;
  columns: ColumnStat[];
};

type MetricKey = keyof Pick<ColumnStat, "mean" | "std" | "min" | "p25" | "median" | "p75" | "max">;

const METRIC_LABELS: Record<MetricKey, string> = {
  mean: "平均",
  std: "標準偏差",
  min: "最小値",
  p25: "第1四分位",
  median: "中央値",
  p75: "第3四分位",
  max: "最大値",
};

const METRIC_KEYS: MetricKey[] = ["mean", "median", "std", "p25", "p75", "min", "max"];

const PRIMARY_COLOR = "#2563eb";
const SECONDARY_COLOR = "#ea580c";

const normalizeNumber = (value: number | null | undefined, digits = 2): number | null => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }
  return Number.parseFloat(value.toFixed(digits));
};

const determineDtype = (column: string, stats: DatasetStats): ColumnStat["dtype"] =>
  stats.numeric_columns.includes(column) ? "number" : "unknown";

const toColumnStat = (column: string, stats: DatasetStats): ColumnStat => {
  const basic = stats.basic_statistics[column];
  return {
    name: column,
    count: stats.row_count,
    mean: basic?.mean ?? null,
    std: basic?.std_dev ?? null,
    min: null,
    p25: null,
    median: basic?.median ?? null,
    p75: null,
    max: null,
    dtype: determineDtype(column, stats),
  };
};

const buildDatasetSummary = (
  id: DatasetSummary["id"],
  dataset: DatasetPreview,
  stats: DatasetStats,
): DatasetSummary => {
  const columnNames = Array.from(new Set([...Object.keys(stats.basic_statistics ?? {}), ...stats.numeric_columns]));
  const columns = columnNames.map((column) => toColumnStat(column, stats));
  return {
    id,
    filename: dataset.original_name,
    columns,
  };
};

const DataComparisonPage = () => {
  const { dataset: primaryDataset, stats: primaryStats, getSeries } = useDataset();
  const [secondaryDataset, setSecondaryDataset] = useState<DatasetPreview | null>(null);
  const [secondaryStats, setSecondaryStats] = useState<DatasetStats | null>(null);
  const secondarySeriesCacheRef = useRef<Record<string, Array<number | null>>>({});
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const primarySummary = useMemo<DatasetSummary | null>(() => {
    if (!primaryDataset || !primaryStats) {
      return null;
    }
    return buildDatasetSummary("A", primaryDataset, primaryStats);
  }, [primaryDataset, primaryStats]);

  const secondarySummary = useMemo<DatasetSummary | null>(() => {
    if (!secondaryDataset || !secondaryStats) {
      return null;
    }
    return buildDatasetSummary("B", secondaryDataset, secondaryStats);
  }, [secondaryDataset, secondaryStats]);

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

  const clearSecondaryDataset = useCallback(() => {
    setSecondaryDataset(null);
    setSecondaryStats(null);
    secondarySeriesCacheRef.current = {};
    setUploadError(null);
  }, []);

  const commonNumericColumns = useMemo(() => {
    if (!primarySummary || !secondarySummary) {
      return [] as string[];
    }
    const primaryNumeric = primarySummary.columns.filter((column) => column.dtype === "number");
    const secondaryNumericSet = new Set(
      secondarySummary.columns.filter((column) => column.dtype === "number").map((column) => column.name),
    );
    return primaryNumeric.map((column) => column.name).filter((name) => secondaryNumericSet.has(name));
  }, [primarySummary, secondarySummary]);

  const metricsByColumn = useMemo(() => {
    if (!primarySummary || !secondarySummary) {
      return new Map<string, ComparisonMetric[]>();
    }
    const primaryMap = new Map(primarySummary.columns.map((column) => [column.name, column]));
    const secondaryMap = new Map(secondarySummary.columns.map((column) => [column.name, column]));
    const map = new Map<string, ComparisonMetric[]>();
    for (const column of commonNumericColumns) {
      const sourceA = primaryMap.get(column);
      const sourceB = secondaryMap.get(column);
      if (!sourceA || !sourceB) {
        continue;
      }
      const metrics: ComparisonMetric[] = METRIC_KEYS.map((key) => {
        const valueA = normalizeNumber(sourceA[key]);
        const valueB = normalizeNumber(sourceB[key]);
        const diff = valueA !== null && valueB !== null ? normalizeNumber(valueB - valueA) : null;
        return {
          key,
          label: METRIC_LABELS[key],
          valueA,
          valueB,
          diff,
        };
      });
      map.set(column, metrics);
    }
    return map;
  }, [primarySummary, secondarySummary, commonNumericColumns]);

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

  const renderUploadCard = () => (
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

  if (!primaryDataset || !primaryStats || !primarySummary) {
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
                {primarySummary.filename}
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
        {secondaryDataset && secondaryStats && secondarySummary ? (
          <div className="flex h-full flex-col justify-between rounded-2xl border border-border/60 bg-background/80 p-6 shadow-sm">
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">比較データセット</p>
              <p className="flex items-center gap-2 text-lg font-semibold" style={{ color: SECONDARY_COLOR }}>
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: SECONDARY_COLOR }} />
                {secondarySummary.filename}
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

      {secondaryStats && secondarySummary ? (
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
                      datasetAName={primarySummary.filename}
                      datasetBName={secondarySummary.filename}
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
