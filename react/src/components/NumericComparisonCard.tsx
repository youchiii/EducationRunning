import { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle } from "lucide-react";

import ComparisonLineChart from "./ComparisonLineChart";

export type ComparisonMetric = {
  key: string;
  label: string;
  valueA: number | null;
  valueB: number | null;
  diff: number | null;
};

type NumericComparisonCardProps = {
  column: string;
  datasetAName: string;
  datasetBName: string;
  colorA: string;
  colorB: string;
  metrics: ComparisonMetric[];
  getSeriesA: (column: string) => Promise<Array<number | null>>;
  getSeriesB: (column: string) => Promise<Array<number | null>>;
};

const toNumericArray = (values: Array<number | null>) =>
  values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));

const formatValue = (value: number | null) => {
  if (value === null || Number.isNaN(value)) {
    return "-";
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
};

const NumericComparisonCard = ({
  column,
  datasetAName,
  datasetBName,
  colorA,
  colorB,
  metrics,
  getSeriesA,
  getSeriesB,
}: NumericComparisonCardProps) => {
  const [seriesA, setSeriesA] = useState<number[]>([]);
  const [seriesB, setSeriesB] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getSeriesARef = useRef(getSeriesA);
  const getSeriesBRef = useRef(getSeriesB);

  useEffect(() => {
    getSeriesARef.current = getSeriesA;
    getSeriesBRef.current = getSeriesB;
  }, [getSeriesA, getSeriesB]);

  useEffect(() => {
    let cancelled = false;
    const fetchSeries = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const [valuesA, valuesB] = await Promise.all([
          getSeriesARef.current(column),
          getSeriesBRef.current(column),
        ]);
        if (cancelled) {
          return;
        }
        setSeriesA(toNumericArray(valuesA));
        setSeriesB(toNumericArray(valuesB));
      } catch (requestError) {
        console.error(requestError);
        if (!cancelled) {
          setError("列データの取得に失敗しました。");
          setSeriesA([]);
          setSeriesB([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    fetchSeries();

    return () => {
      cancelled = true;
    };
  }, [column, datasetAName, datasetBName]);

  const hasChartData = seriesA.length >= 2 || seriesB.length >= 2;

  const diffSummary = useMemo(() => {
    const meanMetric = metrics.find((metric) => metric.key === "mean");
    if (!meanMetric || meanMetric.diff === null) {
      return null;
    }
    const direction = meanMetric.diff > 0 ? "増加" : meanMetric.diff < 0 ? "減少" : "変化なし";
    return {
      text: direction,
      isPositive: meanMetric.diff > 0,
      diff: meanMetric.diff,
    };
  }, [metrics]);

  return (
    <div className="space-y-4 rounded-2xl border border-border/60 bg-background/85 p-5 shadow-sm">
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-foreground">{column}</p>
          {diffSummary && (
            <span
              className={
                diffSummary.isPositive
                  ? "text-xs font-medium text-emerald-500"
                  : diffSummary.diff < 0
                    ? "text-xs font-medium text-destructive"
                    : "text-xs font-medium text-muted-foreground"
              }
            >
              平均 {diffSummary.text}
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          <span style={{ color: colorA }}>{datasetAName}</span>
          <span> と </span>
          <span style={{ color: colorB }}>{datasetBName}</span>
          <span> の分布を比較</span>
        </p>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="h-[220px] animate-pulse rounded-xl bg-muted/40" />
      ) : hasChartData ? (
        <ComparisonLineChart
          seriesA={seriesA}
          seriesB={seriesB}
          labelA={datasetAName}
          labelB={datasetBName}
          colorA={colorA}
          colorB={colorB}
        />
      ) : (
        <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-border/60 bg-muted/20 text-xs text-muted-foreground">
          視覚化に十分なデータがありません。
        </div>
      )}

      <div className="overflow-hidden rounded-xl border border-border/60">
        <table className="min-w-full text-xs">
          <thead className="bg-muted/40">
            <tr>
              <th className="px-3 py-2 text-left font-medium text-muted-foreground">指標</th>
              <th className="px-3 py-2 text-left font-medium" style={{ color: colorA }}>
                {datasetAName}
              </th>
              <th className="px-3 py-2 text-left font-medium" style={{ color: colorB }}>
                {datasetBName}
              </th>
              <th className="px-3 py-2 text-left font-medium text-muted-foreground">差分 (B - A)</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => {
              const diffClass = metric.diff === null
                ? "text-muted-foreground"
                : metric.diff > 0
                  ? "text-emerald-500"
                  : metric.diff < 0
                    ? "text-destructive"
                    : "text-muted-foreground";
              return (
                <tr key={metric.key} className="border-t border-border/50">
                  <td className="px-3 py-2 font-medium text-foreground">{metric.label}</td>
                  <td className="px-3 py-2 text-muted-foreground">{formatValue(metric.valueA)}</td>
                  <td className="px-3 py-2 text-muted-foreground">{formatValue(metric.valueB)}</td>
                  <td className={`px-3 py-2 font-medium ${diffClass}`}>
                    {formatValue(metric.diff)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default NumericComparisonCard;
