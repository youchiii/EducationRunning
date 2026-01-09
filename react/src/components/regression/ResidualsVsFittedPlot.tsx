import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";
import { useIsDarkMode } from "../../hooks/useIsDarkMode";

const Plot = createPlotlyComponent(Plotly);

type ResidualsPlotProps = {
  predicted: Array<number | null>;
  residuals: Array<number | null>;
  stdResiduals: Array<number | null>;
};

const BASE_COLOR = "rgba(233, 84, 142, 0.6)";
const OUTLIER_COLOR = "#ef4444";

export const ResidualsVsFittedPlot = ({ predicted, residuals, stdResiduals }: ResidualsPlotProps) => {
  const isDarkMode = useIsDarkMode();
  const lengthMismatch =
    predicted.length !== residuals.length || predicted.length !== stdResiduals.length;
  const points = predicted.map((value, index) => ({
    predicted: value,
    residual: residuals[index] ?? null,
    stdResidual: stdResiduals[index] ?? null,
  }));
  const finitePoints = points.filter(
    (point): point is { predicted: number; residual: number; stdResidual: number | null } =>
      typeof point.predicted === "number" &&
      Number.isFinite(point.predicted) &&
      typeof point.residual === "number" &&
      Number.isFinite(point.residual),
  );
  const hasEnoughData = finitePoints.length >= 5;

  if (lengthMismatch || !hasEnoughData) {
    return (
      <div className="card card--plot overflow-hidden rounded-xl border border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
        残差を描画するためのデータが不足しています。
      </div>
    );
  }

  const inlierX: number[] = [];
  const inlierY: number[] = [];
  const outlierX: number[] = [];
  const outlierY: number[] = [];

  finitePoints.forEach((point) => {
    const stdValue =
      typeof point.stdResidual === "number" && Number.isFinite(point.stdResidual)
        ? point.stdResidual
        : 0;
    if (Math.abs(stdValue) > 2) {
      outlierX.push(point.predicted);
      outlierY.push(point.residual);
    } else {
      inlierX.push(point.predicted);
      inlierY.push(point.residual);
    }
  });

  const inliers: Data = {
    type: "scatter",
    mode: "markers",
    name: "残差",
    x: inlierX,
    y: inlierY,
    marker: {
      color: BASE_COLOR,
      size: 7,
      opacity: 0.7,
      line: { width: 0 },
    },
    hovertemplate: "予測値: %{x:.3f}<br>残差: %{y:.3f}<extra></extra>",
  };

  const outliers: Data | null = outlierX.length
    ? {
        type: "scatter",
        mode: "markers",
        name: "標準化残差 |z| > 2",
        x: outlierX,
        y: outlierY,
        marker: {
          color: OUTLIER_COLOR,
          size: 8,
          opacity: 0.85,
          line: { width: 0 },
        },
        hovertemplate: "予測値: %{x:.3f}<br>残差: %{y:.3f}<extra></extra>",
      }
    : null;

  const zeroLineColor = isDarkMode ? "#f8fafc" : "#334155";
  const predictedValues = finitePoints.map((point) => point.predicted);
  const minPred = predictedValues.length ? Math.min(...predictedValues) : -1;
  const maxPred = predictedValues.length ? Math.max(...predictedValues) : 1;

  const subtitleColor = isDarkMode ? "#cbd5f5" : "#64748b";

  const layout: Partial<Layout> = {
    margin: { l: 56, r: 24, b: 48, t: 96 },
    title: {
      text: "残差 vs 予測値",
      x: 0.5,
      y: 0.93,
      xanchor: "center",
      yanchor: "top",
      font: { size: 18, color: isDarkMode ? "#f8fafc" : "#0f172a" },
    },
    annotations: [
      {
        text: "分散の偏りや非線形の兆候がないか確認",
        xref: "paper",
        yref: "paper",
        x: 0,
        y: 1.13,
        xanchor: "left",
        yanchor: "bottom",
        showarrow: false,
        font: { size: 12, color: subtitleColor },
      },
    ],
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: isDarkMode ? "#f8fafc" : "#0f172a", size: 14 },
    xaxis: {
      title: "予測値",
      title_standoff: 12,
      automargin: true,
      zeroline: false,
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    yaxis: {
      title: "残差",
      title_standoff: 12,
      automargin: true,
      zeroline: false,
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    shapes: [
      {
        type: "line",
        x0: minPred,
        y0: 0,
        x1: maxPred,
        y1: 0,
        line: {
          color: zeroLineColor,
          width: 2,
          dash: "dot",
        },
      },
    ],
    hovermode: "closest",
    legend: {
      orientation: "h",
      x: 0,
      y: 1.02,
      xanchor: "left",
      yanchor: "bottom",
    },
  };

  const traces = outliers ? [inliers, outliers] : [inliers];

  return (
    <Plot
      aria-label="残差と予測値の散布図"
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: 360, position: "relative", zIndex: 0 }}
      className="relative z-0"
    />
  );
};
