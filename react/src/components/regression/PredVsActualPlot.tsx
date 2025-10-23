import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Layout, Data } from "plotly.js";
import { useIsDarkMode } from "../../hooks/useIsDarkMode";
import { sameScaleRange } from "../../utils/regression";

const Plot = createPlotlyComponent(Plotly);

type PredVsActualPlotProps = {
  actual: Array<number | null>;
  predicted: Array<number | null>;
};

const MARKER_COLOR = "rgba(233, 84, 142, 0.65)";

export const PredVsActualPlot = ({ actual, predicted }: PredVsActualPlotProps) => {
  const isDarkMode = useIsDarkMode();
  const pairs = actual.map((value, index) => ({ actual: value, predicted: predicted[index] ?? null }));
  const validPairs = pairs.filter(
    (pair): pair is { actual: number; predicted: number } =>
      typeof pair.actual === "number" &&
      Number.isFinite(pair.actual) &&
      typeof pair.predicted === "number" &&
      Number.isFinite(pair.predicted),
  );
  const hasEnoughData = validPairs.length >= 5;

  if (!hasEnoughData) {
    return (
      <div className="rounded-xl border border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
        十分なデータがないため、実測値と予測値の散布図を表示できません。
      </div>
    );
  }

  const actualValues = validPairs.map((pair) => pair.actual);
  const predictedValues = validPairs.map((pair) => pair.predicted);

  const [minDomain, maxDomain] = sameScaleRange(actualValues, predictedValues, 0.05);

  const scatter: Data = {
    type: "scatter",
    mode: "markers",
    x: actualValues,
    y: predictedValues,
    marker: {
      color: MARKER_COLOR,
      size: 7,
      opacity: 0.7,
      line: { width: 0 },
    },
    hovertemplate: "実測値: %{x:.3f}<br>予測値: %{y:.3f}<extra></extra>",
    name: "観測値",
  };

  const diagonalColor = isDarkMode ? "#f8fafc" : "#334155";

  const layout: Partial<Layout> = {
    title: { text: "実測値 vs 予測値", font: { size: 18 } },
    margin: { t: 48, r: 24, b: 56, l: 64 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: isDarkMode ? "#f8fafc" : "#0f172a", size: 14 },
    xaxis: {
      title: "実測値",
      zeroline: false,
      range: [minDomain, maxDomain],
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    yaxis: {
      title: "予測値",
      zeroline: false,
      range: [minDomain, maxDomain],
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    shapes: [
      {
        type: "line",
        x0: minDomain,
        y0: minDomain,
        x1: maxDomain,
        y1: maxDomain,
        line: {
          color: diagonalColor,
          width: 3,
        },
      },
    ],
    hovermode: "closest",
    showlegend: false,
  };

  return (
    <Plot
      aria-label="実測値と予測値の散布図"
      data={[scatter]}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "100%" }}
    />
  );
};
