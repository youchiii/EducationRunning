import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";
import { useIsDarkMode } from "../../hooks/useIsDarkMode";

const Plot = createPlotlyComponent(Plotly);

type QQPlotProps = {
  theoretical: Array<number | null>;
  sample: Array<number | null>;
};

const POINT_COLOR = "rgba(233, 84, 142, 0.65)";

export const QQPlot = ({ theoretical, sample }: QQPlotProps) => {
  const isDarkMode = useIsDarkMode();
  if (theoretical.length !== sample.length) {
    return (
      <div className="rounded-xl border border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
        QQプロットを描画するためのデータが不足しています。
      </div>
    );
  }
  const pairs = theoretical.map((value, index) => ({ x: value, y: sample[index] ?? null }));
  const finitePairs = pairs.filter(
    (pair): pair is { x: number; y: number } =>
      typeof pair.x === "number" &&
      Number.isFinite(pair.x) &&
      typeof pair.y === "number" &&
      Number.isFinite(pair.y),
  );

  if (finitePairs.length < 5) {
    return (
      <div className="rounded-xl border border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
        QQプロットを描画するための有効なデータが不足しています。
      </div>
    );
  }

  const finiteX = finitePairs.map((pair) => pair.x);
  const finiteY = finitePairs.map((pair) => pair.y);
  const minValue = Math.min(Math.min(...finiteX), Math.min(...finiteY));
  const maxValue = Math.max(Math.max(...finiteX), Math.max(...finiteY));

  const points: Data = {
    type: "scatter",
    mode: "markers",
    x: finiteX,
    y: finiteY,
    marker: {
      color: POINT_COLOR,
      size: 7,
      opacity: 0.7,
      line: { width: 0 },
    },
    hovertemplate: "理論分位: %{x:.3f}<br>標本分位: %{y:.3f}<extra></extra>",
    name: "残差",
  };

  const guidelineColor = isDarkMode ? "#f8fafc" : "#334155";

  const layout: Partial<Layout> = {
    title: { text: "残差のQQプロット", font: { size: 18 } },
    margin: { t: 48, r: 24, b: 56, l: 64 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: isDarkMode ? "#f8fafc" : "#0f172a", size: 14 },
    xaxis: {
      title: "理論分位",
      zeroline: false,
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    yaxis: {
      title: "標本分位",
      zeroline: false,
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    shapes: [
      {
        type: "line",
        x0: minValue,
        y0: minValue,
        x1: maxValue,
        y1: maxValue,
        line: {
          color: guidelineColor,
          width: 3,
        },
      },
    ],
    hovermode: "closest",
    showlegend: false,
  };

  return (
    <Plot
      aria-label="残差のQQプロット"
      data={[points]}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "100%" }}
    />
  );
};
