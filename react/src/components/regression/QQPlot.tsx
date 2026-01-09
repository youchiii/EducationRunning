import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";
import InfoTooltip from "../InfoTooltip";
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
      <div className="card card--plot overflow-hidden rounded-xl border border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
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
      <div className="card card--plot overflow-hidden rounded-xl border border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
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

  const infoContent = (
    <div className="space-y-2">
      <p className="font-medium text-foreground">
        残差が理論上の正規分布とどれくらい一致しているかを確認する図です。点が45度の対角線に近いほど、残差は正規分布に従うと判断できます。
      </p>
      <p className="text-muted-foreground">
        例えるなら、理想の型に合わせてクッキーを並べる検品のようなものです。溝（対角線）からクッキー（点）がはみ出していれば、形が崩れていて正規性が崩れているサインです。
      </p>
    </div>
  );

  const layout: Partial<Layout> = {
    margin: { l: 56, r: 24, b: 48, t: 32 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: isDarkMode ? "#f8fafc" : "#0f172a", size: 14 },
    xaxis: {
      title: "理論分位",
      title_standoff: 12,
      automargin: true,
      zeroline: false,
      gridcolor: isDarkMode ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.25)",
      linecolor: isDarkMode ? "#f8fafc" : "#0f172a",
    },
    yaxis: {
      title: "標本分位",
      title_standoff: 12,
      automargin: true,
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
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <h3 className="text-base font-semibold text-foreground">残差のQQプロット</h3>
        <InfoTooltip
          ariaLabel="残差のQQプロットの説明"
          content={infoContent}
          asInline
          side="bottom"
        />
      </div>
      <Plot
        aria-label="残差のQQプロット"
        data={[points]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: 360, position: "relative", zIndex: 0 }}
        className="relative z-0"
      />
    </div>
  );
};
