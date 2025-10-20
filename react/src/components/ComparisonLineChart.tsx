import { useMemo } from "react";

type ComparisonLineChartProps = {
  seriesA: number[];
  seriesB: number[];
  labelA: string;
  labelB: string;
  colorA: string;
  colorB: string;
  height?: number;
};

type SampledPoint = {
  index: number;
  value: number;
};

type BuiltPath = {
  path: string | null;
  points: Array<{ x: number; y: number; index: number; value: number }>;
};

const VIEW_WIDTH = 560;
const DEFAULT_HEIGHT = 220;
const padding = { top: 24, right: 32, bottom: 48, left: 64 };

const downsampleSeries = (values: number[], targetCount = 220): SampledPoint[] => {
  if (values.length <= 1) {
    return values.map((value, index) => ({ index, value }));
  }
  if (values.length <= targetCount) {
    return values.map((value, index) => ({ index, value }));
  }
  const step = (values.length - 1) / (targetCount - 1);
  const sampled: SampledPoint[] = [];
  for (let sampleIndex = 0; sampleIndex < targetCount; sampleIndex += 1) {
    const sourceIndex = Math.min(values.length - 1, Math.round(sampleIndex * step));
    sampled.push({ index: sourceIndex, value: values[sourceIndex] });
  }
  return sampled;
};

const buildPath = (
  values: number[],
  totalLength: number,
  minValue: number,
  maxValue: number,
  height: number,
): BuiltPath => {
  if (values.length < 2 || totalLength < 2) {
    return { path: null, points: [] };
  }

  const sampled = downsampleSeries(values);
  const innerWidth = VIEW_WIDTH - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const rangeY = Math.max(1e-9, maxValue - minValue);
  const rangeX = Math.max(1, totalLength - 1);

  const segments: string[] = [];
  const points: Array<{ x: number; y: number; index: number; value: number }> = [];
  sampled.forEach((point, index) => {
    const x = padding.left + (point.index / rangeX) * innerWidth;
    const normalized = (point.value - minValue) / rangeY;
    const y = padding.top + innerHeight - normalized * innerHeight;
    segments.push(`${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
    points.push({ x, y, index: point.index, value: point.value });
  });
  return { path: segments.join(" "), points };
};

const formatMinutes = (minutes: number) => `${minutes}分`;

const ComparisonLineChart = ({
  seriesA,
  seriesB,
  labelA,
  labelB,
  colorA,
  colorB,
  height = DEFAULT_HEIGHT,
}: ComparisonLineChartProps) => {
  const chart = useMemo(() => {
    const usableA = seriesA.filter((value) => Number.isFinite(value)) as number[];
    const usableB = seriesB.filter((value) => Number.isFinite(value)) as number[];
    const combined = [...usableA, ...usableB];
    const totalLength = Math.max(usableA.length, usableB.length);

    if (!combined.length || totalLength < 2) {
      return null;
    }

    const rawMin = Math.min(...combined);
    const rawMax = Math.max(...combined);
    const paddingRatio = 0.12;
    const valuePadding = (rawMax - rawMin || Math.max(1, Math.abs(rawMax))) * paddingRatio;
    const minValue = rawMin - valuePadding;
    const maxValue = rawMax + valuePadding;

    const builtA = buildPath(usableA, totalLength, minValue, maxValue, height);
    const builtB = buildPath(usableB, totalLength, minValue, maxValue, height);

    const innerHeight = height - padding.top - padding.bottom;
    const rangeY = Math.max(1e-9, maxValue - minValue);
    const yTickCount = 5;
    const yTicks = Array.from({ length: yTickCount }, (_, index) => {
      const ratio = index / (yTickCount - 1);
      const value = maxValue - ratio * rangeY;
      const y = padding.top + ratio * innerHeight;
      return { value, y };
    });

    const innerWidth = VIEW_WIDTH - padding.left - padding.right;
    const rangeX = Math.max(1, totalLength - 1);
    const xTickCount = Math.min(6, totalLength);
    const xTicks = Array.from({ length: xTickCount }, (_, index) => {
      const ratio = xTickCount === 1 ? 0 : index / (xTickCount - 1);
      const minute = Math.round(rangeX * ratio);
      const x = padding.left + (minute / rangeX) * innerWidth;
      return { minute, x };
    });

    return {
      pathA: builtA.path,
      pathB: builtB.path,
      pointsA: builtA.points,
      pointsB: builtB.points,
      minValue,
      maxValue,
      yTicks,
      xTicks,
    };
  }, [seriesA, seriesB, height]);

  if (!chart) {
    return (
      <div className="flex h-[220px] w-full items-center justify-center rounded-xl border border-dashed border-border/60 bg-muted/30 text-xs text-muted-foreground">
        比較できる数値データが不足しています。
      </div>
    );
  }

  const { pathA, pathB, pointsA, pointsB, yTicks, xTicks } = chart;

  const lastPoint = (points: Array<{ x: number; y: number }>) => {
    if (!points.length) {
      return null;
    }
    return points[points.length - 1];
  };

  const lastA = lastPoint(pointsA);
  const lastB = lastPoint(pointsB);

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${VIEW_WIDTH} ${height}`} className="w-full">
        <rect
          x={0}
          y={0}
          width={VIEW_WIDTH}
          height={height}
          fill="#ffffff"
          rx={12}
          ry={12}
        />
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={height - padding.bottom}
          stroke="var(--border)"
          strokeWidth={1}
        />
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={VIEW_WIDTH - padding.right}
          y2={height - padding.bottom}
          stroke="var(--border)"
          strokeWidth={1}
        />

        {yTicks.map((tick) => (
          <g key={`y-${tick.value.toFixed(6)}`}>
            <line
              x1={padding.left}
              x2={VIEW_WIDTH - padding.right}
              y1={tick.y}
              y2={tick.y}
              stroke="var(--border)"
              strokeWidth={0.6}
              strokeDasharray="4 6"
            />
            <text x={padding.left - 10} y={tick.y + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
              {tick.value.toFixed(2)}
            </text>
          </g>
        ))}

        {xTicks.map((tick) => (
          <g key={`x-${tick.minute}`}>
            <line
              x1={tick.x}
              x2={tick.x}
              y1={padding.top}
              y2={height - padding.bottom}
              stroke="var(--border)"
              strokeWidth={0.6}
              strokeDasharray="4 6"
            />
            <line
              x1={tick.x}
              x2={tick.x}
              y1={height - padding.bottom}
              y2={height - padding.bottom + 6}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={tick.x}
              y={height - padding.bottom + 18}
              textAnchor="middle"
              className="fill-muted-foreground text-[11px]"
            >
              {formatMinutes(tick.minute)}
            </text>
          </g>
        ))}

        {pathA && (
          <path
            d={pathA}
            fill="none"
            stroke={colorA}
            strokeWidth={2.5}
            strokeLinecap="round"
          />
        )}
        {pathB && (
          <path
            d={pathB}
            fill="none"
            stroke={colorB}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {lastA && (
          <>
            <circle cx={lastA.x} cy={lastA.y} r={4} fill={colorA} stroke="#ffffff" strokeWidth={1.5} />
          </>
        )}
        {lastB && (
          <>
            <circle cx={lastB.x} cy={lastB.y} r={4} fill={colorB} stroke="#ffffff" strokeWidth={1.5} />
          </>
        )}

        <text
          x={VIEW_WIDTH / 2}
          y={height - 10}
          textAnchor="middle"
          className="fill-muted-foreground text-xs"
        >
          経過時間 (分)
        </text>
        <text
          x={padding.left - 44}
          y={padding.top - 8}
          className="fill-muted-foreground text-xs"
        >
          指標値
        </text>
      </svg>
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="h-2 w-4 rounded-full" style={{ backgroundColor: colorA }} />
          <span style={{ color: colorA }}>{labelA}</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-4 rounded-full" style={{ backgroundColor: colorB }} />
          <span style={{ color: colorB }}>{labelB}</span>
        </span>
      </div>
    </div>
  );
};

export default ComparisonLineChart;
