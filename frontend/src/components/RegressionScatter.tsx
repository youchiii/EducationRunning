import { useMemo } from "react";

type RegressionPoint = {
  actual: number;
  predicted: number;
};

type RegressionScatterProps = {
  points: RegressionPoint[];
  width?: number;
  height?: number;
};

const padding = { top: 16, right: 24, bottom: 32, left: 48 };

const RegressionScatter = ({ points, width = 560, height = 320 }: RegressionScatterProps) => {
  const { scale, ticks, linePoints } = useMemo(() => {
    if (!points.length) {
      return {
        scale: { x: () => 0, y: () => 0 },
        ticks: [] as number[],
        linePoints: [] as [number, number][],
      };
    }

    const values = points.flatMap((point) => [point.actual, point.predicted]);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const rangePadding = (maxValue - minValue || 1) * 0.1;
    const domainMin = minValue - rangePadding;
    const domainMax = maxValue + rangePadding;

    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;

    const scaleFn = {
      x: (value: number) => padding.left + ((value - domainMin) / (domainMax - domainMin)) * innerWidth,
      y: (value: number) => padding.top + innerHeight - ((value - domainMin) / (domainMax - domainMin)) * innerHeight,
    };

    const tickCount = 5;
    const ticks = Array.from({ length: tickCount }, (_, index) => {
      const ratio = index / (tickCount - 1);
      return domainMin + ratio * (domainMax - domainMin);
    });

    const linePoints: [number, number][] = [domainMin, domainMax].map((value) => [scaleFn.x(value), scaleFn.y(value)]);

    return {
      scale: scaleFn,
      ticks,
      linePoints,
    };
  }, [points, width, height]);

  if (!points.length) {
    return (
      <div className="flex h-[320px] items-center justify-center rounded-xl border border-border/60 bg-muted/20 text-sm text-muted-foreground">
        可視化できるデータがありません。
      </div>
    );
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
      <rect x={0} y={0} width={width} height={height} fill="var(--background)" rx={12} ry={12} />

      {/* axes */}
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
        x2={width - padding.right}
        y2={height - padding.bottom}
        stroke="var(--border)"
        strokeWidth={1}
      />

      {/* ticks & labels */}
      {ticks.map((tick) => {
        const x = scale.x(tick);
        const y = scale.y(tick);
        return (
          <g key={`tick-${tick}`}>
            {/* y-axis ticks */}
            <line x1={padding.left - 6} x2={padding.left} y1={y} y2={y} stroke="var(--border)" strokeWidth={1} />
            <text
              x={padding.left - 10}
              y={y + 4}
              textAnchor="end"
              className="fill-muted-foreground text-[11px]"
            >
              {tick.toFixed(2)}
            </text>
            {/* x-axis ticks */}
            <line x1={x} x2={x} y1={height - padding.bottom} y2={height - padding.bottom + 6} stroke="var(--border)" strokeWidth={1} />
            <text
              x={x}
              y={height - padding.bottom + 18}
              textAnchor="middle"
              className="fill-muted-foreground text-[11px]"
            >
              {tick.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* y = x line */}
      {linePoints.length === 2 && (
        <line
          x1={linePoints[0][0]}
          y1={linePoints[0][1]}
          x2={linePoints[1][0]}
          y2={linePoints[1][1]}
          stroke="hsl(var(--primary))"
          strokeWidth={1.5}
          strokeDasharray="4 4"
        />
      )}

      {/* data points */}
      {points.map((point, index) => (
        <circle
          key={`point-${index}`}
          cx={scale.x(point.actual)}
          cy={scale.y(point.predicted)}
          r={4.5}
          fill="hsl(var(--primary))"
          fillOpacity={0.75}
        >
          <title>{`実測: ${point.actual.toFixed(2)} / 予測: ${point.predicted.toFixed(2)}`}</title>
        </circle>
      ))}

      <text x={width / 2} y={height - 4} textAnchor="middle" className="fill-muted-foreground text-xs">
        実測値
      </text>
      <text
        x={12}
        y={height / 2}
        transform={`rotate(-90 12 ${height / 2})`}
        textAnchor="middle"
        className="fill-muted-foreground text-xs"
      >
        予測値
      </text>
    </svg>
  );
};

export default RegressionScatter;
