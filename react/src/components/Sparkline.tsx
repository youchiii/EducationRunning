import { useMemo } from "react";

interface SparklineProps {
  values: Array<number | null>;
  width?: number;
  height?: number;
}

const Sparkline = ({ values, width = 160, height = 60 }: SparklineProps) => {
  const path = useMemo(() => {
    const usable = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    if (usable.length < 2) {
      return null;
    }
    const min = Math.min(...usable);
    const max = Math.max(...usable);
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return null;
    }
    const range = max === min ? 1 : max - min;
    const step = width / (usable.length - 1);
    const points = usable.map((value, index) => {
      const x = index * step;
      const normalized = (value - min) / range;
      const y = height - normalized * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return points.join(" ");
  }, [values, width, height]);

  if (!path) {
    return (
      <div className="flex h-[60px] w-full items-center justify-center rounded-md bg-muted text-xs text-muted-foreground">
        データ不足
      </div>
    );
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[60px] w-full rounded-md bg-muted/40">
      <defs>
        <linearGradient id="sparklineGradient" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgb(59 130 246)" stopOpacity="0.9" />
          <stop offset="100%" stopColor="rgb(59 130 246)" stopOpacity="0.2" />
        </linearGradient>
      </defs>
      <path d={path} fill="none" stroke="url(#sparklineGradient)" strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
};

export default Sparkline;
