import { useMemo } from "react";

type DTWChartProps = {
  referenceSeries: number[];
  comparisonSeries: number[];
  distanceSeries?: number[];
  height?: number;
};

const defaultHeight = 240;

const DTWChart = ({
  referenceSeries,
  comparisonSeries,
  distanceSeries,
  height = defaultHeight,
}: DTWChartProps) => {
  const { referencePoints, comparisonPoints, distancePoints } = useMemo(() => {
    const maxLength = Math.max(
      referenceSeries.length,
      comparisonSeries.length,
      distanceSeries?.length ?? 0,
      2,
    );

    const allValues = [
      ...referenceSeries,
      ...comparisonSeries,
      ...(distanceSeries ?? []),
    ];

    const minValue = allValues.length ? Math.min(...allValues) : 0;
    const maxValue = allValues.length ? Math.max(...allValues) : 1;
    const range = maxValue - minValue || 1;

    const yFor = (value: number) => {
      const normalized = (value - minValue) / range;
      // keep a little padding from the edges to make the lines easier to read
      const padded = normalized * 0.9 + 0.05;
      return (1 - padded) * height;
    };

    const xFor = (index: number) => {
      if (maxLength <= 1) {
        return 0;
      }
      return (index / (maxLength - 1)) * 100;
    };

    const mapSeries = (series: number[]) =>
      series.map((value, index) => `${xFor(index)},${yFor(value)}`).join(" ");

    return {
      referencePoints: mapSeries(referenceSeries),
      comparisonPoints: mapSeries(comparisonSeries),
      distancePoints: distanceSeries ? mapSeries(distanceSeries) : "",
    };
  }, [comparisonSeries, distanceSeries, height, referenceSeries]);

  const hasData =
    referenceSeries.length > 0 || comparisonSeries.length > 0 || (distanceSeries?.length ?? 0) > 0;

  if (!hasData) {
    return (
      <div className="flex min-h-[200px] items-center justify-center rounded-lg border border-dashed border-neutral-300 bg-white">
        <p className="text-sm text-neutral-500">Run an analysis to see the DTW chart.</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Dynamic Time Warping</h3>
        <div className="flex gap-4 text-xs uppercase tracking-wide text-neutral-500">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-primary" /> Reference
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-emerald-500" /> Comparison
          </span>
          {distanceSeries && distanceSeries.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-amber-500" /> Distance
            </span>
          )}
        </div>
      </div>
      <div className="relative">
        <svg
          className="h-[240px] w-full"
          viewBox={`0 0 100 ${height}`}
          preserveAspectRatio="none"
          role="img"
          aria-label="Dynamic Time Warping chart"
        >
          <defs>
            <linearGradient id="distanceGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.45" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="100" height={height} fill="#f9fafb" />
          <g stroke="#e5e7eb" strokeWidth="0.5">
            <line x1="0" y1={height} x2="100" y2={height} />
            <line x1="0" y1={height / 2} x2="100" y2={height / 2} />
            <line x1="0" y1={0} x2="100" y2={0} />
          </g>
          {distancePoints && distanceSeries && distanceSeries.length > 0 && (
            <polyline
              points={distancePoints}
              fill="url(#distanceGradient)"
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.8}
            />
          )}
          {referencePoints && (
            <polyline
              points={referencePoints}
              fill="none"
              stroke="#2563eb"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}
          {comparisonPoints && (
            <polyline
              points={comparisonPoints}
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}
        </svg>
      </div>
    </div>
  );
};

export default DTWChart;
