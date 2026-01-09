const finiteValues = (values: number[]) => values.filter((value) => Number.isFinite(value));

export const sameScaleRange = (a: number[], b: number[], padding = 0.05): [number, number] => {
  const values = finiteValues([...a, ...b]);
  if (values.length === 0) {
    return [0, 1];
  }
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (minValue === maxValue) {
    const delta = Math.abs(minValue) > 0 ? Math.abs(minValue) * padding : 1;
    return [minValue - delta, maxValue + delta];
  }
  const span = maxValue - minValue;
  const offset = span * padding;
  return [minValue - offset, maxValue + offset];
};

export const stdResiduals = (residuals: number[]) => {
  if (residuals.length === 0) {
    return [];
  }
  const valid = finiteValues(residuals);
  if (valid.length === 0) {
    return new Array(residuals.length).fill(0);
  }
  const mean = valid.reduce((sum, value) => sum + value, 0) / valid.length;
  const variance =
    valid.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, valid.length - 1);
  const std = variance > 0 ? Math.sqrt(variance) : 1;
  return residuals.map((value) => (Number.isFinite(value) ? (value - mean) / std : 0));
};

export const ci95 = (coefficient: number, se: number): [number, number] => {
  const delta = 1.96 * se;
  return [coefficient - delta, coefficient + delta];
};

export const niceNumber = (value: number | null | undefined, digits = 2): string => {
  if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
    return "—";
  }
  const formatter = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return formatter.format(value);
};

export const nicePercentage = (value: number | null | undefined, digits = 1): string => {
  if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
    return "—";
  }
  const formatter = new Intl.NumberFormat(undefined, {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return formatter.format(value / 100);
};

const isFiniteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

const roundTo = (value: number, digits: number) => {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
};

export type ResidualSummary = {
  mean?: number;
  std?: number;
  skew?: number;
  kurt?: number;
  outliers_gt2?: number;
};

export const computeResidualSummary = (
  residuals: Array<number | null | undefined>,
  standardizedResiduals?: Array<number | null | undefined>,
): ResidualSummary | null => {
  const validResiduals = residuals.filter(isFiniteNumber);
  const n = validResiduals.length;
  if (n === 0) {
    return null;
  }

  const mean = validResiduals.reduce((sum, value) => sum + value, 0) / n;
  const centered = validResiduals.map((value) => value - mean);
  const variance = centered.reduce((sum, value) => sum + value * value, 0) / Math.max(1, n - 1);
  const std = variance > 0 ? Math.sqrt(variance) : 0;

  const summary: ResidualSummary = {
    mean: roundTo(mean, 4),
    std: std > 0 ? roundTo(std, 4) : 0,
  };

  const sum2 = centered.reduce((acc, value) => acc + value * value, 0);

  if (std > 0 && n >= 3) {
    const sum3 = centered.reduce((acc, value) => acc + value ** 3, 0);
    const skew = (n * sum3) / ((n - 1) * (n - 2) * std ** 3);
    if (Number.isFinite(skew)) {
      summary.skew = roundTo(skew, 4);
    }
  }

  if (std > 0 && n >= 4) {
    const sum4 = centered.reduce((acc, value) => acc + value ** 4, 0);
    const numerator = (n * (n + 1) * sum4) - (3 * (sum2 ** 2) * (n - 1));
    const denominator = (n - 1) * (n - 2) * (n - 3) * std ** 4;
    if (denominator !== 0) {
      const kurt = numerator / denominator;
      if (Number.isFinite(kurt)) {
        summary.kurt = roundTo(kurt, 4);
      }
    }
  }

  const stdValues = standardizedResiduals?.filter(isFiniteNumber);
  if (stdValues && stdValues.length) {
    const outliers = stdValues.filter((value) => Math.abs(value) > 2).length;
    summary.outliers_gt2 = outliers;
  } else if (std > 0) {
    const alt = centered.map((value) => value / std);
    const outliers = alt.filter((value) => Math.abs(value) > 2).length;
    summary.outliers_gt2 = outliers;
  }

  return summary;
};
