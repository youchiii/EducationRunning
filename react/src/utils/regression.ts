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

