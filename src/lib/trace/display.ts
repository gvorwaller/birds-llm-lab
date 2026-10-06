export function formatTraceValue(value: number, digits = 4): string {
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 10_000) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
}

export function heatAlpha(value: number): number {
  return Math.min(0.7, 0.08 + Math.abs(value) * 0.25);
}
