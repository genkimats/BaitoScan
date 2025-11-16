import type { WorkRow } from "./parse";

export interface PaySettings {
  weekdayRate: number;
  weekendRate: number;
  deepNightMultiplier: number;
  deepNightStart: string; // "22:00"
  deepNightEnd: string;   // "05:00"
  transitFee: number;     // one-way
}

/** Convert "HH:MM" → minutes since midnight */
function toMinutes(hm: string): number {
  const [h, m] = hm.split(":").map(Number);
  return h * 60 + m;
}

function isWeekend(date: string): boolean {
  const d = new Date(date + "T00:00:00");
  const day = d.getDay();
  return day === 0 || day === 6;
}

/** Compute pay for one shift. */
export function computeDayPay(row: WorkRow, s: PaySettings) {
  const baseRate = isWeekend(row.date) ? s.weekendRate : s.weekdayRate;
  const deepStart = toMinutes(s.deepNightStart);
  const deepEnd = toMinutes(s.deepNightEnd);

  let start = toMinutes(row.start);
  let end = toMinutes(row.end);
  if (end < start) end += 24 * 60; // overnight

  const REG1_0 = 0, REG1_1 = 22 * 60;     // 00:00–22:00
  const NIGHT1_0 = 22 * 60, NIGHT1_1 = 24 * 60; // 22:00–24:00
  const NIGHT2_0 = 24 * 60, NIGHT2_1 = 29 * 60; // 00:00–05:00 next day

  const overlap = (a0: number, a1: number, b0: number, b1: number) =>
    Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));

  const regMins = overlap(start, end, REG1_0, REG1_1);
  const nightMins =
    overlap(start, end, NIGHT1_0, NIGHT1_1) +
    overlap(start, end, NIGHT2_0, NIGHT2_1);

  const perMin = baseRate / 60;
  const regularPay = regMins * perMin;
  const nightPay = nightMins * perMin * s.deepNightMultiplier;

  return { regMins, nightMins, regularPay, nightPay, total: regularPay + nightPay };
}

/** Compute monthly total */
export function computeMonthly(rows: WorkRow[], s: PaySettings) {
  let total = 0, reg = 0, night = 0;
  for (const r of rows) {
    const d = computeDayPay(r, s);
    total += d.total;
    reg += d.regMins;
    night += d.nightMins;
  }

  const transitTotal = s.transitFee * 2 * rows.length;

  return {
    regularMinutes: reg,
    nightMinutes: night,
    regularHours: (reg / 60).toFixed(1),
    nightHours: (night / 60).toFixed(1),
    totalPay: Math.round(total + transitTotal),
    transitTotal
  };
}
