export interface WorkRow {
  date: string;  // "YYYY-MM-DD"
  start: string; // "HH:MM"
  end: string;   // "HH:MM"
}

/**
 * Normalize and parse a recognized line like "3/5 9:00 ~ 17:30".
 */
export function parseWorkLine(line: string, year = new Date().getFullYear()): WorkRow | null {
  const normalized = line.replace(/[〜]/g, "~").replace(/\s+/g, " ").trim();
  const regex = /^(\d{1,2})\/(\d{1,2}) (\d{1,2}):(\d{2}) ?~ ?(\d{1,2}):(\d{2})$/;
  const m = normalized.match(regex);
  if (!m) return null;

  const [, month, day, sh, sm, eh, em] = m.map(Number);
  const pad = (n: number) => (n < 10 ? "0" + n : "" + n);
  return {
    date: `${year}-${pad(month)}-${pad(day)}`,
    start: `${pad(sh)}:${pad(sm)}`,
    end: `${pad(eh)}:${pad(em)}`
  };
}
