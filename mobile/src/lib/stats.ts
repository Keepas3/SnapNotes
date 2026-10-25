import { studyDayStart } from '@/study/day';

/** What was answered on one study day (a day starts at 4 a.m.); only days with at least one answer exist. */
export interface DayStat {
  /** the study day as "YYYY-MM-DD" */
  day: string;
  answers: number;
  /** answers given to cards already in review, which are the ones that measure remembering */
  reviewAnswers: number;
  /** of those, the ones that were not "Missed it" */
  remembered: number;
  /** time spent, each answer capped at a minute */
  seconds: number;
}

/** Fewer review answers than this say little about how well you remember; the page says so instead of showing a shaky percentage as fact. */
export const MIN_RETENTION_ANSWERS = 20;
/** A card whose wait between reviews has grown to this many days counts as mature (Anki's definition). */
export const MATURE_DAYS = 21;
/** One answer never adds more than this to the time studied, so a phone left open on a card does not count. */
export const MAX_ANSWER_SECONDS = 60;

const pad = (n: number) => String(n).padStart(2, '0');

/** "YYYY-MM-DD" of a local calendar date. */
function keyOf(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function dateOf(key: string): Date {
  const [y, m, d] = key.split('-').map(Number);
  return new Date(y, m - 1, d);
}

/** The study day a moment belongs to: 1:30 a.m. still belongs to the day before. */
export function dayKey(moment: Date): string {
  return keyOf(studyDayStart(moment));
}

/** The day `n` days after `key` (negative: before). Calendar arithmetic, so a daylight-saving change cannot skip or repeat a day. */
export function addDays(key: string, n: number): string {
  const d = dateOf(key);
  d.setDate(d.getDate() + n);
  return keyOf(d);
}

/** 0 for Sunday ... 6 for Saturday. */
export function weekdayOf(key: string): number {
  return dateOf(key).getDay();
}

export function retentionPercent(remembered: number, total: number): number | null {
  return total > 0 ? Math.round((remembered / total) * 100) : null;
}

/** The days you answered at least one card. `current` keeps going if you have not studied yet today, as long as yesterday was studied. */
export function streaks(days: DayStat[], today: string): { studied: number; current: number; longest: number } {
  const set = new Set(days.map((d) => d.day));
  let current = 0;
  let cursor = set.has(today) ? today : addDays(today, -1);
  while (set.has(cursor)) {
    current++;
    cursor = addDays(cursor, -1);
  }
  let longest = 0;
  let run = 0;
  let previous: string | null = null;
  for (const day of [...set].sort()) {
    run = previous !== null && addDays(previous, 1) === day ? run + 1 : 1;
    longest = Math.max(longest, run);
    previous = day;
  }
  return { studied: set.size, current, longest };
}

/** Exactly `n` days ending on `today`, oldest first, with zeros for the days nothing was answered. */
export function fillDays(days: DayStat[], today: string, n: number): DayStat[] {
  const byDay = new Map(days.map((d) => [d.day, d]));
  return Array.from({ length: n }, (_, i) => {
    const day = addDays(today, i - (n - 1));
    return byDay.get(day) ?? { day, answers: 0, reviewAnswers: 0, remembered: 0, seconds: 0 };
  });
}

export interface RangeSummary {
  answers: number;
  reviewAnswers: number;
  remembered: number;
  /** null when no card that was already known was answered in the range */
  retention: number | null;
  /** true when there are too few answers for the percentage to mean much */
  retentionIsShaky: boolean;
  daysStudied: number;
  seconds: number;
}

/** The totals over the last `n` days (today included). */
export function summarize(days: DayStat[], today: string, n: number): RangeSummary {
  const inRange = fillDays(days, today, n);
  const sum = (pick: (d: DayStat) => number) => inRange.reduce((total, d) => total + pick(d), 0);
  const reviewAnswers = sum((d) => d.reviewAnswers);
  const remembered = sum((d) => d.remembered);
  return {
    answers: sum((d) => d.answers),
    reviewAnswers,
    remembered,
    retention: retentionPercent(remembered, reviewAnswers),
    retentionIsShaky: reviewAnswers < MIN_RETENTION_ANSWERS,
    daysStudied: inRange.filter((d) => d.answers > 0).length,
    seconds: sum((d) => d.seconds),
  };
}

export interface WeekRetention {
  /** the first day of the week shown, "YYYY-MM-DD" */
  from: string;
  percent: number | null;
  reviewAnswers: number;
}

/** Retention for each of the last `weeks` blocks of seven days, oldest first; the last block ends today. */
export function retentionByWeek(days: DayStat[], today: string, weeks: number): WeekRetention[] {
  const all = fillDays(days, today, weeks * 7);
  return Array.from({ length: weeks }, (_, w) => {
    const block = all.slice(w * 7, w * 7 + 7);
    const reviewAnswers = block.reduce((n, d) => n + d.reviewAnswers, 0);
    const remembered = block.reduce((n, d) => n + d.remembered, 0);
    return { from: block[0].day, percent: retentionPercent(remembered, reviewAnswers), reviewAnswers };
  });
}

/** How dark a calendar square is, 0 (nothing answered) to 4 (a heavy day). */
export function calendarLevel(answers: number): 0 | 1 | 2 | 3 | 4 {
  if (answers <= 0) return 0;
  if (answers < 10) return 1;
  if (answers < 25) return 2;
  if (answers < 50) return 3;
  return 4;
}

export interface CalendarCell {
  day: string;
  answers: number;
  level: 0 | 1 | 2 | 3 | 4;
  /** after today: drawn as an empty gap */
  future: boolean;
}

/** `weeks` columns of seven days (Sunday at the top, the last column holding today), oldest week first. */
export function calendarGrid(days: DayStat[], today: string, weeks: number): CalendarCell[][] {
  const byDay = new Map(days.map((d) => [d.day, d.answers]));
  const lastSaturday = addDays(today, 6 - weekdayOf(today));
  return Array.from({ length: weeks }, (_, w) =>
    Array.from({ length: 7 }, (_, row) => {
      const day = addDays(lastSaturday, -((weeks - 1 - w) * 7 + (6 - row)));
      const answers = byDay.get(day) ?? 0;
      return { day, answers, level: calendarLevel(answers), future: day > today };
    }),
  );
}

/** "42 min", "1 h 05 min", "under a minute", "0 min" */
export function formatDuration(seconds: number): string {
  if (seconds <= 0) return '0 min';
  if (seconds < 60) return 'under a minute';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)} h ${pad(minutes % 60)} min`;
}

const WEEKDAY_LETTERS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** A short label under a bar: the weekday letter for a short range, the day of the month for a longer one. */
export function dayLabel(key: string, style: 'weekday' | 'date'): string {
  return style === 'weekday' ? WEEKDAY_LETTERS[weekdayOf(key)] : String(dateOf(key).getDate());
}

/** "Mar 10" */
export function shortDate(key: string): string {
  const d = dateOf(key);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}`;
}
