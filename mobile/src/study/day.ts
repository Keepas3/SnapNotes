/**
 * A "study day" rolls over at 4 a.m. local time, not midnight, so a late-night session does not count against
 * tomorrow's new-card budget, and cards due "today" stay due until you have slept on them.
 */
export const ROLLOVER_HOUR = 4;

export function studyDayStart(now: Date, rolloverHour = ROLLOVER_HOUR): Date {
  const start = new Date(now);
  if (start.getHours() < rolloverHour) start.setDate(start.getDate() - 1);
  start.setHours(rolloverHour, 0, 0, 0);
  return start;
}

export function studyDayEnd(now: Date, rolloverHour = ROLLOVER_HOUR): Date {
  const end = studyDayStart(now, rolloverHour);
  end.setDate(end.getDate() + 1); // calendar arithmetic, so a daylight-saving change cannot shift the hour
  return end;
}
