import type { FsrsState } from '@/api/types';
import {
  addDays,
  calendarGrid,
  calendarLevel,
  dayKey,
  fillDays,
  formatDuration,
  MIN_RETENTION_ANSWERS,
  retentionByWeek,
  retentionPercent,
  streaks,
  summarize,
  weekdayOf,
  type DayStat,
} from '@/lib/stats';
import { FORECAST_DAYS } from '@/study/store';

import { makeStore, seedCards } from './helpers/fixtures';

const at = (month: number, day: number, hour = 12) => new Date(2026, month - 1, day, hour, 0, 0);
const stat = (day: string, answers: number, reviewAnswers = 0, remembered = 0, seconds = 0): DayStat => ({ day, answers, reviewAnswers, remembered, seconds });

describe('study days', () => {
  it('belongs a late-night answer to the day before, until 4 a.m.', () => {
    expect(dayKey(at(3, 9, 2))).toBe('2026-03-08');
    expect(dayKey(at(3, 9, 3))).toBe('2026-03-08');
    expect(dayKey(at(3, 9, 4))).toBe('2026-03-09');
    expect(dayKey(at(3, 9, 23))).toBe('2026-03-09');
  });

  it('moves along the calendar one day at a time, across month ends and daylight saving changes', () => {
    expect(addDays('2026-02-28', 1)).toBe('2026-03-01');
    expect(addDays('2026-03-01', -1)).toBe('2026-02-28');
    expect(addDays('2026-03-07', 2)).toBe('2026-03-09'); // the US clocks go forward on the 8th
    expect(addDays('2026-10-31', 2)).toBe('2026-11-02');
    expect(weekdayOf('2026-03-08')).toBe(0); // a Sunday
  });
});

describe('streaks', () => {
  const days = ['2026-03-01', '2026-03-02', '2026-03-03', '2026-03-05', '2026-03-09', '2026-03-10'].map((d) => stat(d, 3));

  it('counts the days studied and the longest run', () => {
    expect(streaks(days, '2026-03-10')).toEqual({ studied: 6, current: 2, longest: 3 });
  });

  it('does not break a streak just because today has not been studied yet, but does once a whole day is missed', () => {
    expect(streaks(days.slice(0, 3), '2026-03-04')).toEqual({ studied: 3, current: 3, longest: 3 }); // yesterday was studied
    expect(streaks(days.slice(0, 3), '2026-03-05').current).toBe(0); // the 4th was missed
  });

  it('is zero for no history', () => {
    expect(streaks([], '2026-03-10')).toEqual({ studied: 0, current: 0, longest: 0 });
  });
});

describe('ranges', () => {
  it('fills the days with no answers, ending today', () => {
    const filled = fillDays([stat('2026-03-09', 4)], '2026-03-10', 4);
    expect(filled.map((d) => [d.day, d.answers])).toEqual([
      ['2026-03-07', 0],
      ['2026-03-08', 0],
      ['2026-03-09', 4],
      ['2026-03-10', 0],
    ]);
  });

  it('totals a range and gives retention only from answers to cards that were already known', () => {
    const days = [stat('2026-03-01', 50, 0, 0, 600), stat('2026-03-09', 30, 25, 20, 300), stat('2026-03-10', 10, 5, 5, 120)];
    const week = summarize(days, '2026-03-10', 7); // leaves out the 1st
    expect(week).toMatchObject({ answers: 40, reviewAnswers: 30, remembered: 25, retention: 83, daysStudied: 2, seconds: 420, retentionIsShaky: false });
    expect(summarize(days, '2026-03-10', 30).answers).toBe(90);
  });

  it('says so when there are too few answers for the percentage to mean much, or none at all', () => {
    expect(summarize([stat('2026-03-10', 5, MIN_RETENTION_ANSWERS - 1, 10)], '2026-03-10', 7).retentionIsShaky).toBe(true);
    const none = summarize([stat('2026-03-10', 5, 0, 0)], '2026-03-10', 7);
    expect(none.retention).toBeNull();
    expect(retentionPercent(0, 0)).toBeNull();
    expect(retentionPercent(9, 10)).toBe(90);
  });

  it('gives retention for each of the last weeks, the last ending today', () => {
    const weeks = retentionByWeek([stat('2026-03-10', 10, 10, 9), stat('2026-03-02', 4, 4, 1)], '2026-03-10', 2);
    expect(weeks).toHaveLength(2);
    expect(weeks[0]).toMatchObject({ from: '2026-02-25', percent: 25, reviewAnswers: 4 });
    expect(weeks[1]).toMatchObject({ from: '2026-03-04', percent: 90, reviewAnswers: 10 });
  });

  it('shades a day by how much was answered', () => {
    expect([0, 1, 9, 10, 24, 25, 49, 50, 500].map(calendarLevel)).toEqual([0, 1, 1, 2, 2, 3, 3, 4, 4]);
  });

  it('lays the calendar out in weeks, Sunday on top, with today in the last column and the days after it left empty', () => {
    const grid = calendarGrid([stat('2026-03-10', 12)], '2026-03-10', 3); // a Tuesday
    expect(grid).toHaveLength(3);
    expect(grid.every((column) => column.length === 7)).toBe(true);
    expect(grid[2][0].day).toBe('2026-03-08'); // that Sunday
    expect(grid[2][2]).toMatchObject({ day: '2026-03-10', answers: 12, level: 2, future: false });
    expect(grid[2][3].future).toBe(true); // Wednesday has not happened yet
    expect(grid[0][0].day).toBe('2026-02-22');
  });

  it('writes a time in plain words', () => {
    expect([0, 20, 60, 59 * 60, 65 * 60, 3 * 3600 + 5 * 60].map(formatDuration)).toEqual(['0 min', 'under a minute', '1 min', '59 min', '1 h 05 min', '3 h 05 min']);
  });
});

describe('the stats the phone works out', () => {
  async function setSchedule(db: Awaited<ReturnType<typeof makeStore>>['db'], id: string, over: Partial<FsrsState> & { due?: string }) {
    const row = await db.first<{ fsrs_json: string }>('SELECT fsrs_json FROM cards WHERE id = ?', [id]);
    const next = { ...(JSON.parse(row!.fsrs_json) as FsrsState), ...over };
    await db.run('UPDATE cards SET state = ?, due = ?, fsrs_json = ? WHERE id = ?', [next.state, next.due, JSON.stringify(next), id]);
  }

  it('is all zeros before anything is studied', async () => {
    const { store } = await makeStore();
    const s = await store.stats(at(3, 10));
    expect(s.days).toEqual([]);
    expect(s.cards).toEqual({ total: 0, new: 0, learning: 0, young: 0, mature: 0 });
    expect(s.forecast).toHaveLength(FORECAST_DAYS);
    expect(s.forecast.every((d) => d.due === 0)).toBe(true);
    expect(s.today).toBe('2026-03-10');
  });

  it('counts answers by study day, measures remembering only on cards already known, and caps the time of one answer', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2', 'c3']);
    await store.answer('c1', 4, at(3, 8, 10), 5_000); // new cards: not counted for remembering
    await store.answer('c2', 4, at(3, 8, 11), 5_000);
    await store.answer('c3', 4, at(3, 9, 2), 600_000); // 2 a.m. on the 9th is still the 8th; ten minutes counts as one
    await store.answer('c1', 3, at(3, 10, 15), 4_000); // a review, remembered
    await store.answer('c2', 1, at(3, 10, 15), 4_000); // a review, missed

    const s = await store.stats(at(3, 10, 16));
    expect(s.days).toEqual([
      { day: '2026-03-08', answers: 3, reviewAnswers: 0, remembered: 0, seconds: 70 },
      { day: '2026-03-10', answers: 2, reviewAnswers: 2, remembered: 1, seconds: 8 },
    ]);
    expect(summarize(s.days, s.today, 30)).toMatchObject({ retention: 50, daysStudied: 2, answers: 5 });
    expect(streaks(s.days, s.today)).toEqual({ studied: 2, current: 1, longest: 1 });
  });

  it('splits the cards into new, learning, young and mature, and counts every card once', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['new', 'learning', 'young', 'mature']);
    await store.answer('learning', 3, at(3, 10, 10), 1000); // a first Got it is still learning
    await store.answer('young', 4, at(3, 10, 10), 1000);
    await store.answer('mature', 4, at(3, 10, 10), 1000);
    await setSchedule(db, 'mature', { scheduled_days: 30 });
    const s = await store.stats(at(3, 10, 16));
    expect(s.cards).toEqual({ total: 4, new: 1, learning: 1, young: 1, mature: 1 });
  });

  it('forecasts what falls due, putting overdue cards on today and leaving out paused ones', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['late', 'soon', 'paused', 'far', 'new']);
    for (const id of ['late', 'soon', 'paused', 'far']) await store.answer(id, 4, at(3, 1, 10), 1000);
    await setSchedule(db, 'late', { due: at(3, 2, 10).toISOString() }); // overdue
    await setSchedule(db, 'soon', { due: at(3, 12, 10).toISOString() });
    await setSchedule(db, 'paused', { due: at(3, 12, 11).toISOString() });
    await setSchedule(db, 'far', { due: at(4, 20, 10).toISOString() }); // beyond the 14 days
    await store.pauseCard('paused');

    const s = await store.stats(at(3, 10, 16));
    const due = Object.fromEntries(s.forecast.filter((d) => d.due > 0).map((d) => [d.day, d.due]));
    expect(due).toEqual({ '2026-03-10': 1, '2026-03-12': 1 });
    expect(s.forecast[0].day).toBe('2026-03-10');
    expect(s.forecast[FORECAST_DAYS - 1].day).toBe('2026-03-23');
  });

  it('does not count practice, which records nothing', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['a', 'b']);
    await store.allCardIds(null); // what practice reads
    expect((await store.stats(at(3, 10))).days).toEqual([]);
  });
});
