import { studyDayEnd, studyDayStart } from '@/study/day';
import { ANSWERS, CardState, describeInterval, newState, previewDue, RATINGS, ratingForSwipe, reviewCard } from '@/study/scheduler';

const T0 = new Date('2026-03-01T09:00:00.000Z');
const minutes = (n: number) => new Date(T0.getTime() + n * 60_000);
const days = (n: number) => new Date(T0.getTime() + n * 86_400_000);

describe('FSRS scheduler', () => {
  it('starts a card as new and due immediately', () => {
    const s = newState(T0);
    expect(s.state).toBe(CardState.New);
    expect(s.reps).toBe(0);
    expect(s.due).toBe(T0.toISOString());
    expect(s.last_review).toBeNull();
  });

  it('puts a failed new card back in minutes and an easy one days away', () => {
    const fresh = newState(T0);
    const again = reviewCard(fresh, 1, T0);
    const easy = reviewCard(fresh, 4, T0);
    expect(again.state).toBe(CardState.Learning);
    expect(new Date(again.due).getTime() - T0.getTime()).toBeLessThanOrEqual(5 * 60_000);
    expect(easy.state).toBe(CardState.Review);
    expect(new Date(easy.due).getTime() - T0.getTime()).toBeGreaterThanOrEqual(86_400_000);
    expect(again.last_review).toBe(T0.toISOString());
    expect(again.reps).toBe(1);
  });

  it('graduates a learning card to review after enough successes', () => {
    let s = reviewCard(newState(T0), 3, T0);
    expect(s.state).toBe(CardState.Learning);
    s = reviewCard(s, 3, new Date(s.due));
    expect(s.state).toBe(CardState.Review);
    expect(s.scheduled_days).toBeGreaterThanOrEqual(1);
  });

  it('sends a forgotten review card to relearning and counts the lapse', () => {
    let s = reviewCard(newState(T0), 4, T0); // easy: straight to review
    const lapseAt = new Date(s.due);
    s = reviewCard(s, 1, lapseAt);
    expect(s.state).toBe(CardState.Relearning);
    expect(s.lapses).toBe(1);
    expect(new Date(s.due).getTime() - lapseAt.getTime()).toBeLessThan(86_400_000);
  });

  it('orders the four answers from soonest to latest', () => {
    const reviewCardState = reviewCard(newState(T0), 4, T0);
    const now = new Date(reviewCardState.due);
    const due = previewDue(reviewCardState, now);
    expect(due[1].getTime()).toBeLessThan(due[2].getTime());
    expect(due[2].getTime()).toBeLessThan(due[3].getTime());
    expect(due[3].getTime()).toBeLessThan(due[4].getTime());
  });

  it('gives longer intervals to cards that are remembered over and over', () => {
    let s = reviewCard(newState(T0), 3, T0);
    let now = new Date(s.due);
    const intervals: number[] = [];
    for (let i = 0; i < 5; i++) {
      s = reviewCard(s, 3, now);
      intervals.push(s.scheduled_days);
      now = new Date(s.due);
    }
    expect(intervals.slice(1).every((v, i) => v > intervals[i])).toBe(true);
  });

  it('is deterministic: the same history gives the same schedule', () => {
    const run = () => reviewCard(reviewCard(newState(T0), 3, T0), 3, minutes(10));
    expect(run()).toEqual(run());
  });

  it('does not mutate the state it was given', () => {
    const before = newState(T0);
    const snapshot = JSON.stringify(before);
    reviewCard(before, 3, T0);
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});

describe('describeInterval', () => {
  it.each([
    [0, 'under a minute'],
    [20_000, 'under a minute'],
    [60_000, '1 minute'],
    [10 * 60_000, '10 minutes'],
    [59 * 60_000, '59 minutes'],
    [3_600_000, '1 hour'],
    [3 * 3_600_000, '3 hours'],
    [86_400_000, '1 day'],
    [3 * 86_400_000, '3 days'],
    [45 * 86_400_000, '1.5 months'],
    [400 * 86_400_000, '1.1 years'],
  ])('%d ms -> %s', (ms, label) => {
    expect(describeInterval(T0, new Date(T0.getTime() + ms))).toBe(label);
  });

  it('never goes negative', () => {
    expect(describeInterval(T0, days(-2))).toBe('under a minute');
  });
});

describe('answers', () => {
  it('names the four answers in plain words, worst to best', () => {
    expect(RATINGS.map((r) => [r.value, r.label])).toEqual([
      [1, 'Missed it'],
      [2, 'Hard'],
      [3, 'Got it'],
      [4, 'Easy'],
    ]);
  });

  it('offers only "Missed it" and "Got it" as buttons, the same two a swipe gives, and the scheduler still knows Hard and Easy', () => {
    expect(ANSWERS.map((a) => [a.value, a.label])).toEqual([
      [1, 'Missed it'],
      [3, 'Got it'],
    ]);
    expect(ANSWERS.map((a) => a.value)).toEqual([ratingForSwipe('left'), ratingForSwipe('right')]);
    const s = reviewCard(newState(T0), 3, T0); // one Got it puts a new card in its first learning step
    expect(s.state).toBe(CardState.Learning);
    expect(RATINGS.map((r) => r.value)).toEqual([1, 2, 3, 4]);
  });

  it('turns a swipe into an answer: right is "got it", left is "missed it"', () => {
    expect(ratingForSwipe('right')).toBe(3);
    expect(ratingForSwipe('left')).toBe(1);
  });

  it('keeps a swiped "got it" at the same schedule as tapping Good, and "missed it" as Again', () => {
    const s = reviewCard(newState(T0), 3, T0);
    expect(reviewCard(s, ratingForSwipe('right'), minutes(20))).toEqual(reviewCard(s, 3, minutes(20)));
    expect(reviewCard(s, ratingForSwipe('left'), minutes(20))).toEqual(reviewCard(s, 1, minutes(20)));
  });
});

describe('study day', () => {
  // Local-time constructors on purpose: the rollover is defined in the user's own time zone.
  it('rolls over at 4 a.m., so late-night study belongs to the previous day', () => {
    const lateNight = new Date(2026, 2, 10, 1, 30);
    expect(studyDayStart(lateNight)).toEqual(new Date(2026, 2, 9, 4, 0));
    expect(studyDayEnd(lateNight)).toEqual(new Date(2026, 2, 10, 4, 0));
  });

  it('starts a new day at 4:00 sharp and keeps the afternoon in the same day', () => {
    expect(studyDayStart(new Date(2026, 2, 10, 4, 0))).toEqual(new Date(2026, 2, 10, 4, 0));
    expect(studyDayStart(new Date(2026, 2, 10, 3, 59))).toEqual(new Date(2026, 2, 9, 4, 0));
    expect(studyDayStart(new Date(2026, 2, 10, 15, 0))).toEqual(new Date(2026, 2, 10, 4, 0));
    expect(studyDayEnd(new Date(2026, 2, 10, 15, 0))).toEqual(new Date(2026, 2, 11, 4, 0));
  });

  it('crosses month and year ends', () => {
    expect(studyDayEnd(new Date(2026, 11, 31, 12, 0))).toEqual(new Date(2027, 0, 1, 4, 0));
  });
});
