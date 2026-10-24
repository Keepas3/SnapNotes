import { createEmptyCard, fsrs, generatorParameters, Rating, type Card, type Grade, State } from 'ts-fsrs';

import type { FsrsState } from '@/api/types';

export type RatingValue = 1 | 2 | 3 | 4;

/**
 * The four answers, in the words a student would use. They are FSRS's Again / Hard / Good / Easy:
 * the better you knew it, the longer the card waits before it comes back.
 */
export const RATINGS: { value: RatingValue; label: string }[] = [
  { value: 1, label: 'Missed it' },
  { value: 2, label: 'Hard' },
  { value: 3, label: 'Got it' },
  { value: 4, label: 'Easy' },
];

/** What swiping a card means: right is "got it" (Good), left is "missed it" (Again). */
export function ratingForSwipe(direction: 'left' | 'right'): RatingValue {
  return direction === 'right' ? 3 : 1;
}

export const CardState = { New: 0, Learning: 1, Review: 2, Relearning: 3 } as const;

/**
 * FSRS with its default parameters and 90% target retention. Fuzzing is off so a given history always produces the
 * same schedule, which keeps results reproducible and testable.
 */
const algorithm = fsrs(generatorParameters({ request_retention: 0.9, maximum_interval: 36500, enable_fuzz: false }));

function toCard(s: FsrsState): Card {
  return {
    due: new Date(s.due),
    stability: s.stability,
    difficulty: s.difficulty,
    elapsed_days: s.elapsed_days,
    scheduled_days: s.scheduled_days,
    learning_steps: s.learning_steps,
    reps: s.reps,
    lapses: s.lapses,
    state: s.state as State,
    last_review: s.last_review ? new Date(s.last_review) : undefined,
  };
}

function fromCard(c: Card): FsrsState {
  return {
    due: c.due.toISOString(),
    stability: c.stability,
    difficulty: c.difficulty,
    elapsed_days: c.elapsed_days,
    scheduled_days: c.scheduled_days,
    learning_steps: c.learning_steps,
    reps: c.reps,
    lapses: c.lapses,
    state: c.state,
    last_review: c.last_review ? c.last_review.toISOString() : null,
  };
}

/** The schedule of a card that has never been studied. */
export function newState(now: Date): FsrsState {
  return fromCard(createEmptyCard(now));
}

/** The card's new schedule after answering with `rating` at time `now`. */
export function reviewCard(state: FsrsState, rating: RatingValue, now: Date): FsrsState {
  return fromCard(algorithm.next(toCard(state), now, rating as Grade).card);
}

/** When the card would next be due for each possible answer; used for the button labels. */
export function previewDue(state: FsrsState, now: Date): Record<RatingValue, Date> {
  const preview = algorithm.repeat(toCard(state), now);
  return {
    1: preview[Rating.Again].card.due,
    2: preview[Rating.Hard].card.due,
    3: preview[Rating.Good].card.due,
    4: preview[Rating.Easy].card.due,
  };
}

const count = (n: number, unit: string) => `${n} ${unit}${n === 1 ? '' : 's'}`;

/** How far away `due` is from `now`, in words: "under a minute", "10 minutes", "3 days", "1.5 months", "1.1 years". */
export function describeInterval(now: Date, due: Date): string {
  const ms = Math.max(0, due.getTime() - now.getTime());
  const minutes = Math.round(ms / 60_000);
  if (minutes < 1) return 'under a minute';
  if (minutes < 60) return count(minutes, 'minute');
  const hours = Math.round(ms / 3_600_000);
  if (hours < 24) return count(hours, 'hour');
  const days = Math.round(ms / 86_400_000);
  if (days < 30) return count(days, 'day');
  const months = days / 30.4375;
  if (months < 12) return count(Math.round(months * 10) / 10, 'month');
  return count(Math.round((days / 365.25) * 10) / 10, 'year');
}
