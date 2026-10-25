import { createEmptyCard, fsrs, GenSeedStrategyWithCardId, generatorParameters, get_fuzz_range, Rating, StrategyMode, type Card, type Grade, State } from 'ts-fsrs';

import { seededUnit } from '@/lib/order';

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

/** How sure to be of remembering a card when it comes back, as a share. 0.9 means about 9 in 10 are still remembered. */
export const DEFAULT_RETENTION = 0.9;
export const RETENTION_CHOICES = [0.8, 0.85, 0.9, 0.95] as const;

export interface SchedulingOptions {
  /** the chance of still remembering a card at the moment it is due */
  retention: number;
  /** nudge longer intervals a little so cards learned together do not all come back on the same day */
  spread: boolean;
}

/** Today's behaviour: 90%, no spreading. Anything that passes no options gets exactly this. */
export const DEFAULT_SCHEDULING: SchedulingOptions = { retention: DEFAULT_RETENTION, spread: false };

const algorithms = new Map<string, ReturnType<typeof fsrs>>();

/**
 * FSRS with the default parameters and the given retention. Without spreading the result is a pure function of the card's history,
 * which keeps results reproducible and testable. With it, the nudge is seeded by the card's id and review count, so the time shown
 * under a button is the time the card really gets.
 */
function algorithmFor({ retention, spread }: SchedulingOptions) {
  const key = `${retention}:${spread}`;
  let algorithm = algorithms.get(key);
  if (!algorithm) {
    algorithm = fsrs(generatorParameters({ request_retention: retention, maximum_interval: 36500, enable_fuzz: spread }));
    if (spread) algorithm.useStrategy(StrategyMode.SEED, GenSeedStrategyWithCardId('card_id'));
    algorithms.set(key, algorithm);
  }
  return algorithm;
}

function toCard(s: FsrsState, cardId = ''): Card {
  const card: Card & { card_id: string } = {
    card_id: cardId, // read by the seed strategy when spreading is on; ignored otherwise
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
  return card;
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
export function reviewCard(state: FsrsState, rating: RatingValue, now: Date, options: SchedulingOptions = DEFAULT_SCHEDULING, cardId = ''): FsrsState {
  return fromCard(algorithmFor(options).next(toCard(state, cardId), now, rating as Grade).card);
}

/** When the card would next be due for each possible answer; used for the button labels. */
export function previewDue(state: FsrsState, now: Date, options: SchedulingOptions = DEFAULT_SCHEDULING, cardId = ''): Record<RatingValue, Date> {
  const preview = algorithmFor(options).repeat(toCard(state, cardId), now);
  return {
    1: preview[Rating.Again].card.due,
    2: preview[Rating.Hard].card.due,
    3: preview[Rating.Good].card.due,
    4: preview[Rating.Easy].card.due,
  };
}

const MAX_INTERVAL_DAYS = 36500;
const DAY_MS = 86_400_000;

/**
 * The same card scheduled for another level of retention: the wait after its last review becomes `stability x modifier` days, worked out from
 * the strength the card already has, so nothing about its history changes. With spreading on, a card whose current wait is already inside
 * the allowed spread around the new one is left alone (so applying the setting a card already follows moves nothing), and otherwise the new
 * wait is picked inside that spread, seeded by the card. Returns null for a card this does not apply to (new or still being learned) or that
 * would not move.
 */
export function rescheduleForRetention(state: FsrsState, options: SchedulingOptions, cardId: string): FsrsState | null {
  if (state.state !== CardState.Review || !state.last_review) return null;
  const modifier = algorithmFor({ retention: options.retention, spread: false }).calculate_interval_modifier(options.retention);
  const target = Math.min(MAX_INTERVAL_DAYS, Math.max(1, Math.round(state.stability * modifier)));
  let days = target;
  if (options.spread && target >= 2.5) {
    const { min_ivl, max_ivl } = get_fuzz_range(target, state.elapsed_days, MAX_INTERVAL_DAYS);
    if (state.scheduled_days >= min_ivl && state.scheduled_days <= max_ivl) return null;
    days = Math.floor(seededUnit(`${cardId}:${state.reps}`) * (max_ivl - min_ivl + 1) + min_ivl);
  } else if (days === state.scheduled_days) {
    return null;
  }
  const due = new Date(new Date(state.last_review).getTime() + days * DAY_MS).toISOString();
  return due === state.due && days === state.scheduled_days ? null : { ...state, scheduled_days: days, due };
}

/** The chance, from 0 to 1, that a card that has been studied is still remembered at `now`. A card never studied counts as 0. */
export function chanceRemembered(state: FsrsState, now: Date): number {
  if (state.state === CardState.New || !state.last_review) return 0;
  const days = Math.max(0, (now.getTime() - new Date(state.last_review).getTime()) / 86_400_000);
  return algorithmFor(DEFAULT_SCHEDULING).forgetting_curve(days, Math.max(state.stability, 0.01));
}

/**
 * The card most worth reviewing first: the one least likely to be remembered right now, which weighs how long ago it was seen against how
 * strongly it is known. A card due three days ago that was known well can wait behind one due yesterday that was barely known. Ties go to the
 * older due date. Returns the index in `items`, or -1 when there are none.
 */
export function mostForgotten<T extends { fsrs: FsrsState }>(items: T[], now: Date): number {
  let best = -1;
  let bestChance = Infinity;
  items.forEach((item, i) => {
    const chance = chanceRemembered(item.fsrs, now);
    if (chance < bestChance - 1e-12 || (Math.abs(chance - bestChance) <= 1e-12 && best >= 0 && item.fsrs.due < items[best].fsrs.due)) {
      best = i;
      bestChance = chance;
    }
  });
  return best;
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
