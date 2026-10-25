import type { CardType } from '@/api/types';

/** Longest wrong answer the server keeps. */
export const MAX_CHOICE_LENGTH = 200;
/** Most wrong answers a card can hold. */
export const MAX_WRONG_ANSWERS = 6;
/** Most wrong answer fields the editor offers when adding new ones (the question shows at most 4 beside the right answer). */
export const EDITOR_WRONG_ANSWERS = 4;

export type ChoiceCount = 4 | 5;
export const CHOICE_COUNTS: ChoiceCount[] = [4, 5];

export interface Option {
  text: string;
  correct: boolean;
}

/** The card facts multiple choice needs. The right answer is `back`. */
export interface ChoiceCard {
  id: string;
  type: CardType;
  back: string;
  choices: string[];
}

/** Equal ignoring case, spacing and punctuation: "Mitochondria." and " mitochondria " are the same answer. */
function normalise(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function sameText(a: string, b: string): boolean {
  return normalise(a) === normalise(b);
}

/** True when two answers are really one: the same, one inside the other, or sharing nearly every word. */
function tooAlike(a: string, b: string): boolean {
  const x = normalise(a);
  const y = normalise(b);
  if (!x || !y) return false;
  if (x === y) return true;
  const [short, long] = x.length <= y.length ? [x, y] : [y, x];
  if (short.length >= 4 && long.includes(short)) return true;
  const wordsX = new Set(x.split(' '));
  const wordsY = new Set(y.split(' '));
  const shared = [...wordsX].filter((w) => wordsY.has(w)).length;
  return shared / new Set([...wordsX, ...wordsY]).size >= 0.8;
}

/** Only basic and definition cards with a short enough answer can be asked as multiple choice. */
export function canUseChoices(card: Pick<ChoiceCard, 'type' | 'back'>): boolean {
  const back = card.back.trim();
  return (card.type === 'basic' || card.type === 'definition') && back.length > 0 && back.length <= MAX_CHOICE_LENGTH;
}

// -- a stable shuffle ------------------------------------------------------------------------

/** A small seeded random number generator (mulberry32), so the same card shows its options in the same order while you look at it. */
function seededRandom(seed: string): () => number {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  let state = h >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(items: T[], random: () => number): T[] {
  const copy = items.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

// -- building a question ---------------------------------------------------------------------

/**
 * The options for one question: the right answer and up to `count - 1` wrong ones, shuffled. The card's own wrong answers come first;
 * if it has fewer than needed and `fillFromDeck` is on, answers of the deck's other cards fill the gap (those closest in length to the
 * right answer, never one that is the same as or too alike another option). Returns null when fewer than 3 options are possible, so
 * the card is studied the usual way. `seed` keeps the shuffle steady for one showing of the card.
 */
export function buildOptions(
  card: ChoiceCard,
  others: { id: string; back: string }[],
  { count, fillFromDeck, seed }: { count: ChoiceCount; fillFromDeck: boolean; seed: string },
): Option[] | null {
  if (!canUseChoices(card)) return null;
  const random = seededRandom(`${seed}:${card.id}`);
  const correct = card.back.trim();
  const wrongNeeded = count - 1;

  const taken: string[] = [];
  const accept = (text: string) => {
    const t = text.trim();
    if (!t || t.length > MAX_CHOICE_LENGTH || tooAlike(t, correct) || taken.some((k) => tooAlike(t, k))) return false;
    taken.push(t);
    return true;
  };

  for (const own of shuffled(card.choices, random)) {
    if (taken.length === wrongNeeded) break;
    accept(own);
  }

  if (fillFromDeck && taken.length < wrongNeeded) {
    // Borrow from the other cards. Answers about as long as the right one come first (in a shuffled order, so the same few are not
    // always borrowed), because a much longer or shorter answer would make the right one stand out; others only fill what is left.
    const pool = others
      .filter((o) => o.id !== card.id && o.back.trim() && o.back.trim().length <= MAX_CHOICE_LENGTH)
      .map((o) => ({ text: o.back.trim(), gap: Math.abs(o.back.trim().length - correct.length) }));
    const similarLength = (gap: number) => gap <= 6 || gap <= correct.length * 1.5;
    const ordered = [...shuffled(pool.filter((o) => similarLength(o.gap)), random), ...pool.filter((o) => !similarLength(o.gap)).sort((a, b) => a.gap - b.gap)];
    for (const o of ordered) {
      if (taken.length === wrongNeeded) break;
      accept(o.text);
    }
  }

  if (taken.length < 2) return null; // the right answer plus two wrong ones is the smallest sensible question
  return shuffled([{ text: correct, correct: true }, ...taken.map((text) => ({ text, correct: false }))], random);
}

// -- typing wrong answers --------------------------------------------------------------------

/** Why a set of typed wrong answers cannot be saved with this right answer, or null when they can. Blank fields are ignored. */
export function wrongAnswerProblem(answers: string[], correct: string): string | null {
  const typed = answers.map((a) => a.trim()).filter(Boolean);
  if (typed.length > MAX_WRONG_ANSWERS) return `A card can have at most ${MAX_WRONG_ANSWERS} wrong answers.`;
  for (let i = 0; i < typed.length; i++) {
    if (typed[i].length > MAX_CHOICE_LENGTH) return `Keep each wrong answer under ${MAX_CHOICE_LENGTH} characters.`;
    if (sameText(typed[i], correct)) return `“${typed[i].slice(0, 40)}” is the right answer, so it cannot be one of the wrong ones.`;
  }
  return null;
}

/** The typed wrong answers as they will be saved: trimmed, with blanks and repeats left out. */
export function cleanWrongAnswers(answers: string[]): string[] {
  const kept: string[] = [];
  for (const raw of answers) {
    const text = raw.trim().replace(/\s+/g, ' ');
    if (text && !kept.some((k) => sameText(k, text))) kept.push(text);
  }
  return kept;
}

// -- after the pick --------------------------------------------------------------------------

/** The rating to suggest after an answer was picked: Got it (3) if right, Missed it (1) if wrong, nothing when no answer was picked. */
export function suggestedRating(pickedRight: boolean | null): 1 | 3 | null {
  if (pickedRight === null) return null;
  return pickedRight ? 3 : 1;
}

/** A, B, C... for the options. */
export const optionLetter = (index: number) => String.fromCharCode(65 + index);
