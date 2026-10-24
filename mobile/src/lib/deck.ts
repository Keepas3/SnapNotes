import type { NewCard } from '@/api/types';
import { hasCloze } from '@/lib/cloze';
import { plural } from '@/lib/format';

/** The longest deck name the server accepts (backend DeckIn / DeckPatch). */
export const MAX_DECK_NAME = 80;

/** The longest text the server accepts on one side of a card typed by hand (backend CardIn). */
export const MAX_CARD_TEXT = 2000;

/** Why a deck name cannot be saved, or null when it can. */
export function deckNameProblem(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return 'Give the deck a name.';
  if (trimmed.length > MAX_DECK_NAME) return `Use at most ${MAX_DECK_NAME} characters (this one has ${trimmed.length}).`;
  return null;
}

/** Why a card typed by hand cannot be saved yet, or null when it can. Mirrors what the server checks, so the message arrives before the trip. */
export function newCardProblem(card: NewCard): string | null {
  const front = card.front.trim();
  const back = card.back.trim();
  if (!front) return 'Write the front of the card.';
  if (front.length > MAX_CARD_TEXT || back.length > MAX_CARD_TEXT) return `Use at most ${MAX_CARD_TEXT} characters on each side.`;
  if (card.type === 'cloze') return hasCloze(front) ? null : 'A fill-in-the-blank card needs a blank: wrap the answer like {{c1::answer}}.';
  return back ? null : 'Write the back of the card.';
}

/** The decks another deck could absorb, in alphabetical order; the deck itself is left out. */
export function mergeCandidates<T extends { id: string; name: string }>(decks: T[], selfId: string): T[] {
  return decks.filter((d) => d.id !== selfId).sort((a, b) => a.name.localeCompare(b.name));
}

/** The sentence under "Merge “B” into “A”?". `cards` is everything in B, including cards still waiting for review. */
export function mergeDeckMessage(sourceName: string, intoName: string, cards: number): string {
  if (cards === 0) return `“${sourceName}” has no cards. It will be removed. This cannot be undone.`;
  return `All ${plural(cards, 'card')} in “${sourceName}” move into “${intoName}”, and your study progress on them is kept. “${sourceName}” is then removed. This cannot be undone.`;
}

/** The sentence under "Delete this deck?". `approved` cards are on the phone; `waiting` ones are still to be reviewed. */
export function deleteDeckMessage(approved: number, waiting: number): string {
  const total = approved + waiting;
  if (total === 0) return 'The deck is empty. This cannot be undone.';
  const extra = waiting > 0 ? `, including ${waiting} still waiting for your review` : '';
  return `All ${plural(total, 'card')} in it will be removed${extra}. This cannot be undone.`;
}
