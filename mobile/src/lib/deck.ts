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

/**
 * The decks in the order the user dragged them into. A deck that is not in `order` yet (a new one) comes first, as new decks
 * always have; ids in `order` for decks that no longer exist are ignored, and so is an id listed twice.
 */
export function orderDecks<T extends { id: string }>(decks: T[], order: string[]): T[] {
  const byId = new Map(decks.map((d) => [d.id, d]));
  const placed: T[] = [];
  for (const id of new Set(order)) {
    const deck = byId.get(id);
    if (deck) placed.push(deck);
  }
  const taken = new Set(placed.map((d) => d.id));
  return [...decks.filter((d) => !taken.has(d.id)), ...placed];
}

/** A copy of `items` with the one at `from` moved to index `to`. */
export function moveItem<T>(items: T[], from: number, to: number): T[] {
  if (from === to || from < 0 || from >= items.length) return items.slice();
  const copy = items.slice();
  const [moved] = copy.splice(from, 1);
  copy.splice(Math.max(0, Math.min(to, copy.length)), 0, moved);
  return copy;
}

/** Where a row sits on screen (its layout, before any drag offset). */
export interface RowBox {
  y: number;
  height: number;
}

/** The index the dragged row would take if dropped now: how many of the other rows have their middle above its middle. */
export function dragTarget(rows: RowBox[], from: number, dy: number): number {
  if (!rows[from]) return from;
  const middle = rows[from].y + rows[from].height / 2 + dy;
  return rows.reduce((above, row, i) => (i !== from && row.y + row.height / 2 < middle ? above + 1 : above), 0);
}

/** How far row `i` slides out of the way while row `from` is held over index `to`; `step` is one row's height plus the gap. */
export function dragShift(i: number, from: number, to: number, step: number): number {
  if (i === from) return 0;
  if (from < to && i > from && i <= to) return -step;
  if (from > to && i >= to && i < from) return step;
  return 0;
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
