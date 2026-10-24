import { plural } from '@/lib/format';

/** The longest deck name the server accepts (backend DeckIn / DeckPatch). */
export const MAX_DECK_NAME = 80;

/** Why a deck name cannot be saved, or null when it can. */
export function deckNameProblem(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return 'Give the deck a name.';
  if (trimmed.length > MAX_DECK_NAME) return `Use at most ${MAX_DECK_NAME} characters (this one has ${trimmed.length}).`;
  return null;
}

/** The sentence under "Delete this deck?". `approved` cards are on the phone; `waiting` ones are still to be reviewed. */
export function deleteDeckMessage(approved: number, waiting: number): string {
  const total = approved + waiting;
  if (total === 0) return 'The deck is empty. This cannot be undone.';
  const extra = waiting > 0 ? `, including ${waiting} still waiting for your review` : '';
  return `All ${plural(total, 'card')} in it will be removed${extra}. This cannot be undone.`;
}
