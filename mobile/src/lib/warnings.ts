import type { Card } from '@/api/types';

export interface CardWarnings {
  /** Short labels shown as chips. */
  badges: string[];
  /** The verifier's free-text remark about the card (including a fact-check "May be wrong: ..."), or null. */
  note: string | null;
}

// Remarks that already read as a warning on their own (a fact-check concern, or the label on an AI correction).
const READS_AS_WARNING = /^\s*(May be wrong|Not sourced):/i;

/** Everything the review screen should warn about for one card. */
export function cardWarnings(card: Pick<Card, 'grounded' | 'atomic' | 'issue'>): CardWarnings {
  const badges: string[] = [];
  if (!card.grounded) badges.push('Not clearly supported');
  if (!card.atomic) badges.push('Covers more than one idea');

  const issue = (card.issue ?? '').trim();
  if (!issue) return { badges, note: null };
  // Those already read as a warning; any other remark is introduced as one.
  return { badges, note: READS_AS_WARNING.test(issue) ? issue : `Flagged: ${issue}` };
}
