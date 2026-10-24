import type { Origin } from '@/api/types';
import type { Tone } from '@/lib/steps';
import { domainOf } from '@/lib/format';

/** Where a card's content came from, as the review screen says it. "ai" is a correction with no source behind it. */
export function originLabel(origin: Origin): { label: string; tone: Tone } {
  switch (origin) {
    case 'web':
      return { label: 'From the web', tone: 'info' };
    case 'ai':
      return { label: 'From AI knowledge, not sourced', tone: 'warning' };
    default:
      return { label: 'From your notes', tone: 'neutral' };
  }
}

/** The short chip used in lists. */
export function originBadge(origin: Origin): { label: string; tone: Tone } {
  switch (origin) {
    case 'web':
      return { label: 'Web', tone: 'info' };
    case 'ai':
      return { label: 'AI', tone: 'warning' };
    default:
      return { label: 'Notes', tone: 'neutral' };
  }
}

// Links found by Gemini's Google Search are Google redirect links, so the address says nothing about the real site.
const GOOGLE_REDIRECT_HOST = 'vertexaisearch.cloud.google.com';

/** What to call a web source next to a card: its site, or "Google link" for a redirect. */
export function sourceName(url: string): string {
  const host = domainOf(url);
  return host === GOOGLE_REDIRECT_HOST ? 'the source (via Google)' : host;
}
