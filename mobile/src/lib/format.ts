/** "https://en.wikipedia.org/wiki/Ribosome" -> "en.wikipedia.org" */
export function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

/**
 * Evidence URLs come from web search results, so only ever open plain http(s) links:
 * never javascript:, file:, intent: or custom schemes.
 */
export function safeHttpUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.toString() : null;
  } catch {
    return null;
  }
}

export function formatCost(usd: number): string {
  if (usd <= 0) return '$0';
  return usd < 0.01 ? '< $0.01' : `$${usd.toFixed(2)}`;
}

/** The web research step flags statements that contradict the notes with a leading "CORRECTION:". */
export function splitCorrection(claim: string): { isCorrection: boolean; text: string } {
  const m = /^\s*(?:-\s*)?CORRECTION:\s*/i.exec(claim);
  return m ? { isCorrection: true, text: claim.slice(m[0].length).trim() } : { isCorrection: false, text: claim.replace(/^\s*-\s*/, '').trim() };
}

/** The fact check writes a doubtful statement as "the claim — the correct fact". Without that dash the whole text is the claim. */
export function splitSuspectClaim(text: string): { claim: string; correction: string } {
  const m = /\s+[—–]\s+|\s+-\s+/.exec(text);
  if (!m) return { claim: text.trim(), correction: '' };
  return { claim: text.slice(0, m.index).trim(), correction: text.slice(m.index + m[0].length).trim() };
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** "3 to review · 12 approved" style summary for a deck row. */
export function deckSummary(counts: { draft: number; approved: number; rejected: number }): string {
  const parts: string[] = [];
  if (counts.draft) parts.push(`${counts.draft} to review`);
  parts.push(`${counts.approved} approved`);
  return parts.join(' · ');
}
