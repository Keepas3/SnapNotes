/** Anki-style cloze markers: {{c1::answer}} or {{c1::answer::hint}}. */
export interface ClozeSegment {
  text: string;
  /** true for text inside a cloze marker */
  blank: boolean;
  hint?: string;
}

const CLOZE = /\{\{c\d+::(.*?)(?:::(.*?))?\}\}/g;

export function parseCloze(text: string): ClozeSegment[] {
  const out: ClozeSegment[] = [];
  let last = 0;
  for (const m of text.matchAll(CLOZE)) {
    const start = m.index ?? 0;
    if (start > last) out.push({ text: text.slice(last, start), blank: false });
    out.push({ text: m[1], blank: true, hint: m[2] });
    last = start + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), blank: false });
  return out;
}

/** The question side: every cloze becomes "[...]" (or "[hint]"). */
export function clozeQuestion(text: string): string {
  return parseCloze(text)
    .map((s) => (s.blank ? `[${s.hint ?? '...'}]` : s.text))
    .join('');
}

/** The answer side as plain text, markers removed. */
export function clozeAnswer(text: string): string {
  return parseCloze(text)
    .map((s) => s.text)
    .join('');
}

export function hasCloze(text: string): boolean {
  return parseCloze(text).some((s) => s.blank);
}
