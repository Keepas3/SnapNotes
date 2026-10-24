import { clozeAnswer, clozeQuestion, hasCloze, parseCloze } from '@/lib/cloze';
import { deckSummary, domainOf, formatCost, plural, safeHttpUrl, splitCorrection, splitSuspectClaim } from '@/lib/format';
import {
  deckNameProblem,
  deleteDeckMessage,
  dragShift,
  dragTarget,
  MAX_CARD_TEXT,
  MAX_DECK_NAME,
  mergeCandidates,
  mergeDeckMessage,
  moveItem,
  newCardProblem,
  orderDecks,
} from '@/lib/deck';
import { originBadge, originLabel, sourceName } from '@/lib/origin';
import { cardWarnings } from '@/lib/warnings';
import { currentStepIndex, isActive, stepsFor } from '@/lib/steps';
import { decideSwipe, isTap } from '@/lib/swipe';
import { extOf, formatBytes, photoName, validateSelection } from '@/lib/upload';

describe('cloze', () => {
  it('turns markers into blanks for the question and plain text for the answer', () => {
    const text = 'Water moves from {{c1::low}} to {{c2::high}} solute concentration.';
    expect(clozeQuestion(text)).toBe('Water moves from [...] to [...] solute concentration.');
    expect(clozeAnswer(text)).toBe('Water moves from low to high solute concentration.');
    expect(parseCloze(text).filter((s) => s.blank).map((s) => s.text)).toEqual(['low', 'high']);
  });

  it('uses the optional hint in the blank', () => {
    expect(clozeQuestion('The {{c1::mitochondrion::organelle}} makes ATP.')).toBe('The [organelle] makes ATP.');
  });

  it('leaves text without markers alone', () => {
    expect(hasCloze('Plain sentence.')).toBe(false);
    expect(clozeQuestion('Plain sentence.')).toBe('Plain sentence.');
  });
});

describe('format', () => {
  it('extracts a readable domain', () => {
    expect(domainOf('https://www.en.wikipedia.org/wiki/Ribosome')).toBe('en.wikipedia.org');
    expect(domainOf('not a url')).toBe('not a url');
  });

  it('only allows http(s) links to be opened', () => {
    expect(safeHttpUrl('https://example.org/a?b=1')).toBe('https://example.org/a?b=1');
    expect(safeHttpUrl('javascript:alert(1)')).toBeNull();
    expect(safeHttpUrl('file:///etc/passwd')).toBeNull();
    expect(safeHttpUrl('intent://scan/#Intent;scheme=zxing;end')).toBeNull();
    expect(safeHttpUrl('garbage')).toBeNull();
  });

  it('detects corrections flagged by the research step', () => {
    expect(splitCorrection('- CORRECTION: Ribosomes contain RNA.')).toEqual({ isCorrection: true, text: 'Ribosomes contain RNA.' });
    expect(splitCorrection('- ATP is made in mitochondria.')).toEqual({ isCorrection: false, text: 'ATP is made in mitochondria.' });
  });

  it('splits a doubtful statement from its correction, however the dash was typed', () => {
    const wrong = 'Ribosomes are made of protein only';
    const fix = 'ribosomes also contain ribosomal RNA.';
    expect(splitSuspectClaim(`${wrong} — ${fix}`)).toEqual({ claim: wrong, correction: fix });
    expect(splitSuspectClaim(`${wrong} – ${fix}`)).toEqual({ claim: wrong, correction: fix });
    expect(splitSuspectClaim(`${wrong} - ${fix}`)).toEqual({ claim: wrong, correction: fix });
    // only the first dash separates; a hyphenated word or later dash stays in the correction
    expect(splitSuspectClaim('Water boils at 50 C — at sea level it boils at 100 C (well-known) — always.')).toEqual({
      claim: 'Water boils at 50 C',
      correction: 'at sea level it boils at 100 C (well-known) — always.',
    });
    expect(splitSuspectClaim('Mitochondria make DNA-free proteins')).toEqual({ claim: 'Mitochondria make DNA-free proteins', correction: '' });
  });

  it('lists everything the review screen should warn about for a card', () => {
    const ok = { grounded: true, atomic: true, issue: '' };
    expect(cardWarnings(ok)).toEqual({ badges: [], note: null });
    expect(cardWarnings({ ...ok, grounded: false, atomic: false })).toEqual({
      badges: ['Not clearly supported', 'Covers more than one idea'],
      note: null,
    });
    // a fact-check concern reads as a warning already; any other remark is introduced as one
    expect(cardWarnings({ ...ok, issue: 'May be wrong: ribosomes also contain rRNA.' }).note).toBe('May be wrong: ribosomes also contain rRNA.');
    expect(cardWarnings({ ...ok, issue: '  Back adds a detail not in the notes.  ' }).note).toBe('Flagged: Back adds a detail not in the notes.');
    expect(cardWarnings({ ...ok, issue: 'Minor wording. May be wrong: x.' }).note).toBe('Flagged: Minor wording. May be wrong: x.');
    expect(cardWarnings({ ...ok, issue: "Not sourced: made from the AI's own knowledge. Check it before approving." }).note).toBe(
      "Not sourced: made from the AI's own knowledge. Check it before approving.",
    );
    expect(cardWarnings({ ...ok, issue: undefined as unknown as string }).note).toBeNull(); // an older server may omit it
  });

  it('says where a card came from, and warns when the AI made it without a source', () => {
    expect(originLabel('notes')).toEqual({ label: 'From your notes', tone: 'neutral' });
    expect(originLabel('web')).toEqual({ label: 'From the web', tone: 'info' });
    expect(originLabel('ai')).toEqual({ label: 'From AI knowledge, not sourced', tone: 'warning' });
    expect(originLabel('manual')).toEqual({ label: 'Added by you', tone: 'neutral' });
    expect(originBadge('manual')).toEqual({ label: 'You', tone: 'neutral' });
    expect(originBadge('ai')).toEqual({ label: 'AI', tone: 'warning' });
    expect(originBadge('web').label).toBe('Web');
    expect(originBadge('notes').label).toBe('Notes');
    // an origin an older app has never heard of is treated as the notes, never as a source
    expect(originLabel('mystery' as never).label).toBe('From your notes');
  });

  it('names a web source by its site, and a Google redirect link honestly', () => {
    expect(sourceName('https://www.britannica.com/science/ribosome')).toBe('britannica.com');
    expect(sourceName('https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123')).toBe('the source (via Google)');
    expect(sourceName('not a url')).toBe('not a url');
  });

  it('checks a deck name the way the server does', () => {
    expect(deckNameProblem('General Science Revision')).toBeNull();
    expect(deckNameProblem('  padded  ')).toBeNull();
    expect(deckNameProblem('')).toBe('Give the deck a name.');
    expect(deckNameProblem('   ')).toBe('Give the deck a name.');
    expect(deckNameProblem('x'.repeat(MAX_DECK_NAME))).toBeNull();
    expect(deckNameProblem('x'.repeat(MAX_DECK_NAME + 1))).toBe(`Use at most ${MAX_DECK_NAME} characters (this one has ${MAX_DECK_NAME + 1}).`);
    expect(deckNameProblem(`  ${'x'.repeat(MAX_DECK_NAME)}  `)).toBeNull(); // the spaces around it are trimmed before the server sees it
  });

  it('words the delete question with what will be lost', () => {
    expect(deleteDeckMessage(9, 0)).toBe('All 9 cards in it will be removed. This cannot be undone.');
    expect(deleteDeckMessage(1, 0)).toBe('All 1 card in it will be removed. This cannot be undone.');
    expect(deleteDeckMessage(6, 3)).toBe('All 9 cards in it will be removed, including 3 still waiting for your review. This cannot be undone.');
    expect(deleteDeckMessage(0, 0)).toBe('The deck is empty. This cannot be undone.');
  });

  it('checks a card typed by hand before it is sent', () => {
    expect(newCardProblem({ type: 'basic', front: 'Capital of France?', back: 'Paris' })).toBeNull();
    expect(newCardProblem({ type: 'basic', front: '  ', back: 'Paris' })).toBe('Write the front of the card.');
    expect(newCardProblem({ type: 'basic', front: 'Capital of France?', back: ' ' })).toBe('Write the back of the card.');
    // a fill-in-the-blank card needs its blank, but no explanation
    expect(newCardProblem({ type: 'cloze', front: 'The {{c1::mitochondrion}} makes ATP.', back: '' })).toBeNull();
    expect(newCardProblem({ type: 'cloze', front: 'The mitochondrion makes ATP.', back: 'x' })).toBe(
      'A fill-in-the-blank card needs a blank: wrap the answer like {{c1::answer}}.',
    );
    expect(newCardProblem({ type: 'basic', front: 'x'.repeat(MAX_CARD_TEXT), back: 'y' })).toBeNull();
    expect(newCardProblem({ type: 'basic', front: 'x'.repeat(MAX_CARD_TEXT + 1), back: 'y' })).toBe(`Use at most ${MAX_CARD_TEXT} characters on each side.`);
  });

  it('shows decks in the order the user dragged them, with new decks first', () => {
    const decks = [{ id: 'new' }, { id: 'a' }, { id: 'b' }, { id: 'c' }]; // the default order: newest first
    expect(orderDecks(decks, []).map((d) => d.id)).toEqual(['new', 'a', 'b', 'c']); // nothing dragged yet: unchanged
    expect(orderDecks(decks, ['c', 'a', 'b']).map((d) => d.id)).toEqual(['new', 'c', 'a', 'b']); // a deck not in the saved order goes on top
    expect(orderDecks(decks, ['gone', 'b', 'b', 'a']).map((d) => d.id)).toEqual(['new', 'c', 'b', 'a']); // unknown and repeated ids are ignored
    expect(orderDecks([], ['a'])).toEqual([]);
  });

  it('moves one item to a new position without changing the original list', () => {
    const list = ['a', 'b', 'c', 'd'];
    expect(moveItem(list, 0, 2)).toEqual(['b', 'c', 'a', 'd']);
    expect(moveItem(list, 3, 0)).toEqual(['d', 'a', 'b', 'c']);
    expect(moveItem(list, 1, 1)).toEqual(list);
    expect(moveItem(list, 1, 99)).toEqual(['a', 'c', 'd', 'b']); // past the end means last
    expect(moveItem(list, 9, 0)).toEqual(list); // nothing to move
    expect(list).toEqual(['a', 'b', 'c', 'd']);
  });

  it('works out where a dragged deck would land, and how the others make room', () => {
    // three cards, 100 tall with a 10 gap
    const rows = [
      { y: 0, height: 100 },
      { y: 110, height: 100 },
      { y: 220, height: 100 },
    ];
    expect(dragTarget(rows, 0, 0)).toBe(0); // not moved
    expect(dragTarget(rows, 0, 40)).toBe(0); // middle still above the next card's middle (160)
    expect(dragTarget(rows, 0, 120)).toBe(1); // middle at 170, past card 1's middle
    expect(dragTarget(rows, 0, 400)).toBe(2);
    expect(dragTarget(rows, 2, -120)).toBe(1);
    expect(dragTarget(rows, 2, -400)).toBe(0);
    expect(dragTarget(rows, 5, 10)).toBe(5); // a row that is not there stays where it is

    // holding the first card over the last slot: the other two slide up one card's height
    expect([0, 1, 2].map((i) => dragShift(i, 0, 2, 110))).toEqual([0, -110, -110]);
    // holding the last card over the first slot: the other two slide down
    expect([0, 1, 2].map((i) => dragShift(i, 2, 0, 110))).toEqual([110, 110, 0]);
    expect([0, 1, 2].map((i) => dragShift(i, 1, 1, 110))).toEqual([0, 0, 0]); // held over its own place: nothing moves
  });

  it('offers every other deck for a merge, in alphabetical order', () => {
    const decks = [
      { id: 'c', name: 'Chemistry' },
      { id: 'a', name: 'Anatomy' },
      { id: 'b', name: 'Biology' },
    ];
    expect(mergeCandidates(decks, 'b').map((d) => d.name)).toEqual(['Anatomy', 'Chemistry']);
    expect(mergeCandidates(decks, 'nope')).toHaveLength(3);
    expect(mergeCandidates([], 'a')).toEqual([]);
    expect(decks.map((d) => d.id)).toEqual(['c', 'a', 'b']); // the list that was passed in is left as it was
  });

  it('words the merge question with what moves, what is kept and what is removed', () => {
    expect(mergeDeckMessage('Old notes', 'Biology', 12)).toBe(
      'All 12 cards in “Old notes” move into “Biology”, and your study progress on them is kept. “Old notes” is then removed. This cannot be undone.',
    );
    expect(mergeDeckMessage('Old notes', 'Biology', 1)).toContain('All 1 card in');
    expect(mergeDeckMessage('Empty', 'Biology', 0)).toBe('“Empty” has no cards. It will be removed. This cannot be undone.');
  });

  it('formats costs, plurals and deck summaries', () => {
    expect(formatCost(0)).toBe('$0');
    expect(formatCost(0.004)).toBe('< $0.01');
    expect(formatCost(0.456)).toBe('$0.46');
    expect(plural(1, 'card')).toBe('1 card');
    expect(plural(3, 'card')).toBe('3 cards');
    expect(deckSummary({ draft: 2, approved: 5, rejected: 1 })).toBe('2 to review · 5 approved');
    expect(deckSummary({ draft: 0, approved: 5, rejected: 1 })).toBe('5 approved');
  });
});

describe('job steps', () => {
  const base = { auto_approve: false, stage: null as string | null };

  it('maps server status and stage to the current step', () => {
    expect(currentStepIndex({ ...base, status: 'queued' })).toBe(0);
    expect(currentStepIndex({ ...base, status: 'researching', stage: 'understand' })).toBe(0);
    expect(currentStepIndex({ ...base, status: 'researching', stage: 'enrich' })).toBe(1);
    expect(currentStepIndex({ ...base, status: 'awaiting_approval' })).toBe(2);
    expect(currentStepIndex({ ...base, status: 'approved' })).toBe(3);
    expect(currentStepIndex({ ...base, status: 'generating', stage: 'generate' })).toBe(3);
    expect(currentStepIndex({ ...base, status: 'generating', stage: 'verify' })).toBe(4);
    expect(currentStepIndex({ ...base, status: 'completed' })).toBe(5);
  });

  it('marks earlier steps done, the current one active and later ones pending', () => {
    const states = stepsFor({ ...base, status: 'researching', stage: 'enrich' }).map((s) => s.state);
    expect(states).toEqual(['done', 'active', 'pending', 'pending', 'pending']);
  });

  it('shows the approval step as waiting on the user', () => {
    expect(stepsFor({ ...base, status: 'awaiting_approval' }).map((s) => s.state)).toEqual(['done', 'done', 'waiting', 'pending', 'pending']);
  });

  it('marks everything done when completed, including a skipped approval', () => {
    expect(stepsFor({ ...base, auto_approve: true, status: 'completed' }).every((s) => s.state === 'done')).toBe(true);
  });

  it('marks the step a failed job was on, using the stage the server kept', () => {
    expect(stepsFor({ ...base, status: 'failed', stage: 'generate' }).map((s) => s.state)).toEqual(['done', 'done', 'done', 'failed', 'pending']);
    expect(stepsFor({ ...base, status: 'failed', stage: 'enrich' }).map((s) => s.state)).toEqual(['done', 'failed', 'pending', 'pending', 'pending']);
    expect(stepsFor({ ...base, status: 'failed', stage: null })[0].state).toBe('failed');
  });

  it('polls only while the server is working on its own', () => {
    expect(['queued', 'researching', 'approved', 'generating'].every((s) => isActive(s as never))).toBe(true);
    expect(['awaiting_approval', 'completed', 'failed'].some((s) => isActive(s as never))).toBe(false);
  });
});

describe('swipe gestures', () => {
  it('needs a long drag, or a fast flick, to count as a swipe', () => {
    expect(decideSwipe(150, 0)).toBe('right');
    expect(decideSwipe(-150, 0)).toBe('left');
    expect(decideSwipe(60, 0.1)).toBeNull();
    expect(decideSwipe(60, 0.9)).toBe('right');
    expect(decideSwipe(-60, -0.9)).toBe('left');
    expect(decideSwipe(60, -0.9)).toBeNull(); // moving right but flicking back left
    expect(decideSwipe(10, 2)).toBeNull();
  });

  it('treats a tiny movement as a tap', () => {
    expect(isTap(2, 3)).toBe(true);
    expect(isTap(20, 0)).toBe(false);
  });
});

describe('upload selection', () => {
  const file = (name: string, size?: number, mimeType?: string) => ({ uri: `file:///${name}`, name, size, mimeType });

  it('accepts supported types and rejects the rest', () => {
    expect(validateSelection([file('notes.pdf', 1000), file('slides.pptx', 1000), file('a.md', 10)])).toBeNull();
    expect(validateSelection([file('virus.exe', 10)])).toMatch(/not a supported type/);
    expect(validateSelection([])).toMatch(/at least one/);
  });

  it('enforces the same limits as the server', () => {
    expect(validateSelection(Array.from({ length: 21 }, (_, i) => file(`n${i}.txt`, 1)))).toMatch(/at most 20/);
    expect(validateSelection([file('big.jpg', 6 * 1048576, 'image/jpeg')])).toMatch(/over 5 MB/);
    expect(validateSelection([file('a.pdf', 20 * 1048576), file('b.pdf', 20 * 1048576)])).toMatch(/limit is 30 MB/);
  });

  it('names camera photos by their real type', () => {
    expect(photoName({ uri: 'file:///x/IMG_1.HEIC', mimeType: 'image/png' }, 0, 1)).toBe('photo-1-1.png');
    expect(photoName({ uri: 'file:///x', fileName: 'scan.jpeg' }, 2, 1)).toBe('scan.jpeg');
    expect(photoName({ uri: 'blob:abc' }, 1, 5)).toBe('photo-5-2.jpg');
  });

  it('formats sizes and extensions', () => {
    expect(extOf('Notes.Final.PDF')).toBe('pdf');
    expect(extOf('noext')).toBe('');
    expect(formatBytes(2048)).toBe('2 KB');
    expect(formatBytes(3 * 1048576)).toBe('3.0 MB');
  });
});
