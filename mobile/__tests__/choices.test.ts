import { ApiClient } from '@/api/client';
import { MIGRATIONS, migrate } from '@/db/migrations';
import {
  buildOptions,
  canUseChoices,
  cleanWrongAnswers,
  MAX_CHOICE_LENGTH,
  MAX_WRONG_ANSWERS,
  optionLetter,
  sameText,
  suggestedRating,
  wrongAnswerProblem,
  type ChoiceCard,
} from '@/lib/choices';

import { card, makeStore, response } from './helpers/fixtures';
import { openUnmigratedTestDb } from './helpers/nodeDb';

const T = '2026-03-01T08:00:00.000000+00:00';
const NOW = new Date(2026, 2, 10, 15, 0, 0);
const q = (over: Partial<ChoiceCard> = {}): ChoiceCard => ({ id: 'c1', type: 'basic', back: 'Mitochondria', choices: ['The nucleus', 'The Golgi apparatus', 'Ribosomes'], ...over });
const options = (c: ChoiceCard, others: { id: string; back: string }[] = [], over: Partial<{ count: 4 | 5; fillFromDeck: boolean; seed: string }> = {}) =>
  buildOptions(c, others, { count: 4, fillFromDeck: true, seed: 's1', ...over });

describe('which cards can be asked as multiple choice', () => {
  it('allows questions and definitions with a usable answer, and nothing else', () => {
    expect(canUseChoices({ type: 'basic', back: 'Mitochondria' })).toBe(true);
    expect(canUseChoices({ type: 'definition', back: 'Movement of water' })).toBe(true);
    expect(canUseChoices({ type: 'cloze', back: 'whatever' })).toBe(false); // a blank has no natural set of options
    expect(canUseChoices({ type: 'basic', back: '   ' })).toBe(false);
    expect(canUseChoices({ type: 'basic', back: 'x'.repeat(MAX_CHOICE_LENGTH + 1) })).toBe(false); // too long to show as a button
  });

  it('treats answers as the same ignoring case, spacing and punctuation', () => {
    expect(sameText('Mitochondria.', ' mitochondria ')).toBe(true);
    expect(sameText('The nucleus', 'Nucleus')).toBe(false);
  });
});

describe('building the options for one question', () => {
  it('shows the right answer once, among the card\'s own wrong answers, in the number asked for', () => {
    const built = options(q())!;
    expect(built).toHaveLength(4);
    expect(built.filter((o) => o.correct).map((o) => o.text)).toEqual(['Mitochondria']);
    expect(built.map((o) => o.text).sort()).toEqual(['Mitochondria', 'Ribosomes', 'The Golgi apparatus', 'The nucleus']);
  });

  it('asks for 5 when set to 5, using the card\'s own wrong answers first', () => {
    const own = q({ choices: ['A one', 'B two', 'C three', 'D four', 'E five'] });
    const built = options(own, [{ id: 'x', back: 'From elsewhere' }], { count: 5 })!;
    expect(built).toHaveLength(5);
    expect(built.some((o) => o.text === 'From elsewhere')).toBe(false); // the card had enough of its own
  });

  it('keeps the same order for the same showing of a card and changes it for another', () => {
    const order = (seed: string) => options(q(), [], { seed })!.map((o) => o.text).join('|');
    expect(order('s1')).toBe(order('s1'));
    const seen = new Set(['a', 'b', 'c', 'd', 'e', 'f'].map(order));
    expect(seen.size).toBeGreaterThan(1); // not always the right answer in the same place
  });

  it('fills the gap from the deck\'s other cards when the card has too few of its own', () => {
    const others = [
      { id: 'c2', back: 'Chloroplasts' },
      { id: 'c3', back: 'Lysosomes' },
      { id: 'c4', back: 'Vacuoles' },
    ];
    const built = options(q({ choices: ['The nucleus'] }), others)!;
    expect(built).toHaveLength(4);
    expect(built.filter((o) => !o.correct).map((o) => o.text)).toContain('The nucleus'); // its own one is kept
    expect(built.filter((o) => !o.correct).every((o) => o.text === 'The nucleus' || others.some((x) => x.back === o.text))).toBe(true);
  });

  it('never borrows an answer that is the right one, a copy of it, or a repeat', () => {
    const others = [
      { id: 'c1', back: 'Mitochondria' }, // the card itself
      { id: 'c2', back: 'mitochondria.' }, // same answer on another card
      { id: 'c3', back: 'The mitochondria and the nucleus' }, // contains the right answer
      { id: 'c4', back: 'Chloroplasts' },
      { id: 'c5', back: 'chloroplasts' }, // a repeat of c4
      { id: 'c6', back: 'Lysosomes' },
    ];
    const built = options(q({ choices: [] }), others)!;
    // either spelling of the repeated answer may be the one kept, but never both, and never the right answer's copies
    expect(built.map((o) => o.text.toLowerCase().replace('.', '')).sort()).toEqual(['chloroplasts', 'lysosomes', 'mitochondria']);
    expect(built.filter((o) => o.text.toLowerCase().includes('chloroplasts'))).toHaveLength(1);
    expect(built.filter((o) => o.correct)).toHaveLength(1);
  });

  it('prefers borrowed answers about as long as the right one, so it does not stand out', () => {
    const others = [
      { id: 'a', back: 'A very long answer that goes on and on describing something at great length for no reason' },
      { id: 'b', back: 'Golgi' },
      { id: 'c', back: 'Vacuoles' },
      { id: 'd', back: 'Nucleolus' },
    ];
    const built = options(q({ back: 'Ribosomes', choices: [] }), others, { count: 4 })!;
    expect(built.some((o) => o.text.startsWith('A very long'))).toBe(false);
  });

  it('does not borrow when that is switched off, and then needs the card\'s own', () => {
    const others = [{ id: 'c2', back: 'Chloroplasts' }, { id: 'c3', back: 'Lysosomes' }];
    expect(options(q({ choices: ['The nucleus'] }), others, { fillFromDeck: false })).toBeNull(); // only 1 wrong answer: too few
    expect(options(q({ choices: ['The nucleus', 'Ribosomes'] }), others, { fillFromDeck: false })).toHaveLength(3); // fewer than 4 is fine
  });

  it('gives up (so the card is studied the usual way) when there are too few options or the card cannot use them', () => {
    expect(options(q({ choices: [] }), [])).toBeNull();
    expect(options(q({ type: 'cloze' }))).toBeNull();
    expect(options(q({ back: '' }))).toBeNull();
  });

  it('drops a card\'s own wrong answer that is the right one or a repeat', () => {
    const built = options(q({ choices: ['mitochondria', 'Nucleus', 'nucleus.', 'Ribosomes'] }))!;
    expect(built.map((o) => o.text).sort()).toEqual(['Mitochondria', 'Nucleus', 'Ribosomes']);
  });
});

describe('typing wrong answers', () => {
  it('accepts sensible ones and ignores blank fields', () => {
    expect(wrongAnswerProblem(['The nucleus', '', '  ', 'Ribosomes'], 'Mitochondria')).toBeNull();
    expect(wrongAnswerProblem([], 'x')).toBeNull();
  });

  it('refuses the right answer, an over-long one, and too many', () => {
    expect(wrongAnswerProblem(['Nucleus', ' mitochondria. '], 'Mitochondria')).toBe('“mitochondria.” is the right answer, so it cannot be one of the wrong ones.');
    expect(wrongAnswerProblem(['x'.repeat(MAX_CHOICE_LENGTH + 1)], 'y')).toBe(`Keep each wrong answer under ${MAX_CHOICE_LENGTH} characters.`);
    expect(wrongAnswerProblem(Array.from({ length: MAX_WRONG_ANSWERS + 1 }, (_, i) => `wrong ${i}`), 'right')).toBe(`A card can have at most ${MAX_WRONG_ANSWERS} wrong answers.`);
  });

  it('cleans what is saved: trimmed, no blanks, no repeats', () => {
    expect(cleanWrongAnswers(['  The  nucleus ', '', 'the nucleus.', 'Ribosomes', '   '])).toEqual(['The nucleus', 'Ribosomes']);
  });
});

describe('after an answer is picked', () => {
  it('suggests Got it for a right pick, Missed it for a wrong one, and nothing when none was picked', () => {
    expect(suggestedRating(true)).toBe(3);
    expect(suggestedRating(false)).toBe(1);
    expect(suggestedRating(null)).toBeNull();
  });

  it('labels the options A, B, C...', () => {
    expect([0, 1, 4].map(optionLetter)).toEqual(['A', 'B', 'E']);
  });
});

describe('LocalStore multiple choice', () => {
  it('stores each card\'s wrong answers and follows a change to them', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ decks: [{ id: 'deck-a', name: 'Bio', topic: '', updated_at: T }], cards: [card('c1', { choices: ['x', 'y'] }), card('c2')] }), []);
    expect((await store.getCard('c1'))?.choices).toEqual(['x', 'y']);
    expect((await store.getCard('c2'))?.choices).toEqual([]); // an older server sends none
    await store.applySync(response({ cards: [card('c1', { choices: ['x', 'y', 'z'], updated_at: '2026-03-02T08:00:00.000000+00:00' })] }), []);
    expect((await store.getCard('c1'))?.choices).toEqual(['x', 'y', 'z']);
  });

  it('keeps the schedule of a card whose wrong answers change', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ decks: [{ id: 'deck-a', name: 'Bio', topic: '', updated_at: T }], cards: [card('c1')] }), []);
    await store.answer('c1', 3, NOW, 900);
    const before = (await store.getCard('c1'))!.fsrs;
    await store.applySync(response({ cards: [card('c1', { choices: ['x', 'y'], updated_at: '2026-03-02T08:00:00.000000+00:00' })] }), []);
    expect((await store.getCard('c1'))!.fsrs).toEqual(before);
  });

  it('lists the other answers of a deck, leaving out the card itself, fill-in-the-blank cards and empty answers', async () => {
    const { store } = await makeStore();
    await store.applySync(
      response({
        decks: [{ id: 'deck-a', name: 'Bio', topic: '', updated_at: T }, { id: 'deck-b', name: 'Other', topic: '', updated_at: T }],
        cards: [
          card('c1', { back: 'One' }),
          card('c2', { back: 'Two' }),
          card('c3', { back: 'Three', type: 'definition' }),
          card('c4', { back: 'Four', type: 'cloze' }),
          card('c5', { back: '' }),
          card('c6', { back: 'Elsewhere', deck_id: 'deck-b' }),
        ],
      }),
      [],
    );
    expect((await store.otherAnswers('deck-a', 'c1')).map((o) => o.back).sort()).toEqual(['Three', 'Two']);
  });

  it('remembers how multiple choice is set up, starting off, with 4 options and borrowing on', async () => {
    const { store } = await makeStore();
    expect(await store.choicePrefs()).toEqual({ enabled: false, count: 4, fillFromDeck: true });
    await store.setChoicePrefs({ enabled: true });
    await store.setChoicePrefs({ count: 5, fillFromDeck: false });
    expect(await store.choicePrefs()).toEqual({ enabled: true, count: 5, fillFromDeck: false });
    await store.setChoicePrefs({ count: 4 });
    expect(await store.choicePrefs()).toEqual({ enabled: true, count: 4, fillFromDeck: false });
  });

  it('reads damaged wrong answers as none', async () => {
    const { store, db } = await makeStore();
    await store.applySync(response({ decks: [{ id: 'deck-a', name: 'Bio', topic: '', updated_at: T }], cards: [card('c1')] }), []);
    for (const bad of ['not json', '{"a":1}', '"text"']) {
      await db.run('UPDATE cards SET choices_json = ? WHERE id = ?', [bad, 'c1']);
      expect((await store.getCard('c1'))?.choices).toEqual([]);
    }
    await db.run('UPDATE cards SET choices_json = ? WHERE id = ?', ['["a", 7, null, "b"]', 'c1']);
    expect((await store.getCard('c1'))?.choices).toEqual(['a', 'b']);
  });

  it('steps a version 2 database forward without losing a card', async () => {
    const db = await openUnmigratedTestDb();
    await db.exec(MIGRATIONS[0]);
    await db.exec(MIGRATIONS[1]);
    await db.exec('PRAGMA user_version = 2');
    await db.run("INSERT INTO decks(id, name, topic, updated_at) VALUES ('d1', 'Kept', '', 't')");
    await db.run("INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, created_at, updated_at, due, fsrs_json) VALUES ('c1','d1','basic','f','b','notes','','t','t','t','{}')");

    await migrate(db);

    expect((await db.first<{ user_version: number }>('PRAGMA user_version'))?.user_version).toBe(MIGRATIONS.length);
    expect(await db.first('SELECT back, choices_json FROM cards WHERE id = ?', ['c1'])).toEqual({ back: 'b', choices_json: '[]' });
  });
});

describe('the client', () => {
  type Call = { url: string; init: RequestInit };
  function fakeFetch(responses: { status?: number; body?: unknown }[]) {
    const calls: Call[] = [];
    const impl = jest.fn(async (url: string, init: RequestInit = {}) => {
      calls.push({ url, init });
      const next = responses.shift() ?? {};
      const status = next.status ?? 200;
      return { ok: status < 300, status, json: async () => next.body } as unknown as Response;
    });
    return { impl: impl as unknown as typeof fetch, calls };
  }

  it('asks for suggestions without saving, naming the cards only when asked to', async () => {
    const { impl, calls } = fakeFetch([{ body: { suggestions: [], skipped: [] } }, { body: { suggestions: [], skipped: [] } }]);
    const api = new ApiClient('http://srv', 't', impl);
    await api.suggestChoices('d1');
    expect(calls[0].url).toBe('http://srv/decks/d1/suggest-choices');
    expect(JSON.parse(calls[0].init.body as string)).toEqual({ wrong_per_card: 3 }); // no card_ids: the server picks the deck's cards that lack them
    await api.suggestChoices('d1', { cardIds: ['c1'], wrongPerCard: 4 });
    expect(JSON.parse(calls[1].init.body as string)).toEqual({ card_ids: ['c1'], wrong_per_card: 4 });
    expect(() => api.suggestChoices('')).toThrow(/needs a deck id/);
  });

  it('saves checked wrong answers as one batch and shows the server\'s sentence when one is refused', async () => {
    const { impl, calls } = fakeFetch([{ body: { saved: 2 } }, { status: 422, body: { detail: '“A1” is the right answer, so it cannot be one of the wrong ones.' } }]);
    const api = new ApiClient('http://srv', 't', impl);
    await expect(api.saveChoices([{ card_id: 'c1', choices: ['x', 'y'] }, { card_id: 'c2', choices: ['p'] }])).resolves.toEqual({ saved: 2 });
    expect(calls[0].url).toBe('http://srv/cards/choices');
    expect(JSON.parse(calls[0].init.body as string)).toEqual({ items: [{ card_id: 'c1', choices: ['x', 'y'] }, { card_id: 'c2', choices: ['p'] }] });
    await expect(api.saveChoices([{ card_id: 'c1', choices: ['A1'] }])).rejects.toMatchObject({ status: 422, message: expect.stringContaining('right answer') });
  });
});
