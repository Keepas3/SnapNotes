import type { FsrsState } from '@/api/types';
import { CardState, newState, reviewCard } from '@/study/scheduler';
import { DEFAULT_NEW_PER_DAY } from '@/study/store';

import { card, counterIds, deckA, makeStore, response, seedCards } from './helpers/fixtures';

// A weekday afternoon, local time (the study day rolls over at 4 a.m. local).
const NOW = new Date(2026, 2, 10, 15, 0, 0);
const plusMin = (n: number) => new Date(NOW.getTime() + n * 60_000);
const plusDays = (n: number) => new Date(NOW.getTime() + n * 86_400_000);

describe('LocalStore deck order', () => {
  it('remembers the order the user arranged, and starts empty', async () => {
    const { store } = await makeStore();
    expect(await store.deckOrder()).toEqual([]);
    await store.setDeckOrder(['b', 'a', 'c']);
    expect(await store.deckOrder()).toEqual(['b', 'a', 'c']);
    await store.setDeckOrder(['c']);
    expect(await store.deckOrder()).toEqual(['c']);
  });

  it('reads damaged saved data as no order at all', async () => {
    const { store } = await makeStore();
    for (const bad of ['not json', '{"a":1}', '"text"']) {
      await store.setMeta('deck_order', bad);
      expect(await store.deckOrder()).toEqual([]);
    }
    await store.setMeta('deck_order', '["a", 7, null, "b"]');
    expect(await store.deckOrder()).toEqual(['a', 'b']); // only the ids are kept
  });

  it('survives clearing the cached library, since it is only a preference', async () => {
    const { store } = await makeStore();
    await store.setDeckOrder(['b', 'a']);
    await store.clearData();
    expect(await store.deckOrder()).toEqual(['b', 'a']);
  });
});

describe('LocalStore.applySync', () => {
  it('stores decks and approved cards, each starting as a new card', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    const decks = await store.listDecks(NOW, 10);
    expect(decks).toHaveLength(1);
    expect(decks[0]).toMatchObject({ id: 'deck-a', name: 'Biology', total: 2, new: 2, learning: 0, review: 0 });
    const c1 = await store.getCard('c1');
    expect(c1?.fsrs.state).toBe(CardState.New);
    expect(c1?.front).toBe('Front c1');
  });

  it('keeps only approved cards, and drops one that stops being approved', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ decks: [deckA], cards: [card('ok'), card('draft', { status: 'draft' }), card('no', { status: 'rejected' })] }), []);
    expect((await store.listDecks(NOW, 10))[0].total).toBe(1);

    await store.applySync(response({ cards: [card('ok', { status: 'rejected' })] }), []);
    expect(await store.getCard('ok')).toBeNull();
  });

  it('updates card text without touching the schedule the phone has built up', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    await store.answer('c1', 3, NOW, 1200);
    const studied = (await store.getCard('c1'))!.fsrs;

    await store.applySync(response({ cursor: 2, cards: [card('c1', { front: 'Edited front', updated_at: '2026-03-02T00:00:00.000Z' })] }), []);
    const after = await store.getCard('c1');
    expect(after?.front).toBe('Edited front');
    expect(after?.fsrs).toEqual(studied);
  });

  it('takes a newer schedule from the server but never overwrites a newer local review', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    const local = await store.answer('c1', 3, NOW, 900); // reviewed at NOW

    const serverState = (reviewedAt: Date, rating: 1 | 4): FsrsState => reviewCard(newState(reviewedAt), rating, reviewedAt);
    const older = serverState(plusMin(-60), 4);
    await store.applySync(response({ cursor: 2, states: [{ card_id: 'c1', last_review: older.last_review!, state: older }] }), []);
    expect((await store.getCard('c1'))!.fsrs).toEqual(local.fsrs); // older: ignored

    const newer = serverState(plusMin(30), 4);
    await store.applySync(response({ cursor: 3, states: [{ card_id: 'c1', last_review: newer.last_review!, state: newer }] }), []);
    expect((await store.getCard('c1'))!.fsrs).toEqual(newer); // newer: adopted
  });

  it('ignores schedules for cards it does not hold', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ states: [{ card_id: 'ghost', last_review: NOW.toISOString(), state: newState(NOW) }] }), []);
    expect(await store.getCard('ghost')).toBeNull();
  });

  it('removes deleted cards, and a deleted deck with everything in it', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.applySync(response({ cursor: 2, deleted: [{ kind: 'card', id: 'c1' }] }), []);
    expect(await store.getCard('c1')).toBeNull();
    expect(await store.getCard('c2')).not.toBeNull();

    await store.applySync(response({ cursor: 3, deleted: [{ kind: 'deck', id: 'deck-a' }] }), []);
    expect(await store.listDecks(NOW, 10)).toEqual([]);
    expect(await store.getCard('c2')).toBeNull();
  });

  it('is idempotent: applying a response twice changes nothing', async () => {
    const { store } = await makeStore();
    const res = response({ cursor: 5, decks: [deckA], cards: [card('c1'), card('c2')] });
    await store.applySync(res, []);
    await store.answer('c1', 3, NOW, 1000);
    const before = JSON.stringify([await store.listDecks(NOW, 10), await store.getCard('c1'), await store.cursor()]);
    await store.applySync(res, []);
    expect(JSON.stringify([await store.listDecks(NOW, 10), await store.getCard('c1'), await store.cursor()])).toBe(before);
  });

  it('records the cursor and marks only the logs that were uploaded as synced', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.answer('c1', 3, NOW, 1);
    await store.answer('c2', 3, plusMin(1), 1);
    const pending = await store.pendingLogs(10);
    expect(pending).toHaveLength(2);

    await store.applySync(response({ cursor: 42 }), [pending[0].id]);
    expect(await store.cursor()).toBe(42);
    expect((await store.pendingLogs(10)).map((l) => l.id)).toEqual([pending[1].id]);
    expect(await store.pendingCount()).toBe(1);
  });

  it('rolls back everything if the response cannot be applied', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    const cursorBefore = await store.cursor();
    const bad = response({ cursor: 9, decks: [{ ...deckA, name: 'Changed' }], cards: [{ ...card('c9'), front: null as unknown as string }] });
    await expect(store.applySync(bad, [])).rejects.toThrow();
    expect((await store.listDecks(NOW, 10))[0].name).toBe('Biology'); // the deck rename did not stick
    expect(await store.getCard('c9')).toBeNull();
    expect(await store.cursor()).toBe(cursorBefore); // and the cursor did not move, so the page is fetched again
  });
});

describe('LocalStore.applySync when the server asks for a reset', () => {
  it('replaces the cached library, keeps the identity and settings, and re-opens every review for upload', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.setNewPerDay(7);
    const device = await store.deviceId();
    await store.answer('c1', 3, NOW, 1);
    await store.answer('c2', 3, plusMin(1), 1);
    await store.applySync(response({ cursor: 9, epoch: 'epoch-1' }), (await store.pendingLogs(10)).map((l) => l.id));
    expect(await store.pendingCount()).toBe(0);

    // The server now knows one deck with one of the two cards, under a new epoch, at a lower position.
    await store.applySync(
      response({ reset: true, epoch: 'epoch-2', cursor: 3, decks: [{ ...deckA, name: 'Restored' }], cards: [card('c1')] }),
      [],
    );

    expect((await store.listDecks(NOW, 10)).map((d) => d.name)).toEqual(['Restored']);
    expect(await store.getCard('c2')).toBeNull(); // not on the server any more: gone from the phone
    expect(await store.cursor()).toBe(3);
    expect(await store.epoch()).toBe('epoch-2');
    expect(await store.pendingCount()).toBe(2); // the whole history waits to be uploaded again
    expect(await store.deviceId()).toBe(device);
    expect(await store.newPerDay()).toBe(7);
  });

  it('counts the reviews carried by the reset request itself as uploaded, and re-opens the rest', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.answer('c1', 3, NOW, 1);
    await store.answer('c2', 3, plusMin(1), 1);
    const [sentNow, other] = await store.pendingLogs(10);
    await store.applySync(response({ reset: true, epoch: 'epoch-2', cursor: 3, decks: [deckA], cards: [card('c1'), card('c2')] }), [sentNow.id]);
    expect((await store.pendingLogs(10)).map((l) => l.id)).toEqual([other.id]);
  });

  it('an ordinary response leaves the cache and history alone but still records the epoch', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    await store.answer('c1', 3, NOW, 1);
    await store.applySync(response({ epoch: 'epoch-9', cursor: 2 }), []);
    expect(await store.getCard('c1')).not.toBeNull();
    expect(await store.pendingCount()).toBe(1);
    expect(await store.epoch()).toBe('epoch-9');
  });

  it('forgets the epoch together with the rest of the cache when the server address changes', async () => {
    const { store } = await makeStore();
    await store.bindServer('http://one');
    await store.applySync(response({ epoch: 'epoch-1', cursor: 4 }), []);
    expect(await store.epoch()).toBe('epoch-1');
    await store.bindServer('http://two');
    expect(await store.epoch()).toBeNull();
  });
});

describe('LocalStore.answer', () => {
  it('updates the schedule and appends a log with the before and after state', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    const updated = await store.answer('c1', 3, NOW, 2345);

    expect(updated.fsrs.state).toBe(CardState.Learning);
    expect(updated.fsrs.last_review).toBe(NOW.toISOString());
    const [log] = await store.pendingLogs(10);
    expect(log).toMatchObject({ card_id: 'c1', rating: 3, reviewed_at: NOW.toISOString(), elapsed_ms: 2345, state_before: CardState.New });
    expect(log.state_after).toEqual(updated.fsrs);
    expect(log.id).toMatch(/^id-/);
  });

  it('is atomic: if the log cannot be written, the card keeps its old schedule', async () => {
    let calls = 0;
    const { store } = await makeStore(() => {
      if (++calls > 1) throw new Error('no ids left'); // the first id is spent on the device id below
      return 'device';
    });
    await seedCards(store, ['c1']);
    await store.deviceId();
    await expect(store.answer('c1', 3, NOW, 1)).rejects.toThrow('no ids left');
    expect((await store.getCard('c1'))!.fsrs.state).toBe(CardState.New);
    expect(await store.pendingCount()).toBe(0);
  });

  it('rejects an answer for a card that is gone', async () => {
    const { store } = await makeStore();
    await expect(store.answer('missing', 3, NOW, 1)).rejects.toThrow(/no longer/);
  });

  it('numbers logs in the order they were answered', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2', 'c3']);
    for (const [i, id] of ['c1', 'c2', 'c3'].entries()) await store.answer(id, 3, plusMin(i), 1);
    expect((await store.pendingLogs(10)).map((l) => l.card_id)).toEqual(['c1', 'c2', 'c3']);
    expect((await store.pendingLogs(2)).map((l) => l.card_id)).toEqual(['c1', 'c2']);
  });
});

describe('LocalStore.nextCard', () => {
  it('offers new cards oldest first and finishes when there are none left', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('c1');
    await store.answer('c1', 4, NOW, 1); // easy: graduates, due in days
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('c2');
    await store.answer('c2', 4, NOW, 1);
    const done = await store.nextCard(null, NOW, 10);
    expect(done.card).toBeNull();
    expect(done.counts).toEqual({ new: 0, learning: 0, review: 0 });
  });

  it('brings a failed card back within the session, ahead of new cards', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.answer('c1', 1, NOW, 1); // again: due in a minute or so
    // not yet due, but within learn-ahead: new cards come first
    expect((await store.nextCard(null, plusMin(0), 10)).card?.id).toBe('c2');
    await store.answer('c2', 4, NOW, 1);
    // nothing else to do, so the learning card is offered early (learn-ahead) instead of ending the session
    expect((await store.nextCard(null, plusMin(0), 10)).card?.id).toBe('c1');
    // once due it jumps the queue
    await seedCards(store, ['c3'], deckA, '2026-03-05T00:00:00.000Z');
    expect((await store.nextCard(null, plusMin(5), 10)).card?.id).toBe('c1');
  });

  it('shows reviews due today before new cards, but not ones due tomorrow', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['old', 'fresh']);
    await store.answer('old', 4, plusDays(-30), 1); // easy a month ago -> due long ago
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('old');

    await store.answer('old', 4, NOW, 1); // reviewed today: due in days
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('fresh');
    expect(await store.counts(null, NOW, 10)).toEqual({ new: 1, learning: 0, review: 0 });
  });

  it('treats a card due later today as due now (day granularity for reviews)', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['c1']);
    await store.answer('c1', 4, plusDays(-10), 1);
    const laterToday = new Date(2026, 2, 10, 22, 0, 0); // before the 4 a.m. rollover
    await db.run('UPDATE cards SET due = ? WHERE id = ?', [laterToday.toISOString(), 'c1']);
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('c1');
    const tomorrow = new Date(2026, 2, 11, 9, 0, 0);
    await db.run('UPDATE cards SET due = ? WHERE id = ?', [tomorrow.toISOString(), 'c1']);
    expect((await store.nextCard(null, NOW, 10)).card).toBeNull();
  });

  it('stops offering new cards once the daily budget is used, and resets it the next study day', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2', 'c3', 'c4']);
    for (const id of ['c1', 'c2']) await store.answer(id, 4, NOW, 1);
    expect(await store.counts(null, NOW, 2)).toMatchObject({ new: 0 });
    expect((await store.nextCard(null, NOW, 2)).card).toBeNull(); // budget of 2 is spent

    expect((await store.nextCard(null, NOW, 3)).card?.id).toBe('c3'); // raising the limit lets more through

    const nextMorning = new Date(2026, 2, 11, 9, 0, 0);
    expect((await store.nextCard(null, nextMorning, 2)).card?.id).toBe('c3'); // a new study day
  });

  it('counts a card against the budget only the first time it is answered', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.answer('c1', 1, NOW, 1); // again, still learning
    await store.answer('c1', 1, plusMin(2), 1); // and again
    expect(await store.counts(null, plusMin(3), 2)).toMatchObject({ new: 1 }); // only one slot used
  });

  it('can be limited to one deck', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['a1'], deckA);
    await seedCards(store, ['b1'], { id: 'deck-b', name: 'Chem', topic: '', updated_at: '2026-03-01T08:00:00.000Z' }, '2026-03-01T07:00:00.000Z');
    expect((await store.nextCard('deck-a', NOW, 10)).card?.id).toBe('a1');
    expect((await store.nextCard('deck-b', NOW, 10)).card?.id).toBe('b1');
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('b1'); // oldest first across decks
    expect(await store.counts('deck-a', NOW, 10)).toEqual({ new: 1, learning: 0, review: 0 });
  });

  it('reports when the next studied card comes due', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    expect(await store.nextDue(null)).toBeNull(); // nothing studied yet
    const a = await store.answer('c1', 4, NOW, 1);
    await store.answer('c2', 4, plusDays(1), 1);
    expect(await store.nextDue(null)).toBe(a.fsrs.due);
  });
});

describe('LocalStore settings and identity', () => {
  it('keeps one device id for good', async () => {
    const { store } = await makeStore(counterIds('dev'));
    const first = await store.deviceId();
    expect(await store.deviceId()).toBe(first);
    expect(first).toBe('dev-1');
  });

  it('defaults and clamps the daily new-card limit', async () => {
    const { store } = await makeStore();
    expect(await store.newPerDay()).toBe(DEFAULT_NEW_PER_DAY);
    await store.setNewPerDay(25);
    expect(await store.newPerDay()).toBe(25);
    await store.setNewPerDay(-4);
    expect(await store.newPerDay()).toBe(0);
    await store.setNewPerDay(99999);
    expect(await store.newPerDay()).toBe(500);
  });

  it('clears the cached library when the server changes, but keeps the device id and preferences', async () => {
    const { store } = await makeStore();
    expect(await store.bindServer('http://one')).toBe('new');
    expect(await store.bindServer('http://one')).toBe('same');
    await seedCards(store, ['c1']);
    await store.answer('c1', 3, NOW, 1);
    await store.setNewPerDay(7);
    const device = await store.deviceId();
    await store.applySync(response({ cursor: 12 }), []);

    expect(await store.bindServer('http://two')).toBe('reset');
    expect(await store.listDecks(NOW, 10)).toEqual([]);
    expect(await store.pendingCount()).toBe(0);
    expect(await store.cursor()).toBe(0);
    expect(await store.deviceId()).toBe(device);
    expect(await store.newPerDay()).toBe(7);
  });
});
