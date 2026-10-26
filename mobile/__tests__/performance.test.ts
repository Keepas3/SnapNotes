/**
 * Keeps studying quick on a slow phone or emulator, where every call to the database is a round trip between JavaScript and the phone's
 * SQLite. These count calls instead of timing them, so they do not depend on the speed of the machine running the tests.
 */
import { runSync } from '@/sync/engine';

import { card, deckA, makeStore, response, seedCards } from './helpers/fixtures';

const NOW = new Date(2026, 2, 10, 15, 0, 0);

/** Count the calls the store makes to its database from now on. */
function countCalls(db: Awaited<ReturnType<typeof makeStore>>['db']) {
  const counter = { calls: 0 };
  for (const name of ['all', 'first', 'run', 'exec', 'transaction'] as const) {
    const original = (db as unknown as Record<string, (...a: unknown[]) => unknown>)[name].bind(db);
    (db as unknown as Record<string, unknown>)[name] = (...args: unknown[]) => {
      counter.calls++;
      return original(...args);
    };
  }
  return counter;
}

describe('preferences are read from memory after the first time', () => {
  it('asks the database once, and sees every change made through the store', async () => {
    const { store, db } = await makeStore();
    const counter = countCalls(db);
    expect(await store.newPerDay()).toBe(10);
    const afterFirst = counter.calls;
    expect(afterFirst).toBe(1);
    await store.newPerDay();
    await store.newPerDay();
    expect(counter.calls).toBe(afterFirst); // nothing more

    await store.setNewPerDay(25);
    const afterWrite = counter.calls;
    expect(await store.newPerDay()).toBe(25); // the new value, with no read
    expect(counter.calls).toBe(afterWrite);
  });

  it('covers the settings a card reads: new per day, order, scheduling and multiple choice', async () => {
    const { store, db } = await makeStore();
    await store.setOrderPrefs({ newOrder: 'order', mixNew: false });
    await store.setSchedulingPrefs({ retention: 0.95, spread: false });
    await store.setChoicePrefs({ enabled: true, count: 5, fillFromDeck: false });
    const counter = countCalls(db);
    expect(await store.orderPrefs()).toEqual({ newOrder: 'order', mixNew: false });
    expect(await store.schedulingPrefs()).toEqual({ retention: 0.95, spread: false });
    expect(await store.choicePrefs()).toEqual({ enabled: true, count: 5, fillFromDeck: false });
    expect(counter.calls).toBe(0);
  });

  it('is not fooled by a different phone database being opened (a new store reads for itself)', async () => {
    const first = await makeStore();
    await first.store.setNewPerDay(3);
    const second = await makeStore();
    expect(await second.store.newPerDay()).toBe(10);
  });
});

describe('what one study tap costs', () => {
  it('stays within a budget of database calls (it was 26 before the preferences were kept in memory and the counts took one trip)', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, Array.from({ length: 40 }, (_, i) => `c${i}`));
    // the same steps as the study screen: show the next card, answer it, show the one after
    const tap = async () => {
      const next = await store.nextCard(null, NOW, 20);
      await store.schedulingPrefs();
      await store.otherAnswers(next.card!.deck_id, next.card!.id);
      await store.answer(next.card!.id, 3, NOW, 1000);
      const after = await store.nextCard(null, NOW, 20);
      await store.schedulingPrefs();
      await store.otherAnswers(after.card!.deck_id, after.card!.id);
    };
    await tap(); // the first tap reads the preferences
    const counter = countCalls(db);
    await tap();
    expect(counter.calls).toBeLessThanOrEqual(13);
  });
});

describe('a sync says whether it brought anything new', () => {
  function fakeApi(replies: ReturnType<typeof response>[]) {
    return { async sync() { return replies.shift()!; } };
  }

  it('is false when the server only confirmed the reviews, and true for a card, a deck, a deletion or a fresh download', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    await store.answer('c1', 3, NOW, 800);
    expect((await runSync(fakeApi([response({ accepted: 1 })]), store)).changed).toBe(false);
    expect((await runSync(fakeApi([response({ cards: [card('c2')] })]), store)).changed).toBe(true);
    expect((await runSync(fakeApi([response({ decks: [{ ...deckA, name: 'Renamed' }] })]), store)).changed).toBe(true);
    expect((await runSync(fakeApi([response({ deleted: [{ kind: 'card', id: 'c2' }] })]), store)).changed).toBe(true);
    expect((await runSync(fakeApi([response({ reset: true }), response()]), store)).changed).toBe(true); // after a reset the history is sent again, hence a second page
  });
});
