import type { FsrsState } from '@/api/types';
import { migrate, MIGRATIONS } from '@/db/migrations';
import type { Db } from '@/db/types';
import { chanceRemembered, mostForgotten, previewDue, reviewCard } from '@/study/scheduler';
import { PROBLEM_LAPSES } from '@/study/store';

import { card, deckA, makeStore, response, seedCards } from './helpers/fixtures';
import { openUnmigratedTestDb } from './helpers/nodeDb';

const NOW = new Date(2026, 2, 10, 15, 0, 0);
const daysAgo = (n: number) => new Date(NOW.getTime() - n * 86_400_000);

/** A card that has been studied: it is in review, was last seen `seenDaysAgo` ago, is as strong as `stability` days, and is due `dueDaysAgo` ago. */
function studied(seenDaysAgo: number, stability: number, dueDaysAgo: number, lapses = 0): FsrsState {
  return {
    due: daysAgo(dueDaysAgo).toISOString(),
    stability,
    difficulty: 5,
    elapsed_days: seenDaysAgo,
    scheduled_days: Math.max(1, seenDaysAgo - dueDaysAgo),
    learning_steps: 0,
    reps: 5,
    lapses,
    state: 2,
    last_review: daysAgo(seenDaysAgo).toISOString(),
  };
}

async function setSchedule(db: Db, id: string, s: FsrsState) {
  await db.run('UPDATE cards SET state = ?, due = ?, last_review = ?, fsrs_json = ? WHERE id = ?', [s.state, s.due, s.last_review, JSON.stringify(s), id]);
}

describe('which card is most worth reviewing', () => {
  it('prefers the card least likely to be remembered over the one that is simply older', () => {
    const wellKnown = studied(33, 30, 3); // due 3 days ago, but known for a month at a time
    const barelyKnown = studied(3, 2, 1); // due yesterday, but only just learned
    expect(chanceRemembered(barelyKnown, NOW)).toBeLessThan(chanceRemembered(wellKnown, NOW));
    expect(mostForgotten([{ fsrs: wellKnown }, { fsrs: barelyKnown }], NOW)).toBe(1);
  });

  it('breaks a tie with the older due date, and finds nothing in an empty list', () => {
    const a = studied(5, 10, 2);
    const b = { ...a, due: daysAgo(4).toISOString() };
    expect(mostForgotten([{ fsrs: a }, { fsrs: b }], NOW)).toBe(1);
    expect(mostForgotten([], NOW)).toBe(-1);
  });

  it('counts a card never studied as not remembered at all', () => {
    expect(chanceRemembered({ ...studied(1, 5, 0), state: 0, last_review: null }, NOW)).toBe(0);
  });

  it('offers learning cards first, then the most forgotten review, not the oldest due date', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['old-strong', 'new-weak', 'learning']);
    await setSchedule(db, 'old-strong', studied(33, 30, 3));
    await setSchedule(db, 'new-weak', studied(3, 2, 1));
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('new-weak');

    await setSchedule(db, 'learning', { ...studied(0.01, 0.4, 0.001), state: 1 });
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('learning');
  });
});

describe('retention and spread', () => {
  const base = studied(10, 10, 0);

  it('brings cards back sooner when you aim to remember more of them', () => {
    const days = (retention: number) => reviewCard(base, 3, NOW, { retention, spread: false }).scheduled_days;
    expect(days(0.95)).toBeLessThan(days(0.9));
    expect(days(0.9)).toBeLessThan(days(0.8));
  });

  it('keeps today\'s exact schedules when no options are given', () => {
    expect(reviewCard(base, 3, NOW)).toEqual(reviewCard(base, 3, NOW, { retention: 0.9, spread: false }));
  });

  it('shows the time that the card really gets, even when spreading', () => {
    const options = { retention: 0.9, spread: true };
    for (const id of ['a', 'b', 'c', 'd', 'e', 'f']) {
      const shown = previewDue(base, NOW, options, id)[3].toISOString();
      expect(reviewCard(base, 3, NOW, options, id).due).toBe(shown);
    }
  });

  it('spreads cards with the same history by their id, and leaves short intervals alone', () => {
    const options = { retention: 0.9, spread: true };
    const longer = studied(60, 60, 0);
    const intervals = new Set('abcdefghijklmnop'.split('').map((id) => reviewCard(longer, 3, NOW, options, id).scheduled_days));
    expect(intervals.size).toBeGreaterThan(1);
    const fresh = reviewCard(studied(0.5, 0.5, 0), 3, NOW, options, 'a').scheduled_days;
    expect(fresh).toBe(reviewCard(studied(0.5, 0.5, 0), 3, NOW, { retention: 0.9, spread: false }, 'a').scheduled_days);
  });

  it('remembers the settings on the phone, spreading on and 90% by default, and ignores a retention it does not offer', async () => {
    const { store } = await makeStore();
    expect(await store.schedulingPrefs()).toEqual({ retention: 0.9, spread: true });
    await store.setSchedulingPrefs({ retention: 0.95, spread: false });
    expect(await store.schedulingPrefs()).toEqual({ retention: 0.95, spread: false });
    await store.setSchedulingPrefs({ retention: 0.5 });
    expect((await store.schedulingPrefs()).retention).toBe(0.95);
  });

  it('uses the saved settings when a card is answered', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['x']);
    await setSchedule(db, 'x', studied(10, 10, 0));
    await store.setSchedulingPrefs({ retention: 0.95, spread: false });
    const answered = await store.answer('x', 3, NOW, 1000);
    expect(answered.fsrs.scheduled_days).toBe(reviewCard(studied(10, 10, 0), 3, NOW, { retention: 0.95, spread: false }).scheduled_days);
  });
});

describe('pausing cards', () => {
  it('leaves a paused card out of studying, counting, practice and the next-due time, and brings it back unchanged', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['a', 'b']);
    await setSchedule(db, 'a', studied(5, 3, 1));
    await setSchedule(db, 'b', studied(5, 3, 0.5));
    expect((await store.counts(null, NOW, 10)).review).toBe(2);

    await store.pauseCard('a');
    expect((await store.counts(null, NOW, 10)).review).toBe(1);
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('b');
    expect(await store.allCardIds(null)).toEqual(['b']);
    const deck = (await store.listDecks(NOW, 10))[0];
    expect([deck.total, deck.review, deck.paused]).toEqual([2, 1, 1]); // still part of the deck, just not offered

    await store.resumeCard('a');
    expect((await store.counts(null, NOW, 10)).review).toBe(2);
    expect((await store.getCard('a'))?.fsrs.stability).toBe(3); // its schedule was kept
  });

  it('forgets the pause when the card or its deck is deleted, or the library is cleared', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['a', 'b', 'c']);
    for (const id of ['a', 'b', 'c']) await store.pauseCard(id);
    const flags = async () => (await db.all<{ card_id: string }>('SELECT card_id FROM card_flags ORDER BY card_id')).map((r) => r.card_id);

    await store.applySync(response({ deleted: [{ kind: 'card', id: 'a' }] }), []);
    expect(await flags()).toEqual(['b', 'c']);
    await store.applySync(response({ cards: [card('b', { status: 'rejected' })] }), []);
    expect(await flags()).toEqual(['c']);
    await store.applySync(response({ deleted: [{ kind: 'deck', id: deckA.id }] }), []);
    expect(await flags()).toEqual([]);

    await seedCards(store, ['z']);
    await store.pauseCard('z');
    await store.clearData();
    expect(await flags()).toEqual([]);
  });

  it('keeps a pause when the cards are downloaded again', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['a']);
    await store.pauseCard('a');
    await store.applySync(response({ reset: true, decks: [deckA], cards: [card('a')] }), []);
    expect((await store.cardStats(deckA.id)).find((c) => c.id === 'a')?.paused).toBe(true);
  });
});

describe('problem cards', () => {
  it('counts the cards missed many times, but not paused ones, and reports each card\'s misses', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['fine', 'problem', 'paused-problem']);
    await setSchedule(db, 'fine', studied(5, 3, 1, PROBLEM_LAPSES - 1));
    await setSchedule(db, 'problem', studied(5, 3, 1, PROBLEM_LAPSES));
    await setSchedule(db, 'paused-problem', studied(5, 3, 1, PROBLEM_LAPSES + 2));
    await store.pauseCard('paused-problem');

    const deck = (await store.listDecks(NOW, 10))[0];
    expect(deck.problem).toBe(1);
    const stats = await store.cardStats(deckA.id);
    expect(stats.find((c) => c.id === 'problem')).toEqual({ id: 'problem', lapses: PROBLEM_LAPSES, paused: false });
    expect(stats.find((c) => c.id === 'paused-problem')?.paused).toBe(true);
  });
});

describe('studying ahead', () => {
  it('offers cards due in the next few days only when asked, and schedules them like any other answer', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['soon', 'later']);
    await setSchedule(db, 'soon', studied(4, 6, -2)); // due in 2 days
    await setSchedule(db, 'later', studied(4, 12, -8)); // due in 8 days

    expect((await store.counts(null, NOW, 10)).review).toBe(0);
    expect((await store.nextCard(null, NOW, 10)).card).toBeNull();

    expect((await store.counts(null, NOW, 10, 3)).review).toBe(1);
    const ahead = await store.nextCard(null, NOW, 10, 3);
    expect(ahead.card?.id).toBe('soon');
    expect((await store.counts(null, NOW, 10, 10)).review).toBe(2);

    const answered = await store.answer('soon', 3, NOW, 1000);
    expect(answered.fsrs.reps).toBe(6);
    expect(new Date(answered.fsrs.due).getTime()).toBeGreaterThan(NOW.getTime());
    expect(await store.pendingCount()).toBe(1); // a real review, to be uploaded
  });

  it('shows today\'s reviews before the ones studied ahead', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['today', 'ahead']);
    await setSchedule(db, 'today', studied(40, 100, 0.5));
    await setSchedule(db, 'ahead', studied(1, 0.5, -1)); // far less remembered, but not due yet
    expect((await store.nextCard(null, NOW, 10, 3)).card?.id).toBe('today');
  });
});

describe('the phone database', () => {
  it('steps a version 3 database forward to pausing without losing its cards', async () => {
    const db = await openUnmigratedTestDb();
    for (const sql of MIGRATIONS.slice(0, 3)) await db.exec(sql);
    await db.exec('PRAGMA user_version = 3');
    await db.run("INSERT INTO decks(id, name, topic, updated_at) VALUES ('d1', 'Kept', '', 't')");
    await db.run(
      "INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, created_at, updated_at, due, fsrs_json) VALUES ('c1','d1','basic','f','b','notes','','t','t','t','{}')",
    );
    await migrate(db);
    expect((await db.first<{ user_version: number }>('PRAGMA user_version'))?.user_version).toBe(MIGRATIONS.length);
    expect((await db.first<{ n: number }>('SELECT COUNT(*) AS n FROM cards'))?.n).toBe(1);
    await db.run("INSERT INTO card_flags(card_id, paused) VALUES ('c1', 1)");
  });
});
