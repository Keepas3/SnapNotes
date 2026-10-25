import type { FsrsState } from '@/api/types';
import { migrate, MIGRATIONS } from '@/db/migrations';
import { seededFirst, seededUnit } from '@/lib/order';
import { newState, rescheduleForRetention, reviewCard } from '@/study/scheduler';
import { runSync } from '@/sync/engine';

import { makeStore, response, seedCards } from './helpers/fixtures';
import { openUnmigratedTestDb } from './helpers/nodeDb';

const NOW = new Date(2026, 2, 10, 15, 0, 0);
const at = (days: number, hour = 15) => new Date(2026, 2, 10 + days, hour, 0, 0);

describe('undoing an answer', () => {
  async function setup() {
    const ctx = await makeStore();
    await ctx.store.setOrderPrefs({ newOrder: 'order', mixNew: false });
    await seedCards(ctx.store, ['a', 'b', 'c']);
    ctx.store.startUndoSession(at(0, 9));
    return ctx;
  }

  it('gives the card back exactly as it was, takes the answer out of the history, and gives the new-card slot back', async () => {
    const { store } = await setup();
    const before = (await store.getCard('a'))!.fsrs;
    await store.answer('a', 4, NOW, 1500);
    expect((await store.getCard('a'))!.fsrs).not.toEqual(before);
    expect(await store.counts(null, NOW, 1)).toMatchObject({ new: 0 }); // a limit of one new card a day is used up

    const restored = await store.undoLast();
    expect(restored?.id).toBe('a');
    expect(restored?.fsrs).toEqual(before);
    expect((await store.getCard('a'))!.fsrs).toEqual(before);
    expect(await store.pendingCount()).toBe(0);
    expect(await store.counts(null, NOW, 1)).toMatchObject({ new: 1 });
    expect((await store.nextCard(null, NOW, 10)).card?.id).toBe('a'); // and it is the next card again
  });

  it('goes back one answer at a time, most recent first, and then has nothing left', async () => {
    const { store } = await setup();
    await store.answer('a', 3, at(0, 10), 1);
    await store.answer('b', 4, at(0, 11), 1);
    await store.answer('a', 3, at(0, 12), 1); // a second answer to the same card
    expect(await store.undoCount()).toBe(3);
    expect((await store.undoLast())?.id).toBe('a');
    expect((await store.getCard('a'))!.fsrs.reps).toBe(1); // back to after its first answer
    expect((await store.undoLast())?.id).toBe('b');
    expect((await store.undoLast())?.id).toBe('a');
    expect((await store.getCard('a'))!.fsrs.reps).toBe(0);
    expect(await store.undoLast()).toBeNull();
    expect(await store.undoCount()).toBe(0);
  });

  it('holds answers back from upload while the session is open, and releases them when it ends', async () => {
    const { store } = await setup();
    await store.answer('a', 3, at(0, 10), 1);
    expect(await store.pendingLogs(10)).toEqual([]); // held
    expect(await store.pendingCount()).toBe(1); // but still waiting to sync, as far as Home is concerned
    expect(await store.sendableCount()).toBe(0);

    store.endUndoSession();
    expect((await store.pendingLogs(10)).map((l) => l.card_id)).toEqual(['a']);
    expect(await store.sendableCount()).toBe(1);
    expect(await store.undoLast()).toBeNull(); // too late
    expect(await store.undoCount()).toBe(0);
  });

  it('still uploads what was answered before the screen was opened', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['a', 'b']);
    await store.answer('a', 3, at(0, 8), 1); // before
    store.startUndoSession(at(0, 9));
    await store.answer('b', 3, at(0, 10), 1); // during
    expect((await store.pendingLogs(10)).map((l) => l.card_id)).toEqual(['a']);
    expect(await store.undoCount()).toBe(1);
  });

  it('does not stall the sync while answers are held (a sync would otherwise keep looking for more to send)', async () => {
    const { store } = await setup();
    await store.answer('a', 3, at(0, 10), 1);
    const sync = jest.fn(async (req: { logs: unknown[] }) => response({ accepted: req.logs.length }));
    const summary = await runSync({ sync } as never, store);
    expect(sync).toHaveBeenCalledTimes(1);
    expect(summary.pushed).toBe(0);
    expect(await store.pendingCount()).toBe(1); // still held, still undoable
    expect(await store.undoCount()).toBe(1);
  });

  it('leaves a card alone if it has been answered since', async () => {
    const { store, db } = await setup();
    await store.answer('a', 3, at(0, 10), 1);
    await db.run("UPDATE cards SET last_review = '2026-03-10T22:00:00.000Z' WHERE id = 'a'"); // something else changed it
    expect(await store.undoLast()).toBeNull();
    expect(await store.pendingCount()).toBe(1); // nothing was thrown away
  });

  it('cannot undo answers that were made before this feature (no saved schedule)', async () => {
    const { store, db } = await setup();
    await store.answer('a', 3, at(0, 10), 1);
    await db.run('UPDATE review_logs SET fsrs_before_json = NULL');
    expect(await store.undoCount()).toBe(0);
    expect(await store.undoLast()).toBeNull();
  });

  it('steps a version 4 database forward without losing its answers', async () => {
    const db = await openUnmigratedTestDb();
    for (const sql of MIGRATIONS.slice(0, 4)) await db.exec(sql);
    await db.exec('PRAGMA user_version = 4');
    await db.run("INSERT INTO review_logs(id, card_id, rating, reviewed_at, elapsed_ms, state_before, state_after_json, synced) VALUES ('l1','c1',3,'t',100,0,'{}',1)");
    await migrate(db);
    expect((await db.first<{ user_version: number }>('PRAGMA user_version'))?.user_version).toBe(MIGRATIONS.length);
    expect((await db.first<{ fsrs_before_json: string | null }>("SELECT fsrs_before_json FROM review_logs WHERE id = 'l1'"))?.fsrs_before_json).toBeNull();
  });
});

describe('the order of new cards', () => {
  const deckB = { id: 'deck-b', name: 'Chem', topic: '', updated_at: '2026-03-01T08:00:00.000000+00:00' };

  async function twelveNewCards() {
    const ctx = await makeStore();
    const a = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'];
    const b = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6'];
    await seedCards(ctx.store, a);
    await seedCards(ctx.store, b, deckB, '2026-03-01T09:00:00.000Z'); // made later, so "in order" shows all of deck A first
    return { ...ctx, inOrder: [...a, ...b] };
  }

  /** The cards in the order they are offered, answering each "Easy" so it leaves the new cards. */
  async function sequence(store: Awaited<ReturnType<typeof makeStore>>['store'], day: Date, count: number) {
    const seen: string[] = [];
    for (let i = 0; i < count; i++) {
      const next = await store.nextCard(null, day, 100);
      if (!next.card) break;
      seen.push(next.card.id);
      await store.answer(next.card.id, 4, day, 1);
    }
    return seen;
  }

  it('offers the cards in the order they were made when asked to', async () => {
    const { store, inOrder } = await twelveNewCards();
    await store.setOrderPrefs({ newOrder: 'order' });
    expect(await sequence(store, NOW, 12)).toEqual(inOrder);
  });

  it('shuffles by default: every card once, not in creation order, mixing the decks, and the same all day', async () => {
    const { store, inOrder } = await twelveNewCards();
    expect(await store.orderPrefs()).toEqual({ newOrder: 'shuffled', mixNew: true });
    const first = (await store.nextCard(null, NOW, 100)).card?.id;
    expect((await store.nextCard(null, at(0, 20), 100)).card?.id).toBe(first); // same study day, same next card
    const seen = await sequence(store, NOW, 12);
    expect([...seen].sort()).toEqual([...inOrder].sort());
    expect(seen).not.toEqual(inOrder);
    expect(seen.slice(0, 6).some((id) => id.startsWith('a')) && seen.slice(0, 6).some((id) => id.startsWith('b'))).toBe(true);
  });

  it('shuffles differently on another day', async () => {
    const { store } = await twelveNewCards();
    const picks = new Set<string | undefined>();
    for (let day = 0; day < 6; day++) picks.add((await store.nextCard(null, at(day), 100)).card?.id);
    expect(picks.size).toBeGreaterThan(1);
  });

  it('gives the same shuffle for the same seed and a different one for another', () => {
    const ids = Array.from({ length: 30 }, (_, i) => `card-${i}`);
    expect(seededFirst(ids, 'day-1')).toBe(seededFirst(ids, 'day-1'));
    expect(new Set(['d1', 'd2', 'd3', 'd4', 'd5', 'd6'].map((d) => seededFirst(ids, d))).size).toBeGreaterThan(1);
    expect(seededFirst([], 'x')).toBeNull();
    const units = Array.from({ length: 200 }, (_, i) => seededUnit(`seed-${i}`));
    expect(Math.min(...units)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...units)).toBeLessThan(1);
    expect(units.filter((u) => u < 0.5).length).toBeGreaterThan(70); // roughly half
    expect(units.filter((u) => u < 0.5).length).toBeLessThan(130);
  });

  describe('mixed in with reviews', () => {
    async function reviewsAndNew() {
      const ctx = await makeStore();
      await seedCards(ctx.store, ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']);
      for (const id of ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']) await ctx.store.answer(id, 4, at(-40), 1); // reviewed long ago: due now
      return ctx;
    }

    it('shows reviews first, then new cards, when mixing is off', async () => {
      const { store } = await reviewsAndNew();
      await store.setOrderPrefs({ newOrder: 'order', mixNew: false });
      const seen = await sequence(store, NOW, 12);
      expect(seen.slice(0, 6).every((id) => id.startsWith('r'))).toBe(true);
      expect(seen.slice(6).every((id) => id.startsWith('n'))).toBe(true);
    });

    it('spreads the new cards through the reviews when mixing is on', async () => {
      const { store } = await reviewsAndNew();
      await store.setOrderPrefs({ newOrder: 'order', mixNew: true });
      const seen = await sequence(store, NOW, 12);
      expect([...seen].sort()).toEqual(['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']);
      const firstNew = seen.findIndex((id) => id.startsWith('n'));
      const lastReview = seen.map((id) => id.startsWith('r')).lastIndexOf(true);
      expect(firstNew).toBeLessThan(6); // a new card turns up before the reviews are finished...
      expect(lastReview).toBeGreaterThan(5); // ...and a review turns up after new cards have started
    });

    it('still shows a learning card that is due before either', async () => {
      const { store } = await reviewsAndNew();
      await store.answer('n1', 1, NOW, 1); // missed: due again in about a minute
      expect((await store.nextCard(null, at(0, 15) /* the same minute */, 100)).card?.id).not.toBe('n1'); // not yet due
      const later = new Date(NOW.getTime() + 20 * 60_000);
      expect((await store.nextCard(null, later, 100)).card?.id).toBe('n1'); // now due, ahead of reviews and new cards
    });

    it('still respects the daily limit of new cards', async () => {
      const { store } = await reviewsAndNew();
      const seen: string[] = [];
      for (let i = 0; i < 20; i++) {
        const next = await store.nextCard(null, NOW, 2);
        if (!next.card) break;
        seen.push(next.card.id);
        await store.answer(next.card.id, 4, NOW, 1);
      }
      expect(seen.filter((id) => id.startsWith('n'))).toHaveLength(2);
      expect(seen.filter((id) => id.startsWith('r'))).toHaveLength(6);
    });
  });
});

describe('changing the retention of cards already scheduled', () => {
  const T0 = new Date(2026, 0, 5, 9, 0, 0);
  const exact = (retention: number) => ({ retention, spread: false });

  /** A card in review with some history behind it. */
  function reviewState(): FsrsState {
    let s = reviewCard(newState(T0), 4, T0, exact(0.9));
    for (let i = 0; i < 3; i++) s = reviewCard(s, 3, new Date(s.due), exact(0.9));
    return s;
  }

  it('gives the wait that answering would have given at that retention, from the strength the card already has', () => {
    const base = reviewState();
    const now = new Date(base.due);
    for (const [from, to] of [[0.9, 0.95], [0.9, 0.8], [0.95, 0.85], [0.8, 0.9]] as const) {
      const answeredAt = (r: number) => reviewCard(base, 3, now, exact(r));
      const moved = rescheduleForRetention(answeredAt(from), exact(to), 'card');
      expect(moved?.scheduled_days).toBe(answeredAt(to).scheduled_days);
      expect(moved?.due).toBe(answeredAt(to).due);
      expect(moved?.stability).toBe(answeredAt(from).stability); // its history is untouched
    }
  });

  it('moves a card sooner for a higher retention and later for a lower one', () => {
    const s = reviewState();
    const days = (r: number) => (rescheduleForRetention(s, exact(r), 'x') ?? s).scheduled_days;
    expect(days(0.95)).toBeLessThan(days(0.9));
    expect(days(0.8)).toBeGreaterThan(days(0.9));
  });

  it('leaves a card alone that already follows the setting, and one not in review', () => {
    const s = reviewState();
    expect(rescheduleForRetention(s, exact(0.9), 'x')).toBeNull();
    expect(rescheduleForRetention(newState(T0), exact(0.95), 'x')).toBeNull();
    expect(rescheduleForRetention(reviewCard(newState(T0), 1, T0, exact(0.9)), exact(0.95), 'x')).toBeNull(); // still learning
  });

  it('with spreading on, keeps a wait that is inside the allowed spread, and settles on one that stays put when applied again', () => {
    const s = reviewState();
    const spread = { retention: 0.95, spread: true };
    const once = rescheduleForRetention(s, spread, 'card-1');
    expect(once).not.toBeNull();
    expect(rescheduleForRetention(once!, spread, 'card-1')).toBeNull(); // applying the same setting again moves nothing
    const waits = new Set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].map((id) => rescheduleForRetention(s, spread, id)?.scheduled_days));
    expect(waits.size).toBeGreaterThan(1); // cards with the same strength do not all land on one day
  });

  describe('on the phone', () => {
    async function cardsInReview() {
      const ctx = await makeStore();
      await ctx.store.setSchedulingPrefs({ spread: false });
      await seedCards(ctx.store, ['r1', 'r2', 'r3', 'learning', 'new']);
      for (const id of ['r1', 'r2', 'r3']) {
        await ctx.store.answer(id, 4, at(-30), 1);
        await ctx.store.answer(id, 3, at(-12), 1);
      }
      await ctx.store.answer('learning', 1, NOW, 1);
      return ctx;
    }
    const dues = async (store: Awaited<ReturnType<typeof makeStore>>['store']) =>
      Object.fromEntries(await Promise.all(['r1', 'r2', 'r3', 'learning', 'new'].map(async (id) => [id, (await store.getCard(id))!.fsrs.due])));

    it('describes what would change before anything does', async () => {
      const { store } = await cardsInReview();
      const before = await dues(store);
      const preview = await store.previewRetentionChange(0.95, NOW);
      expect(preview).toMatchObject({ review: 3, changed: 3, earlier: 3, later: 0 });
      expect(preview.averageDays).toBeLessThan(0);
      expect(await dues(store)).toEqual(before); // nothing moved
      expect((await store.previewRetentionChange(0.8, NOW)).later).toBe(3);
      expect(await store.previewRetentionChange(0.9, NOW)).toMatchObject({ review: 3, changed: 0 }); // they were scheduled at 90%
    });

    it('moves only the cards in review, keeps their strength and history, and can be changed back', async () => {
      const { store } = await cardsInReview();
      const before = await dues(store);
      const logs = await store.pendingCount();
      const strength = (await store.getCard('r1'))!.fsrs.stability;

      expect(await store.applyRetentionChange(0.95)).toBe(3);
      const after = await dues(store);
      for (const id of ['r1', 'r2', 'r3']) expect(new Date(after[id]).getTime()).toBeLessThan(new Date(before[id]).getTime());
      expect(after.learning).toBe(before.learning);
      expect(after.new).toBe(before.new);
      expect((await store.getCard('r1'))!.fsrs.stability).toBe(strength);
      expect(await store.pendingCount()).toBe(logs); // it is not an answer

      expect(await store.applyRetentionChange(0.95)).toBe(0); // again: nothing left to move
      expect(await store.applyRetentionChange(0.9)).toBe(3);
      expect(await dues(store)).toEqual(before); // back where they were
    });

    it('counts the cards that would become due today', async () => {
      const { store } = await cardsInReview();
      const preview = await store.previewRetentionChange(0.95, new Date(NOW.getTime() + 200 * 86_400_000)); // far in the future: everything is due then anyway
      expect(preview.dueToday).toBe(0);
    });
  });
});
