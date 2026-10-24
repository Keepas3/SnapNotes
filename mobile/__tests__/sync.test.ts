import type { SyncRequest, SyncResponse } from '@/api/types';
import { LOG_BATCH, runSync } from '@/sync/engine';

import { card, deckA, makeStore, response, seedCards } from './helpers/fixtures';

const NOW = new Date(2026, 2, 10, 15, 0, 0);

/** Scripted server: answers each sync call from a queue and records the requests it got. */
function fakeApi(replies: (SyncResponse | Error)[]) {
  const requests: SyncRequest[] = [];
  return {
    requests,
    api: {
      async sync(req: SyncRequest): Promise<SyncResponse> {
        requests.push(JSON.parse(JSON.stringify(req)));
        const next = replies.shift();
        if (!next) throw new Error('unexpected sync call');
        if (next instanceof Error) throw next;
        return next;
      },
    },
  };
}

describe('runSync', () => {
  it('uploads pending reviews, applies what comes back, and marks the reviews as synced', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    await store.answer('c1', 3, NOW, 800);
    const { api, requests } = fakeApi([response({ cursor: 7, accepted: 1, cards: [card('c3')] })]);

    const summary = await runSync(api, store);

    expect(summary).toEqual({ pushed: 1, cards: 1, pages: 1, reset: false });
    expect(requests).toHaveLength(1);
    expect(requests[0].logs.map((l) => l.card_id)).toEqual(['c1']);
    expect(requests[0].cursor).toBe(1); // seeded earlier
    expect(requests[0].device_id).toMatch(/^id-/);
    expect(await store.pendingCount()).toBe(0);
    expect(await store.cursor()).toBe(7);
    expect(await store.getCard('c3')).not.toBeNull();
  });

  it('keeps reviews pending when the request fails, and resends the very same ones later', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    await store.answer('c1', 3, NOW, 800);
    const sentIds: string[][] = [];
    const { api } = fakeApi([new Error('Cannot reach the server'), response({ cursor: 3, accepted: 1 })]);
    const spy = { sync: async (req: SyncRequest) => (sentIds.push(req.logs.map((l) => l.id)), api.sync(req)) };

    await expect(runSync(spy, store)).rejects.toThrow('Cannot reach the server');
    expect(await store.pendingCount()).toBe(1);
    expect(await store.cursor()).toBe(1); // unchanged

    await runSync(spy, store);
    expect(await store.pendingCount()).toBe(0);
    expect(sentIds[1]).toEqual(sentIds[0]); // same review id, so the server can recognise the retry
  });

  it('keeps pulling while the server says there is more, following its cursor', async () => {
    const { store } = await makeStore();
    const { api, requests } = fakeApi([
      response({ cursor: 10, has_more: true, decks: [deckA], cards: [card('c1')] }),
      response({ cursor: 20, has_more: true, cards: [card('c2')] }),
      response({ cursor: 25, has_more: false, cards: [card('c3')] }),
    ]);
    const summary = await runSync(api, store);
    expect(summary).toMatchObject({ pages: 3, cards: 3 });
    expect(requests.map((r) => r.cursor)).toEqual([0, 10, 20]);
    expect(await store.cursor()).toBe(25);
    expect((await store.listDecks(NOW, 10))[0].total).toBe(3);
  });

  it('sends a large backlog of reviews in batches', async () => {
    const { store, db } = await makeStore();
    await seedCards(store, ['c1']);
    const total = LOG_BATCH * 2 + 50;
    await db.transaction(async (tx) => {
      for (let i = 0; i < total; i++) {
        await tx.run(
          "INSERT INTO review_logs(id, card_id, rating, reviewed_at, elapsed_ms, state_before, state_after_json, synced) VALUES (?, 'c1', 3, ?, 1, 2, '{}', 0)",
          [`log-${String(i).padStart(4, '0')}`, new Date(NOW.getTime() + i * 1000).toISOString()],
        );
      }
    });
    // '{}' is not a valid state, but the fake server does not care and the engine only forwards it.
    const { api, requests } = fakeApi([response({ cursor: 2 }), response({ cursor: 3 }), response({ cursor: 4 })]);
    await runSync(api, store);
    expect(requests.map((r) => r.logs.length)).toEqual([LOG_BATCH, LOG_BATCH, 50]);
    expect(requests[0].logs[0].id).toBe('log-0000'); // oldest first
    expect(await store.pendingCount()).toBe(0);
  });

  it('stops with an error if the server claims more data but never advances the cursor', async () => {
    const { store } = await makeStore();
    const { api } = fakeApi([response({ cursor: 0, has_more: true }), response({ cursor: 0, has_more: true })]);
    await expect(runSync(api, store)).rejects.toThrow(/stalled/);
  });

  it('works with nothing to send and nothing new', async () => {
    const { store } = await makeStore();
    const { api, requests } = fakeApi([response({ cursor: 0 })]);
    expect(await runSync(api, store)).toEqual({ pushed: 0, cards: 0, pages: 1, reset: false });
    expect(requests[0].logs).toEqual([]);
  });

  it('tells the server which database it last synced with', async () => {
    const { store } = await makeStore();
    const { api, requests } = fakeApi([response({ epoch: 'epoch-A', cursor: 3 }), response({ epoch: 'epoch-A', cursor: 3 })]);
    await runSync(api, store);
    await runSync(api, store);
    expect(requests[0].epoch).toBeNull(); // the very first sync has nothing to say
    expect(requests[1].epoch).toBe('epoch-A');
    expect(await store.epoch()).toBe('epoch-A');
  });

  it('studying offline then syncing: the review made while offline reaches the server and survives a refresh', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    const studied = await store.answer('c1', 3, NOW, 500); // no network involved

    // server accepts the log, and its refreshed copy of the card carries the same schedule back
    const { api } = fakeApi([
      response({ cursor: 5, accepted: 1, cards: [card('c1')], states: [{ card_id: 'c1', last_review: studied.fsrs.last_review!, state: studied.fsrs }] }),
    ]);
    await runSync(api, store);
    expect((await store.getCard('c1'))!.fsrs).toEqual(studied.fsrs);
    expect(await store.pendingCount()).toBe(0);
  });
});

describe('runSync when the server is not the database the phone last synced with', () => {
  /** A phone that has studied two cards and has everything uploaded, as of epoch-1 / cursor 9. */
  async function syncedPhone() {
    const { store } = await makeStore();
    await seedCards(store, ['c1', 'c2']);
    const first = await store.answer('c1', 3, NOW, 1);
    await store.answer('c2', 3, new Date(NOW.getTime() + 60_000), 1);
    const logs = await store.pendingLogs(10);
    await store.applySync(response({ cursor: 9, epoch: 'epoch-1' }), logs.map((l) => l.id));
    expect(await store.pendingCount()).toBe(0);
    return { store, first, logs };
  }

  it('drops cards the server no longer has, re-uploads the whole history, and restores schedules from the answer', async () => {
    const { store, first, logs } = await syncedPhone();
    const c2Log = logs.find((l) => l.card_id === 'c2')!;
    const { api, requests } = fakeApi([
      // the restored server only knows deck-a and c1, under a new identity
      response({ reset: true, epoch: 'epoch-2', cursor: 2, decks: [deckA], cards: [card('c1')] }),
      // ...and then receives the history: c1's reviews are news to it, c2's card is gone
      response({
        epoch: 'epoch-2',
        cursor: 3,
        accepted: 1,
        rejected: [{ id: c2Log.id, reason: 'unknown_card' }],
        states: [{ card_id: 'c1', last_review: first.fsrs.last_review!, state: first.fsrs }],
      }),
    ]);

    const summary = await runSync(api, store);

    expect(summary.reset).toBe(true);
    expect(requests).toHaveLength(2);
    expect(requests[0]).toMatchObject({ cursor: 9, epoch: 'epoch-1' });
    expect(requests[0].logs).toEqual([]); // nothing was pending when the reset was discovered
    expect(requests[1]).toMatchObject({ cursor: 2, epoch: 'epoch-2' });
    expect(requests[1].logs.map((l) => l.id).sort()).toEqual(logs.map((l) => l.id).sort()); // the full history, not just new reviews

    expect(await store.getCard('c2')).toBeNull(); // the phantom is gone
    expect((await store.getCard('c1'))!.fsrs).toEqual(first.fsrs); // and c1 has its schedule back
    expect(await store.pendingCount()).toBe(0);
    expect(await store.cursor()).toBe(3);
    expect(await store.epoch()).toBe('epoch-2');
  });

  it('is only a reset once: the next sync carries the new epoch and is an ordinary one', async () => {
    const { store } = await syncedPhone();
    const { api, requests } = fakeApi([
      response({ reset: true, epoch: 'epoch-2', cursor: 4, decks: [deckA], cards: [card('c1'), card('c2')] }),
      response({ epoch: 'epoch-2', cursor: 5 }), // answers the re-uploaded history
      response({ epoch: 'epoch-2', cursor: 5 }), // the next, ordinary sync
    ]);
    await runSync(api, store);
    const second = await runSync(api, store);
    expect(second.reset).toBe(false);
    expect(requests[2]).toMatchObject({ cursor: 5, epoch: 'epoch-2' });
    expect(requests[2].logs).toEqual([]);
  });

  it('does not mistake a reset that moves the cursor backwards for a stalled server', async () => {
    const { store } = await syncedPhone(); // cursor 9
    const { api } = fakeApi([
      response({ reset: true, epoch: 'epoch-2', cursor: 2, has_more: true, decks: [deckA], cards: [card('c1')] }),
      response({ epoch: 'epoch-2', cursor: 4, has_more: true, cards: [card('c2')] }),
      response({ epoch: 'epoch-2', cursor: 6 }),
      response({ epoch: 'epoch-2', cursor: 6 }), // the re-uploaded history
    ]);
    await expect(runSync(api, store)).resolves.toMatchObject({ reset: true });
    expect(await store.cursor()).toBe(6);
  });

  it('keeps going after a failure mid-way: the history is still marked as needing upload', async () => {
    const { store } = await syncedPhone();
    const { api } = fakeApi([
      response({ reset: true, epoch: 'epoch-2', cursor: 2, decks: [deckA], cards: [card('c1'), card('c2')] }),
      new Error('Cannot reach the server'), // the phone loses signal right after the reset page
    ]);
    await expect(runSync(api, store)).rejects.toThrow('Cannot reach the server');
    expect(await store.pendingCount()).toBe(2); // nothing is lost: the next sync picks up where this one stopped
    expect(await store.epoch()).toBe('epoch-2');

    const { api: retry, requests } = fakeApi([response({ epoch: 'epoch-2', cursor: 3 })]);
    await runSync(retry, store);
    expect(requests[0].logs).toHaveLength(2);
    expect(requests[0].epoch).toBe('epoch-2');
    expect(await store.pendingCount()).toBe(0);
  });
});
