import type { SyncCard, SyncResponse } from '@/api/types';
import type { Db } from '@/db/types';
import { LocalStore } from '@/study/store';

import { openTestDb } from './nodeDb';

/** Deterministic ids: "id-1", "id-2", ... */
export function counterIds(prefix = 'id') {
  let n = 0;
  return () => `${prefix}-${++n}`;
}

export async function makeStore(newId = counterIds()): Promise<{ store: LocalStore; db: Db }> {
  const db = await openTestDb();
  return { store: new LocalStore(db, newId), db };
}

export function card(id: string, over: Partial<SyncCard> = {}): SyncCard {
  return {
    id,
    deck_id: 'deck-a',
    type: 'basic',
    front: `Front ${id}`,
    back: `Back ${id}`,
    origin: 'notes',
    source_ref: 'p.1',
    source_snippet: 'snippet',
    status: 'approved',
    created_at: '2026-03-01T08:00:00.000000+00:00',
    updated_at: '2026-03-01T08:00:00.000000+00:00',
    ...over,
  };
}

export function response(over: Partial<SyncResponse> = {}): SyncResponse {
  return {
    epoch: 'epoch-1',
    reset: false,
    cursor: 1,
    has_more: false,
    server_time: '2026-03-01T09:00:00.000Z',
    accepted: 0,
    duplicates: 0,
    rejected: [],
    decks: [],
    cards: [],
    states: [],
    deleted: [],
    ...over,
  };
}

export const deckA = { id: 'deck-a', name: 'Biology', topic: 'biology', updated_at: '2026-03-01T08:00:00.000000+00:00' };

/** Create a deck with `n` approved, never-studied cards, through the same path real data takes. */
export async function seedCards(store: LocalStore, ids: string[], deck = deckA, created = '2026-03-01T08:00:00.000000+00:00') {
  await store.applySync(
    response({ decks: [deck], cards: ids.map((id, i) => card(id, { deck_id: deck.id, created_at: new Date(Date.parse(created) + i * 1000).toISOString() })) }),
    [],
  );
}
