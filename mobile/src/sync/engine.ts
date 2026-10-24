import type { ApiClient } from '@/api/client';
import type { LocalStore } from '@/study/store';

/** Logs sent per request; the server accepts at most 500. */
export const LOG_BATCH = 200;
/** Safety net against a server that never says it is done. */
const MAX_PAGES = 200;

export interface SyncSummary {
  /** reviews the server accepted for the first time */
  pushed: number;
  /** cards received or changed */
  cards: number;
  pages: number;
  /** the server was not the database this phone last synced with, so the library was downloaded afresh */
  reset: boolean;
}

/**
 * One full sync: upload pending reviews, then pull changes page by page until caught up.
 *
 * Reviews are only marked as uploaded after the server's response has been applied locally, in the same
 * transaction. If the request fails, or the app dies mid-sync, they are simply sent again next time, and the
 * server ignores the duplicates (reviews have client-generated ids).
 *
 * If the server says `reset`, the cached library is replaced and the whole review history becomes pending again,
 * so the loop keeps going until that history has been re-uploaded too.
 */
export async function runSync(api: Pick<ApiClient, 'sync'>, store: LocalStore): Promise<SyncSummary> {
  const summary: SyncSummary = { pushed: 0, cards: 0, pages: 0, reset: false };
  const deviceId = await store.deviceId();

  for (let page = 0; page < MAX_PAGES; page++) {
    const cursor = await store.cursor();
    const logs = await store.pendingLogs(LOG_BATCH);
    const res = await api.sync({ device_id: deviceId, cursor, epoch: await store.epoch(), logs });
    await store.applySync(res, logs.map((l) => l.id));

    summary.pages++;
    summary.pushed += res.accepted;
    summary.cards += res.cards.length;
    summary.reset ||= res.reset;

    const remaining = await store.pendingCount(); // more than one batch of reviews, or reviews re-opened by a reset
    if (!res.has_more && remaining === 0) return summary;
    // A reset legitimately moves the cursor backwards; any other page that claims "more" must move it forward.
    if (res.has_more && !res.reset && res.cursor <= cursor && remaining === 0) throw new Error('Sync stalled: the server did not advance the cursor.');
  }
  throw new Error('Sync did not finish.');
}
