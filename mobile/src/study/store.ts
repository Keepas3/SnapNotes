import type { CardType, FsrsState, Origin, SyncLogWire, SyncResponse } from '@/api/types';
import type { Db, SqlValue } from '@/db/types';

import { studyDayEnd, studyDayStart } from './day';
import { newState, reviewCard, type RatingValue } from './scheduler';

/** Cards still in (re)learning that are due within this long are offered early rather than ending the session. */
export const LEARN_AHEAD_MS = 20 * 60_000;
export const DEFAULT_NEW_PER_DAY = 10;

export interface LocalCard {
  id: string;
  deck_id: string;
  type: CardType;
  front: string;
  back: string;
  origin: Origin;
  source_ref: string;
  source_snippet: string;
  created_at: string;
  updated_at: string;
  fsrs: FsrsState;
}

export interface StudyCounts {
  new: number;
  learning: number;
  review: number;
}

export interface DeckSummary extends StudyCounts {
  id: string;
  name: string;
  topic: string;
  /** The folder the deck is filed in, or null. */
  folder_id: string | null;
  updated_at: string;
  total: number;
}

/** A folder with how much it holds; what is due in it comes from `counts` with the folder as the scope. */
export interface FolderSummary {
  id: string;
  name: string;
  updated_at: string;
  decks: number;
  total: number;
}

/** What to study or count: one deck (its id), every deck in a folder, or everything (null). */
export type StudyScope = string | { folderId: string } | null;

/** The SQL that narrows a query on `cards` to a scope, with its arguments. */
function scopeClause(scope: StudyScope): { sql: string; args: SqlValue[] } {
  if (scope === null) return { sql: '', args: [] };
  if (typeof scope === 'string') return { sql: ' AND deck_id = ?', args: [scope] };
  return { sql: ' AND deck_id IN (SELECT id FROM decks WHERE folder_id = ?)', args: [scope.folderId] };
}

interface CardRow {
  id: string;
  deck_id: string;
  type: CardType;
  front: string;
  back: string;
  origin: Origin;
  source_ref: string;
  source_snippet: string;
  created_at: string;
  updated_at: string;
  fsrs_json: string;
}

const iso = (value: string) => new Date(value).toISOString();
const rowToCard = ({ fsrs_json, ...rest }: CardRow): LocalCard => ({ ...rest, fsrs: JSON.parse(fsrs_json) as FsrsState });
const chunks = <T>(items: T[], size: number): T[][] => Array.from({ length: Math.ceil(items.length / size) }, (_, i) => items.slice(i * size, i * size + size));

/**
 * Everything the phone knows locally: decks, approved cards with their FSRS schedule, and the reviews waiting to be
 * uploaded. All study actions run against this store, so studying never needs the network.
 */
export class LocalStore {
  constructor(
    private readonly db: Db,
    private readonly newId: () => string,
  ) {}

  // -- key/value ------------------------------------------------------------------------

  async getMeta(key: string): Promise<string | null> {
    return (await this.db.first<{ value: string }>('SELECT value FROM meta WHERE key = ?', [key]))?.value ?? null;
  }

  async setMeta(key: string, value: string): Promise<void> {
    await this.db.run('INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value', [key, value]);
  }

  async deviceId(): Promise<string> {
    const existing = await this.getMeta('device_id');
    if (existing) return existing;
    const id = this.newId();
    await this.setMeta('device_id', id);
    return id;
  }

  async cursor(): Promise<number> {
    return Number((await this.getMeta('cursor')) ?? 0);
  }

  /** Identity of the server database this data came from; null before the first sync. */
  epoch(): Promise<string | null> {
    return this.getMeta('epoch');
  }

  async newPerDay(): Promise<number> {
    const stored = await this.getMeta('new_per_day');
    const n = stored === null ? DEFAULT_NEW_PER_DAY : Number(stored);
    return Number.isFinite(n) && n >= 0 ? n : DEFAULT_NEW_PER_DAY;
  }

  setNewPerDay(n: number): Promise<void> {
    return this.setMeta('new_per_day', String(Math.max(0, Math.min(500, Math.round(n)))));
  }

  /** A saved list of ids; damaged or missing data reads as an empty list. */
  private async idList(key: string): Promise<string[]> {
    try {
      const parsed: unknown = JSON.parse((await this.getMeta(key)) ?? '[]');
      return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [];
    } catch {
      return [];
    }
  }

  /** The deck ids in the order the user arranged them on this phone (empty until they drag one). */
  deckOrder(): Promise<string[]> {
    return this.idList('deck_order');
  }

  setDeckOrder(ids: string[]): Promise<void> {
    return this.setMeta('deck_order', JSON.stringify(ids));
  }

  /** The folders the user has collapsed on Home (a preference of this phone; folders start open). */
  closedFolders(): Promise<string[]> {
    return this.idList('folders_closed');
  }

  setClosedFolders(ids: string[]): Promise<void> {
    return this.setMeta('folders_closed', JSON.stringify(ids));
  }

  /**
   * Ties the local data to one server. The first call records it; a different server means the cached decks and
   * cursor belong to someone else, so they are cleared. Returns what happened.
   */
  async bindServer(url: string): Promise<'new' | 'same' | 'reset'> {
    const current = await this.getMeta('server_url');
    if (current === url) return 'same';
    if (current !== null) await this.clearData();
    await this.setMeta('server_url', url);
    return current === null ? 'new' : 'reset';
  }

  /** Remove decks, cards, reviews and the sync cursor, keeping this device's identity and preferences. */
  async clearData(): Promise<void> {
    await this.db.transaction(async (tx) => {
      await tx.run('DELETE FROM review_logs');
      await tx.run('DELETE FROM cards');
      await tx.run('DELETE FROM decks');
      await tx.run('DELETE FROM folders');
      await tx.run("DELETE FROM meta WHERE key IN ('cursor', 'epoch', 'server_url')");
    });
  }

  // -- decks and counts ----------------------------------------------------------------

  /** New cards still allowed today across all decks (the daily budget is shared). */
  private async newRemaining(now: Date, newPerDay: number): Promise<number> {
    const row = await this.db.first<{ n: number }>('SELECT COUNT(*) AS n FROM review_logs WHERE state_before = 0 AND reviewed_at >= ?', [
      studyDayStart(now).toISOString(),
    ]);
    return Math.max(0, newPerDay - (row?.n ?? 0));
  }

  async listDecks(now: Date, newPerDay: number): Promise<DeckSummary[]> {
    const end = studyDayEnd(now).toISOString();
    const remaining = await this.newRemaining(now, newPerDay);
    const rows = await this.db.all<{ id: string; name: string; topic: string; folder_id: string | null; updated_at: string; total: number; new_total: number; learning: number; review: number }>(
      `SELECT d.id, d.name, d.topic, d.folder_id, d.updated_at,
              COUNT(c.id) AS total,
              COALESCE(SUM(c.state = 0), 0) AS new_total,
              COALESCE(SUM(c.state IN (1, 3) AND c.due <= ?), 0) AS learning,
              COALESCE(SUM(c.state = 2 AND c.due <= ?), 0) AS review
         FROM decks d LEFT JOIN cards c ON c.deck_id = d.id
        GROUP BY d.id ORDER BY d.updated_at DESC, d.name`,
      [end, end],
    );
    return rows.map(({ new_total, ...r }) => ({ ...r, new: Math.min(new_total, remaining) }));
  }

  /** The folders on this phone, A to Z, with how many decks and cards each holds. */
  async listFolders(): Promise<FolderSummary[]> {
    return this.db.all<FolderSummary>(
      `SELECT f.id, f.name, f.updated_at, COUNT(DISTINCT d.id) AS decks, COUNT(c.id) AS total
         FROM folders f LEFT JOIN decks d ON d.folder_id = f.id LEFT JOIN cards c ON c.deck_id = d.id
        GROUP BY f.id ORDER BY lower(f.name), f.id`,
    );
  }

  /** Counts for one deck, a whole folder, or every deck (null). The daily new-card limit is shared, so a folder is counted as a whole, not summed. */
  async counts(scope: StudyScope, now: Date, newPerDay: number): Promise<StudyCounts> {
    const end = studyDayEnd(now).toISOString();
    const { sql: scoped, args: extra } = scopeClause(scope);
    const row = await this.db.first<{ new_total: number; learning: number; review: number }>(
      `SELECT COALESCE(SUM(state = 0), 0) AS new_total,
              COALESCE(SUM(state IN (1, 3) AND due <= ?), 0) AS learning,
              COALESCE(SUM(state = 2 AND due <= ?), 0) AS review
         FROM cards WHERE 1 = 1${scoped}`,
      [end, end, ...extra],
    );
    const remaining = await this.newRemaining(now, newPerDay);
    return { new: Math.min(row?.new_total ?? 0, remaining), learning: row?.learning ?? 0, review: row?.review ?? 0 };
  }

  /**
   * The next card to study: learning cards that are due, then reviews due today, then new cards (within the daily
   * budget), and finally learning cards due in the next few minutes. Null means the session is finished.
   */
  async nextCard(scope: StudyScope, now: Date, newPerDay: number): Promise<{ card: LocalCard | null; counts: StudyCounts }> {
    const { sql: scoped, args: extra } = scopeClause(scope);
    const pick = async (where: string, args: SqlValue[], order: string) =>
      this.db.first<CardRow>(`SELECT * FROM cards WHERE ${where}${scoped} ORDER BY ${order} LIMIT 1`, [...args, ...extra]);

    const counts = await this.counts(scope, now, newPerDay);
    let row = await pick('state IN (1, 3) AND due <= ?', [now.toISOString()], 'due');
    row ??= await pick('state = 2 AND due <= ?', [studyDayEnd(now).toISOString()], 'due');
    if (!row && counts.new > 0) row = await pick('state = 0', [], 'created_at, id');
    row ??= await pick('state IN (1, 3) AND due <= ?', [new Date(now.getTime() + LEARN_AHEAD_MS).toISOString()], 'due');
    return { card: row ? rowToCard(row) : null, counts };
  }

  /** Earliest time any already-studied card comes due again, for the "all caught up" screen. */
  async nextDue(scope: StudyScope): Promise<string | null> {
    const { sql: scoped, args } = scopeClause(scope);
    const row = await this.db.first<{ due: string | null }>(`SELECT MIN(due) AS due FROM cards WHERE state != 0${scoped}`, args);
    return row?.due ?? null;
  }

  async getCard(id: string): Promise<LocalCard | null> {
    const row = await this.db.first<CardRow>('SELECT * FROM cards WHERE id = ?', [id]);
    return row ? rowToCard(row) : null;
  }

  // -- studying ------------------------------------------------------------------------

  /** Record an answer: update the card's schedule and append a review log, atomically. */
  async answer(cardId: string, rating: RatingValue, now: Date, elapsedMs: number): Promise<LocalCard> {
    return this.db.transaction(async (tx) => {
      const row = await tx.first<CardRow & { state: number }>('SELECT * FROM cards WHERE id = ?', [cardId]);
      if (!row) throw new Error('That card is no longer in this deck.');
      const before = JSON.parse(row.fsrs_json) as FsrsState;
      const after = reviewCard(before, rating, now);

      await tx.run('UPDATE cards SET state = ?, due = ?, last_review = ?, fsrs_json = ? WHERE id = ?', [
        after.state,
        after.due,
        after.last_review,
        JSON.stringify(after),
        cardId,
      ]);
      await tx.run(
        'INSERT INTO review_logs(id, card_id, rating, reviewed_at, elapsed_ms, state_before, state_after_json, synced) VALUES (?, ?, ?, ?, ?, ?, ?, 0)',
        [this.newId(), cardId, rating, now.toISOString(), Math.max(0, Math.round(elapsedMs)), before.state, JSON.stringify(after)],
      );
      return rowToCard({ ...row, fsrs_json: JSON.stringify(after) });
    });
  }

  // -- sync ----------------------------------------------------------------------------

  async pendingLogs(limit: number): Promise<SyncLogWire[]> {
    const rows = await this.db.all<{ id: string; card_id: string; rating: number; reviewed_at: string; elapsed_ms: number; state_before: number; state_after_json: string }>(
      'SELECT id, card_id, rating, reviewed_at, elapsed_ms, state_before, state_after_json FROM review_logs WHERE synced = 0 ORDER BY reviewed_at, rowid LIMIT ?',
      [limit],
    );
    return rows.map(({ state_after_json, ...r }) => ({ ...r, state_after: JSON.parse(state_after_json) as FsrsState }));
  }

  async pendingCount(): Promise<number> {
    return (await this.db.first<{ n: number }>('SELECT COUNT(*) AS n FROM review_logs WHERE synced = 0'))?.n ?? 0;
  }

  /**
   * Apply one sync response in a single transaction: merge decks, cards, schedules and deletions, mark the uploaded
   * logs as synced, and advance the cursor. Applying the same response twice changes nothing.
   */
  async applySync(res: SyncResponse, sentLogIds: string[]): Promise<void> {
    await this.db.transaction(async (tx) => {
      if (res.reset) {
        // The server is not the database this cache came from (restored backup, rebuilt volume). What the server now
        // sends is a full download, so drop the cache, including cards the server no longer has. Review history is
        // kept and re-opened for upload: whatever the server lost is re-sent, and what it still has is ignored as a
        // duplicate (reviews carry client-made ids).
        await tx.run('DELETE FROM cards');
        await tx.run('DELETE FROM decks');
        await tx.run('DELETE FROM folders');
        await tx.run('UPDATE review_logs SET synced = 0');
      }

      for (const f of res.folders ?? []) {
        await tx.run('INSERT INTO folders(id, name, updated_at) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at', [
          f.id,
          f.name,
          iso(f.updated_at),
        ]);
      }

      for (const d of res.decks) {
        await tx.run(
          'INSERT INTO decks(id, name, topic, folder_id, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, topic = excluded.topic, folder_id = excluded.folder_id, updated_at = excluded.updated_at',
          [d.id, d.name, d.topic, d.folder_id ?? null, iso(d.updated_at)],
        );
      }

      for (const c of res.cards) {
        if (c.status !== 'approved') {
          // Only approved cards are studied; one that was un-approved (or rejected) disappears from the device.
          await tx.run('DELETE FROM cards WHERE id = ?', [c.id]);
          continue;
        }
        const fresh = newState(new Date(c.created_at));
        await tx.run(
          `INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, source_snippet, created_at, updated_at, state, due, last_review, fsrs_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
           ON CONFLICT(id) DO UPDATE SET deck_id = excluded.deck_id, type = excluded.type, front = excluded.front, back = excluded.back,
             origin = excluded.origin, source_ref = excluded.source_ref, source_snippet = excluded.source_snippet, updated_at = excluded.updated_at`,
          [c.id, c.deck_id, c.type, c.front, c.back, c.origin, c.source_ref, c.source_snippet, iso(c.created_at), iso(c.updated_at), fresh.state, fresh.due, JSON.stringify(fresh)],
        );
      }

      for (const s of res.states) {
        const local = await tx.first<{ last_review: string | null }>('SELECT last_review FROM cards WHERE id = ?', [s.card_id]);
        if (!local) continue; // schedule for a card this device does not hold (not approved, or deleted)
        // The most recent review wins, so a stale server copy never overwrites a review made on this device.
        if (local.last_review !== null && new Date(local.last_review).getTime() >= new Date(s.last_review).getTime()) continue;
        const st = { ...s.state, due: iso(s.state.due), last_review: iso(s.last_review) };
        await tx.run('UPDATE cards SET state = ?, due = ?, last_review = ?, fsrs_json = ? WHERE id = ?', [st.state, st.due, st.last_review, JSON.stringify(st), s.card_id]);
      }

      for (const gone of res.deleted) {
        if (gone.kind === 'card') {
          await tx.run('DELETE FROM cards WHERE id = ?', [gone.id]);
        } else if (gone.kind === 'folder') {
          // The folder goes, its decks stay: take them out of it (the server also sends each deck's own change).
          await tx.run('UPDATE decks SET folder_id = NULL WHERE folder_id = ?', [gone.id]);
          await tx.run('DELETE FROM folders WHERE id = ?', [gone.id]);
        } else {
          await tx.run('DELETE FROM cards WHERE deck_id = ?', [gone.id]);
          await tx.run('DELETE FROM decks WHERE id = ?', [gone.id]);
        }
      }

      for (const ids of chunks(sentLogIds, 200)) {
        await tx.run(`UPDATE review_logs SET synced = 1 WHERE id IN (${ids.map(() => '?').join(',')})`, ids);
      }
      await tx.run("INSERT INTO meta(key, value) VALUES ('cursor', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", [String(res.cursor)]);
      await tx.run("INSERT INTO meta(key, value) VALUES ('epoch', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", [res.epoch]);
    });
  }
}
