import type { CardType, FsrsState, Origin, SyncLogWire, SyncResponse } from '@/api/types';
import type { Db, SqlValue } from '@/db/types';

import { studyDayEnd, studyDayStart } from './day';
import { DEFAULT_RETENTION, mostForgotten, newState, RETENTION_CHOICES, reviewCard, type RatingValue, type SchedulingOptions } from './scheduler';

/** Cards still in (re)learning that are due within this long are offered early rather than ending the session. */
export const LEARN_AHEAD_MS = 20 * 60_000;
export const DEFAULT_NEW_PER_DAY = 10;
/** A card missed this many times is a problem card (Anki's default "leech" threshold). */
export const PROBLEM_LAPSES = 8;
/** How many of the most overdue reviews are weighed when picking the one most worth showing next; far more than a day's study. */
const REVIEW_CANDIDATES = 300;

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
  /** Wrong answers for multiple choice; the right answer is `back`. */
  choices: string[];
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
  /** cards the user has paused (left out of the counts above) */
  paused: number;
  /** cards, not paused, that have been missed PROBLEM_LAPSES times or more */
  problem: number;
}

/** A folder with how much it holds; what is due in it comes from `counts` with the folder as the scope. */
export interface FolderSummary {
  id: string;
  name: string;
  updated_at: string;
  decks: number;
  total: number;
  /** cards in it that are paused */
  paused: number;
}

/** How multiple choice is set up on this phone (a preference of this phone, not synced). */
export interface ChoicePrefs {
  enabled: boolean;
  count: 4 | 5;
  fillFromDeck: boolean;
}

/** What one card shows in a deck's list: how often it has been missed and whether it is paused. */
export interface CardStat {
  id: string;
  lapses: number;
  paused: boolean;
}

/** What to study or count: one deck (its id), every deck in a folder, or everything (null). */
export type StudyScope = string | { folderId: string } | null;

/** Leaves paused cards out of a query on `cards` (used unqualified, so not in the deck list's join). */
const NOT_PAUSED = ' AND id NOT IN (SELECT card_id FROM card_flags WHERE paused = 1)';

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
  choices_json: string;
}

/** A saved list of wrong answers; anything damaged reads as none. */
function parseChoices(json: string | null | undefined): string[] {
  try {
    const parsed: unknown = JSON.parse(json ?? '[]');
    return Array.isArray(parsed) ? parsed.filter((c): c is string => typeof c === 'string') : [];
  } catch {
    return [];
  }
}

const iso = (value: string) => new Date(value).toISOString();
const rowToCard = ({ fsrs_json, choices_json, ...rest }: CardRow): LocalCard => ({ ...rest, choices: parseChoices(choices_json), fsrs: JSON.parse(fsrs_json) as FsrsState });
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

  /** How multiple choice is set up on this phone: switched on or off, 4 or 5 options, and whether gaps are filled from other cards. */
  async choicePrefs(): Promise<ChoicePrefs> {
    const [on, count, fill] = await Promise.all([this.getMeta('mc_on'), this.getMeta('mc_count'), this.getMeta('mc_fill')]);
    return { enabled: on === '1', count: count === '5' ? 5 : 4, fillFromDeck: fill !== '0' };
  }

  async setChoicePrefs(change: Partial<ChoicePrefs>): Promise<void> {
    if (change.enabled !== undefined) await this.setMeta('mc_on', change.enabled ? '1' : '0');
    if (change.count !== undefined) await this.setMeta('mc_count', change.count === 5 ? '5' : '4');
    if (change.fillFromDeck !== undefined) await this.setMeta('mc_fill', change.fillFromDeck ? '1' : '0');
  }

  /** How cards are scheduled on this phone: the chance of remembering to aim for (default 90%) and whether to spread longer intervals (default on). */
  async schedulingPrefs(): Promise<SchedulingOptions> {
    const [retention, spread] = await Promise.all([this.getMeta('retention'), this.getMeta('spread')]);
    const wanted = Number(retention);
    return { retention: (RETENTION_CHOICES as readonly number[]).includes(wanted) ? wanted : DEFAULT_RETENTION, spread: spread !== '0' };
  }

  async setSchedulingPrefs(change: Partial<SchedulingOptions>): Promise<void> {
    if (change.retention !== undefined && (RETENTION_CHOICES as readonly number[]).includes(change.retention)) await this.setMeta('retention', String(change.retention));
    if (change.spread !== undefined) await this.setMeta('spread', change.spread ? '1' : '0');
  }

  // -- pausing cards ---------------------------------------------------------------------

  /** Stop offering a card (it keeps its schedule and history). Kept on this phone only. */
  async pauseCard(cardId: string): Promise<void> {
    await this.db.run('INSERT INTO card_flags(card_id, paused) VALUES (?, 1) ON CONFLICT(card_id) DO UPDATE SET paused = 1', [cardId]);
  }

  async resumeCard(cardId: string): Promise<void> {
    await this.db.run('DELETE FROM card_flags WHERE card_id = ?', [cardId]);
  }

  /** Missed-count and paused state of each card in a deck, for its card list. */
  async cardStats(deckId: string): Promise<CardStat[]> {
    const rows = await this.db.all<{ id: string; lapses: number | null; paused: number | null }>(
      `SELECT c.id, json_extract(c.fsrs_json, '$.lapses') AS lapses, f.paused AS paused
         FROM cards c LEFT JOIN card_flags f ON f.card_id = c.id WHERE c.deck_id = ?`,
      [deckId],
    );
    return rows.map((r) => ({ id: r.id, lapses: r.lapses ?? 0, paused: r.paused === 1 }));
  }

  /** The answers of a deck's other studyable cards, to borrow as wrong answers (questions and definitions only). */
  otherAnswers(deckId: string, exceptCardId: string): Promise<{ id: string; back: string }[]> {
    return this.db.all<{ id: string; back: string }>("SELECT id, back FROM cards WHERE deck_id = ? AND id != ? AND type IN ('basic', 'definition') AND back != ''", [deckId, exceptCardId]);
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
      await tx.run('DELETE FROM card_flags');
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
    const rows = await this.db.all<{ id: string; name: string; topic: string; folder_id: string | null; updated_at: string; total: number; paused: number; problem: number; new_total: number; learning: number; review: number }>(
      `SELECT d.id, d.name, d.topic, d.folder_id, d.updated_at,
              COUNT(c.id) AS total,
              COUNT(p.card_id) AS paused,
              COALESCE(SUM(p.card_id IS NULL AND json_extract(c.fsrs_json, '$.lapses') >= ?), 0) AS problem,
              COALESCE(SUM(p.card_id IS NULL AND c.state = 0), 0) AS new_total,
              COALESCE(SUM(p.card_id IS NULL AND c.state IN (1, 3) AND c.due <= ?), 0) AS learning,
              COALESCE(SUM(p.card_id IS NULL AND c.state = 2 AND c.due <= ?), 0) AS review
         FROM decks d LEFT JOIN cards c ON c.deck_id = d.id LEFT JOIN card_flags p ON p.card_id = c.id AND p.paused = 1
        GROUP BY d.id ORDER BY d.updated_at DESC, d.name`,
      [PROBLEM_LAPSES, end, end],
    );
    return rows.map(({ new_total, ...r }) => ({ ...r, new: Math.min(new_total, remaining) }));
  }

  /** The folders on this phone, A to Z, with how many decks and cards each holds. */
  async listFolders(): Promise<FolderSummary[]> {
    return this.db.all<FolderSummary>(
      `SELECT f.id, f.name, f.updated_at, COUNT(DISTINCT d.id) AS decks, COUNT(c.id) AS total, COUNT(p.card_id) AS paused
         FROM folders f LEFT JOIN decks d ON d.folder_id = f.id LEFT JOIN cards c ON c.deck_id = d.id LEFT JOIN card_flags p ON p.card_id = c.id AND p.paused = 1
        GROUP BY f.id ORDER BY lower(f.name), f.id`,
    );
  }

  /** The end of the study day, moved on by `aheadDays` when studying cards before they are due. */
  private reviewHorizon(now: Date, aheadDays: number): string {
    const end = studyDayEnd(now);
    end.setDate(end.getDate() + Math.max(0, Math.round(aheadDays)));
    return end.toISOString();
  }

  /**
   * Counts for one deck, a whole folder, or every deck (null). The daily new-card limit is shared, so a folder is counted as a whole, not summed.
   * Paused cards are left out. With `aheadDays`, reviews due that many days after today are counted as well.
   */
  async counts(scope: StudyScope, now: Date, newPerDay: number, aheadDays = 0): Promise<StudyCounts> {
    const end = studyDayEnd(now).toISOString();
    const { sql: scoped, args: extra } = scopeClause(scope);
    const row = await this.db.first<{ new_total: number; learning: number; review: number }>(
      `SELECT COALESCE(SUM(state = 0), 0) AS new_total,
              COALESCE(SUM(state IN (1, 3) AND due <= ?), 0) AS learning,
              COALESCE(SUM(state = 2 AND due <= ?), 0) AS review
         FROM cards WHERE 1 = 1${scoped}${NOT_PAUSED}`,
      [end, this.reviewHorizon(now, aheadDays), ...extra],
    );
    const remaining = await this.newRemaining(now, newPerDay);
    return { new: Math.min(row?.new_total ?? 0, remaining), learning: row?.learning ?? 0, review: row?.review ?? 0 };
  }

  /**
   * The next card to study: learning cards that are due, then reviews due today, then new cards (within the daily
   * budget), and finally learning cards due in the next few minutes. Null means the session is finished.
   * Among reviews the one least likely to be remembered right now comes first. With `aheadDays`, once today's reviews
   * are done the reviews due within that many days are offered too, the same way.
   */
  async nextCard(scope: StudyScope, now: Date, newPerDay: number, aheadDays = 0): Promise<{ card: LocalCard | null; counts: StudyCounts }> {
    const { sql: scoped, args: extra } = scopeClause(scope);
    const pick = async (where: string, args: SqlValue[], order: string) =>
      this.db.first<CardRow>(`SELECT * FROM cards WHERE ${where}${scoped}${NOT_PAUSED} ORDER BY ${order} LIMIT 1`, [...args, ...extra]);
    const forgotten = async (until: string): Promise<CardRow | null> => {
      const rows = await this.db.all<CardRow>(`SELECT * FROM cards WHERE state = 2 AND due <= ?${scoped}${NOT_PAUSED} ORDER BY due LIMIT ${REVIEW_CANDIDATES}`, [until, ...extra]);
      const best = mostForgotten(rows.map(rowToCard), now);
      return best >= 0 ? rows[best] : null;
    };

    const counts = await this.counts(scope, now, newPerDay, aheadDays);
    let row = await pick('state IN (1, 3) AND due <= ?', [now.toISOString()], 'due');
    row ??= await forgotten(studyDayEnd(now).toISOString());
    if (!row && counts.new > 0) row = await pick('state = 0', [], 'created_at, id');
    row ??= await pick('state IN (1, 3) AND due <= ?', [new Date(now.getTime() + LEARN_AHEAD_MS).toISOString()], 'due');
    if (!row && aheadDays > 0) row = await forgotten(this.reviewHorizon(now, aheadDays));
    return { card: row ? rowToCard(row) : null, counts };
  }

  /** Every card in the scope that is not paused, in the order they were made, for practice that ignores what is due. */
  async allCardIds(scope: StudyScope): Promise<string[]> {
    const { sql: scoped, args } = scopeClause(scope);
    const rows = await this.db.all<{ id: string }>(`SELECT id FROM cards WHERE 1 = 1${scoped}${NOT_PAUSED} ORDER BY deck_id, created_at, id`, args);
    return rows.map((r) => r.id);
  }

  /** Earliest time any already-studied card comes due again, for the "all caught up" screen. */
  async nextDue(scope: StudyScope): Promise<string | null> {
    const { sql: scoped, args } = scopeClause(scope);
    const row = await this.db.first<{ due: string | null }>(`SELECT MIN(due) AS due FROM cards WHERE state != 0${scoped}${NOT_PAUSED}`, args);
    return row?.due ?? null;
  }

  async getCard(id: string): Promise<LocalCard | null> {
    const row = await this.db.first<CardRow>('SELECT * FROM cards WHERE id = ?', [id]);
    return row ? rowToCard(row) : null;
  }

  // -- studying ------------------------------------------------------------------------

  /** Record an answer: update the card's schedule and append a review log, atomically. */
  async answer(cardId: string, rating: RatingValue, now: Date, elapsedMs: number): Promise<LocalCard> {
    const options = await this.schedulingPrefs();
    return this.db.transaction(async (tx) => {
      const row = await tx.first<CardRow & { state: number }>('SELECT * FROM cards WHERE id = ?', [cardId]);
      if (!row) throw new Error('That card is no longer in this deck.');
      const before = JSON.parse(row.fsrs_json) as FsrsState;
      const after = reviewCard(before, rating, now, options, cardId);

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
          await tx.run('DELETE FROM card_flags WHERE card_id = ?', [c.id]);
          continue;
        }
        const fresh = newState(new Date(c.created_at));
        await tx.run(
          `INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, source_snippet, created_at, updated_at, state, due, last_review, fsrs_json, choices_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
           ON CONFLICT(id) DO UPDATE SET deck_id = excluded.deck_id, type = excluded.type, front = excluded.front, back = excluded.back,
             origin = excluded.origin, source_ref = excluded.source_ref, source_snippet = excluded.source_snippet, updated_at = excluded.updated_at,
             choices_json = excluded.choices_json`,
          [c.id, c.deck_id, c.type, c.front, c.back, c.origin, c.source_ref, c.source_snippet, iso(c.created_at), iso(c.updated_at), fresh.state, fresh.due, JSON.stringify(fresh), JSON.stringify(c.choices ?? [])],
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
          await tx.run('DELETE FROM card_flags WHERE card_id = ?', [gone.id]);
        } else if (gone.kind === 'folder') {
          // The folder goes, its decks stay: take them out of it (the server also sends each deck's own change).
          await tx.run('UPDATE decks SET folder_id = NULL WHERE folder_id = ?', [gone.id]);
          await tx.run('DELETE FROM folders WHERE id = ?', [gone.id]);
        } else {
          await tx.run('DELETE FROM card_flags WHERE card_id IN (SELECT id FROM cards WHERE deck_id = ?)', [gone.id]);
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
