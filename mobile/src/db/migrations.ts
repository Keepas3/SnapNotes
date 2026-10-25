import type { Db } from './types';

/**
 * Schema versions, applied in order and recorded in `PRAGMA user_version`. Only add to the end: devices in the
 * field are on older versions and must be able to step forward.
 *
 * The device keeps only what it needs to study offline: every deck, and the *approved* cards with their FSRS state.
 * Review logs are the device's own append-only history; `synced` marks the ones the server has acknowledged.
 */
export const MIGRATIONS: string[] = [
  `
  CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

  CREATE TABLE decks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
  );

  CREATE TABLE cards (
    id TEXT PRIMARY KEY,
    deck_id TEXT NOT NULL,
    type TEXT NOT NULL,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    origin TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_snippet TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- FSRS schedule. state/due/last_review are copied out of fsrs_json so they can be queried and indexed.
    state INTEGER NOT NULL DEFAULT 0,
    due TEXT NOT NULL,
    last_review TEXT,
    fsrs_json TEXT NOT NULL
  );
  CREATE INDEX idx_cards_deck ON cards(deck_id);
  CREATE INDEX idx_cards_due ON cards(state, due);

  CREATE TABLE review_logs (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    reviewed_at TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    state_before INTEGER NOT NULL,
    state_after_json TEXT NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
  );
  CREATE INDEX idx_logs_pending ON review_logs(synced, reviewed_at);
  CREATE INDEX idx_logs_reviewed ON review_logs(reviewed_at);
  `,
  // Version 2: folders that group decks. A deck names its folder in folder_id (NULL: not in one).
  `
  CREATE TABLE folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
  ALTER TABLE decks ADD COLUMN folder_id TEXT;
  CREATE INDEX idx_decks_folder ON decks(folder_id);
  `,
  // Version 3: the wrong answers of a card, for multiple choice (a JSON list of strings; the right answer is the card's back).
  `
  ALTER TABLE cards ADD COLUMN choices_json TEXT NOT NULL DEFAULT '[]';
  `,
  // Version 4: cards the user has paused. Kept on this phone only, apart from the cards so a re-download of them leaves it alone.
  `
  CREATE TABLE card_flags (
    card_id TEXT PRIMARY KEY,
    paused INTEGER NOT NULL DEFAULT 0
  );
  `,
  // Version 5: the schedule a card had before each answer, so the answer can be undone. Older rows stay NULL and cannot be.
  `
  ALTER TABLE review_logs ADD COLUMN fsrs_before_json TEXT;
  `,
];

export async function migrate(db: Db): Promise<void> {
  const row = await db.first<{ user_version: number }>('PRAGMA user_version');
  const current = row?.user_version ?? 0;
  for (let version = current; version < MIGRATIONS.length; version++) {
    await db.transaction(async (tx) => {
      await tx.exec(MIGRATIONS[version]);
      await tx.exec(`PRAGMA user_version = ${version + 1}`);
    });
  }
}
