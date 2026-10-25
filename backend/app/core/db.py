"""SQLite access. A fresh short-lived connection per operation keeps this safe across the request threadpool
and the job worker threads; WAL mode lets readers and the single writer overlap."""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,
    stored_names_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- status: queued -> researching -> awaiting_approval -> approved -> generating -> completed | failed
-- ("approved" means: sources approved, waiting for the generate phase)
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    stage TEXT,
    phase TEXT,
    error TEXT,
    auto_approve INTEGER NOT NULL DEFAULT 0,
    understanding_json TEXT,
    usage_json TEXT NOT NULL DEFAULT '[]',
    cost_usd REAL NOT NULL DEFAULT 0,
    deck_id TEXT REFERENCES decks(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    claim TEXT NOT NULL,
    snippet TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_evidence_job ON evidence(job_id);

-- status: draft | approved | rejected
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    deck_id TEXT NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    type TEXT NOT NULL,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    origin TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_snippet TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    status TEXT NOT NULL,
    atomic INTEGER NOT NULL,
    grounded INTEGER NOT NULL,
    issue TEXT NOT NULL DEFAULT '',
    duplicate_of TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_id, status);
CREATE INDEX IF NOT EXISTS idx_cards_job ON cards(job_id);
CREATE INDEX IF NOT EXISTS idx_cards_updated ON cards(updated_at);
"""


def _seq_triggers(table: str) -> list[str]:
    """Stamp every insert/update of `table` with the next value of the global change counter.

    The update trigger only fires when `seq` itself was not changed by the statement, which is also what stops
    its own `UPDATE ... SET seq` from re-triggering it."""
    bump = "UPDATE sync_counter SET seq = seq + 1 WHERE id = 1"
    stamp = f"UPDATE {table} SET seq = (SELECT seq FROM sync_counter WHERE id = 1) WHERE rowid = NEW.rowid"
    return [
        f"CREATE TRIGGER {table}_seq_ins AFTER INSERT ON {table} BEGIN {bump}; {stamp}; END",
        f"CREATE TRIGGER {table}_seq_upd AFTER UPDATE ON {table} WHEN NEW.seq = OLD.seq BEGIN {bump}; {stamp}; END",
    ]


def _tombstone_trigger(table: str, kind: str) -> str:
    return (
        f"CREATE TRIGGER {table}_tombstone AFTER DELETE ON {table} BEGIN "
        "UPDATE sync_counter SET seq = seq + 1 WHERE id = 1; "
        f"INSERT INTO tombstones(seq, kind, id) VALUES ((SELECT seq FROM sync_counter WHERE id = 1), '{kind}', OLD.id); END"
    )


# Schema version 2: everything a client needs to sync incrementally.
#  - sync_counter: one global, strictly increasing change counter
#  - decks/cards/card_state carry the counter value of their last change (`seq`), maintained by triggers
#  - tombstones record deletions (including cascades) so clients can remove what no longer exists
#  - review_logs: append-only reviews pushed by devices; card_state: the latest FSRS state per card
SYNC_MIGRATION_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS sync_counter (id INTEGER PRIMARY KEY CHECK (id = 1), seq INTEGER NOT NULL)",
    "INSERT OR IGNORE INTO sync_counter(id, seq) VALUES (1, 0)",
    "ALTER TABLE decks ADD COLUMN seq INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE cards ADD COLUMN seq INTEGER NOT NULL DEFAULT 0",
    "CREATE INDEX IF NOT EXISTS idx_decks_seq ON decks(seq)",
    "CREATE INDEX IF NOT EXISTS idx_cards_seq ON cards(seq)",
    """CREATE TABLE IF NOT EXISTS tombstones (
        seq INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        id TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_logs (
        id TEXT PRIMARY KEY,
        card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
        rating INTEGER NOT NULL,
        reviewed_at TEXT NOT NULL,
        elapsed_ms INTEGER NOT NULL,
        state_before INTEGER NOT NULL,
        state_after_json TEXT NOT NULL,
        device_id TEXT NOT NULL,
        received_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_review_logs_card ON review_logs(card_id, reviewed_at)",
    """CREATE TABLE IF NOT EXISTS card_state (
        card_id TEXT PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
        state_json TEXT NOT NULL,
        due TEXT NOT NULL,
        last_review TEXT NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_card_state_seq ON card_state(seq)",
]
SYNC_TRIGGER_STATEMENTS = [
    *_seq_triggers("decks"),
    *_seq_triggers("cards"),
    *_seq_triggers("card_state"),
    _tombstone_trigger("decks", "deck"),
    _tombstone_trigger("cards", "card"),
]
SCHEMA_VERSION = 5

# Schema version 5: folders that group decks. A folder syncs like a deck (a change counter stamp, a tombstone on delete), and a deck
# names its folder in `folder_id` (NULL: not in a folder). Written so running it twice is harmless.
FOLDER_MIGRATION_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS folders (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_folders_seq ON folders(seq)",
]
FOLDER_TRIGGER_STATEMENTS = [
    stmt.replace("CREATE TRIGGER", "CREATE TRIGGER IF NOT EXISTS", 1)
    for stmt in (*_seq_triggers("folders"), _tombstone_trigger("folders", "folder"))
]


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None -> autocommit; transactions are explicit in tx().
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            self._migrate(conn)
        finally:
            conn.close()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Bring databases created by earlier versions up to SCHEMA_VERSION, in one transaction."""
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            if version < 2:
                for statement in SYNC_MIGRATION_STATEMENTS:
                    conn.execute(statement)
                # Rows that predate sync get distinct counter values (oldest first) before the triggers exist.
                seq = 0
                for table in ("decks", "cards"):
                    for (rowid,) in conn.execute(f"SELECT rowid FROM {table} ORDER BY rowid").fetchall():
                        seq += 1
                        conn.execute(f"UPDATE {table} SET seq = ? WHERE rowid = ?", (seq, rowid))
                conn.execute("UPDATE sync_counter SET seq = ? WHERE id = 1", (seq,))
                for statement in SYNC_TRIGGER_STATEMENTS:
                    conn.execute(statement)
            if version < 3:
                # Version 3: the database's identity, so devices can tell when it is not the one they synced with.
                conn.execute("CREATE TABLE IF NOT EXISTS sync_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute("INSERT OR IGNORE INTO sync_meta(key, value) VALUES ('epoch', ?)", (uuid.uuid4().hex,))
            if version < 4:
                # Version 4: a job can be told to add its cards to an existing deck instead of making a new one.
                if "target_deck_id" not in {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}:
                    conn.execute("ALTER TABLE jobs ADD COLUMN target_deck_id TEXT REFERENCES decks(id) ON DELETE SET NULL")
            if version < 5:
                for statement in FOLDER_MIGRATION_STATEMENTS:
                    conn.execute(statement)
                if "folder_id" not in {row[1] for row in conn.execute("PRAGMA table_info(decks)")}:
                    conn.execute("ALTER TABLE decks ADD COLUMN folder_id TEXT REFERENCES folders(id) ON DELETE SET NULL")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_decks_folder ON decks(folder_id)")
                for statement in FOLDER_TRIGGER_STATEMENTS:
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        """Autocommit connection for single statements and reads."""
        c = self._connect()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Write transaction that takes the write lock up front (no mid-transaction upgrade deadlocks)."""
        c = self._connect()
        try:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            else:
                c.execute("COMMIT")
        finally:
            c.close()
