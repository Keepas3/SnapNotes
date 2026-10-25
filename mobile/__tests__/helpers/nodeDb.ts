import { migrate } from '@/db/migrations';
import { serialize } from '@/db/serial';
import type { Db, RawDb, SqlValue } from '@/db/types';

// Just the part of Node's built-in SQLite driver used here (the project does not pull in @types/node).
interface NodeStatement {
  run(...params: SqlValue[]): { changes: number | bigint };
  all(...params: SqlValue[]): unknown[];
}
interface NodeDatabase {
  exec(sql: string): void;
  prepare(sql: string): NodeStatement;
}

/**
 * A real SQLite database for tests, using the driver built into Node. It runs the same SQL and migrations as the
 * app does on a phone, so the store and sync code is tested against real SQL, not a mock.
 */
export async function openTestDb(): Promise<Db> {
  const db = await openUnmigratedTestDb();
  await migrate(db);
  return db;
}

/** The same database before any migration has run, for tests that build an older version by hand and step it forward. */
export async function openUnmigratedTestDb(): Promise<Db> {
  // `process.getBuiltinModule` avoids Jest's module resolver, which does not know the `node:sqlite` scheme.
  const { DatabaseSync } = (process as unknown as { getBuiltinModule(id: string): { DatabaseSync: new (path: string) => NodeDatabase } }).getBuiltinModule('node:sqlite');
  const sqlite = new DatabaseSync(':memory:');
  const raw: RawDb = {
    async exec(sql) {
      sqlite.exec(sql);
    },
    async run(sql, params = []) {
      const result = sqlite.prepare(sql).run(...params);
      return { changes: Number(result.changes) };
    },
    async all<T>(sql: string, params: SqlValue[] = []) {
      return sqlite.prepare(sql).all(...params) as T[];
    },
  };
  return serialize(raw);
}
