import { openDatabaseAsync, type SQLiteDatabase } from 'expo-sqlite';

import { migrate } from './migrations';
import { serialize } from './serial';
import type { Db, RawDb } from './types';

function expoRaw(db: SQLiteDatabase): RawDb {
  return {
    exec: (sql) => db.execAsync(sql),
    async run(sql, params = []) {
      const result = await db.runAsync(sql, params);
      return { changes: result.changes };
    },
    all: (sql, params = []) => db.getAllAsync(sql, params),
  };
}

/** Opens (and migrates) the on-device database. Native implementation; the web preview has its own in open.web.ts. */
export async function openLocalDb(name = 'snapnotes.db'): Promise<Db> {
  const sqlite = await openDatabaseAsync(name);
  await sqlite.execAsync('PRAGMA journal_mode = WAL');
  const db = serialize(expoRaw(sqlite));
  await migrate(db);
  return db;
}
