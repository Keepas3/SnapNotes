import initSqlJs from 'sql.js/dist/sql-asm.js';

import { migrate } from './migrations';
import { serialize } from './serial';
import type { Db, RawDb, SqlValue } from './types';

/**
 * Web preview only. expo-sqlite's web build needs a worker and cross-origin isolation, so the browser preview
 * uses sql.js (SQLite compiled to plain JavaScript) and keeps a copy of the database in localStorage.
 * The app itself ships as a native app, where open.ts and expo-sqlite are used instead.
 */
const STORAGE_KEY = 'snapnotes.localdb';

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function fromBase64(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function openLocalDb(name = STORAGE_KEY): Promise<Db> {
  const SQL = await initSqlJs();
  let saved: Uint8Array | undefined;
  try {
    const text = globalThis.localStorage?.getItem(name);
    if (text) saved = fromBase64(text);
  } catch {
    /* unreadable copy: start empty */
  }
  const sqlite = new SQL.Database(saved);

  let timer: ReturnType<typeof setTimeout> | undefined;
  const persistSoon = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      try {
        globalThis.localStorage?.setItem(name, toBase64(sqlite.export()));
      } catch {
        /* storage full or unavailable: the preview just loses persistence */
      }
    }, 250);
  };

  const raw: RawDb = {
    async exec(sql) {
      sqlite.exec(sql);
      persistSoon();
    },
    async run(sql, params = []) {
      sqlite.run(sql, params as SqlValue[]);
      persistSoon();
      return { changes: sqlite.getRowsModified() };
    },
    async all<T>(sql: string, params: SqlValue[] = []) {
      const stmt = sqlite.prepare(sql);
      try {
        stmt.bind(params);
        const rows: T[] = [];
        while (stmt.step()) rows.push(stmt.getAsObject() as T);
        return rows;
      } finally {
        stmt.free();
      }
    },
  };

  const db = serialize(raw);
  await migrate(db);
  return db;
}
