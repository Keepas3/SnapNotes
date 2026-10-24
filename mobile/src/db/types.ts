export type SqlValue = string | number | null;

/** The few operations the app needs from a SQLite driver. Implemented for expo-sqlite (native), sql.js (web preview) and node:sqlite (tests). */
export interface RawDb {
  /** Run one or more statements without parameters. */
  exec(sql: string): Promise<void>;
  run(sql: string, params?: SqlValue[]): Promise<{ changes: number }>;
  all<T = Record<string, SqlValue>>(sql: string, params?: SqlValue[]): Promise<T[]>;
}

export interface Db extends RawDb {
  first<T = Record<string, SqlValue>>(sql: string, params?: SqlValue[]): Promise<T | null>;
  /**
   * Run `fn` atomically. Every other operation on this database waits until it finishes, so a study answer and a
   * sync can never interleave half-written state.
   */
  transaction<T>(fn: (tx: Db) => Promise<T>): Promise<T>;
}
