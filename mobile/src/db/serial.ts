import type { Db, RawDb, SqlValue } from './types';

function withFirst(raw: RawDb): Db {
  return {
    exec: (sql) => raw.exec(sql),
    run: (sql, params) => raw.run(sql, params),
    all: (sql, params) => raw.all(sql, params),
    async first<T>(sql: string, params?: SqlValue[]) {
      const rows = await raw.all<T>(sql, params);
      return rows[0] ?? null;
    },
    transaction() {
      throw new Error('Transactions cannot be nested');
    },
  };
}

/**
 * Wraps a driver so that all operations run one at a time, and `transaction()` holds the line for its whole
 * duration. SQLite connections have no isolation between callers sharing them, so without this a read issued
 * from another part of the app could land in the middle of someone else's transaction.
 */
export function serialize(raw: RawDb): Db {
  let tail: Promise<unknown> = Promise.resolve();
  const enqueue = <T>(task: () => Promise<T>): Promise<T> => {
    const result = tail.then(task, task);
    tail = result.catch(() => undefined);
    return result;
  };

  const inner = withFirst(raw);
  return {
    exec: (sql) => enqueue(() => raw.exec(sql)),
    run: (sql, params) => enqueue(() => raw.run(sql, params)),
    all: <T>(sql: string, params?: SqlValue[]) => enqueue(() => raw.all<T>(sql, params)),
    first: <T>(sql: string, params?: SqlValue[]) => enqueue(() => inner.first<T>(sql, params)),
    transaction: <T>(fn: (tx: Db) => Promise<T>) =>
      enqueue(async () => {
        await raw.exec('BEGIN IMMEDIATE');
        try {
          const out = await fn(inner);
          await raw.exec('COMMIT');
          return out;
        } catch (e) {
          await raw.exec('ROLLBACK');
          throw e;
        }
      }),
  };
}
