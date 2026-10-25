import { MIGRATIONS } from '@/db/migrations';

import { openTestDb } from './helpers/nodeDb';

describe('local database', () => {
  it('migrates a fresh database to the latest version', async () => {
    const db = await openTestDb();
    const v = await db.first<{ user_version: number }>('PRAGMA user_version');
    expect(v?.user_version).toBe(MIGRATIONS.length);
    const tables = await db.all<{ name: string }>("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name");
    expect(tables.map((t) => t.name)).toEqual(['card_flags', 'cards', 'decks', 'folders', 'meta', 'review_logs']);
  });

  it('rolls a transaction back when it throws', async () => {
    const db = await openTestDb();
    await expect(
      db.transaction(async (tx) => {
        await tx.run("INSERT INTO meta(key, value) VALUES ('a', '1')");
        throw new Error('boom');
      }),
    ).rejects.toThrow('boom');
    expect(await db.first('SELECT * FROM meta')).toBeNull();
  });

  it('commits a transaction that succeeds and returns its result', async () => {
    const db = await openTestDb();
    const out = await db.transaction(async (tx) => {
      await tx.run("INSERT INTO meta(key, value) VALUES ('a', '1')");
      return tx.first<{ value: string }>("SELECT value FROM meta WHERE key = 'a'");
    });
    expect(out).toEqual({ value: '1' });
    expect(await db.all('SELECT * FROM meta')).toHaveLength(1);
  });

  it('keeps other operations out of a running transaction', async () => {
    const db = await openTestDb();
    const order: string[] = [];
    const tx = db.transaction(async (t) => {
      await t.run("INSERT INTO meta(key, value) VALUES ('inside', '1')");
      await new Promise((r) => setTimeout(r, 20)); // a slow step in the middle
      order.push('tx-done');
    });
    // Issued while the transaction is open: must wait for it, so it sees the committed row.
    const read = db.all('SELECT key FROM meta').then((rows) => {
      order.push(`read-saw-${rows.length}`);
    });
    await Promise.all([tx, read]);
    expect(order).toEqual(['tx-done', 'read-saw-1']);
  });

  it('refuses nested transactions instead of deadlocking', async () => {
    const db = await openTestDb();
    await expect(db.transaction(async (tx) => tx.transaction(async () => undefined))).rejects.toThrow(/nested/);
  });

  it('keeps working after a failed operation', async () => {
    const db = await openTestDb();
    await expect(db.run('INSERT INTO nope VALUES (1)')).rejects.toThrow();
    await db.run("INSERT INTO meta(key, value) VALUES ('ok', '1')");
    expect(await db.all('SELECT * FROM meta')).toHaveLength(1);
  });
});
