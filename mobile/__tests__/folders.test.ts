import { MIGRATIONS, migrate } from '@/db/migrations';
import {
  dropTarget,
  folderNameProblem,
  folderSubtitle,
  homeRows,
  MAX_FOLDER_NAME,
  parseStudyScope,
  reorderWithin,
  siblingIndexes,
  studyParam,
  type HomeRow,
} from '@/lib/folders';

import { card, deckA, makeStore, response, seedCards } from './helpers/fixtures';
import { openUnmigratedTestDb } from './helpers/nodeDb';

const T = '2026-03-01T08:00:00.000000+00:00';
const NOW = new Date(2026, 2, 10, 15, 0, 0);
const folder = (id: string, name = id) => ({ id, name, updated_at: T });
const deckIn = (id: string, folder_id: string | null, name = id) => ({ id, name, topic: '', folder_id, updated_at: T });

describe('folder names and study addresses', () => {
  it('checks a folder name the way the server does', () => {
    expect(folderNameProblem('Biology')).toBeNull();
    expect(folderNameProblem('  padded  ')).toBeNull();
    expect(folderNameProblem('')).toBe('Give the folder a name.');
    expect(folderNameProblem('   ')).toBe('Give the folder a name.');
    expect(folderNameProblem('x'.repeat(MAX_FOLDER_NAME))).toBeNull();
    expect(folderNameProblem('x'.repeat(MAX_FOLDER_NAME + 1))).toBe(`Use at most ${MAX_FOLDER_NAME} characters (this one has ${MAX_FOLDER_NAME + 1}).`);
  });

  it('turns a study address into what to study, and back', () => {
    expect(parseStudyScope('all')).toBeNull();
    expect(parseStudyScope(undefined)).toBeNull();
    expect(parseStudyScope('9f2c4e')).toBe('9f2c4e');
    expect(parseStudyScope('folder-abc123')).toEqual({ folderId: 'abc123' });
    for (const scope of [null, 'deck-1', { folderId: 'f9' }]) expect(parseStudyScope(studyParam(scope))).toEqual(scope);
    expect(studyParam({ folderId: 'f9' })).toBe('folder-f9');
  });

  it('describes a folder', () => {
    expect(folderSubtitle(3, 42)).toBe('3 decks · 42 cards');
    expect(folderSubtitle(1, 1)).toBe('1 deck · 1 card');
    expect(folderSubtitle(0, 0)).toBe('0 decks · 0 cards');
  });
});

describe('the Home list', () => {
  const decks = [
    { id: 'new', folder_id: null },
    { id: 'a', folder_id: 'bio' },
    { id: 'b', folder_id: 'bio' },
    { id: 'c', folder_id: 'prog' },
    { id: 'd', folder_id: null },
  ];
  const folders = [{ id: 'bio' }, { id: 'prog' }];
  const ids = (rows: HomeRow[]) => rows.map((r) => (r.kind === 'unfiled' ? '--' : r.kind === 'folder' ? `[${r.id}]` : r.id));

  it('lists folders with their decks, then the decks that are in no folder', () => {
    expect(ids(homeRows(decks, folders, [], new Set()))).toEqual(['[bio]', 'a', 'b', '[prog]', 'c', '--', 'new', 'd']);
  });

  it('keeps the order the user dragged the decks into, inside each list', () => {
    expect(ids(homeRows(decks, folders, ['b', 'a', 'd', 'new'], new Set()))).toEqual(['[bio]', 'b', 'a', '[prog]', 'c', '--', 'd', 'new']);
  });

  it('hides the decks of a collapsed folder but still shows the folder', () => {
    expect(ids(homeRows(decks, folders, [], new Set(['bio'])))).toEqual(['[bio]', '[prog]', 'c', '--', 'new', 'd']);
  });

  it('shows a plain list, with no heading, when there are no folders', () => {
    expect(ids(homeRows(decks.map((d) => ({ ...d, folder_id: null })), [], [], new Set()))).toEqual(['new', 'a', 'b', 'c', 'd']);
  });

  it('treats a deck in a folder this phone has not heard of as not in a folder', () => {
    const rows = homeRows([{ id: 'x', folder_id: 'ghost' }], [{ id: 'bio' }], [], new Set());
    expect(rows).toEqual([{ kind: 'folder', id: 'bio' }, { kind: 'unfiled' }, { kind: 'deck', id: 'x', folderId: null }]);
  });

  it('finds the decks that share a list', () => {
    const rows = homeRows(decks, folders, [], new Set());
    expect(siblingIndexes(rows, 1)).toEqual([1, 2]); // a and b, the Biology decks
    expect(siblingIndexes(rows, 6)).toEqual([6, 7]); // new and d, the loose ones
    expect(siblingIndexes(rows, 0)).toEqual([]); // a folder row has no siblings
  });

  it('rearranges one list inside the whole order, leaving the other decks where they were', () => {
    expect(reorderWithin(['a', 'x', 'b', 'y', 'c'], ['c', 'a', 'b'])).toEqual(['c', 'x', 'a', 'y', 'b']);
    expect(reorderWithin(['a', 'b'], ['b', 'a'])).toEqual(['b', 'a']);
  });
});

describe('dropping a held deck', () => {
  // [bio] folder, a, b (in bio), [prog] folder, c (in prog), unfiled heading, d (loose); each 100 tall with a 10 gap
  const rows: HomeRow[] = [
    { kind: 'folder', id: 'bio' },
    { kind: 'deck', id: 'a', folderId: 'bio' },
    { kind: 'deck', id: 'b', folderId: 'bio' },
    { kind: 'folder', id: 'prog' },
    { kind: 'deck', id: 'c', folderId: 'prog' },
    { kind: 'unfiled' },
    { kind: 'deck', id: 'd', folderId: null },
  ];
  const boxes = rows.map((_, i) => ({ y: i * 110, height: 100 }));
  const at = (index: number, dy: number) => dropTarget(rows, boxes, index, dy);

  it('files a deck dropped over the middle of another folder', () => {
    expect(at(6, 3 * 110 - 6 * 110)).toEqual({ type: 'folder', folderId: 'prog' }); // d, carried up to the Programming row
    expect(at(1, 3 * 110 - 1 * 110)).toEqual({ type: 'folder', folderId: 'prog' }); // a, carried down to it
  });

  it('does nothing when a deck is dropped over its own folder', () => {
    expect(at(1, -110)).toEqual({ type: 'none' }); // a over [bio]
  });

  it('takes a deck out of its folder when it is dropped on the "not in a folder" heading', () => {
    expect(at(4, 110)).toEqual({ type: 'unfile' }); // c down to the heading
    expect(at(1, 4 * 110)).toEqual({ type: 'unfile' }); // a
  });

  it('does nothing for a loose deck dropped on that heading', () => {
    expect(at(6, -110)).toEqual({ type: 'none' });
  });

  it('only counts the middle of a folder row: the very edge is not a drop into it', () => {
    expect(at(6, 3 * 110 - 6 * 110 - 45)).not.toEqual({ type: 'folder', folderId: 'prog' }); // middle 5px into the row's top 15%
  });

  it('reorders a deck among the decks of its own list', () => {
    expect(at(1, 120)).toEqual({ type: 'reorder', to: 1 }); // a carried past b: its middle (280) is below b's middle (270), so it becomes second of the two
  });

  it('does nothing for a small move, and for a card that is not a deck', () => {
    expect(at(1, 10)).toEqual({ type: 'none' });
    expect(at(0, 200)).toEqual({ type: 'none' }); // a folder row is not draggable
    expect(dropTarget(rows, boxes, 99, 0)).toEqual({ type: 'none' });
  });
});

describe('LocalStore folders', () => {
  it('stores folders and each deck\'s folder, and lists the folders A to Z with what they hold', async () => {
    const { store } = await makeStore();
    await store.applySync(
      response({
        folders: [folder('f2', 'Programming'), folder('f1', 'biology')],
        decks: [deckIn('d1', 'f1', 'Bio I'), deckIn('d2', 'f1', 'Bio II'), deckIn('d3', null, 'Loose')],
        cards: [card('c1', { deck_id: 'd1' }), card('c2', { deck_id: 'd1' }), card('c3', { deck_id: 'd2' }), card('c4', { deck_id: 'd3' })],
      }),
      [],
    );
    expect((await store.listFolders()).map((f) => [f.name, f.decks, f.total])).toEqual([
      ['biology', 2, 3],
      ['Programming', 0, 0],
    ]);
    const byId = Object.fromEntries((await store.listDecks(NOW, 10)).map((d) => [d.id, d.folder_id]));
    expect(byId).toEqual({ d1: 'f1', d2: 'f1', d3: null });
  });

  it('follows a deck moved to another folder, or taken out of one', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ folders: [folder('f1'), folder('f2')], decks: [deckIn('d1', 'f1')] }), []);
    await store.applySync(response({ decks: [deckIn('d1', 'f2')] }), []);
    expect((await store.listDecks(NOW, 10))[0].folder_id).toBe('f2');
    await store.applySync(response({ decks: [deckIn('d1', null)] }), []);
    expect((await store.listDecks(NOW, 10))[0].folder_id).toBeNull();
  });

  it('keeps decks and cards when their folder is deleted, and takes them out of it', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ folders: [folder('f1')], decks: [deckIn('d1', 'f1')], cards: [card('c1', { deck_id: 'd1' })] }), []);
    await store.applySync(response({ deleted: [{ kind: 'folder', id: 'f1' }] }), []); // even before the deck's own update arrives
    expect(await store.listFolders()).toEqual([]);
    const decks = await store.listDecks(NOW, 10);
    expect(decks.map((d) => [d.id, d.folder_id, d.total])).toEqual([['d1', null, 1]]);
  });

  it('copes with an older server that sends no folders', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ decks: [{ id: 'd1', name: 'Old', topic: '', updated_at: T }] }), []);
    expect((await store.listDecks(NOW, 10))[0].folder_id).toBeNull();
    expect(await store.listFolders()).toEqual([]);
  });

  it('drops folders with the rest of the cached library on a reset or a server switch', async () => {
    const { store } = await makeStore();
    await store.applySync(response({ folders: [folder('f1')], decks: [deckIn('d1', 'f1')] }), []);
    await store.applySync(response({ reset: true, folders: [folder('f9')], decks: [deckIn('d9', 'f9')] }), []);
    expect((await store.listFolders()).map((f) => f.id)).toEqual(['f9']);
    await store.clearData();
    expect(await store.listFolders()).toEqual([]);
  });

  it('counts a folder as a whole, sharing the daily new-card limit instead of adding up its decks', async () => {
    const { store } = await makeStore();
    await store.applySync(
      response({
        folders: [folder('f1')],
        decks: [deckIn('d1', 'f1'), deckIn('d2', 'f1'), deckIn('d3', null)],
        cards: [...['a', 'b', 'c'].map((id) => card(id, { deck_id: 'd1' })), ...['d', 'e', 'f'].map((id) => card(id, { deck_id: 'd2' })), card('z', { deck_id: 'd3' })],
      }),
      [],
    );
    expect(await store.counts({ folderId: 'f1' }, NOW, 4)).toEqual({ new: 4, learning: 0, review: 0 }); // 6 new cards, a budget of 4
    expect((await store.counts({ folderId: 'f1' }, NOW, 10)).new).toBe(6); // the loose deck's card is not in it
    expect((await store.counts('d1', NOW, 10)).new).toBe(3);
    expect((await store.counts(null, NOW, 10)).new).toBe(7);
  });

  it('studies only the folder\'s cards, and reports when its next studied card is due', async () => {
    const { store } = await makeStore();
    await store.applySync(
      response({ folders: [folder('f1')], decks: [deckIn('d1', 'f1'), deckIn('d2', null)], cards: [card('in', { deck_id: 'd1' }), card('out', { deck_id: 'd2' })] }),
      [],
    );
    const seen = new Set<string>();
    for (let i = 0; i < 2; i++) {
      const next = await store.nextCard({ folderId: 'f1' }, NOW, 10);
      if (!next.card) break;
      seen.add(next.card.id);
      await store.answer(next.card.id, 3, NOW, 1000);
    }
    expect([...seen]).toEqual(['in']);
    expect(await store.nextDue({ folderId: 'f1' })).not.toBeNull();
    expect(await store.nextDue('d2')).toBeNull(); // the other deck was never studied
  });

  it('lists every card in the scope for practice, whatever the daily limit or schedule says, and changes nothing', async () => {
    const { store } = await makeStore();
    const ids = ['a', 'b', 'c', 'd', 'e'];
    await store.applySync(response({ folders: [folder('f1')], decks: [deckIn('d1', 'f1'), deckIn('d2', null)], cards: [...ids.map((id) => card(id, { deck_id: 'd1' })), card('z', { deck_id: 'd2' })] }), []);
    await store.answer('a', 3, NOW, 1000); // studied, so not due again today
    expect((await store.counts('d1', NOW, 1)).new).toBe(0); // the daily limit of 1 is used up...
    expect(['a', undefined]).toContain((await store.nextCard('d1', NOW, 1)).card?.id); // ...so normal study offers no other new card
    expect((await store.allCardIds('d1')).sort()).toEqual(ids); // practice still has all five
    expect((await store.allCardIds({ folderId: 'f1' })).sort()).toEqual(ids);
    expect((await store.allCardIds(null)).sort()).toEqual([...ids, 'z']);
    expect(await store.pendingCount()).toBe(1); // only the one real answer was logged
  });
});

describe('the phone database', () => {
  it('steps a version 1 database forward to folders without losing its decks, cards or reviews', async () => {
    const db = await openUnmigratedTestDb();
    await db.exec(MIGRATIONS[0]);
    await db.exec('PRAGMA user_version = 1');
    await db.run("INSERT INTO decks(id, name, topic, updated_at) VALUES ('d1', 'Kept', '', 't')");
    await db.run(
      "INSERT INTO cards(id, deck_id, type, front, back, origin, source_ref, created_at, updated_at, due, fsrs_json) VALUES ('c1','d1','basic','f','b','notes','','t','t','t','{}')",
    );

    await migrate(db);

    expect((await db.first<{ user_version: number }>('PRAGMA user_version'))?.user_version).toBe(MIGRATIONS.length);
    expect(await db.first('SELECT name, folder_id FROM decks WHERE id = ?', ['d1'])).toEqual({ name: 'Kept', folder_id: null });
    expect((await db.first<{ n: number }>('SELECT COUNT(*) AS n FROM cards'))?.n).toBe(1);
    await db.run("INSERT INTO folders(id, name, updated_at) VALUES ('f1', 'New', 't')"); // the table exists
  });

  it('still seeds a deck the way every other test does', async () => {
    const { store } = await makeStore();
    await seedCards(store, ['c1']);
    expect((await store.listDecks(NOW, 10))[0]).toMatchObject({ id: deckA.id, folder_id: null });
  });
});
