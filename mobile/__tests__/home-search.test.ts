import { deckCaption } from '@/lib/deck';
import { filterHomeRows, folderCaption, homeRows, matchesQuery, type HomeRow } from '@/lib/folders';

import { makeStore } from './helpers/fixtures';

const deck = (id: string, folder_id: string | null, name = id, topic = '') => ({ id, name, topic, folder_id });
const folder = (id: string, name = id) => ({ id, name });
const ids = (rows: HomeRow[]) => rows.map((r) => (r.kind === 'unfiled' ? 'unfiled' : `${r.kind}:${r.id}`));

const decks = [
  deck('cells', 'bio', 'Cell biology', 'biology'),
  deck('genetics', 'bio', 'Genetics', 'biology'),
  deck('python', 'prog', 'Python basics', 'programming'),
  deck('sql', 'prog', 'SQL joins', 'programming'),
  deck('french', null, 'French verbs', 'language'),
  deck('chem', null, 'Chemistry', 'science'),
];
const folders = [folder('bio', 'Biology'), folder('prog', 'Programming')];

describe('searching the Home list', () => {
  it('is the ordinary list when nothing is typed', () => {
    const closed = new Set(['prog']);
    expect(filterHomeRows(decks, folders, [], closed, '')).toEqual(homeRows(decks, folders, [], closed));
    expect(filterHomeRows(decks, folders, [], closed, '   ')).toEqual(homeRows(decks, folders, [], closed));
  });

  it('matches the deck name or topic, ignoring case, and keeps the folder of a matching deck with only the matching decks', () => {
    expect(ids(filterHomeRows(decks, folders, [], new Set(), 'sql'))).toEqual(['folder:prog', 'deck:sql']);
    expect(ids(filterHomeRows(decks, folders, [], new Set(), 'GENETICS'))).toEqual(['folder:bio', 'deck:genetics']);
    expect(ids(filterHomeRows(decks, folders, [], new Set(), 'language'))).toEqual(['deck:french']); // by topic; no folder is listed, so no "not in a folder" heading
    expect(ids(filterHomeRows(decks, folders, [], new Set(), '  chem  '))).toEqual(['deck:chem']);
  });

  it('shows every deck of a folder whose own name matches', () => {
    expect(ids(filterHomeRows(decks, folders, [], new Set(), 'program'))).toEqual(['folder:prog', 'deck:python', 'deck:sql']);
  });

  it('lists folders open while searching, even when they were closed', () => {
    expect(ids(filterHomeRows(decks, folders, [], new Set(['bio', 'prog']), 'cell'))).toEqual(['folder:bio', 'deck:cells']);
  });

  it('combines matches from folders and loose decks, in the order the decks were arranged', () => {
    const order = ['chem', 'french', 'sql', 'python', 'genetics', 'cells'];
    expect(ids(filterHomeRows(decks, folders, order, new Set(), 'i'))).toEqual([
      'folder:bio', 'deck:genetics', 'deck:cells', // "Biology" matches, so both its decks, in the arranged order
      'folder:prog', 'deck:sql', 'deck:python', // "Programming" matches
      'unfiled', 'deck:chem', // "Chemistry" has an i; "French verbs" and its topic do not
    ]);
  });

  it('gives nothing when nothing matches', () => {
    expect(filterHomeRows(decks, folders, [], new Set(), 'zzz')).toEqual([]);
  });

  it('treats a deck whose folder is unknown to this phone as loose', () => {
    expect(ids(filterHomeRows([deck('a', 'gone', 'Algebra')], [], [], new Set(), 'alg'))).toEqual(['deck:a']);
  });

  it('matches text without being fooled by case or spaces', () => {
    expect(matchesQuery('Cell Biology', 'cell')).toBe(true);
    expect(matchesQuery('Cell Biology', ' BIOLOGY ')).toBe(true);
    expect(matchesQuery('Cell Biology', 'physics')).toBe(false);
  });
});

describe('which folders are open', () => {
  it('starts with none open, remembers what was opened, and ignores damaged data', async () => {
    const { store } = await makeStore();
    expect(await store.openFolders()).toEqual([]);
    await store.setOpenFolders(['a', 'b']);
    expect(await store.openFolders()).toEqual(['a', 'b']);
    await store.setMeta('folders_open', '{not json');
    expect(await store.openFolders()).toEqual([]);
    await store.setMeta('folders_open', '"text"');
    expect(await store.openFolders()).toEqual([]);
  });

  it('does not read the old list of closed folders', async () => {
    const { store } = await makeStore();
    await store.setMeta('folders_closed', JSON.stringify(['a']));
    expect(await store.openFolders()).toEqual([]);
  });
});

describe('the words on a deck and folder line', () => {
  const base = { total: 24, paused: 0, problem: 0, new: 0, learning: 0, review: 0 };

  it('says how many cards and what is due', () => {
    expect(deckCaption({ ...base, review: 3, learning: 1, new: 5 })).toBe('24 cards · 4 due · 5 new');
    expect(deckCaption({ ...base, new: 1 })).toBe('24 cards · 1 new');
    expect(deckCaption(base)).toBe('24 cards · nothing due');
    expect(deckCaption({ ...base, total: 1 })).toBe('1 card · nothing due');
  });

  it('adds problem cards, paused cards and cards waiting for review only when there are some', () => {
    expect(deckCaption({ ...base, review: 2, problem: 2, paused: 3 }, 4)).toBe('24 cards · 2 due · 2 problem cards · 3 paused · 4 to review');
    expect(deckCaption({ ...base, total: 3, paused: 3 })).toBe('3 cards · all paused');
    expect(deckCaption({ ...base, total: 0 })).toBe('0 cards · none approved yet');
  });

  it('says how big a folder is and what is due in it', () => {
    expect(folderCaption({ decks: 3, total: 42 }, 8, 5)).toBe('3 decks · 42 cards · 8 due · 5 new');
    expect(folderCaption({ decks: 1, total: 1 }, 0, 0)).toBe('1 deck · 1 card · nothing due');
    expect(folderCaption({ decks: 0, total: 0 }, 0, 0)).toBe('0 decks · 0 cards · empty');
  });
});
