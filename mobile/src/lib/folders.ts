import type { StudyScope } from '@/study/store';

import { dragTarget, orderDecks, type RowBox } from './deck';

/** The longest folder name the server accepts. */
export const MAX_FOLDER_NAME = 80;

/** Why a folder name cannot be saved, or null when it can. (The server also refuses a name another folder already uses.) */
export function folderNameProblem(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return 'Give the folder a name.';
  if (trimmed.length > MAX_FOLDER_NAME) return `Use at most ${MAX_FOLDER_NAME} characters (this one has ${trimmed.length}).`;
  return null;
}

// -- studying a folder ---------------------------------------------------------------------

/** In a study address a folder is `folder-<id>`; a deck is its bare id (always hex, so the two cannot be confused) and everything is `all`. */
const FOLDER_PARAM_PREFIX = 'folder-';

export function parseStudyScope(param: string | undefined): StudyScope {
  if (!param || param === 'all') return null;
  if (param.startsWith(FOLDER_PARAM_PREFIX)) return { folderId: param.slice(FOLDER_PARAM_PREFIX.length) };
  return param;
}

export function studyParam(scope: StudyScope): string {
  if (scope === null) return 'all';
  return typeof scope === 'string' ? scope : FOLDER_PARAM_PREFIX + scope.folderId;
}

// -- the Home list -------------------------------------------------------------------------

/** One line of the Home list, top to bottom. Folders, then (once any folder exists) a "not in a folder" heading, then the loose decks. */
export type HomeRow = { kind: 'folder'; id: string } | { kind: 'unfiled' } | { kind: 'deck'; id: string; folderId: string | null };

/**
 * The rows to show. `folders` are in the order they are to appear; a folder in `closed` hides its decks. Decks keep the order the user
 * dragged them into. A deck that names a folder this phone does not know about (not synced yet) is shown as not in a folder.
 */
export function homeRows(decks: { id: string; folder_id: string | null }[], folders: { id: string }[], order: string[], closed: ReadonlySet<string>): HomeRow[] {
  const ordered = orderDecks(decks, order);
  const known = new Set(folders.map((f) => f.id));
  const rows: HomeRow[] = [];
  for (const folder of folders) {
    rows.push({ kind: 'folder', id: folder.id });
    if (closed.has(folder.id)) continue;
    for (const deck of ordered) if (deck.folder_id === folder.id) rows.push({ kind: 'deck', id: deck.id, folderId: folder.id });
  }
  if (folders.length > 0) rows.push({ kind: 'unfiled' });
  for (const deck of ordered) if (!deck.folder_id || !known.has(deck.folder_id)) rows.push({ kind: 'deck', id: deck.id, folderId: null });
  return rows;
}

/** True when `text` contains `query` (ignoring case and the spaces around the query). */
export function matchesQuery(text: string, query: string): boolean {
  return text.toLowerCase().includes(query.trim().toLowerCase());
}

/**
 * The Home list narrowed by a search. With nothing typed it is exactly `homeRows`. With a search, a deck is kept when its name or topic
 * contains the text; a folder is kept when its own name does (with all its decks) or when it holds a matching deck (with just those decks).
 * Folders are listed open while searching, whatever was closed before, because the point is to see the matches.
 */
export function filterHomeRows(
  decks: { id: string; name: string; topic?: string; folder_id: string | null }[],
  folders: { id: string; name: string }[],
  order: string[],
  closed: ReadonlySet<string>,
  query: string,
): HomeRow[] {
  if (!query.trim()) return homeRows(decks, folders, order, closed);
  const ordered = orderDecks(decks, order);
  const known = new Set(folders.map((f) => f.id));
  const deckMatches = (d: { name: string; topic?: string }) => matchesQuery(d.name, query) || matchesQuery(d.topic ?? '', query);
  const rows: HomeRow[] = [];
  for (const folder of folders) {
    const inside = ordered.filter((d) => d.folder_id === folder.id);
    const folderMatches = matchesQuery(folder.name, query);
    const shown = folderMatches ? inside : inside.filter(deckMatches);
    if (!folderMatches && shown.length === 0) continue;
    rows.push({ kind: 'folder', id: folder.id });
    for (const deck of shown) rows.push({ kind: 'deck', id: deck.id, folderId: folder.id });
  }
  const loose = ordered.filter((d) => (!d.folder_id || !known.has(d.folder_id)) && deckMatches(d));
  if (loose.length > 0) {
    if (rows.length > 0) rows.push({ kind: 'unfiled' });
    for (const deck of loose) rows.push({ kind: 'deck', id: deck.id, folderId: null });
  }
  return rows;
}

/** "3 decks · 42 cards" */
export function folderSubtitle(decks: number, cards: number): string {
  return `${decks} ${decks === 1 ? 'deck' : 'decks'} · ${cards} ${cards === 1 ? 'card' : 'cards'}`;
}

/** What a folder's line says under its name: its size and what is due in it. */
export function folderCaption(folder: { decks: number; total: number }, due: number, fresh: number): string {
  const parts = [due > 0 && `${due} due`, fresh > 0 && `${fresh} new`].filter(Boolean) as string[];
  return `${folderSubtitle(folder.decks, folder.total)} · ${parts.length ? parts.join(' · ') : folder.total > 0 ? 'nothing due' : 'empty'}`;
}

// -- dragging a deck -----------------------------------------------------------------------

/** What letting go would do: nothing, move the deck within its own list, put it in a folder, or take it out of its folder. */
export type DropTarget = { type: 'none' } | { type: 'reorder'; to: number } | { type: 'folder'; folderId: string } | { type: 'unfile' };

/** Indexes (in `rows`) of the decks that share a list with the deck at `index`, in order; the deck itself included. */
export function siblingIndexes(rows: HomeRow[], index: number): number[] {
  const held = rows[index];
  if (!held || held.kind !== 'deck') return [];
  return rows.flatMap((r, i) => (r.kind === 'deck' && r.folderId === held.folderId ? [i] : []));
}

/**
 * Where a held deck would land. `boxes` are the rows' places in the list (same indexes as `rows`) and `dy` how far the finger has moved.
 * Over the middle of a folder's row (or the "not in a folder" heading) it files (or unfiles) the deck; anywhere else it reorders among
 * the decks of its own list, and `to` is the index within that list.
 */
export function dropTarget(rows: HomeRow[], boxes: RowBox[], from: number, dy: number): DropTarget {
  const held = rows[from];
  if (!held || held.kind !== 'deck' || !boxes[from]) return { type: 'none' };
  const middle = boxes[from].y + boxes[from].height / 2 + dy;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const box = boxes[i];
    if (row.kind === 'deck' || !box) continue;
    if (middle < box.y + box.height * 0.15 || middle > box.y + box.height * 0.85) continue;
    if (row.kind === 'folder') return held.folderId === row.id ? { type: 'none' } : { type: 'folder', folderId: row.id };
    return held.folderId === null ? { type: 'none' } : { type: 'unfile' };
  }
  const siblings = siblingIndexes(rows, from);
  const position = siblings.indexOf(from);
  const to = dragTarget(
    siblings.map((i) => boxes[i]),
    position,
    dy,
  );
  return to === position ? { type: 'none' } : { type: 'reorder', to };
}

/** The whole deck order with the decks of one list rearranged: they take, in turn, the places those decks already held. */
export function reorderWithin(all: string[], newSiblingOrder: string[]): string[] {
  const moving = new Set(newSiblingOrder);
  let next = 0;
  return all.map((id) => (moving.has(id) ? newSiblingOrder[next++] : id));
}
