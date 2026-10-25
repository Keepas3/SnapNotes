import { useQueryClient } from '@tanstack/react-query';
import { Redirect, router, useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { Pressable, RefreshControl, StyleSheet, View } from 'react-native';

import { ApiError } from '@/api/client';
import { useDecks, useDeleteDeck, useJobs, useSetDeckFolder } from '@/api/hooks';
import type { Job } from '@/api/types';
import { DeckSettingsSheet } from '@/components/deck-settings';
import { DraggableRow } from '@/components/draggable-row';
import { FolderSheet, type FolderSheetTarget } from '@/components/folder-sheet';
import { ThemedText } from '@/components/themed-text';
import { Badge, Banner, Button, Card, Centered, EmptyState, IconButton, Screen, Section, Spinner } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { confirm } from '@/lib/confirm';
import { deleteDeckMessage, dragShift, moveItem, orderDecks, type RowBox } from '@/lib/deck';
import { dropTarget, folderSubtitle, homeRows, reorderWithin, siblingIndexes, studyParam, type DropTarget, type HomeRow } from '@/lib/folders';
import { plural } from '@/lib/format';
import { useSettings } from '@/lib/settings';
import { JOB_STATUS_LABEL } from '@/lib/steps';
import { useClosedFolders, useDeckOrder, useLocalDecks, useLocalFolders, usePendingReviews, useSetClosedFolders, useSetDeckOrder, useStudyCounts } from '@/study/hooks';
import type { DeckSummary, FolderSummary } from '@/study/store';
import { LOCAL_KEY, useSync } from '@/sync/provider';

export default function Home() {
  const { ready, configured } = useSettings();
  if (!ready) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }
  if (!configured) return <Redirect href="/settings" />;
  return <HomeContent />;
}

function HomeContent() {
  const queryClient = useQueryClient();
  const { sync, status, storeError } = useSync();
  const serverDecks = useDecks(); // online only: how many drafts are waiting for review in each deck
  const jobs = useJobs(); // online only: uploads in progress
  const local = useLocalDecks(); // on-device: what can be studied, with or without a connection
  const pending = usePendingReviews();
  const savedOrder = useDeckOrder(); // the order the user dragged the decks into, kept on this phone
  const saveOrder = useSetDeckOrder();
  const localFolders = useLocalFolders(); // folders as the phone knows them
  const closedFolders = useClosedFolders(); // which folders are collapsed (a preference of this phone)
  const saveClosed = useSetClosedFolders();
  const fileDeck = useSetDeckFolder(); // dropping a deck on a folder changes the library on the server
  const [folderSheet, setFolderSheet] = useState<FolderSheetTarget | null>(null);
  const [moveError, setMoveError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  // While a deck is being dragged: which row, where it would land, how far the finger moved, and how far the others slide.
  const [drag, setDrag] = useState<{ from: number; dy: number; step: number; target: DropTarget } | null>(null);
  const [held, setHeld] = useState(false); // a finger has been held on a deck long enough that it may be dragged: the page must not scroll
  // Each row's height, filled in as they are laid out. Heights only: a row's position is worked out from them, because a
  // layout event is not sent when a row merely moves (it would be stale after a reorder).
  const heights = useRef(new Map<string, number>());

  // Whenever Home comes into view, refresh what the phone knows and catch up with the server.
  useFocusEffect(
    useCallback(() => {
      queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
      void sync({ minIntervalMs: 15_000 });
    }, [queryClient, sync]),
  );

  async function refresh() {
    setRefreshing(true);
    await Promise.all([sync(), serverDecks.refetch(), jobs.refetch()]);
    setRefreshing(false);
  }

  const inProgress = (jobs.data ?? []).filter((j) => j.status !== 'completed');
  const draftCounts = new Map((serverDecks.data ?? []).map((d) => [d.id, d.counts.draft]));
  const decks = local.data ?? [];
  const folders = localFolders.data ?? [];
  const closed = new Set(closedFolders.data ?? []);
  const orderedIds = orderDecks(decks, savedOrder.data ?? []).map((d) => d.id); // every deck, in the order the user arranged them
  const rows = homeRows(decks, folders, savedOrder.data ?? [], closed);
  const deckById = new Map(decks.map((d) => [d.id, d]));
  const folderById = new Map(folders.map((f) => [f.id, f]));
  const waiting = decks.reduce((n, d) => n + d.new + d.learning + d.review, 0);

  const rowKey = (row: HomeRow) => (row.kind === 'unfiled' ? 'unfiled' : `${row.kind}:${row.id}`);

  /** Where each row sits, top to bottom as shown now (relative to the first). */
  function rowBoxes(): RowBox[] {
    let y = 0;
    return rows.map((row) => {
      const height = heights.current.get(rowKey(row)) ?? 0;
      const box = { y, height };
      y += height + DECK_GAP;
      return box;
    });
  }

  function dragStart(from: number) {
    setDrag({ from, dy: 0, step: (rowBoxes()[from]?.height ?? 0) + DECK_GAP, target: { type: 'none' } });
  }
  function dragMove(from: number, dy: number) {
    setDrag((d) => (d ? { ...d, from, dy, target: dropTarget(rows, rowBoxes(), from, dy) } : d));
  }
  function saveReorder(from: number, to: number) {
    const siblings = siblingIndexes(rows, from);
    const ids = siblings.map((i) => (rows[i] as Extract<HomeRow, { kind: 'deck' }>).id);
    saveOrder.mutate(reorderWithin(orderedIds, moveItem(ids, siblings.indexOf(from), to)));
  }
  async function dragEnd(from: number, dy: number) {
    const target = dropTarget(rows, rowBoxes(), from, dy);
    const heldRow = rows[from];
    setDrag(null);
    if (target.type === 'reorder') saveReorder(from, target.to);
    else if ((target.type === 'folder' || target.type === 'unfile') && heldRow.kind === 'deck') {
      setMoveError(null);
      try {
        await fileDeck.mutateAsync({ deckId: heldRow.id, folderId: target.type === 'folder' ? target.folderId : null });
        await refreshPhone();
      } catch (e) {
        setMoveError(e instanceof Error ? e.message : 'Could not move the deck.');
      }
    }
  }
  function nudge(from: number, direction: -1 | 1) {
    const siblings = siblingIndexes(rows, from);
    const to = siblings.indexOf(from) + direction;
    if (to >= 0 && to < siblings.length) saveReorder(from, to);
  }
  function toggleFolder(id: string) {
    saveClosed.mutate(closed.has(id) ? [...closed].filter((x) => x !== id) : [...closed, id]);
  }

  /** The phone's list is its own copy of the library, so bring it up to date as soon as the server has changed. */
  async function refreshPhone() {
    queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
    await sync();
  }

  /** How a row moves while a deck is held: the held one follows the finger, and the decks it passes make room. */
  function rowMotion(i: number) {
    if (drag === null) return undefined;
    if (drag.from === i) return { transform: [{ translateY: drag.dy }], zIndex: 10, opacity: 0.92 };
    if (drag.target.type !== 'reorder') return undefined;
    const siblings = siblingIndexes(rows, drag.from);
    const place = siblings.indexOf(i);
    if (place < 0) return undefined;
    return { transform: [{ translateY: dragShift(place, siblings.indexOf(drag.from), drag.target.to, drag.step) }] };
  }

  const onlineError = serverDecks.error ?? jobs.error;
  const offline = (onlineError instanceof ApiError && onlineError.isNetwork) || status.offline;

  return (
    <Screen
      scrollEnabled={drag === null && !held}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} />}
      footer={<Button title="New deck from notes" variant={waiting > 0 ? 'secondary' : 'primary'} onPress={() => router.push('/new')} />}>
      {storeError ? <Banner message={`Offline study is unavailable: ${storeError}`} /> : null}
      {offline ? (
        <Banner tone="warning" message="You're offline. Studying still works; uploading notes and reviewing new cards need a connection." />
      ) : onlineError ? (
        <Banner message={onlineError.message} actionLabel="Server settings" onAction={() => router.push('/settings')} />
      ) : null}

      {waiting > 0 ? (
        <Card>
          <ThemedText type="heading">{plural(waiting, 'card')} waiting</ThemedText>
          <Button title="Study now" onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: 'all' } })} />
        </Card>
      ) : null}

      {inProgress.length > 0 ? (
        <Section title="In progress">
          {inProgress.map((job) => (
            <JobRow key={job.id} job={job} />
          ))}
        </Section>
      ) : null}

      <Section title="Your decks" right={<Button title="New folder" variant="ghost" onPress={() => setFolderSheet({ mode: 'new' })} />}>
        {moveError ? <Banner message={moveError} /> : null}
        {local.isPending ? (
          <Spinner />
        ) : rows.length === 0 ? (
          <EmptyState
            title={status.syncing ? 'Loading your decks…' : 'No decks yet'}
            message="Upload a PDF, slides, a document, or photos of your notes and SnapNotes will turn them into flashcards."
          />
        ) : (
          rows.map((row, i) => {
            const key = rowKey(row);
            const onLayout = (e: { nativeEvent: { layout: { height: number } } }) => heights.current.set(key, e.nativeEvent.layout.height);
            if (row.kind === 'folder') {
              const folder = folderById.get(row.id);
              if (!folder) return null;
              return (
                <View key={key} onLayout={onLayout}>
                  <FolderRow
                    folder={folder}
                    open={!closed.has(folder.id)}
                    highlighted={drag?.target.type === 'folder' && drag.target.folderId === folder.id}
                    onToggle={() => toggleFolder(folder.id)}
                    onSettings={() => setFolderSheet({ mode: 'edit', folder: { id: folder.id, name: folder.name }, deckCount: folder.decks })}
                  />
                </View>
              );
            }
            if (row.kind === 'unfiled') {
              return (
                <View key={key} onLayout={onLayout} style={[styles.unfiled, drag?.target.type === 'unfile' && styles.unfiledActive]}>
                  <ThemedText type="smallBold" themeColor="textSecondary">
                    Not in a folder
                  </ThemedText>
                </View>
              );
            }
            const deck = deckById.get(row.id);
            if (!deck) return null;
            const siblings = siblingIndexes(rows, i);
            return (
              // Press and hold a deck, then drag: the held deck follows the finger and the ones it passes make room.
              <DraggableRow
                key={key}
                onLayout={onLayout}
                onHold={() => setHeld(true)}
                onLift={() => dragStart(i)}
                onMove={(dy) => dragMove(i, dy)}
                onDrop={(dy) => {
                  void dragEnd(i, dy);
                  setHeld(false);
                }}
                onCancel={() => {
                  setDrag(null);
                  setHeld(false);
                }}
                style={[row.folderId ? styles.indented : null, rowMotion(i)]}>
                <DeckRow deck={deck} drafts={draftCounts.get(deck.id) ?? 0} onMove={siblings.length > 1 ? (direction) => nudge(i, direction) : undefined} />
              </DraggableRow>
            );
          })
        )}
      </Section>

      <FolderSheet target={folderSheet} onClose={() => setFolderSheet(null)} onChanged={refreshPhone} />

      <SyncFooter pending={pending.data ?? 0} />
    </Screen>
  );
}

/** A folder: its name and what it holds, a Study button for everything in it, and a chevron to hide or show its decks. */
function FolderRow({
  folder,
  open,
  highlighted,
  onToggle,
  onSettings,
}: {
  folder: FolderSummary;
  open: boolean;
  highlighted: boolean;
  onToggle: () => void;
  onSettings: () => void;
}) {
  const theme = useTheme();
  const counts = useStudyCounts({ folderId: folder.id }); // counted for the folder as a whole, so the shared daily new-card limit holds
  const due = (counts.data?.learning ?? 0) + (counts.data?.review ?? 0);
  const fresh = counts.data?.new ?? 0;
  const parts = [due > 0 && `${due} due`, fresh > 0 && `${fresh} new`].filter(Boolean) as string[];
  return (
    <Card style={highlighted ? { borderColor: theme.tint, borderWidth: 2, backgroundColor: theme.backgroundSelected } : undefined}>
      <View style={styles.deckHeader}>
        <IconButton glyph="☰" label={`Settings for folder ${folder.name}`} onPress={onSettings} />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`${folder.name}, ${folderSubtitle(folder.decks, folder.total)}`}
          accessibilityState={{ expanded: open }}
          onPress={onToggle}
          style={({ pressed }) => [styles.flex, styles.folderTitle, pressed && styles.pressed]}>
          <ThemedText type="heading" numberOfLines={1} style={styles.flex}>
            {folder.name}
          </ThemedText>
          <ThemedText themeColor="textSecondary">{open ? '⌃' : '⌄'}</ThemedText>
        </Pressable>
      </View>
      <ThemedText type="small" themeColor="textSecondary">
        {folderSubtitle(folder.decks, folder.total)} · {parts.length ? parts.join(' · ') : folder.total > 0 ? 'nothing due' : 'empty'}
      </ThemedText>
      {folder.total - folder.paused > 0 ? (
        <View style={styles.studyButtons}>
          {due + fresh > 0 ? (
            <Button title={`Study folder (${due + fresh})`} variant="secondary" style={styles.flex} onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: studyParam({ folderId: folder.id }) } })} />
          ) : null}
          <Button title={`All ${folder.total - folder.paused}`} variant="ghost" style={styles.flex} onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: studyParam({ folderId: folder.id }), mode: 'all' } })} />
        </View>
      ) : null}
    </Card>
  );
}

function SyncFooter({ pending }: { pending: number }) {
  const { status, sync } = useSync();
  const when = status.lastSyncedAt?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return (
    <View style={styles.syncRow}>
      <ThemedText type="caption" themeColor={status.error ? 'danger' : 'textSecondary'} style={styles.flex}>
        {status.syncing
          ? 'Syncing…'
          : status.error
            ? `Sync problem: ${status.error}`
            : status.offline
              ? `Offline${pending > 0 ? `, ${plural(pending, 'review')} waiting to sync` : ''}`
              : pending > 0
                ? `${plural(pending, 'review')} waiting to sync`
                : when
                  ? `Synced at ${when}`
                  : 'Not synced yet'}
      </ThemedText>
      <Button title="Sync now" variant="ghost" onPress={() => sync()} disabled={status.syncing} />
    </View>
  );
}

function JobRow({ job }: { job: Job }) {
  const s = JOB_STATUS_LABEL[job.status];
  return (
    <Card accessibilityLabel={`${job.filename}, ${s.label}`} onPress={() => router.push({ pathname: '/jobs/[id]', params: { id: job.id } })}>
      <View style={styles.rowBetween}>
        <ThemedText type="smallBold" numberOfLines={1} style={styles.flex}>
          {job.understanding?.main_topic ?? job.filename}
        </ThemedText>
        <Badge label={s.label} tone={s.tone} />
      </View>
      {job.status === 'failed' && job.error ? (
        <ThemedText type="caption" themeColor="danger" numberOfLines={2}>
          {job.error}
        </ThemedText>
      ) : (
        <ThemedText type="caption" themeColor="textSecondary" numberOfLines={1}>
          {job.filename}
        </ThemedText>
      )}
    </Card>
  );
}

function DeckRow({ deck, drafts, onMove }: { deck: DeckSummary; drafts: number; onMove?: (direction: -1 | 1) => void }) {
  const queryClient = useQueryClient();
  const { sync } = useSync();
  const remove = useDeleteDeck();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const due = deck.learning + deck.review;
  const studyable = due + deck.new;
  const parts = [due > 0 && `${due} due`, deck.new > 0 && `${deck.new} new`].filter(Boolean) as string[];
  const open = () => router.push({ pathname: '/decks/[id]', params: { id: deck.id } });

  /** The phone's list is its own copy of the library, so bring it up to date as soon as the server has changed. */
  async function refreshPhone() {
    queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
    await sync();
  }

  async function onDelete() {
    if (!(await confirm(`Delete “${deck.name}”?`, deleteDeckMessage(deck.total, drafts), 'Delete', true))) return;
    setError(null);
    try {
      await remove.mutateAsync(deck.id);
      await refreshPhone();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete the deck.');
    }
  }

  return (
    <Card>
      {/* Buttons and the tappable parts are siblings: nesting one pressable in another is invalid on the web. */}
      <View style={styles.deckHeader}>
        <IconButton glyph="☰" label={`Settings for ${deck.name}`} onPress={() => setSettingsOpen(true)} />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`${deck.name}, ${plural(deck.total, 'card')}`}
          accessibilityActions={onMove ? [{ name: 'decrement', label: 'Move up' }, { name: 'increment', label: 'Move down' }] : undefined}
          onAccessibilityAction={(e) => onMove?.(e.nativeEvent.actionName === 'decrement' ? -1 : 1)}
          onPress={open}
          style={({ pressed }) => [styles.flex, pressed && styles.pressed]}>
          <ThemedText type="heading" numberOfLines={1}>
            {deck.name}
          </ThemedText>
        </Pressable>
        <IconButton glyph={'✕'} label={`Delete ${deck.name}`} tone="danger" onPress={onDelete} loading={remove.isPending} />
      </View>
      <Pressable accessibilityRole="button" accessibilityLabel={`Open ${deck.name}`} onPress={open} style={({ pressed }) => [styles.summary, pressed && styles.pressed]}>
        {deck.topic ? (
          <ThemedText type="caption" themeColor="textSecondary">
            {deck.topic}
          </ThemedText>
        ) : null}
        <View style={styles.rowBetween}>
          <ThemedText type="small" themeColor="textSecondary" style={styles.flex}>
            {plural(deck.total, 'card')} · {parts.length ? parts.join(' · ') : deck.total > deck.paused ? 'nothing due' : deck.total > 0 ? 'all paused' : 'none approved yet'}
            {deck.problem > 0 ? ` · ${plural(deck.problem, 'problem card')}` : ''}
            {deck.paused > 0 && deck.total > deck.paused ? ` · ${deck.paused} paused` : ''}
          </ThemedText>
          {drafts > 0 ? <Badge label={`${drafts} to review`} tone="warning" /> : null}
        </View>
      </Pressable>
      {error ? <Banner message={error} /> : null}
      {deck.total - deck.paused > 0 ? (
        <View style={styles.studyButtons}>
          {studyable > 0 ? (
            <Button title={`Study (${studyable})`} variant="secondary" style={styles.flex} onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: deck.id } })} />
          ) : null}
          <Button title={`All ${deck.total - deck.paused}`} variant="ghost" style={styles.flex} onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: deck.id, mode: 'all' } })} />
        </View>
      ) : null}
      <DeckSettingsSheet deck={settingsOpen ? { id: deck.id, name: deck.name, folderId: deck.folder_id } : null} onClose={() => setSettingsOpen(false)} onChanged={refreshPhone} />
    </Card>
  );
}

/** The space between deck cards in the list; the same as the gap `Section` puts between its children. */
const DECK_GAP = Spacing.two;

const styles = StyleSheet.create({
  indented: { marginLeft: Spacing.three },
  folderTitle: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two },
  unfiled: { paddingTop: Spacing.two, paddingHorizontal: Spacing.one, borderRadius: 8 },
  unfiledActive: { backgroundColor: 'rgba(120, 130, 255, 0.18)' },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
  deckHeader: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two },
  flex: { flex: 1 },
  studyButtons: { flexDirection: 'row', gap: 8 },
  summary: { gap: Spacing.two },
  pressed: { opacity: 0.75 },
  syncRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
});
