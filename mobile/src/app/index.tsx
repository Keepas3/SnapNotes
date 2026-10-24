import { useQueryClient } from '@tanstack/react-query';
import { Redirect, Stack, router, useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { Pressable, RefreshControl, StyleSheet, View } from 'react-native';

import { ApiError } from '@/api/client';
import { useDecks, useDeleteDeck, useJobs } from '@/api/hooks';
import type { Job } from '@/api/types';
import { DeckSettingsSheet } from '@/components/deck-settings';
import { DraggableRow } from '@/components/draggable-row';
import { ThemedText } from '@/components/themed-text';
import { Badge, Banner, Button, Card, Centered, EmptyState, IconButton, Screen, Section, Spinner } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { confirm } from '@/lib/confirm';
import { deleteDeckMessage, dragShift, dragTarget, moveItem, orderDecks, type RowBox } from '@/lib/deck';
import { plural } from '@/lib/format';
import { useSettings } from '@/lib/settings';
import { JOB_STATUS_LABEL } from '@/lib/steps';
import { useDeckOrder, useLocalDecks, usePendingReviews, useSetDeckOrder } from '@/study/hooks';
import type { DeckSummary } from '@/study/store';
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

function HeaderSettingsButton() {
  return (
    <Pressable accessibilityRole="button" accessibilityLabel="Settings" onPress={() => router.push('/settings')} hitSlop={12} style={styles.menuButton}>
      <ThemedText type="linkPrimary" style={styles.menuGlyph}>
        ☰
      </ThemedText>
    </Pressable>
  );
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
  const [refreshing, setRefreshing] = useState(false);
  // While a deck is being dragged: which one, where it would land, how far the finger moved, and how far the others slide.
  const [drag, setDrag] = useState<{ from: number; to: number; dy: number; step: number } | null>(null);
  const [held, setHeld] = useState(false); // a finger has been held on a deck long enough that it may be dragged: the page must not scroll
  // Each deck card's height, filled in as they are laid out. Heights only: a card's position is worked out from them, because a
  // layout event is not sent when a card merely moves (it would be stale after a reorder).
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
  const decks = orderDecks(local.data ?? [], savedOrder.data ?? []);
  const waiting = decks.reduce((n, d) => n + d.new + d.learning + d.review, 0);

  /** Where each card sits, top to bottom in the order shown now (relative to the first). */
  function rowBoxes(): RowBox[] {
    let y = 0;
    return decks.map((d) => {
      const height = heights.current.get(d.id) ?? 0;
      const box = { y, height };
      y += height + DECK_GAP;
      return box;
    });
  }

  function dragStart(from: number) {
    setDrag({ from, to: from, dy: 0, step: (rowBoxes()[from]?.height ?? 0) + DECK_GAP });
  }
  function dragMove(from: number, dy: number) {
    setDrag((d) => (d ? { ...d, from, dy, to: dragTarget(rowBoxes(), from, dy) } : d));
  }
  function dragEnd(from: number, dy: number) {
    const to = dragTarget(rowBoxes(), from, dy);
    setDrag(null);
    if (to !== from) saveOrder.mutate(moveItem(decks, from, to).map((d) => d.id));
  }
  function nudge(from: number, direction: -1 | 1) {
    const to = from + direction;
    if (to >= 0 && to < decks.length) saveOrder.mutate(moveItem(decks, from, to).map((d) => d.id));
  }

  const onlineError = serverDecks.error ?? jobs.error;
  const offline = (onlineError instanceof ApiError && onlineError.isNetwork) || status.offline;

  return (
    <Screen
      scrollEnabled={drag === null && !held}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} />}
      footer={<Button title="New deck from notes" variant={waiting > 0 ? 'secondary' : 'primary'} onPress={() => router.push('/new')} />}>
      <Stack.Screen options={{ title: 'SnapNotes', headerLeft: HeaderSettingsButton }} />

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

      <Section title="Your decks">
        {local.isPending ? (
          <Spinner />
        ) : decks.length === 0 ? (
          <EmptyState
            title={status.syncing ? 'Loading your decks…' : 'No decks yet'}
            message="Upload a PDF, slides, a document, or photos of your notes and SnapNotes will turn them into flashcards."
          />
        ) : (
          decks.map((deck, i) => (
            // Press and hold a deck, then drag: the held deck follows the finger and the ones it passes make room.
            <DraggableRow
              key={deck.id}
              onLayout={(e) => heights.current.set(deck.id, e.nativeEvent.layout.height)}
              onHold={() => setHeld(true)}
              onLift={() => dragStart(i)}
              onMove={(dy) => dragMove(i, dy)}
              onDrop={(dy) => {
                dragEnd(i, dy);
                setHeld(false);
              }}
              onCancel={() => {
                setDrag(null);
                setHeld(false);
              }}
              style={
                drag === null
                  ? undefined
                  : drag.from === i
                    ? { transform: [{ translateY: drag.dy }], zIndex: 10, opacity: 0.92 }
                    : { transform: [{ translateY: dragShift(i, drag.from, drag.to, drag.step) }] }
              }>
              <DeckRow deck={deck} drafts={draftCounts.get(deck.id) ?? 0} onMove={decks.length > 1 ? (direction) => nudge(i, direction) : undefined} />
            </DraggableRow>
          ))
        )}
      </Section>

      <SyncFooter pending={pending.data ?? 0} />
    </Screen>
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
            {plural(deck.total, 'card')} · {parts.length ? parts.join(' · ') : deck.total > 0 ? 'nothing due' : 'none approved yet'}
          </ThemedText>
          {drafts > 0 ? <Badge label={`${drafts} to review`} tone="warning" /> : null}
        </View>
      </Pressable>
      {error ? <Banner message={error} /> : null}
      {studyable > 0 ? (
        <Button title={`Study (${studyable})`} variant="secondary" onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: deck.id } })} />
      ) : null}
      <DeckSettingsSheet deck={settingsOpen ? { id: deck.id, name: deck.name } : null} onClose={() => setSettingsOpen(false)} onChanged={refreshPhone} />
    </Card>
  );
}

/** The space between deck cards in the list; the same as the gap `Section` puts between its children. */
const DECK_GAP = Spacing.two;

const styles = StyleSheet.create({
  menuButton: { paddingHorizontal: Spacing.two, paddingVertical: Spacing.one },
  menuGlyph: { fontSize: 24, lineHeight: 28 },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
  deckHeader: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two },
  flex: { flex: 1 },
  summary: { gap: Spacing.two },
  pressed: { opacity: 0.75 },
  syncRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
});
