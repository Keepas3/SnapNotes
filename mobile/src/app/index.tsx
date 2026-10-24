import { useQueryClient } from '@tanstack/react-query';
import { Redirect, Stack, router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Pressable, RefreshControl, StyleSheet, View } from 'react-native';

import { ApiError } from '@/api/client';
import { useDecks, useDeleteDeck, useJobs } from '@/api/hooks';
import type { Job } from '@/api/types';
import { DeckSettingsSheet } from '@/components/deck-settings';
import { ThemedText } from '@/components/themed-text';
import { Badge, Banner, Button, Card, Centered, EmptyState, IconButton, Screen, Section, Spinner } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { confirm } from '@/lib/confirm';
import { deleteDeckMessage } from '@/lib/deck';
import { plural } from '@/lib/format';
import { useSettings } from '@/lib/settings';
import { JOB_STATUS_LABEL } from '@/lib/steps';
import { useLocalDecks, usePendingReviews } from '@/study/hooks';
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
    <Pressable accessibilityRole="button" accessibilityLabel="Settings" onPress={() => router.push('/settings')} hitSlop={12}>
      <ThemedText type="linkPrimary">Settings</ThemedText>
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
  const [refreshing, setRefreshing] = useState(false);

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
  const waiting = decks.reduce((n, d) => n + d.new + d.learning + d.review, 0);

  const onlineError = serverDecks.error ?? jobs.error;
  const offline = (onlineError instanceof ApiError && onlineError.isNetwork) || status.offline;

  return (
    <Screen
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} />}
      footer={<Button title="New deck from notes" variant={waiting > 0 ? 'secondary' : 'primary'} onPress={() => router.push('/new')} />}>
      <Stack.Screen options={{ title: 'SnapNotes', headerRight: HeaderSettingsButton }} />

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
          decks.map((deck) => <DeckRow key={deck.id} deck={deck} drafts={draftCounts.get(deck.id) ?? 0} />)
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

function DeckRow({ deck, drafts }: { deck: DeckSummary; drafts: number }) {
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
        <IconButton glyph={'⚙︎'} label={`Settings for ${deck.name}`} onPress={() => setSettingsOpen(true)} />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`${deck.name}, ${plural(deck.total, 'card')}`}
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
      <DeckSettingsSheet deck={settingsOpen ? { id: deck.id, name: deck.name } : null} onClose={() => setSettingsOpen(false)} onRenamed={refreshPhone} />
    </Card>
  );
}

const styles = StyleSheet.create({
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
  deckHeader: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two },
  flex: { flex: 1 },
  summary: { gap: Spacing.two },
  pressed: { opacity: 0.75 },
  syncRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
});
