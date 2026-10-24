import { useQueryClient } from '@tanstack/react-query';
import { Stack, router, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { useDeck, useDeckCards, useDeleteDeck } from '@/api/hooks';
import type { Card, CardStatus } from '@/api/types';
import { CardEditor } from '@/components/card-editor';
import { ThemedText } from '@/components/themed-text';
import { Badge, Banner, Button, Card as Panel, Centered, Chip, EmptyState, Screen, Section, Spinner } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { clozeQuestion } from '@/lib/cloze';
import { confirm } from '@/lib/confirm';
import { exportDeck } from '@/lib/exportDeck';
import { describeExportFailure, EXPORT_FORMATS, exportFileName, type ExportFormat } from '@/lib/exportFile';
import { plural } from '@/lib/format';
import { originBadge } from '@/lib/origin';
import { useApi } from '@/lib/settings';
import { useStudyCounts } from '@/study/hooks';
import { LOCAL_KEY, useSync } from '@/sync/provider';

export default function DeckScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const deck = useDeck(id);
  // Until the user picks a tab, open on "To review" if anything is waiting there.
  const [picked, setPicked] = useState<CardStatus | null>(null);
  const filter: CardStatus = picked ?? ((deck.data?.counts.draft ?? 0) > 0 ? 'draft' : 'approved');
  const setFilter = setPicked;
  const cards = useDeckCards(id, filter);
  const drafts = useDeckCards(id, 'draft'); // also tells us which job to open for review
  const study = useStudyCounts(id); // from the on-device database
  const remove = useDeleteDeck();

  // Cards approved or edited a moment ago reach the phone through a sync.
  const queryClient = useQueryClient();
  const { sync } = useSync();
  useFocusEffect(
    useCallback(() => {
      queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
      void sync({ minIntervalMs: 5_000 });
    }, [queryClient, sync]),
  );
  const [editing, setEditing] = useState<Card | null>(null);
  const api = useApi();
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportMessage, setExportMessage] = useState<{ tone: 'success' | 'danger'; text: string } | null>(null);

  if (deck.isPending) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }
  if (deck.isError) {
    return (
      <Screen>
        <Banner message={deck.error.message} actionLabel="Try again" onAction={() => deck.refetch()} />
      </Screen>
    );
  }

  const d = deck.data;
  const reviewJobId = drafts.data?.find((c) => c.job_id)?.job_id ?? null;
  const studyable = (study.data?.new ?? 0) + (study.data?.learning ?? 0) + (study.data?.review ?? 0);

  async function onExport(format: ExportFormat) {
    setExporting(format);
    setExportMessage(null);
    try {
      const outcome = await exportDeck(api, d.id, d.name, format);
      // The share sheet is its own confirmation; a plain download (browser preview, or no share sheet) needs a note.
      if (outcome === 'saved') setExportMessage({ tone: 'success', text: `Downloaded ${exportFileName(d.name, format)}.` });
    } catch (e) {
      setExportMessage({ tone: 'danger', text: describeExportFailure(e) });
    } finally {
      setExporting(null);
    }
  }

  async function onDelete() {
    if (!(await confirm(`Delete “${d.name}”?`, `All ${plural(d.counts.approved + d.counts.draft + d.counts.rejected, 'card')} in it will be removed.`, 'Delete', true))) return;
    await remove.mutateAsync(d.id);
    router.replace('/');
  }

  return (
    <Screen
      footer={
        <>
          {reviewJobId && d.counts.draft > 0 ? (
            <Button
              title={`Review ${plural(d.counts.draft, 'new card')}`}
              variant={studyable > 0 ? 'secondary' : 'primary'}
              onPress={() => router.push({ pathname: '/jobs/[id]/review', params: { id: reviewJobId } })}
            />
          ) : null}
          <Button
            title={studyable > 0 ? `Study (${studyable})` : study.data ? 'Nothing to study right now' : 'Study'}
            disabled={studyable === 0}
            onPress={() => router.push({ pathname: '/study/[deckId]', params: { deckId: id } })}
          />
        </>
      }>
      <Stack.Screen options={{ title: d.name }} />

      {d.topic ? (
        <ThemedText type="small" themeColor="textSecondary">
          {d.topic}
        </ThemedText>
      ) : null}

      <View style={styles.chips}>
        <Chip label={`Approved (${d.counts.approved})`} selected={filter === 'approved'} onPress={() => setFilter('approved')} />
        <Chip label={`To review (${d.counts.draft})`} selected={filter === 'draft'} onPress={() => setFilter('draft')} />
        <Chip label={`Set aside (${d.counts.rejected})`} selected={filter === 'rejected'} onPress={() => setFilter('rejected')} />
      </View>

      {cards.isPending ? (
        <Spinner />
      ) : cards.isError ? (
        <Banner message={cards.error.message} actionLabel="Try again" onAction={() => cards.refetch()} />
      ) : cards.data.length === 0 ? (
        <EmptyState
          title={filter === 'approved' ? 'No approved cards yet' : filter === 'draft' ? 'Nothing to review' : 'Nothing set aside'}
          message={filter === 'approved' ? 'Approve cards while reviewing and they will appear here.' : undefined}
        />
      ) : (
        cards.data.map((card) => (
          <Panel key={card.id} accessibilityLabel={`Edit card: ${card.front}`} onPress={() => setEditing(card)}>
            <ThemedText type="smallBold">{card.type === 'cloze' ? clozeQuestion(card.front) : card.front}</ThemedText>
            <View style={styles.badges}>
              <Badge {...originBadge(card.origin)} />
              <Badge label={card.type} />
              {card.issue ? <Badge label={card.issue} tone="warning" /> : null}
            </View>
          </Panel>
        ))
      )}

      <Section title="Export">
        <ThemedText type="caption" themeColor="textSecondary">
          Approved cards only. In Anki they arrive as new cards (your study history stays in SnapNotes), and importing the same deck again updates the cards
          instead of duplicating them.
        </ThemedText>
        {(['apkg', 'csv'] as const).map((format) => (
          <Button
            key={format}
            title={EXPORT_FORMATS[format].label}
            variant="secondary"
            disabled={d.counts.approved === 0 || (exporting !== null && exporting !== format)}
            loading={exporting === format}
            onPress={() => onExport(format)}
          />
        ))}
        {d.counts.approved === 0 ? (
          <ThemedText type="caption" themeColor="textSecondary">
            Approve some cards to export them.
          </ThemedText>
        ) : null}
        {exportMessage ? <Banner tone={exportMessage.tone} message={exportMessage.text} /> : null}
      </Section>

      <Button title="Delete deck" variant="ghost" onPress={onDelete} loading={remove.isPending} />

      <CardEditor card={editing} onClose={() => setEditing(null)} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
});
