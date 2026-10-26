import * as Haptics from 'expo-haptics';
import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Linking, Platform, Pressable, StyleSheet, View } from 'react-native';

import { useInvalidateLibrary, useJob, useJobCards, usePatchCard, useReviewCards } from '@/api/hooks';
import type { Card } from '@/api/types';
import { CardFace, CardMeta } from '@/components/card-face';
import { CardEditor } from '@/components/card-editor';
import { CardBackdrop, SwipeCard, type SwipeCardHandle } from '@/components/swipe-card';
import { SuspectClaims } from '@/components/suspect-claims';
import { ThemedText } from '@/components/themed-text';
import { useLoadGate } from '@/components/load-gate';
import { Badge, Banner, Button, Card as Panel, EmptyState, Screen, Section } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { clozeQuestion } from '@/lib/cloze';
import { plural, safeHttpUrl } from '@/lib/format';
import { sourceName } from '@/lib/origin';
import type { SwipeDirection } from '@/lib/swipe';
import { cardWarnings } from '@/lib/warnings';

interface Decision {
  card: Card;
  decision: 'approved' | 'rejected';
}

function buzz() {
  if (Platform.OS === 'web') return; // the browser's vibrate API is blocked until the user has tapped, and is pointless on desktop
  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
}

export default function ReviewScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const job = useJob(id);
  const cards = useJobCards(id);

  const gate = useLoadGate({ pending: cards.isPending, error: cards.error }, 'these cards', () => void cards.refetch());
  if (gate) return gate;
  if (!cards.data) return null;
  const understanding = job.data?.understanding;
  return <ReviewSession cards={cards.data} deckId={job.data?.deck_id ?? null} suspectClaims={understanding?.suspect_claims} researchNote={understanding?.research_note} />;
}

/** One pass through a job's draft cards. It starts from the cards as loaded and then keeps its own queue. */
function ReviewSession({ cards, deckId, suspectClaims, researchNote }: { cards: Card[]; deckId: string | null; suspectClaims?: string[]; researchNote?: string }) {
  const review = useReviewCards();
  const patch = usePatchCard();
  const invalidate = useInvalidateLibrary();

  // Snapshots, so edits, swipes and background refetches never reshuffle the pile under the user's thumb.
  const [queue, setQueue] = useState<Card[]>(() => cards.filter((c) => c.status === 'draft'));
  const [setAside, setSetAside] = useState<Card[]>(() => cards.filter((c) => c.status === 'rejected')); // filtered out by the checker
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [history, setHistory] = useState<Decision[]>([]);
  const [unsaved, setUnsaved] = useState<Decision[]>([]);
  const [editing, setEditing] = useState<Card | null>(null);
  const swipeRef = useRef<SwipeCardHandle>(null);

  // Home and the deck screen show counts, so refresh them when leaving.
  const invalidateRef = useRef(invalidate);
  useEffect(() => {
    invalidateRef.current = invalidate;
  });
  useEffect(() => () => invalidateRef.current(), []);

  const current = queue[index];
  const done = index >= queue.length;

  function save(d: Decision) {
    review.mutate(d.decision === 'approved' ? { approve: [d.card.id] } : { reject: [d.card.id] }, {
      onError: () => setUnsaved((u) => [...u, d]),
    });
  }

  function onSwiped(direction: SwipeDirection) {
    const d: Decision = { card: queue[index], decision: direction === 'right' ? 'approved' : 'rejected' };
    buzz();
    setHistory((h) => [...h, d]);
    setIndex((i) => i + 1);
    setFlipped(false);
    save(d);
  }

  function undo() {
    const last = history[history.length - 1];
    if (!last) return;
    setHistory((h) => h.slice(0, -1));
    setUnsaved((u) => u.filter((x) => x.card.id !== last.card.id));
    setIndex((i) => Math.max(0, i - 1));
    setFlipped(false);
    patch.mutate({ id: last.card.id, patch: { status: 'draft' } });
  }

  async function retryUnsaved() {
    const batch = unsaved;
    setUnsaved([]);
    try {
      await review.mutateAsync({
        approve: batch.filter((d) => d.decision === 'approved').map((d) => d.card.id),
        reject: batch.filter((d) => d.decision === 'rejected').map((d) => d.card.id),
      });
    } catch {
      setUnsaved(batch);
    }
  }

  function restore(card: Card) {
    patch.mutate(
      { id: card.id, patch: { status: 'draft' } },
      {
        onSuccess: (saved) => {
          setSetAside((s) => s.filter((c) => c.id !== card.id));
          setQueue((q) => [...q, saved]);
        },
      },
    );
  }

  const approvedCount = history.filter((d) => d.decision === 'approved').length;
  const rejectedCount = history.length - approvedCount;

  return (
    <Screen
      scroll
      footer={
        !done ? (
          <>
            <View style={styles.actions}>
              <Button title="✕  Reject" variant="danger" onPress={() => swipeRef.current?.swipe('left')} style={styles.actionBtn} />
              <Button title="Edit" variant="secondary" onPress={() => setEditing(current)} style={styles.editBtn} />
              <Button title="✓  Approve" onPress={() => swipeRef.current?.swipe('right')} style={styles.actionBtn} />
            </View>
            {history.length > 0 ? <Button title="Undo last" variant="ghost" onPress={undo} /> : null}
          </>
        ) : undefined
      }>
      <Stack.Screen options={{ title: 'Review cards' }} />

      {unsaved.length > 0 ? (
        <Banner message={`${plural(unsaved.length, 'decision')} could not be saved.`} actionLabel="Retry" onAction={retryUnsaved} />
      ) : null}

      {/* The job screen hands over to this one as soon as the job finishes, so the warning has to be here too. */}
      <SuspectClaims claims={suspectClaims} note={researchNote} collapsible />

      {!done ? (
        <>
          <ProgressBar done={index} total={queue.length} />
          <View style={styles.stack}>
            <CardBackdrop />
            <SwipeCard key={current.id} ref={swipeRef} onSwiped={onSwiped} onTap={() => setFlipped((f) => !f)}>
              <CardMeta card={current} />
              <View style={styles.face}>
                <CardFace card={current} side={flipped ? 'back' : 'front'} />
              </View>
              <CardFooter card={current} flipped={flipped} />
            </SwipeCard>
          </View>
          <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
            Swipe right to keep, left to discard. Tap the card to flip it.
          </ThemedText>
        </>
      ) : (
        <>
          <EmptyState
            title={queue.length === 0 ? 'No cards to review' : 'All reviewed'}
            message={queue.length === 0 ? 'Nothing is waiting for approval.' : `${plural(approvedCount, 'card')} approved, ${rejectedCount} discarded.`}
          />
          <View style={styles.doneActions}>
            {deckId ? <Button title="Open deck" onPress={() => router.replace({ pathname: '/decks/[id]', params: { id: deckId } })} /> : null}
            <Button title="Back to home" variant="secondary" onPress={() => router.replace('/')} />
            {history.length > 0 ? <Button title="Undo last" variant="ghost" onPress={undo} /> : null}
          </View>

          {setAside.length > 0 ? (
            <Section title={`Set aside by the checker (${setAside.length})`}>
              <ThemedText type="caption" themeColor="textSecondary">
                These were left out because they looked duplicated or were not backed by your notes or the approved sources. Restore any you want to review.
              </ThemedText>
              {setAside.map((card) => (
                <Panel key={card.id}>
                  <ThemedText type="smallBold">{card.type === 'cloze' ? clozeQuestion(card.front) : card.front}</ThemedText>
                  <Badge label={card.duplicate_of ? 'Looks like a duplicate' : card.issue || 'Not verified'} tone="warning" />
                  <Button title="Restore to review" variant="secondary" onPress={() => restore(card)} loading={patch.isPending && patch.variables?.id === card.id} />
                </Panel>
              ))}
            </Section>
          ) : null}
        </>
      )}

      <CardEditor
        card={editing}
        onClose={() => setEditing(null)}
        onSaved={(saved) => setQueue((q) => q.map((c) => (c.id === saved.id ? saved : c)))}
        onDeleted={(gone) => setQueue((q) => q.filter((c) => c.id !== gone.id))}
      />
    </Screen>
  );
}

function ProgressBar({ done, total }: { done: number; total: number }) {
  const theme = useTheme();
  const pct = total === 0 ? 0 : Math.min(100, (done / total) * 100);
  return (
    <View style={styles.progress}>
      <ThemedText type="small" themeColor="textSecondary">
        Card {Math.min(done + 1, total)} of {total}
      </ThemedText>
      <View style={[styles.track, { backgroundColor: theme.backgroundSelected }]}>
        <View style={[styles.fill, { width: `${pct}%`, backgroundColor: theme.tint }]} />
      </View>
    </View>
  );
}

function CardFooter({ card, flipped }: { card: Card; flipped: boolean }) {
  const link = card.origin === 'web' ? safeHttpUrl(card.source_ref) : null;
  const { badges, note } = cardWarnings(card);
  return (
    <View style={styles.footerStack}>
      {badges.map((w) => (
        <Badge key={w} label={w} tone="warning" />
      ))}
      {note ? <Banner tone="warning" message={note} /> : null}
      {flipped && card.source_snippet ? (
        <ThemedText type="caption" themeColor="textSecondary">
          Source: “{card.source_snippet}”
        </ThemedText>
      ) : null}
      {flipped && link ? (
        <Pressable accessibilityRole="link" onPress={() => Linking.openURL(link)} hitSlop={8}>
          <ThemedText type="linkPrimary">Open {sourceName(link)}</ThemedText>
        </Pressable>
      ) : null}
      {!flipped ? (
        <ThemedText type="caption" themeColor="textSecondary">
          Tap to see the answer
        </ThemedText>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  progress: { gap: Spacing.one },
  track: { height: 6, borderRadius: 3, overflow: 'hidden' },
  fill: { height: 6, borderRadius: 3 },
  stack: { width: '100%', maxWidth: 520, alignSelf: 'center', paddingBottom: Spacing.three },
  face: { flexGrow: 1, justifyContent: 'center', paddingVertical: Spacing.three },
  footerStack: { gap: Spacing.two, alignItems: 'flex-start' },
  center: { textAlign: 'center' },
  actions: { flexDirection: 'row', gap: Spacing.two },
  actionBtn: { flex: 2 },
  editBtn: { flex: 1 },
  doneActions: { gap: Spacing.two },
});
