import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { CardFace, CardMeta } from '@/components/card-face';
import { SwipeCard } from '@/components/swipe-card';
import { ThemedText } from '@/components/themed-text';
import { Banner, Button, Centered, EmptyState, Screen, Spinner } from '@/components/ui';
import { Spacing, type ThemeColor } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { plural } from '@/lib/format';
import type { SwipeDirection } from '@/lib/swipe';
import { describeInterval, previewDue, RATINGS, ratingForSwipe, type RatingValue } from '@/study/scheduler';
import type { LocalCard, StudyCounts } from '@/study/store';
import { useInvalidateLocal } from '@/study/hooks';
import { useSync } from '@/sync/provider';

interface Step {
  card: LocalCard | null;
  counts: StudyCounts;
  /** how long each answer would postpone the card, in words, e.g. { 1: "1 minute", 2: "6 minutes", 3: "10 minutes", 4: "3 days" } */
  intervals: Record<RatingValue, string> | null;
  /** when the next studied card comes due ("in 3d"), for the all-caught-up screen */
  nextDue: string | null;
}

const TONE: Record<RatingValue, ThemeColor> = { 1: 'danger', 2: 'warning', 3: 'success', 4: 'tint' };

/** Called from the answer handler only, never while rendering. */
const msSince = (startedAt: number) => Date.now() - startedAt;

export default function StudyScreen() {
  const { deckId } = useLocalSearchParams<{ deckId: string }>();
  const scope = deckId === 'all' ? null : deckId;
  const { store, storeError, status, sync } = useSync();
  const invalidateLocal = useInvalidateLocal();

  const [step, setStep] = useState<Step | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [answered, setAnswered] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shownAt = useRef(0);

  /** Work out what to show next, from the on-device database only. */
  const fetchStep = useCallback(async (): Promise<Step> => {
    const now = new Date();
    const { card, counts } = await store!.nextCard(scope, now, await store!.newPerDay());
    if (card) {
      const due = previewDue(card.fsrs, now);
      const intervals = { 1: describeInterval(now, due[1]), 2: describeInterval(now, due[2]), 3: describeInterval(now, due[3]), 4: describeInterval(now, due[4]) };
      return { card, counts, intervals, nextDue: null };
    }
    const next = await store!.nextDue(scope);
    const nextDue = next ? (new Date(next) <= now ? 'now' : `in ${describeInterval(now, new Date(next))}`) : null;
    return { card: null, counts, intervals: null, nextDue };
  }, [store, scope]);

  const show = useCallback((next: Step) => {
    shownAt.current = Date.now();
    setRevealed(false);
    setStep(next);
  }, []);

  useEffect(() => {
    if (!store) return;
    let cancelled = false;
    fetchStep().then((next) => {
      if (!cancelled) show(next);
    });
    return () => {
      cancelled = true;
    };
  }, [store, fetchStep, show]);

  // Start catching up in the background; studying never waits for it.
  useEffect(() => {
    void sync({ minIntervalMs: 30_000 });
  }, [sync]);

  // When the session is over, push the reviews right away rather than waiting for the next trigger.
  const finished = step !== null && step.card === null;
  useEffect(() => {
    if (finished) void sync();
  }, [finished, sync]);

  async function answer(rating: RatingValue) {
    if (!store || !step?.card || busy) return;
    setBusy(true);
    setError(null);
    try {
      await store.answer(step.card.id, rating, new Date(), msSince(shownAt.current));
      setAnswered((n) => n + 1);
      invalidateLocal();
      show(await fetchStep());
      void sync({ minIntervalMs: 20_000 });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save that answer.');
    } finally {
      setBusy(false);
    }
  }

  /** A swipe is the same as tapping a button: right is "got it", left is "missed it". */
  function onSwiped(direction: SwipeDirection) {
    void answer(ratingForSwipe(direction));
  }

  if (storeError) {
    return (
      <Screen>
        <Banner message={`Offline study is unavailable: ${storeError}`} />
      </Screen>
    );
  }
  if (!step) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }

  const { card, counts, intervals, nextDue } = step;
  const total = counts.new + counts.learning + counts.review;

  if (!card) {
    const never = counts.new === 0 && nextDue === null && answered === 0;
    return (
      <Screen footer={<Button title="Done" onPress={() => (router.canGoBack() ? router.back() : router.replace('/'))} />}>
        <Stack.Screen options={{ title: 'Study' }} />
        <EmptyState
          title={never ? 'Nothing to study yet' : 'All caught up'}
          message={
            never
              ? 'Approve some cards, then sync, and they will show up here.'
              : answered > 0
                ? `You gave ${plural(answered, 'answer')} this session.${nextDue ? ` The next card is due ${nextDue}.` : ''}`
                : nextDue
                  ? `Nothing is due right now. The next card is due ${nextDue}.`
                  : 'Nothing is due right now.'
          }
        />
        <SyncLine />
      </Screen>
    );
  }

  const kind = card.fsrs.state === 0 ? 'new' : card.fsrs.state === 2 ? 'review' : 'learning';

  return (
    <Screen
      footer={
        revealed && intervals ? (
          <View style={styles.footerStack}>
            <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
              How well did you know it? Under each button is when you will see this card again.
            </ThemedText>
            <View style={styles.ratings}>
              {RATINGS.map((r) => (
                <RatingButton key={r.value} label={r.label} interval={intervals[r.value]} tone={TONE[r.value]} disabled={busy} onPress={() => answer(r.value)} />
              ))}
            </View>
          </View>
        ) : (
          <Button title="Show answer" onPress={() => setRevealed(true)} />
        )
      }>
      <Stack.Screen options={{ title: 'Study' }} />

      <View style={styles.counts} accessibilityLabel={`${counts.new} new, ${counts.learning} learning, ${counts.review} to review`}>
        <Count label="New" value={counts.new} active={kind === 'new'} />
        <Count label="Learning" value={counts.learning} active={kind === 'learning'} />
        <Count label="Review" value={counts.review} active={kind === 'review'} />
      </View>
      {total === 0 ? (
        <ThemedText type="caption" themeColor="textSecondary">
          Finishing the cards you are still learning.
        </ThemedText>
      ) : null}

      {error ? <Banner message={error} /> : null}

      {/* Keyed by the answer count too, so a card that comes straight back is drawn afresh instead of staying flung off-screen. */}
      <SwipeCard
        key={`${card.id}-${answered}`}
        swipeable={revealed && !busy}
        rightLabel="GOT IT"
        leftLabel="MISSED IT"
        accessibilityHint={revealed ? 'Swipe right if you knew it, left if you missed it.' : 'Tap to see the answer.'}
        onTap={() => setRevealed(true)}
        onSwiped={onSwiped}>
        <CardMeta card={card} />
        <View style={styles.face}>
          <CardFace card={card} side={revealed ? 'back' : 'front'} />
        </View>
        {revealed && card.source_snippet ? (
          <ThemedText type="caption" themeColor="textSecondary">
            Source: “{card.source_snippet}”
          </ThemedText>
        ) : null}
      </SwipeCard>
      <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
        {revealed ? 'Swipe the card right if you knew it, left if you missed it.' : 'Tap the card to see the answer.'}
      </ThemedText>

      {status.offline ? (
        <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
          Offline: your answers are saved on this device and will sync later.
        </ThemedText>
      ) : null}
    </Screen>
  );
}

function Count({ label, value, active }: { label: string; value: number; active: boolean }) {
  return (
    <View style={styles.count}>
      <ThemedText type="heading" themeColor={active ? 'tint' : 'text'} style={active ? styles.underline : undefined}>
        {value}
      </ThemedText>
      <ThemedText type="caption" themeColor="textSecondary">
        {label}
      </ThemedText>
    </View>
  );
}

function RatingButton({ label, interval, tone, disabled, onPress }: { label: string; interval: string; tone: ThemeColor; disabled: boolean; onPress: () => void }) {
  const theme = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={`${label}. You will see this card again in ${interval}.`}
      accessibilityState={{ disabled }}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [styles.rating, { backgroundColor: theme.backgroundElement, borderColor: theme.border }, disabled && styles.dim, pressed && styles.pressed]}>
      <ThemedText type="smallBold" themeColor={tone}>
        {label}
      </ThemedText>
      <ThemedText type="caption" themeColor="textSecondary" style={styles.ratingTime}>
        {interval}
      </ThemedText>
    </Pressable>
  );
}

function SyncLine() {
  const { status } = useSync();
  if (status.offline) {
    return (
      <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
        Offline. Your answers will sync when the server is reachable.
      </ThemedText>
    );
  }
  if (status.error) return <Banner tone="warning" message={`Sync problem: ${status.error}`} />;
  return (
    <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
      {status.syncing ? 'Syncing…' : status.lastSyncedAt ? 'Synced.' : ''}
    </ThemedText>
  );
}

const styles = StyleSheet.create({
  counts: { flexDirection: 'row', justifyContent: 'space-around' },
  count: { alignItems: 'center', minWidth: 64 },
  underline: { textDecorationLine: 'underline' },
  face: { flexGrow: 1, justifyContent: 'center', paddingVertical: Spacing.three },
  footerStack: { gap: Spacing.two },
  ratings: { flexDirection: 'row', gap: Spacing.two },
  rating: { flex: 1, minHeight: 60, borderRadius: 14, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: 2, alignItems: 'center', justifyContent: 'center', gap: 2 },
  ratingTime: { textAlign: 'center' },
  dim: { opacity: 0.5 },
  pressed: { opacity: 0.75 },
  center: { textAlign: 'center' },
});
