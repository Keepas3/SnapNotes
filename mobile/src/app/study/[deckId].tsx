import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Pressable, StyleSheet, Switch, View } from 'react-native';

import { CardFace, CardMeta } from '@/components/card-face';
import { SwipeCard } from '@/components/swipe-card';
import { ThemedText } from '@/components/themed-text';
import { Banner, Button, Centered, EmptyState, Screen, Spinner } from '@/components/ui';
import { Spacing, type ThemeColor } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { buildOptions, optionLetter, suggestedRating, type Option } from '@/lib/choices';
import { plural } from '@/lib/format';
import { parseStudyScope } from '@/lib/folders';
import type { SwipeDirection } from '@/lib/swipe';
import { describeInterval, previewDue, RATINGS, ratingForSwipe, type RatingValue } from '@/study/scheduler';
import type { LocalCard, StudyCounts } from '@/study/store';
import { useChoicePrefs, useInvalidateLocal, useSetChoicePrefs } from '@/study/hooks';
import { useSync } from '@/sync/provider';

interface Step {
  card: LocalCard | null;
  counts: StudyCounts;
  /** how long each answer would postpone the card, in words, e.g. { 1: "1 minute", 2: "6 minutes", 3: "10 minutes", 4: "3 days" } */
  intervals: Record<RatingValue, string> | null;
  /** when the next studied card comes due ("in 3d"), for the all-caught-up screen */
  nextDue: string | null;
  /** the answers of the deck's other cards, which multiple choice can borrow as wrong answers */
  others: { id: string; back: string }[];
}

const TONE: Record<RatingValue, ThemeColor> = { 1: 'danger', 2: 'warning', 3: 'success', 4: 'tint' };

/** Called from the answer handler only, never while rendering. */
const msSince = (startedAt: number) => Date.now() - startedAt;

export default function StudyScreen() {
  const { deckId } = useLocalSearchParams<{ deckId: string }>();
  // Kept as one value per address, so the effects that depend on it do not run again on every render.
  const scope = useMemo(() => parseStudyScope(deckId), [deckId]);
  const { store, storeError, status, sync } = useSync();
  const invalidateLocal = useInvalidateLocal();

  const [step, setStep] = useState<Step | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [answered, setAnswered] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<number | null>(null); // the option chosen in multiple choice, if any
  const shownAt = useRef(0);
  const choicePrefs = useChoicePrefs();
  const setChoicePrefs = useSetChoicePrefs();
  const prefs = choicePrefs.data;
  const mcOn = prefs?.enabled ?? false;

  // The question as multiple choice, or null when it is off or this card has too few options (then it is studied the usual way).
  // Re-made only when the card or the settings change, so the options hold still while the card is on screen.
  const options = useMemo<Option[] | null>(() => {
    const current = step?.card;
    if (!current || !prefs?.enabled) return null;
    return buildOptions(current, step.others, { count: prefs.count, fillFromDeck: prefs.fillFromDeck, seed: String(answered) });
  }, [step, prefs, answered]);

  /** Work out what to show next, from the on-device database only. */
  const fetchStep = useCallback(async (): Promise<Step> => {
    const now = new Date();
    const { card, counts } = await store!.nextCard(scope, now, await store!.newPerDay());
    if (card) {
      const due = previewDue(card.fsrs, now);
      const intervals = { 1: describeInterval(now, due[1]), 2: describeInterval(now, due[2]), 3: describeInterval(now, due[3]), 4: describeInterval(now, due[4]) };
      return { card, counts, intervals, nextDue: null, others: await store!.otherAnswers(card.deck_id, card.id) };
    }
    const next = await store!.nextDue(scope);
    const nextDue = next ? (new Date(next) <= now ? 'now' : `in ${describeInterval(now, new Date(next))}`) : null;
    return { card: null, counts, intervals: null, nextDue, others: [] };
  }, [store, scope]);

  const show = useCallback((next: Step) => {
    shownAt.current = Date.now();
    setRevealed(false);
    setPicked(null);
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

  /** Choosing an option shows the answer. It does not rate the card: the student still says how well they knew it. */
  function pick(index: number) {
    if (!options || revealed || busy) return;
    setPicked(index);
    setRevealed(true);
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
  const pickedRight = picked !== null && options ? options[picked].correct : null;
  const suggested = suggestedRating(pickedRight);

  return (
    <Screen
      footer={
        revealed && intervals ? (
          <View style={styles.footerStack}>
            <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
              {suggested === 3
                ? 'You picked the right answer. Got it is suggested, but how well did you really know it?'
                : suggested === 1
                  ? 'Not quite. Missed it is suggested, but you decide.'
                  : 'How well did you know it? Under each button is when you will see this card again.'}
            </ThemedText>
            <View style={styles.ratings}>
              {RATINGS.map((r) => (
                <RatingButton
                  key={r.value}
                  label={r.label}
                  interval={intervals[r.value]}
                  tone={TONE[r.value]}
                  suggested={r.value === suggested}
                  disabled={busy}
                  onPress={() => answer(r.value)}
                />
              ))}
            </View>
          </View>
        ) : options ? (
          <Button title="I don't know" variant="secondary" onPress={() => setRevealed(true)} />
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

      <View style={styles.toggleRow}>
        <View style={styles.flex}>
          <ThemedText type="smallBold">Multiple choice</ThemedText>
          <ThemedText type="caption" themeColor="textSecondary">
            {!mcOn
              ? `Turn on to answer from ${prefs?.count ?? 4} options.`
              : options
                ? 'Pick the right answer.'
                : 'No options for this card, so use Show answer.'}
          </ThemedText>
        </View>
        <Switch accessibilityLabel="Multiple choice" value={mcOn} onValueChange={(on) => setChoicePrefs.mutate({ enabled: on })} />
      </View>

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

      {options ? (
        <View style={styles.options} accessibilityRole="radiogroup" accessibilityLabel="Answer options">
          {options.map((option, i) => (
            <OptionButton
              key={i}
              letter={optionLetter(i)}
              text={option.text}
              state={!revealed ? 'open' : option.correct ? 'right' : picked === i ? 'wrong' : 'dim'}
              onPress={() => pick(i)}
            />
          ))}
          {revealed && picked !== null ? (
            <ThemedText type="smallBold" themeColor={pickedRight ? 'success' : 'danger'}>
              {pickedRight ? 'Correct!' : 'Not quite. The right answer is highlighted.'}
            </ThemedText>
          ) : null}
        </View>
      ) : null}

      <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
        {revealed
          ? 'Swipe the card right if you knew it, left if you missed it.'
          : options
            ? 'Pick an answer, or tap I don’t know.'
            : 'Tap the card to see the answer.'}
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

/** One answer choice. After the answer is shown the right one turns green and a wrong pick turns red. */
function OptionButton({ letter, text, state, onPress }: { letter: string; text: string; state: 'open' | 'right' | 'wrong' | 'dim'; onPress: () => void }) {
  const theme = useTheme();
  const colours =
    state === 'right'
      ? { backgroundColor: theme.successBg, borderColor: theme.success }
      : state === 'wrong'
        ? { backgroundColor: theme.dangerBg, borderColor: theme.danger }
        : { backgroundColor: theme.backgroundElement, borderColor: theme.border };
  return (
    <Pressable
      accessibilityRole="radio"
      accessibilityLabel={`${letter}. ${text}${state === 'right' ? '. The right answer.' : state === 'wrong' ? '. Your answer, not correct.' : ''}`}
      accessibilityState={{ disabled: state !== 'open', selected: state === 'wrong' || state === 'right' }}
      disabled={state !== 'open'}
      onPress={onPress}
      style={({ pressed }) => [styles.option, colours, state === 'dim' && styles.dim, pressed && styles.pressed]}>
      <View style={[styles.letter, { backgroundColor: theme.backgroundSelected }]}>
        <ThemedText type="smallBold">{letter}</ThemedText>
      </View>
      <ThemedText style={styles.optionText}>{text}</ThemedText>
    </Pressable>
  );
}

function RatingButton({
  label,
  interval,
  tone,
  suggested,
  disabled,
  onPress,
}: {
  label: string;
  interval: string;
  tone: ThemeColor;
  suggested: boolean;
  disabled: boolean;
  onPress: () => void;
}) {
  const theme = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={`${label}. You will see this card again in ${interval}.${suggested ? ' Suggested.' : ''}`}
      accessibilityState={{ disabled }}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.rating,
        { backgroundColor: theme.backgroundElement, borderColor: suggested ? theme[tone] : theme.border },
        suggested && styles.ratingSuggested,
        disabled && styles.dim,
        pressed && styles.pressed,
      ]}>
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
  ratingSuggested: { borderWidth: 2 },
  toggleRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.three },
  flex: { flex: 1 },
  options: { gap: Spacing.two },
  option: { flexDirection: 'row', alignItems: 'center', gap: Spacing.three, minHeight: 56, borderRadius: 14, borderWidth: 1, paddingVertical: Spacing.two, paddingHorizontal: Spacing.three },
  letter: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  optionText: { flex: 1 },
  dim: { opacity: 0.5 },
  pressed: { opacity: 0.75 },
  center: { textAlign: 'center' },
});
