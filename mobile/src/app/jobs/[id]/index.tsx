import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';

import { useDeck, useDeleteDocument, useEvidence, useJob, useJobCards, useRetryJob } from '@/api/hooks';
import { SuspectClaims } from '@/components/suspect-claims';
import { ThemedText } from '@/components/themed-text';
import { useLoadGate } from '@/components/load-gate';
import { Badge, Banner, Button, Card, Screen, Section } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { confirm } from '@/lib/confirm';
import { formatCost, plural } from '@/lib/format';
import { isActive, JOB_STATUS_LABEL, stepsFor, type Step } from '@/lib/steps';

export default function JobScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const job = useJob(id);
  const status = job.data?.status;

  const evidence = useEvidence(id, status === 'awaiting_approval');
  const target = useDeck(job.data?.target_deck_id); // the deck this upload is adding to, when it is not making a new one
  const jobCards = useJobCards(id, undefined, status === 'completed'); // this job's own cards: the deck may hold others too
  const retry = useRetryJob(id);
  const discard = useDeleteDocument();

  // Take the user to the next screen when a job *they were watching* needs them or finishes.
  const previous = useRef<typeof status>(undefined);
  useEffect(() => {
    if (!job.data) return;
    const prev = previous.current;
    previous.current = job.data.status;
    if (prev === undefined || !isActive(prev)) return;
    if (job.data.status === 'awaiting_approval') router.push({ pathname: '/jobs/[id]/sources', params: { id } });
    else if (job.data.status === 'completed') router.push({ pathname: '/jobs/[id]/review', params: { id } });
  }, [job.data, id]);

  const gate = useLoadGate({ pending: job.isPending, error: job.error }, 'this upload', () => void job.refetch());
  if (gate) return gate;
  if (!job.data) return null;

  const j = job.data;
  const label = JOB_STATUS_LABEL[j.status];
  const steps = stepsFor(j);

  async function onDiscard() {
    if (!(await confirm('Discard this upload?', 'The uploaded files and any cards you have not approved will be deleted.', 'Discard', true))) return;
    await discard.mutateAsync(j.document_id);
    router.replace('/');
  }

  const generated = (jobCards.data ?? []).filter((c) => c.status !== 'rejected'); // set-aside cards (repeats, unsupported ones) are not counted
  const draftCount = generated.filter((c) => c.status === 'draft').length;
  const addingTo = target.data?.name;
  const title = j.target_deck_id ? (addingTo ? `Adding to “${addingTo}”` : 'Adding to a deck') : (j.understanding?.main_topic ?? 'New deck');

  return (
    <Screen
      footer={
        j.status === 'awaiting_approval' ? (
          <Button
            title={`Review ${plural(evidence.data?.evidence.length ?? 0, 'web source')}`}
            onPress={() => router.push({ pathname: '/jobs/[id]/sources', params: { id } })}
          />
        ) : j.status === 'completed' ? (
          <>
            {draftCount > 0 ? (
              <Button title={`Review ${plural(draftCount, 'new card')}`} onPress={() => router.push({ pathname: '/jobs/[id]/review', params: { id } })} />
            ) : null}
            {j.deck_id ? (
              <Button
                title="Open deck"
                variant={draftCount > 0 ? 'secondary' : 'primary'}
                onPress={() => router.replace({ pathname: '/decks/[id]', params: { id: j.deck_id! } })}
              />
            ) : null}
          </>
        ) : j.status === 'failed' ? (
          <Button title="Try again" onPress={() => retry.mutate()} loading={retry.isPending} />
        ) : undefined
      }>
      <Stack.Screen options={{ title }} />

      <Card>
        <View style={styles.between}>
          <ThemedText type="smallBold" numberOfLines={1} style={styles.flex}>
            {j.filename}
          </ThemedText>
          <Badge label={label.label} tone={label.tone} />
        </View>
        {j.understanding ? (
          <ThemedText type="caption" themeColor="textSecondary">
            {j.understanding.subject_area} · {plural(j.understanding.subtopics.length, 'subtopic')}
          </ThemedText>
        ) : null}
      </Card>

      {j.status === 'failed' ? <Banner message={j.error ?? 'Something went wrong.'} /> : null}
      {retry.isError ? <Banner message={retry.error.message} /> : null}

      <SuspectClaims claims={j.understanding?.suspect_claims} note={j.understanding?.research_note} />

      <Section title="Progress">
        <Card>
          {steps.map((step) => (
            <StepRow key={step.key} step={step} />
          ))}
        </Card>
        {isActive(j.status) ? (
          <ThemedText type="caption" themeColor="textSecondary">
            This usually takes a minute or two. You can leave this screen; the job keeps running on the server.
          </ThemedText>
        ) : null}
      </Section>

      {j.status === 'completed' && jobCards.isSuccess ? (
        generated.length === 0 && addingTo ? (
          <Banner tone="info" message={`No new cards were added to “${addingTo}”. Anything that repeated a card already there was set aside.`} />
        ) : (
          <Banner
            tone="success"
            message={`${plural(generated.length, 'card')} ${addingTo ? `added to “${addingTo}”` : 'generated'}${draftCount ? `, ${draftCount} waiting for your review` : ''}.`}
          />
        )
      ) : null}

      <ThemedText type="caption" themeColor="textSecondary">
        Estimated AI cost so far: {formatCost(j.cost_usd)}
      </ThemedText>

      {j.status !== 'completed' ? <Button title="Discard this upload" variant="ghost" onPress={onDiscard} loading={discard.isPending} /> : null}
    </Screen>
  );
}

function StepRow({ step }: { step: Step }) {
  const theme = useTheme();
  const glyph: Record<Step['state'], { text: string; color: string }> = {
    done: { text: '✓', color: theme.success },
    waiting: { text: '!', color: theme.warning },
    failed: { text: '✕', color: theme.danger },
    pending: { text: '○', color: theme.textSecondary },
    active: { text: '', color: theme.tint },
  };
  const g = glyph[step.state];
  return (
    <View style={styles.step} accessibilityLabel={`${step.label}: ${step.state}`}>
      <View style={styles.glyph}>
        {step.state === 'active' ? <ActivityIndicator size="small" color={g.color} /> : <ThemedText style={{ color: g.color, fontWeight: 700 }}>{g.text}</ThemedText>}
      </View>
      <ThemedText themeColor={step.state === 'pending' ? 'textSecondary' : 'text'} style={step.state === 'active' ? styles.activeLabel : undefined}>
        {step.label}
      </ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  between: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
  flex: { flex: 1 },
  step: { flexDirection: 'row', alignItems: 'center', gap: Spacing.three, minHeight: 36 },
  glyph: { width: 24, alignItems: 'center' },
  activeLabel: { fontWeight: 700 },
});
