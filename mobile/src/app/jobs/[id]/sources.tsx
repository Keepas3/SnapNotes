import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { Linking, Pressable, StyleSheet, Switch, View } from 'react-native';

import { useApproveSources, useEvidence } from '@/api/hooks';
import { ApiError } from '@/api/client';
import type { EvidenceItem } from '@/api/types';
import { SuspectClaims } from '@/components/suspect-claims';
import { ThemedText } from '@/components/themed-text';
import { useLoadGate } from '@/components/load-gate';
import { Badge, Banner, Button, Card, EmptyState, Screen, Section } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { domainOf, plural, safeHttpUrl, splitCorrection } from '@/lib/format';

export default function SourcesScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const evidence = useEvidence(id);
  const approve = useApproveSources(id);
  const [rejected, setRejected] = useState<ReadonlySet<number>>(new Set());

  const gate = useLoadGate({ pending: evidence.isPending, error: evidence.error }, 'the sources for this upload', () => void evidence.refetch());
  if (gate) return gate;
  if (!evidence.data) return null;

  const items = evidence.data.evidence;
  const understanding = evidence.data.understanding;
  const kept = items.length - rejected.size;

  const toggle = (itemId: number) =>
    setRejected((cur) => {
      const next = new Set(cur);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });

  async function submit() {
    try {
      await approve.mutateAsync([...rejected]);
      router.replace({ pathname: '/jobs/[id]', params: { id } });
    } catch (e) {
      // 409 means it was already approved (e.g. from another screen): just show the job.
      if (e instanceof ApiError && e.status === 409) router.replace({ pathname: '/jobs/[id]', params: { id } });
    }
  }

  return (
    <Screen
      footer={
        <Button
          title={kept === 0 ? 'Generate cards from my notes only' : `Generate cards using ${plural(kept, 'source')}`}
          onPress={submit}
          loading={approve.isPending}
        />
      }>
      <Stack.Screen options={{ title: 'Approve sources' }} />

      {understanding ? (
        <Card>
          <ThemedText type="heading">{understanding.main_topic}</ThemedText>
          {understanding.gaps.length > 0 ? (
            <>
              <ThemedText type="smallBold">Things worth checking in your notes</ThemedText>
              {understanding.gaps.map((g, i) => (
                <ThemedText key={i} type="small" themeColor="textSecondary">
                  • {g}
                </ThemedText>
              ))}
            </>
          ) : null}
        </Card>
      ) : null}

      <SuspectClaims claims={understanding?.suspect_claims} note={understanding?.research_note} />

      <Section title={`Web sources (${items.length})`}>
        <ThemedText type="caption" themeColor="textSecondary">
          These were found online to fill gaps in your notes. Switch off anything you do not trust; cards will only use the sources left on.
        </ThemedText>
        {items.length === 0 ? (
          <EmptyState title="No web sources found" message="Cards will be written from your notes alone." />
        ) : (
          items.map((item) => <SourceRow key={item.id} item={item} included={!rejected.has(item.id)} onToggle={() => toggle(item.id)} />)
        )}
      </Section>

      {approve.isError && !(approve.error instanceof ApiError && approve.error.status === 409) ? <Banner message={approve.error.message} /> : null}
    </Screen>
  );
}

function SourceRow({ item, included, onToggle }: { item: EvidenceItem; included: boolean; onToggle: () => void }) {
  const { isCorrection, text } = splitCorrection(item.claim);
  const link = safeHttpUrl(item.url);
  return (
    <Card style={!included ? styles.dimmed : undefined}>
      <View style={styles.row}>
        <View style={styles.flex}>
          {isCorrection ? <Badge label="Corrects your notes" tone="warning" /> : null}
          <ThemedText type="smallBold">{text}</ThemedText>
        </View>
        <Switch accessibilityLabel={`Use source: ${item.title}`} value={included} onValueChange={onToggle} />
      </View>
      {item.snippet ? (
        <ThemedText type="caption" themeColor="textSecondary" numberOfLines={3}>
          “{item.snippet}”
        </ThemedText>
      ) : null}
      {link ? (
        <Pressable accessibilityRole="link" accessibilityLabel={`Open ${item.title}`} onPress={() => Linking.openURL(link)}>
          <ThemedText type="linkPrimary" numberOfLines={1}>
            {item.title} · {domainOf(link)}
          </ThemedText>
        </Pressable>
      ) : (
        <ThemedText type="caption" themeColor="textSecondary">
          {item.title}
        </ThemedText>
      )}
    </Card>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'flex-start', gap: Spacing.three },
  flex: { flex: 1, gap: Spacing.one },
  dimmed: { opacity: 0.55 },
});
