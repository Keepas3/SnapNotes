import { useQueryClient } from '@tanstack/react-query';
import { Stack, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { StyleSheet, Switch, View } from 'react-native';

import { useDeck, useDeckCards, useSaveChoices, useSuggestChoices } from '@/api/hooks';
import type { Card } from '@/api/types';
import { CardEditor } from '@/components/card-editor';
import { ThemedText } from '@/components/themed-text';
import { Badge, Banner, Button, Card as Panel, Centered, Screen, Section, Spinner } from '@/components/ui';
import { WrongAnswers } from '@/components/wrong-answers';
import { Spacing } from '@/constants/theme';
import { canUseChoices, cleanWrongAnswers, wrongAnswerProblem } from '@/lib/choices';
import { plural } from '@/lib/format';
import { useChoicePrefs } from '@/study/hooks';
import { LOCAL_KEY, useSync } from '@/sync/provider';

/** What the AI suggested for one card, which the user can edit and either keep or leave out. */
interface Draft {
  use: boolean;
  answers: string[];
}

/** Write the wrong answers for a whole deck: ask the AI, check and edit what it suggests, then save only what you approve. */
export default function ChoicesScreen() {
  const { deckId } = useLocalSearchParams<{ deckId: string }>();
  const queryClient = useQueryClient();
  const { sync } = useSync();
  const deck = useDeck(deckId);
  const cards = useDeckCards(deckId, 'approved');
  const prefs = useChoicePrefs();
  const suggest = useSuggestChoices();
  const save = useSaveChoices();
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [skipped, setSkipped] = useState<{ card_id: string; reason: string }[]>([]);
  const [message, setMessage] = useState<{ tone: 'success' | 'danger' | 'info'; text: string } | null>(null);
  const [editing, setEditing] = useState<Card | null>(null);

  if (cards.isPending || deck.isPending) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }
  if (cards.isError || deck.isError) {
    return (
      <Screen>
        <Banner message={(cards.error ?? deck.error)?.message ?? 'Could not load this deck.'} actionLabel="Try again" onAction={() => cards.refetch()} />
      </Screen>
    );
  }

  const wrongWanted = (prefs.data?.count ?? 4) - 1;
  const usable = cards.data.filter(canUseChoices);
  const ready = usable.filter((c) => (c.choices ?? []).length >= wrongWanted);
  const lacking = usable.filter((c) => (c.choices ?? []).length < wrongWanted);
  const byId = new Map(cards.data.map((c) => [c.id, c]));
  const draftIds = Object.keys(drafts);
  const kept = draftIds.filter((id) => drafts[id].use);

  async function askTheAi() {
    setMessage(null);
    try {
      const res = await suggest.mutateAsync({ deckId, wrongPerCard: wrongWanted });
      setDrafts(Object.fromEntries(res.suggestions.map((s) => [s.card_id, { use: true, answers: [...s.wrong_answers, ...Array(Math.max(0, 3 - s.wrong_answers.length)).fill('')] }])));
      setSkipped(res.skipped);
      setMessage(
        res.suggestions.length
          ? { tone: 'info', text: `Suggestions for ${plural(res.suggestions.length, 'card')}. Read each one: the AI can get them wrong, and nothing is saved until you press Save.` }
          : { tone: 'info', text: 'The AI had no suggestions it could stand behind. You can still write wrong answers yourself, or let the app borrow them from other cards.' },
      );
    } catch (e) {
      setMessage({ tone: 'danger', text: e instanceof Error ? e.message : 'Could not get suggestions.' });
    }
  }

  async function saveKept() {
    const items = [];
    for (const id of kept) {
      const card = byId.get(id);
      if (!card) continue;
      const problem = wrongAnswerProblem(drafts[id].answers, card.back);
      if (problem) {
        setMessage({ tone: 'danger', text: `“${card.front.slice(0, 40)}”: ${problem}` });
        return;
      }
      items.push({ card_id: id, choices: cleanWrongAnswers(drafts[id].answers) });
    }
    try {
      const { saved } = await save.mutateAsync(items);
      setDrafts({});
      setSkipped([]);
      setMessage({ tone: 'success', text: `Saved wrong answers for ${plural(saved, 'card')}.` });
      queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
      void sync(); // so the phone has them for studying
    } catch (e) {
      setMessage({ tone: 'danger', text: e instanceof Error ? e.message : 'Could not save.' });
    }
  }

  return (
    <Screen footer={kept.length > 0 ? <Button title={`Save ${plural(kept.length, 'card')}`} onPress={saveKept} loading={save.isPending} /> : undefined}>
      <Stack.Screen options={{ title: 'Multiple choice answers' }} />

      <Panel>
        <ThemedText type="heading">{deck.data.name}</ThemedText>
        <ThemedText type="small" themeColor="textSecondary">
          {ready.length} of {usable.length} question and definition {usable.length === 1 ? 'card has' : 'cards have'} {plural(wrongWanted, 'wrong answer')} written. Fill-in-the-blank cards are left out because they
          have no options.
        </ThemedText>
        <ThemedText type="caption" themeColor="textSecondary">
          While studying, a card with too few uses its own first and then borrows from the deck’s other cards, if that is on in Settings. Writing them yourself gives the best questions.
        </ThemedText>
        <Button
          title={lacking.length ? `Suggest with AI for ${Math.min(lacking.length, 25)} ${lacking.length === 1 ? 'card' : 'cards'}` : 'Every card has enough'}
          variant="secondary"
          onPress={askTheAi}
          loading={suggest.isPending}
          disabled={lacking.length === 0 || suggest.isPending}
        />
      </Panel>

      {message ? <Banner tone={message.tone} message={message.text} /> : null}

      {draftIds.length > 0 ? (
        <Section title="Check the suggestions">
          {draftIds.map((id) => {
            const card = byId.get(id);
            if (!card) return null;
            const draft = drafts[id];
            return (
              <Panel key={id} style={!draft.use ? styles.off : undefined}>
                <ThemedText type="smallBold">{card.front}</ThemedText>
                <ThemedText type="small" themeColor="success">
                  Right answer: {card.back}
                </ThemedText>
                <WrongAnswers compact values={draft.answers} onChange={(answers) => setDrafts((d) => ({ ...d, [id]: { ...d[id], answers } }))} />
                <View style={styles.useRow}>
                  <ThemedText type="small" style={styles.flex}>
                    Keep these for this card
                  </ThemedText>
                  <Switch accessibilityLabel={`Keep these wrong answers for ${card.front}`} value={draft.use} onValueChange={(use) => setDrafts((d) => ({ ...d, [id]: { ...d[id], use } }))} />
                </View>
              </Panel>
            );
          })}
        </Section>
      ) : null}

      {skipped.length > 0 ? (
        <Section title="No suggestion made">
          {skipped.map((s) => (
            <Panel key={s.card_id}>
              <ThemedText type="smallBold">{byId.get(s.card_id)?.front ?? 'A card'}</ThemedText>
              <ThemedText type="caption" themeColor="textSecondary">
                {s.reason}
              </ThemedText>
            </Panel>
          ))}
        </Section>
      ) : null}

      <Section title="Cards">
        {usable.length === 0 ? (
          <ThemedText type="small" themeColor="textSecondary">
            This deck has no approved question or definition cards yet.
          </ThemedText>
        ) : (
          usable.map((card) => (
            <Panel key={card.id} accessibilityLabel={`Edit wrong answers: ${card.front}`} onPress={() => setEditing(card)}>
              <ThemedText type="smallBold">{card.front}</ThemedText>
              <View style={styles.useRow}>
                <ThemedText type="caption" themeColor="textSecondary" style={styles.flex} numberOfLines={1}>
                  {card.back}
                </ThemedText>
                <Badge label={`${plural((card.choices ?? []).length, 'wrong answer')}`} tone={(card.choices ?? []).length >= wrongWanted ? 'success' : 'neutral'} />
              </View>
            </Panel>
          ))
        )}
      </Section>

      <CardEditor card={editing} onClose={() => setEditing(null)} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  useRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.three },
  flex: { flex: 1 },
  off: { opacity: 0.5 },
});
