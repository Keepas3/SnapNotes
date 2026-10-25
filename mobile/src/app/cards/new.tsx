import { useQueryClient } from '@tanstack/react-query';
import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { StyleSheet, TextInput, View } from 'react-native';

import { useCreateCard } from '@/api/hooks';
import type { NewCard } from '@/api/types';
import { ThemedText } from '@/components/themed-text';
import { Banner, Button, Chip, Screen } from '@/components/ui';
import { WrongAnswers } from '@/components/wrong-answers';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { cleanWrongAnswers, wrongAnswerProblem } from '@/lib/choices';
import { hasCloze } from '@/lib/cloze';
import { newCardProblem } from '@/lib/deck';
import { plural } from '@/lib/format';
import { LOCAL_KEY, useSync } from '@/sync/provider';

/** Type a card yourself: no AI and no upload. It is approved at once, so it can be studied straight away. */
export default function NewCardScreen() {
  const { deckId, deckName } = useLocalSearchParams<{ deckId?: string; deckName?: string }>();
  const theme = useTheme();
  const queryClient = useQueryClient();
  const { sync } = useSync();
  const create = useCreateCard();
  const [type, setType] = useState<NewCard['type']>('basic');
  const [front, setFront] = useState('');
  const [back, setBack] = useState('');
  const [wrong, setWrong] = useState<string[]>(['', '', '']); // wrong answers for multiple choice (not for a fill-in-the-blank card)
  const [error, setError] = useState<string | null>(null);
  const [added, setAdded] = useState(0);
  const isCloze = type === 'cloze';
  const where = deckName || 'this deck';
  const inputStyle = [styles.input, { color: theme.text, backgroundColor: theme.backgroundElement, borderColor: theme.border }];

  async function save() {
    if (!deckId) return;
    const choices = isCloze ? [] : cleanWrongAnswers(wrong);
    const card: NewCard = { type, front: front.trim(), back: back.trim(), ...(choices.length ? { choices } : {}) };
    const problem = newCardProblem(card) ?? (isCloze ? null : wrongAnswerProblem(wrong, card.back));
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    try {
      await create.mutateAsync({ deckId, card });
      setAdded((n) => n + 1);
      setFront('');
      setBack('');
      setWrong(['', '', '']);
      // Bring the phone up to date so the card can be studied on it right now.
      queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
      void sync();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not add the card.');
    }
  }

  if (!deckId) {
    return (
      <Screen>
        <Stack.Screen options={{ title: 'Add a card' }} />
        <Banner message="Open this from a deck's settings so it knows which deck to add to." actionLabel="Back" onAction={() => router.back()} />
      </Screen>
    );
  }

  return (
    <Screen footer={<Button title="Add card" onPress={save} loading={create.isPending} />}>
      <Stack.Screen options={{ title: 'Add a card' }} />

      <ThemedText type="small" themeColor="textSecondary">
        Added to “{where}”. It is approved straight away, so you can study it after the next sync (which starts as soon as you save).
      </ThemedText>

      <View style={styles.row}>
        <Chip label="Question and answer" selected={!isCloze} onPress={() => setType('basic')} />
        <Chip label="Fill in the blank" selected={isCloze} onPress={() => setType('cloze')} />
      </View>

      <ThemedText type="smallBold">{isCloze ? 'Sentence (use {{c1::answer}} for blanks)' : 'Front (the question)'}</ThemedText>
      <TextInput
        accessibilityLabel="Front of card"
        style={inputStyle}
        value={front}
        onChangeText={setFront}
        multiline
        placeholder={isCloze ? 'The {{c1::mitochondrion}} makes most of the cell’s ATP.' : 'What does the mitochondrion do?'}
        placeholderTextColor={theme.textSecondary}
      />
      {isCloze && front.trim() && !hasCloze(front) ? <Banner tone="warning" message="No blank found. Wrap the answer like {{c1::answer}}." /> : null}

      <ThemedText type="smallBold">{isCloze ? 'Explanation (optional)' : 'Back (the answer)'}</ThemedText>
      <TextInput
        accessibilityLabel="Back of card"
        style={inputStyle}
        value={back}
        onChangeText={setBack}
        multiline
        placeholder={isCloze ? 'Anything you want shown with the answer' : 'It makes most of the cell’s ATP.'}
        placeholderTextColor={theme.textSecondary}
      />

      {isCloze ? null : <WrongAnswers values={wrong} onChange={setWrong} />}

      {error ? <Banner message={error} /> : null}
      {added > 0 ? <Banner tone="success" message={`${plural(added, 'card')} added to “${where}”. Add another, or go back.`} /> : null}
      {added > 0 ? <Button title="Done" variant="secondary" onPress={() => router.back()} /> : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  input: { minHeight: 72, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, padding: Spacing.three, fontSize: 16, textAlignVertical: 'top' },
});
