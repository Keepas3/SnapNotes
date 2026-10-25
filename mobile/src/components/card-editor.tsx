import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useDeleteCard, usePatchCard, useSuggestChoices } from '@/api/hooks';
import type { Card, Difficulty } from '@/api/types';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { cleanWrongAnswers, wrongAnswerProblem } from '@/lib/choices';
import { hasCloze } from '@/lib/cloze';
import { confirm } from '@/lib/confirm';

import { CardMeta } from './card-face';
import { ThemedText } from './themed-text';
import { Banner, Button, Chip } from './ui';
import { WrongAnswers } from './wrong-answers';

interface Props {
  /** The card being edited; the editor is hidden while this is null. */
  card: Card | null;
  onClose: () => void;
  /** Called with the saved card so a queue holding a copy can update it. */
  onSaved?: (card: Card) => void;
  onDeleted?: (card: Card) => void;
}

const DIFFICULTIES: Difficulty[] = ['easy', 'medium', 'hard'];

export function CardEditor({ card, onClose, onSaved, onDeleted }: Props) {
  const theme = useTheme();
  return (
    <Modal visible={card !== null} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={[styles.fill, { backgroundColor: theme.background }]}>
        {/* Keyed by card id so each card starts from its own saved text. */}
        {card ? <EditorBody key={card.id} card={card} onClose={onClose} onSaved={onSaved} onDeleted={onDeleted} /> : null}
      </KeyboardAvoidingView>
    </Modal>
  );
}

function EditorBody({ card, onClose, onSaved, onDeleted }: Props & { card: Card }) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const patch = usePatchCard();
  const remove = useDeleteCard();
  const [front, setFront] = useState(card.front);
  const [back, setBack] = useState(card.back);
  const [difficulty, setDifficulty] = useState<Difficulty>(card.difficulty);
  const [error, setError] = useState<string | null>(null);
  // The wrong answers for multiple choice, with a few empty fields to write into.
  const [wrong, setWrong] = useState<string[]>(() => {
    const own = [...(card.choices ?? [])];
    while (own.length < 3) own.push('');
    return own;
  });
  const suggest = useSuggestChoices();

  const inputStyle = [styles.input, { color: theme.text, backgroundColor: theme.backgroundElement, borderColor: theme.border }];
  const isCloze = card.type === 'cloze';
  const clozeBroken = isCloze && !hasCloze(front);

  /** Ask the AI for wrong answers for this card and put them in the fields to be edited. Nothing is saved until Save changes. */
  async function suggestWrong() {
    setError(null);
    try {
      const res = await suggest.mutateAsync({ deckId: card.deck_id, cardIds: [card.id], wrongPerCard: 3 });
      const found = res.suggestions[0]?.wrong_answers;
      if (found) setWrong([...found, ...Array(Math.max(0, 3 - found.length)).fill('')]);
      else setError(res.skipped[0]?.reason ?? 'The AI could not suggest wrong answers for this card.');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not get suggestions.');
    }
  }

  async function save() {
    if (!front.trim() || (!isCloze && !back.trim())) {
      setError('Both sides need some text.');
      return;
    }
    const problem = isCloze ? null : wrongAnswerProblem(wrong, back);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    try {
      const saved = await patch.mutateAsync({
        id: card.id,
        patch: { front: front.trim(), back: back.trim(), difficulty, ...(isCloze ? {} : { choices: cleanWrongAnswers(wrong) }) },
      });
      onSaved?.(saved);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the card.');
    }
  }

  async function del() {
    if (!(await confirm('Delete this card?', 'This cannot be undone.', 'Delete', true))) return;
    try {
      await remove.mutateAsync(card.id);
      onDeleted?.(card);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete the card.');
    }
  }

  return (
    <ScrollView contentContainerStyle={[styles.content, { paddingTop: Spacing.four, paddingBottom: insets.bottom + Spacing.four }]} keyboardShouldPersistTaps="handled">
      <View style={styles.header}>
        <ThemedText type="heading">Edit card</ThemedText>
        <Button title="Cancel" variant="ghost" onPress={onClose} />
      </View>
      <CardMeta card={card} />
      {card.issue ? <Banner tone="warning" message={`Flagged: ${card.issue}`} /> : null}
      {card.source_snippet ? (
        <ThemedText type="caption" themeColor="textSecondary">
          Source: “{card.source_snippet}”
        </ThemedText>
      ) : null}

      <ThemedText type="smallBold">{isCloze ? 'Sentence (use {{c1::answer}} for blanks)' : 'Front'}</ThemedText>
      <TextInput accessibilityLabel="Front of card" style={inputStyle} value={front} onChangeText={setFront} multiline placeholderTextColor={theme.textSecondary} />
      {clozeBroken ? <Banner tone="warning" message="No blank found. Wrap the answer like {{c1::answer}}." /> : null}

      <ThemedText type="smallBold">{isCloze ? 'Explanation (optional)' : 'Back'}</ThemedText>
      <TextInput accessibilityLabel="Back of card" style={inputStyle} value={back} onChangeText={setBack} multiline placeholderTextColor={theme.textSecondary} />

      {isCloze ? (
        <ThemedText type="caption" themeColor="textSecondary">
          Multiple choice is not available for fill-in-the-blank cards.
        </ThemedText>
      ) : (
        <WrongAnswers values={wrong} onChange={setWrong} onSuggest={card.status === 'approved' ? suggestWrong : undefined} suggesting={suggest.isPending} />
      )}

      <ThemedText type="smallBold">Difficulty</ThemedText>
      <View style={styles.row}>
        {DIFFICULTIES.map((d) => (
          <Chip key={d} label={d} selected={difficulty === d} onPress={() => setDifficulty(d)} />
        ))}
      </View>

      {error ? <Banner message={error} /> : null}
      <Button title="Save changes" onPress={save} loading={patch.isPending} />
      <Button title="Delete card" variant="danger" onPress={del} loading={remove.isPending} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  content: { paddingHorizontal: Spacing.three, gap: Spacing.three, width: '100%', maxWidth: 720, alignSelf: 'center' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  input: { minHeight: 56, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, padding: Spacing.three, fontSize: 16, textAlignVertical: 'top' },
  row: { flexDirection: 'row', gap: Spacing.two },
});
