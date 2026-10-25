import { Pressable, StyleSheet, TextInput, View } from 'react-native';

import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { EDITOR_WRONG_ANSWERS } from '@/lib/choices';

import { ThemedText } from './themed-text';
import { Button } from './ui';

interface Props {
  /** The wrong answers as typed so far, with blank fields for the ones still to write. */
  values: string[];
  onChange: (next: string[]) => void;
  /** Offered only when the card already exists, so the AI has something to write wrong answers for. */
  onSuggest?: () => void;
  suggesting?: boolean;
  /** Leave out the explanation, for a list where it would repeat on every card. */
  compact?: boolean;
}

/** The wrong answers of a card, for multiple choice. The right answer is the card's back; these are the options that are not. */
export function WrongAnswers({ values, onChange, onSuggest, suggesting, compact }: Props) {
  const theme = useTheme();
  const set = (i: number, text: string) => onChange(values.map((v, j) => (j === i ? text : v)));
  const remove = (i: number) => onChange(values.filter((_, j) => j !== i));
  const canAdd = values.length < EDITOR_WRONG_ANSWERS; // a card that already has more keeps them, but no more fields are offered

  return (
    <View style={styles.block}>
      <ThemedText type="smallBold">Wrong answers{compact ? '' : ' (for multiple choice)'}</ThemedText>
      {compact ? null : (
        <ThemedText type="caption" themeColor="textSecondary">
          Write answers that look believable but are wrong; the card’s answer is the right one. Two or more are enough. Leave them empty to borrow answers from your other cards, if that is switched on in Settings.
        </ThemedText>
      )}
      {values.map((value, i) => (
        <View key={i} style={styles.row}>
          <TextInput
            accessibilityLabel={`Wrong answer ${i + 1}`}
            style={[styles.input, { color: theme.text, backgroundColor: theme.backgroundElement, borderColor: theme.border }]}
            value={value}
            onChangeText={(text) => set(i, text)}
            placeholder={`Wrong answer ${i + 1}`}
            placeholderTextColor={theme.textSecondary}
            multiline
          />
          <Pressable accessibilityRole="button" accessibilityLabel={`Remove wrong answer ${i + 1}`} hitSlop={10} onPress={() => remove(i)} style={styles.remove}>
            <ThemedText themeColor="textSecondary">✕</ThemedText>
          </Pressable>
        </View>
      ))}
      <View style={styles.actions}>
        {canAdd ? <Button title="Add another" variant="ghost" onPress={() => onChange([...values, ''])} /> : null}
        {onSuggest ? <Button title="Suggest with AI" variant="secondary" onPress={onSuggest} loading={suggesting} disabled={suggesting} /> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  block: { gap: Spacing.two },
  row: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two },
  input: { flex: 1, minHeight: 48, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two, fontSize: 16, textAlignVertical: 'top' },
  remove: { width: 32, alignItems: 'center', justifyContent: 'center' },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
});
