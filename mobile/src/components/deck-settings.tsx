import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useRenameDeck } from '@/api/hooks';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { deckNameProblem, MAX_DECK_NAME } from '@/lib/deck';

import { ThemedText } from './themed-text';
import { Badge, Banner, Button, Card } from './ui';

interface Props {
  /** The deck being configured; the sheet is hidden while this is null. */
  deck: { id: string; name: string } | null;
  onClose: () => void;
  /** Called after the new name has been saved on the server, so the screen can bring the phone up to date. */
  onRenamed?: () => void | Promise<void>;
}

/** Settings for one deck: rename it now; adding cards and merging decks are announced here but not built yet. */
export function DeckSettingsSheet({ deck, onClose, onRenamed }: Props) {
  const theme = useTheme();
  return (
    <Modal visible={deck !== null} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={[styles.fill, { backgroundColor: theme.background }]}>
        {/* Keyed by deck id so each deck starts from its own saved name. */}
        {deck ? <SheetBody key={deck.id} deck={deck} onClose={onClose} onRenamed={onRenamed} /> : null}
      </KeyboardAvoidingView>
    </Modal>
  );
}

function SheetBody({ deck, onClose, onRenamed }: Props & { deck: { id: string; name: string } }) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const rename = useRenameDeck();
  const [name, setName] = useState(deck.name);
  const [error, setError] = useState<string | null>(null);
  const unchanged = name.trim() === deck.name;

  async function save() {
    const problem = deckNameProblem(name);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    try {
      await rename.mutateAsync({ id: deck.id, name: name.trim() });
      await onRenamed?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not rename the deck.');
    }
  }

  return (
    <ScrollView contentContainerStyle={[styles.content, { paddingTop: Spacing.four, paddingBottom: insets.bottom + Spacing.four }]} keyboardShouldPersistTaps="handled">
      <View style={styles.header}>
        <ThemedText type="heading">Deck settings</ThemedText>
        <Button title="Close" variant="ghost" onPress={onClose} />
      </View>

      <ThemedText type="smallBold">Deck title</ThemedText>
      <TextInput
        accessibilityLabel="Deck title"
        style={[styles.input, { color: theme.text, backgroundColor: theme.backgroundElement, borderColor: theme.border }]}
        value={name}
        onChangeText={setName}
        maxLength={MAX_DECK_NAME + 20} // let a too-long name be typed so the message can explain, instead of silently cutting it
        returnKeyType="done"
        onSubmitEditing={save}
        autoCapitalize="sentences"
        placeholderTextColor={theme.textSecondary}
      />
      {error ? <Banner message={error} /> : null}
      <Button title="Save title" onPress={save} loading={rename.isPending} disabled={unchanged} />

      <ThemedText type="smallBold">Grow this deck</ThemedText>
      <Card>
        <View style={styles.between}>
          <ThemedText type="small" style={styles.flex}>
            Add more cards to this deck
          </ThemedText>
          <Badge label="Coming soon" tone="info" />
        </View>
        <View style={styles.between}>
          <ThemedText type="small" style={styles.flex}>
            Merge this deck with another one
          </ThemedText>
          <Badge label="Coming soon" tone="info" />
        </View>
        <ThemedText type="caption" themeColor="textSecondary">
          Later you will be able to add new notes to a deck you already have, or combine two decks into one, without losing your study history.
        </ThemedText>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  content: { paddingHorizontal: Spacing.three, gap: Spacing.three, width: '100%', maxWidth: 720, alignSelf: 'center' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  input: { minHeight: 48, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two, fontSize: 16 },
  between: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: Spacing.two },
  flex: { flex: 1 },
});
