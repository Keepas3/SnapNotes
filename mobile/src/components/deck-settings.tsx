import { router } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useDecks, useMergeDecks, useRenameDeck } from '@/api/hooks';
import type { Deck } from '@/api/types';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { confirm } from '@/lib/confirm';
import { deckNameProblem, MAX_DECK_NAME, mergeCandidates, mergeDeckMessage } from '@/lib/deck';
import { plural } from '@/lib/format';

import { ThemedText } from './themed-text';
import { Banner, Button, Card, Spinner } from './ui';

interface Props {
  /** The deck being configured; the sheet is hidden while this is null. */
  deck: { id: string; name: string } | null;
  onClose: () => void;
  /** Called after the server's copy of the library has changed (renamed, or another deck merged in), so the screen can bring the phone up to date. */
  onChanged?: () => void | Promise<void>;
}

/** Settings for one deck: rename it, add notes or a typed card to it, or merge another deck into it. */
export function DeckSettingsSheet({ deck, onClose, onChanged }: Props) {
  const theme = useTheme();
  return (
    <Modal visible={deck !== null} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={[styles.fill, { backgroundColor: theme.background }]}>
        {/* Keyed by deck id so each deck starts from its own saved name. */}
        {deck ? <SheetBody key={deck.id} deck={deck} onClose={onClose} onChanged={onChanged} /> : null}
      </KeyboardAvoidingView>
    </Modal>
  );
}

function SheetBody({ deck, onClose, onChanged }: Props & { deck: { id: string; name: string } }) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const rename = useRenameDeck();
  const [name, setName] = useState(deck.name);
  const [error, setError] = useState<string | null>(null);
  const [merging, setMerging] = useState(false);
  const unchanged = name.trim() === deck.name;
  const target = { deckId: deck.id, deckName: deck.name };

  async function save() {
    const problem = deckNameProblem(name);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    try {
      await rename.mutateAsync({ id: deck.id, name: name.trim() });
      await onChanged?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not rename the deck.');
    }
  }

  /** These two leave the sheet for their own screens, which come back to the deck when done. */
  function addNotes() {
    onClose();
    router.push({ pathname: '/new', params: target });
  }
  function typeCard() {
    onClose();
    router.push({ pathname: '/cards/new', params: target });
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
        <ActionRow title="Add notes to this deck" hint="Upload more notes. The new cards join this deck for you to review; repeats are set aside." onPress={addNotes} />
        <ActionRow title="Type a card" hint="Write one question and answer yourself." onPress={typeCard} />
        <ActionRow
          title="Merge another deck into this one"
          hint="Moves all its cards here, with your study progress. The other deck is removed."
          onPress={() => setMerging((open) => !open)}
          expanded={merging}
        />
      </Card>

      {merging ? (
        <MergePicker
          deck={deck}
          onMerged={async () => {
            await onChanged?.();
            onClose();
          }}
        />
      ) : null}
    </ScrollView>
  );
}

function ActionRow({ title, hint, onPress, expanded }: { title: string; hint: string; onPress: () => void; expanded?: boolean }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      accessibilityState={expanded === undefined ? undefined : { expanded }}
      onPress={onPress}
      style={({ pressed }) => [styles.action, pressed && styles.pressed]}>
      <View style={styles.flex}>
        <ThemedText type="smallBold">{title}</ThemedText>
        <ThemedText type="caption" themeColor="textSecondary">
          {hint}
        </ThemedText>
      </View>
      <ThemedText themeColor="textSecondary">{expanded === undefined ? '›' : expanded ? '⌃' : '⌄'}</ThemedText>
    </Pressable>
  );
}

/** The other decks, one tap each. Asks "are you sure" first, because the chosen deck is removed. */
function MergePicker({ deck, onMerged }: { deck: { id: string; name: string }; onMerged: () => Promise<void> }) {
  const decks = useDecks();
  const merge = useMergeDecks();
  const [error, setError] = useState<string | null>(null);
  const others = mergeCandidates(decks.data ?? [], deck.id);

  async function mergeIn(other: Deck) {
    if (merge.isPending) return;
    const cards = other.counts.draft + other.counts.approved + other.counts.rejected;
    if (!(await confirm(`Merge “${other.name}” into “${deck.name}”?`, mergeDeckMessage(other.name, deck.name, cards), 'Merge', true))) return;
    setError(null);
    try {
      await merge.mutateAsync({ intoId: deck.id, fromId: other.id });
      await onMerged();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not merge the decks.');
    }
  }

  return (
    <View style={styles.picker}>
      <ThemedText type="smallBold">Merge which deck into “{deck.name}”?</ThemedText>
      {decks.isPending ? (
        <Spinner />
      ) : decks.isError ? (
        <Banner message={decks.error.message} actionLabel="Try again" onAction={() => decks.refetch()} />
      ) : others.length === 0 ? (
        <ThemedText type="caption" themeColor="textSecondary">
          There are no other decks to merge in.
        </ThemedText>
      ) : (
        others.map((other) => {
          const total = other.counts.draft + other.counts.approved + other.counts.rejected;
          return (
            <Card key={other.id} accessibilityLabel={`Merge ${other.name} into ${deck.name}`} onPress={() => mergeIn(other)}>
              <ThemedText type="smallBold" numberOfLines={1}>
                {other.name}
              </ThemedText>
              <ThemedText type="caption" themeColor="textSecondary">
                {plural(total, 'card')}
                {other.counts.draft > 0 ? `, ${other.counts.draft} waiting for review` : ''}
              </ThemedText>
            </Card>
          );
        })
      )}
      {merge.isPending ? <Spinner /> : null}
      {error ? <Banner message={error} /> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  flex: { flex: 1 },
  content: { paddingHorizontal: Spacing.three, gap: Spacing.three, width: '100%', maxWidth: 720, alignSelf: 'center' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  input: { minHeight: 48, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two, fontSize: 16 },
  action: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two, minHeight: 56, paddingVertical: Spacing.one },
  picker: { gap: Spacing.two },
  pressed: { opacity: 0.75 },
});
