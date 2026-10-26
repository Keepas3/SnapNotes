import { router } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { deckCaption } from '@/lib/deck';
import { filterHomeRows, folderCaption, studyParam } from '@/lib/folders';
import { useStudyCounts } from '@/study/hooks';
import type { DeckSummary, FolderSummary } from '@/study/store';

import { ThemedText } from './themed-text';
import { Button, Card, IconButton } from './ui';

interface Props {
  visible: boolean;
  onClose: () => void;
  decks: DeckSummary[];
  folders: FolderSummary[];
  /** The order the user arranged the decks in (empty until they drag one). */
  order: string[];
}

/**
 * Every folder and deck in one searchable list, folders open, to find one among many and jump to it: tap a deck to open it, or its
 * Study button to start studying it. Works from what the phone has, so it works offline.
 */
export function DeckPickerSheet({ visible, onClose, decks, folders, order }: Props) {
  const theme = useTheme();
  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={[styles.fill, { backgroundColor: theme.background }]}>
        {/* Mounted only while shown, so every visit starts with an empty search. */}
        {visible ? <PickerBody decks={decks} folders={folders} order={order} onClose={onClose} /> : null}
      </KeyboardAvoidingView>
    </Modal>
  );
}

function PickerBody({ decks, folders, order, onClose }: Omit<Props, 'visible'>) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const [query, setQuery] = useState('');
  const rows = filterHomeRows(decks, folders, order, new Set(), query);
  const deckById = new Map(decks.map((d) => [d.id, d]));
  const folderById = new Map(folders.map((f) => [f.id, f]));

  const openDeck = (id: string) => {
    onClose();
    router.push({ pathname: '/decks/[id]', params: { id } });
  };
  const study = (deckId: string) => {
    onClose();
    router.push({ pathname: '/study/[deckId]', params: { deckId } });
  };

  return (
    <ScrollView contentContainerStyle={[styles.content, { paddingTop: Spacing.four, paddingBottom: insets.bottom + Spacing.four }]} keyboardShouldPersistTaps="handled">
      <View style={styles.header}>
        <ThemedText type="heading">All decks</ThemedText>
        <Button title="Close" variant="ghost" onPress={onClose} />
      </View>

      <View style={[styles.searchBox, { backgroundColor: theme.backgroundElement, borderColor: theme.border }]}>
        <TextInput
          accessibilityLabel="Search all decks"
          style={[styles.searchInput, { color: theme.text }]}
          value={query}
          onChangeText={setQuery}
          placeholder="Search decks and folders"
          placeholderTextColor={theme.textSecondary}
          autoCorrect={false}
          returnKeyType="search"
        />
        {query ? <IconButton glyph="✕" label="Clear search" onPress={() => setQuery('')} /> : null}
      </View>

      {rows.length === 0 ? (
        <ThemedText themeColor="textSecondary" style={styles.center}>
          {query.trim() ? `No deck or folder matches “${query.trim()}”.` : 'No decks yet.'}
        </ThemedText>
      ) : (
        rows.map((row) => {
          if (row.kind === 'unfiled') {
            return (
              <ThemedText key="unfiled" type="smallBold" themeColor="textSecondary" style={styles.heading}>
                Not in a folder
              </ThemedText>
            );
          }
          if (row.kind === 'folder') {
            const folder = folderById.get(row.id);
            return folder ? <PickerFolder key={`folder:${row.id}`} folder={folder} onStudy={() => study(studyParam({ folderId: folder.id }))} /> : null;
          }
          const deck = deckById.get(row.id);
          if (!deck) return null;
          const due = deck.learning + deck.review + deck.new;
          return (
            <View key={`deck:${row.id}`} style={[styles.line, row.folderId ? styles.indented : null]}>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`Open ${deck.name}`}
                onPress={() => openDeck(deck.id)}
                style={({ pressed }) => [styles.flex, pressed && styles.pressed]}>
                <ThemedText type="smallBold" numberOfLines={1}>
                  {deck.name}
                </ThemedText>
                <ThemedText type="caption" themeColor="textSecondary" numberOfLines={1}>
                  {deckCaption(deck)}
                </ThemedText>
              </Pressable>
              {due > 0 ? <Button title={`Study ${due}`} size="small" variant="secondary" onPress={() => study(deck.id)} /> : null}
            </View>
          );
        })
      )}
    </ScrollView>
  );
}

/** A folder heading in the list, with what is due in it as a whole (the shared daily new-card limit counted once). */
function PickerFolder({ folder, onStudy }: { folder: FolderSummary; onStudy: () => void }) {
  const counts = useStudyCounts({ folderId: folder.id });
  const due = (counts.data?.learning ?? 0) + (counts.data?.review ?? 0);
  const fresh = counts.data?.new ?? 0;
  return (
    <Card style={styles.folderCard}>
      <View style={styles.line}>
        <View style={styles.flex}>
          <ThemedText type="smallBold" numberOfLines={1}>
            {folder.name}
          </ThemedText>
          <ThemedText type="caption" themeColor="textSecondary" numberOfLines={1}>
            {folderCaption(folder, due, fresh)}
          </ThemedText>
        </View>
        {due + fresh > 0 ? <Button title={`Study ${due + fresh}`} size="small" variant="secondary" onPress={onStudy} /> : null}
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  flex: { flex: 1 },
  content: { paddingHorizontal: Spacing.three, gap: Spacing.two, width: '100%', maxWidth: 720, alignSelf: 'center' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: Spacing.one },
  searchBox: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two, minHeight: 44, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: Spacing.three },
  searchInput: { flex: 1, fontSize: 16, paddingVertical: Spacing.two },
  heading: { marginTop: Spacing.two },
  line: { flexDirection: 'row', alignItems: 'center', gap: Spacing.two, minHeight: 48, paddingVertical: Spacing.one },
  indented: { marginLeft: Spacing.three },
  folderCard: { padding: Spacing.two, gap: 0, borderRadius: 12, marginTop: Spacing.one },
  pressed: { opacity: 0.75 },
  center: { textAlign: 'center', paddingVertical: Spacing.four },
});
