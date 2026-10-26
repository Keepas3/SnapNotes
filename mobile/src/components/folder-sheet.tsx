import { router } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useCreateFolder, useDeleteFolder, useRenameFolder } from '@/api/hooks';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { confirm } from '@/lib/confirm';
import { plural } from '@/lib/format';
import { folderNameProblem, MAX_FOLDER_NAME, studyParam } from '@/lib/folders';

import { OfflineNotice } from './load-gate';
import { ThemedText } from './themed-text';
import { Banner, Button } from './ui';

/** Make a new folder, or rename or delete an existing one. */
export type FolderSheetTarget = { mode: 'new' } | { mode: 'edit'; folder: { id: string; name: string }; deckCount: number; cards?: number };

interface Props {
  /** What the sheet is for; it is hidden while this is null. */
  target: FolderSheetTarget | null;
  onClose: () => void;
  /** Called after the server's copy changed, so the screen can bring the phone up to date. */
  onChanged?: () => void | Promise<void>;
}

export function FolderSheet({ target, onClose, onChanged }: Props) {
  const theme = useTheme();
  return (
    <Modal visible={target !== null} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={[styles.fill, { backgroundColor: theme.background }]}>
        {/* Keyed so each folder (and "new") starts from its own text. */}
        {target ? <SheetBody key={target.mode === 'edit' ? target.folder.id : 'new'} target={target} onClose={onClose} onChanged={onChanged} /> : null}
      </KeyboardAvoidingView>
    </Modal>
  );
}

function SheetBody({ target, onClose, onChanged }: Props & { target: FolderSheetTarget }) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const create = useCreateFolder();
  const rename = useRenameFolder();
  const remove = useDeleteFolder();
  const editing = target.mode === 'edit' ? target : null;
  const [name, setName] = useState(editing?.folder.name ?? '');
  const [error, setError] = useState<string | null>(null);
  const unchanged = editing !== null && name.trim() === editing.folder.name;
  const busy = create.isPending || rename.isPending || remove.isPending;

  async function save() {
    const problem = folderNameProblem(name);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    try {
      if (editing) await rename.mutateAsync({ id: editing.folder.id, name: name.trim() });
      else await create.mutateAsync(name.trim());
      await onChanged?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the folder.');
    }
  }

  async function del() {
    if (!editing) return;
    const kept = editing.deckCount === 0 ? 'It is empty.' : `Its ${plural(editing.deckCount, 'deck')} ${editing.deckCount === 1 ? 'is' : 'are'} kept and will no longer be in a folder.`;
    if (!(await confirm(`Delete the folder “${editing.folder.name}”?`, `${kept} Only the folder is removed.`, 'Delete folder', true))) return;
    setError(null);
    try {
      await remove.mutateAsync(editing.folder.id);
      await onChanged?.();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete the folder.');
    }
  }

  return (
    <ScrollView contentContainerStyle={[styles.content, { paddingTop: Spacing.four, paddingBottom: insets.bottom + Spacing.four }]} keyboardShouldPersistTaps="handled">
      <View style={styles.header}>
        <ThemedText type="heading">{editing ? 'Folder settings' : 'New folder'}</ThemedText>
        <Button title="Close" variant="ghost" onPress={onClose} />
      </View>
      <OfflineNotice doing="Changing a folder" />

      <ThemedText type="smallBold">Folder name</ThemedText>
      <TextInput
        accessibilityLabel="Folder name"
        style={[styles.input, { color: theme.text, backgroundColor: theme.backgroundElement, borderColor: theme.border }]}
        value={name}
        onChangeText={setName}
        maxLength={MAX_FOLDER_NAME + 20} // let a too-long name be typed so the message can explain, instead of silently cutting it
        placeholder="For example Biology"
        placeholderTextColor={theme.textSecondary}
        returnKeyType="done"
        onSubmitEditing={save}
        autoCapitalize="sentences"
        autoFocus={!editing}
      />
      {error ? <Banner message={error} /> : null}
      <Button title={editing ? 'Save name' : 'Create folder'} onPress={save} loading={create.isPending || rename.isPending} disabled={unchanged || busy} />

      {editing ? (
        <>
          {(editing.cards ?? 0) > 0 ? (
            <Button
              title={`Practice all ${editing.cards} cards`}
              variant="secondary"
              onPress={() => {
                onClose();
                router.push({ pathname: '/study/[deckId]', params: { deckId: studyParam({ folderId: editing.folder.id }), mode: 'all' } });
              }}
            />
          ) : null}
          <ThemedText type="caption" themeColor="textSecondary">
            Put decks in this folder from a deck’s ☰ menu, or by holding a deck and dropping it on the folder. Deleting a folder never deletes its decks.
          </ThemedText>
          <Button title="Delete folder" variant="danger" onPress={del} loading={remove.isPending} disabled={busy} />
        </>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  content: { paddingHorizontal: Spacing.three, gap: Spacing.three, width: '100%', maxWidth: 720, alignSelf: 'center' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  input: { minHeight: 48, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two, fontSize: 16 },
});
