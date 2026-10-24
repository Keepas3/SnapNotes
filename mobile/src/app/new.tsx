import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';
import { Stack, router, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { Image, Pressable, StyleSheet, Switch, View } from 'react-native';

import { useUpload } from '@/api/hooks';
import type { UploadFile } from '@/api/types';
import { ThemedText } from '@/components/themed-text';
import { Banner, Button, Card, Screen, Section } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { formatBytes, isImage, MAX_FILES, photoName, validateSelection } from '@/lib/upload';

export default function NewDeckScreen() {
  // Opened from a deck's settings with the deck to add to; opened from Home with nothing, it makes a new deck.
  const { deckId, deckName } = useLocalSearchParams<{ deckId?: string; deckName?: string }>();
  const adding = Boolean(deckId);
  const upload = useUpload();
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [skipApproval, setSkipApproval] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const add = (more: UploadFile[]) => {
    setError(null);
    setFiles((cur) => [...cur, ...more].slice(0, MAX_FILES));
  };

  async function pickDocuments() {
    const res = await DocumentPicker.getDocumentAsync({ type: '*/*', multiple: true, copyToCacheDirectory: true });
    if (res.canceled) return;
    add(res.assets.map((a) => ({ uri: a.uri, name: a.name, mimeType: a.mimeType, size: a.size })));
  }

  const fromAssets = (assets: ImagePicker.ImagePickerAsset[]): UploadFile[] =>
    assets.map((a, i) => ({ uri: a.uri, name: photoName(a, files.length + i), mimeType: a.mimeType ?? undefined, size: a.fileSize }));

  async function takePhoto() {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      setError('Camera access is turned off. Allow it in your device settings to photograph notes.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.7 });
    if (!res.canceled) add(fromAssets(res.assets));
  }

  async function pickPhotos() {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsMultipleSelection: true,
      selectionLimit: MAX_FILES,
      quality: 0.7,
    });
    if (!res.canceled) add(fromAssets(res.assets));
  }

  async function submit() {
    const problem = validateSelection(files);
    if (problem) return setError(problem);
    setError(null);
    try {
      const { job_id } = await upload.mutateAsync({ files, autoApprove: skipApproval, deckId: deckId || undefined });
      router.replace({ pathname: '/jobs/[id]', params: { id: job_id } });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.');
    }
  }

  return (
    <Screen
      footer={
        <Button
          title={
            files.length ? `Upload ${files.length} ${files.length === 1 ? 'file' : 'files'} and ${adding ? 'add cards' : 'generate'}` : 'Choose notes to start'
          }
          onPress={submit}
          disabled={files.length === 0}
          loading={upload.isPending}
        />
      }>
      <Stack.Screen options={{ title: adding ? `Add to “${deckName || 'deck'}”` : 'New deck' }} />

      <Section title="Add your notes">
        <Button title="Take a photo of your notes" onPress={takePhoto} />
        <Button title="Choose photos" variant="secondary" onPress={pickPhotos} />
        <Button title="Choose files (PDF, Word, slides, text)" variant="secondary" onPress={pickDocuments} />
        <ThemedText type="caption" themeColor="textSecondary">
          {adding
            ? `The new cards are added to “${deckName || 'this deck'}” for you to review; any that repeat a card already there are set aside. Several photos of consecutive pages are read as one set of notes. Up to ${MAX_FILES} files, 30 MB in total.`
            : `Several photos of consecutive pages become one deck. Up to ${MAX_FILES} files, 30 MB in total.`}
        </ThemedText>
      </Section>

      {files.length > 0 ? (
        <Section title="Selected">
          {files.map((f, i) => (
            <Card key={`${f.uri}-${i}`}>
              <View style={styles.fileRow}>
                {isImage(f) ? <Image source={{ uri: f.uri }} style={styles.thumb} accessibilityIgnoresInvertColors /> : null}
                <View style={styles.flex}>
                  <ThemedText type="smallBold" numberOfLines={1}>
                    {f.name}
                  </ThemedText>
                  <ThemedText type="caption" themeColor="textSecondary">
                    {formatBytes(f.size)}
                  </ThemedText>
                </View>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel={`Remove ${f.name}`}
                  hitSlop={12}
                  onPress={() => setFiles((cur) => cur.filter((_, j) => j !== i))}>
                  <ThemedText type="linkPrimary">Remove</ThemedText>
                </Pressable>
              </View>
            </Card>
          ))}
        </Section>
      ) : null}

      <Card>
        <View style={styles.fileRow}>
          <View style={styles.flex}>
            <ThemedText type="smallBold">Skip source approval</ThemedText>
            <ThemedText type="caption" themeColor="textSecondary">
              By default you choose which web sources to trust before cards are written.
            </ThemedText>
          </View>
          <Switch accessibilityLabel="Skip source approval" value={skipApproval} onValueChange={setSkipApproval} />
        </View>
      </Card>

      {error ? <Banner message={error} /> : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  fileRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.three },
  flex: { flex: 1 },
  thumb: { width: 48, height: 48, borderRadius: 8 },
});
