import { Stack, router } from 'expo-router';
import { useState } from 'react';
import { Platform, StyleSheet, Switch, TextInput, View } from 'react-native';

import { ApiClient, ApiError, insecureUrlProblem, normalizeBaseUrl } from '@/api/client';
import { ThemedText } from '@/components/themed-text';
import { Banner, Button, Card, Centered, Chip, Screen, Spinner } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { APPEARANCES } from '@/lib/appearance';
import { useAppearance } from '@/lib/appearance-provider';
import { confirm } from '@/lib/confirm';
import { plural } from '@/lib/format';
import { useSettings } from '@/lib/settings';
import { CHOICE_COUNTS } from '@/lib/choices';
import { useChoicePrefs, useNewPerDay, usePendingReviews, useSetChoicePrefs, useSetNewPerDay } from '@/study/hooks';
import { useSync } from '@/sync/provider';

const URL_HINT = Platform.select({ web: 'http://localhost:8000', android: 'http://10.0.2.2:8000', default: 'http://192.168.1.20:8000' });
const NEW_PER_DAY_CHOICES = [5, 10, 20, 30, 50];

export default function SettingsScreen() {
  const { ready } = useSettings();
  if (!ready) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }
  // Mounted only once the saved values are known, so the form can start from them.
  return <SettingsForm />;
}

function SettingsForm() {
  const theme = useTheme();
  const appearance = useAppearance();
  const { baseUrl, token, configured, save, clear } = useSettings();
  const { status: sync, sync: syncNow } = useSync();
  const newPerDay = useNewPerDay();
  const setNewPerDay = useSetNewPerDay();
  const choicePrefs = useChoicePrefs();
  const setChoicePrefs = useSetChoicePrefs();
  const pending = usePendingReviews();
  const [urlInput, setUrlInput] = useState(baseUrl);
  const [tokenInput, setTokenInput] = useState(token);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: 'success' | 'danger'; text: string } | null>(null);

  async function saveAndTest() {
    setMessage(null);
    const url = normalizeBaseUrl(urlInput);
    if (!url) return setMessage({ tone: 'danger', text: `Enter the server address, for example ${URL_HINT}` });
    const insecure = insecureUrlProblem(url, __DEV__);
    if (insecure) return setMessage({ tone: 'danger', text: insecure });
    if (!tokenInput.trim()) return setMessage({ tone: 'danger', text: 'Enter the access token.' });

    setBusy(true);
    try {
      const probe = new ApiClient(url, tokenInput.trim());
      await probe.health(); // is anything there?
      await probe.listJobs(1); // does it accept the token?
      // Decks and reviews on this phone belong to the server they came from; a different server replaces them.
      const waiting = pending.data ?? 0;
      if (configured && url !== baseUrl && waiting > 0) {
        const ok = await confirm(
          'Switch servers?',
          `${plural(waiting, 'review')} on this phone ${waiting === 1 ? 'has' : 'have'} not been synced to the current server. Switching servers will discard ${waiting === 1 ? 'it' : 'them'}.`,
          'Switch',
          true,
        );
        if (!ok) return;
      }
      await save(url, tokenInput);
      setMessage({ tone: 'success', text: 'Connected.' });
      if (router.canGoBack()) router.back();
      else router.replace('/');
    } catch (e) {
      const text =
        e instanceof ApiError && e.status === 401
          ? 'The server is reachable, but it rejected this access token.'
          : e instanceof Error
            ? e.message
            : 'Could not connect.';
      setMessage({ tone: 'danger', text });
    } finally {
      setBusy(false);
    }
  }

  async function disconnect() {
    if (!(await confirm('Disconnect?', 'This removes the saved server address and token from this device.', 'Disconnect', true))) return;
    await clear();
    setUrlInput('');
    setTokenInput('');
    setMessage(null);
  }

  const input = [styles.input, { color: theme.text, backgroundColor: theme.backgroundElement, borderColor: theme.border }];
  const waiting = pending.data ?? 0;

  return (
    <Screen>
      <Stack.Screen options={{ title: 'Settings' }} />
      {!configured ? (
        <Card>
          <ThemedText type="heading">Connect to your SnapNotes server</ThemedText>
          <ThemedText themeColor="textSecondary">
            Enter the address of the backend you are running and the access token you set for it (SNAPNOTES_TOKEN). Both stay on this device.
          </ThemedText>
        </Card>
      ) : null}

      <ThemedText type="smallBold">Server address</ThemedText>
      <TextInput
        accessibilityLabel="Server address"
        style={input}
        value={urlInput}
        onChangeText={setUrlInput}
        placeholder={URL_HINT}
        placeholderTextColor={theme.textSecondary}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
      />
      <ThemedText type="caption" themeColor="textSecondary">
        On a physical phone use your computer’s Wi-Fi address (not localhost). The Android emulator reaches your computer at 10.0.2.2.
      </ThemedText>

      <ThemedText type="smallBold">Access token</ThemedText>
      <TextInput
        accessibilityLabel="Access token"
        style={input}
        value={tokenInput}
        onChangeText={setTokenInput}
        placeholder="Paste the token"
        placeholderTextColor={theme.textSecondary}
        autoCapitalize="none"
        autoCorrect={false}
        secureTextEntry
      />

      {message ? <Banner tone={message.tone} message={message.text} /> : null}
      <Button title="Save and test connection" onPress={saveAndTest} loading={busy} />

      <ThemedText type="heading" style={styles.sectionTitle}>
        Appearance
      </ThemedText>
      <View style={styles.choices} accessibilityRole="radiogroup" accessibilityLabel="Appearance">
        {APPEARANCES.map((a) => (
          <Chip key={a.value} label={a.label} selected={appearance.preference === a.value} onPress={() => appearance.setPreference(a.value)} />
        ))}
      </View>
      <ThemedText type="caption" themeColor="textSecondary">
        Light or dark colours for the whole app. “Same as phone” follows your phone’s own light or dark setting.
      </ThemedText>

      {configured ? (
        <>
          <ThemedText type="heading" style={styles.sectionTitle}>
            Studying
          </ThemedText>
          <ThemedText type="smallBold">New cards per day</ThemedText>
          <View style={styles.choices}>
            {NEW_PER_DAY_CHOICES.map((n) => (
              <Chip key={n} label={String(n)} selected={newPerDay.data === n} onPress={() => setNewPerDay.mutate(n)} />
            ))}
          </View>
          <ThemedText type="caption" themeColor="textSecondary">
            How many cards you have never seen before are introduced each day. Reviews of cards you already know are not limited. The day starts at 4 a.m.
          </ThemedText>

          <ThemedText type="smallBold">Multiple choice: options per question</ThemedText>
          <View style={styles.choices}>
            {CHOICE_COUNTS.map((n) => (
              <Chip key={n} label={`${n} options`} selected={choicePrefs.data?.count === n} onPress={() => setChoicePrefs.mutate({ count: n })} />
            ))}
          </View>
          <View style={styles.switchRow}>
            <ThemedText type="small" style={styles.flex}>
              Fill missing options from other cards in the deck
            </ThemedText>
            <Switch
              accessibilityLabel="Fill missing options from other cards in the deck"
              value={choicePrefs.data?.fillFromDeck ?? true}
              onValueChange={(on) => setChoicePrefs.mutate({ fillFromDeck: on })}
            />
          </View>
          <ThemedText type="caption" themeColor="textSecondary">
            Turn multiple choice on or off with the switch on the study screen. A card uses its own wrong answers first; when it has fewer than needed, this borrows the answers of the deck’s other cards, so every card can be a question without typing anything.
          </ThemedText>

          <ThemedText type="heading" style={styles.sectionTitle}>
            Sync
          </ThemedText>
          <ThemedText type="small" themeColor="textSecondary">
            {sync.syncing
              ? 'Syncing…'
              : sync.offline
                ? 'Offline: the server cannot be reached.'
                : sync.error
                  ? `Last sync failed: ${sync.error}`
                  : sync.lastSyncedAt
                    ? `Last synced at ${sync.lastSyncedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}.`
                    : 'Not synced yet.'}{' '}
            {waiting > 0 ? `${plural(waiting, 'review')} waiting to upload.` : 'Everything you studied has been uploaded.'}
          </ThemedText>
          <Button title="Sync now" variant="secondary" onPress={() => syncNow()} loading={sync.syncing} />

          <Button title="Disconnect this device" variant="danger" onPress={disconnect} />
        </>
      ) : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  input: { minHeight: 48, borderRadius: 12, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: Spacing.three, fontSize: 16 },
  sectionTitle: { marginTop: Spacing.three },
  choices: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  switchRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.three },
  flex: { flex: 1 },
});
