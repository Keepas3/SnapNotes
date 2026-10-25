import { router } from 'expo-router';
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
import { RETENTION_CHOICES } from '@/study/scheduler';
import { useChoicePrefs, useNewPerDay, useOrderPrefs, usePendingReviews, useRetentionChange, useSchedulingPrefs, useSetChoicePrefs, useSetNewPerDay, useSetOrderPrefs, useSetSchedulingPrefs } from '@/study/hooks';
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
  const scheduling = useSchedulingPrefs();
  const setScheduling = useSetSchedulingPrefs();
  const orderPrefs = useOrderPrefs();
  const setOrderPrefs = useSetOrderPrefs();
  const retentionChange = useRetentionChange();
  const [applying, setApplying] = useState(false);
  const [applyNote, setApplyNote] = useState<{ tone: 'success' | 'danger'; text: string } | null>(null);
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
      router.replace('/'); // Settings is a tab now: connected, so go to the decks
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

  /** Give the cards already scheduled the dates the chosen retention implies, after showing what that would do. */
  async function updateScheduledCards() {
    const retention = scheduling.data?.retention ?? 0.9;
    const percent = `${Math.round(retention * 100)}%`;
    setApplyNote(null);
    setApplying(true);
    try {
      const p = await retentionChange.preview(retention);
      if (p.changed === 0) {
        setApplyNote({ tone: 'success', text: p.review === 0 ? 'No cards have been studied enough to be rescheduled yet.' : `Nothing to change: the ${plural(p.review, 'card')} already scheduled follow ${percent}.` });
        return;
      }
      const days = Math.abs(p.averageDays);
      const lines = [
        `${p.earlier > 0 ? `${plural(p.earlier, 'card')} will come back sooner` : ''}${p.earlier > 0 && p.later > 0 ? ', ' : ''}${p.later > 0 ? `${plural(p.later, 'card')} later` : ''} (${days} days ${p.averageDays < 0 ? 'sooner' : 'later'} on average).`,
        p.dueToday > 0 ? `${plural(p.dueToday, 'card')} will be due today.` : '',
        `${plural(p.review - p.changed, 'card')} stay as they are. Cards you are still learning and your answers are not touched. You can change it back the same way.`,
      ].filter(Boolean);
      if (!(await confirm(`Reschedule for ${percent}?`, lines.join(' '), 'Reschedule'))) return;
      const moved = await retentionChange.apply(retention);
      setApplyNote({ tone: 'success', text: `${plural(moved, 'card')} rescheduled for ${percent}.` });
    } catch (e) {
      setApplyNote({ tone: 'danger', text: e instanceof Error ? e.message : 'Could not reschedule the cards.' });
    } finally {
      setApplying(false);
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

          <ThemedText type="smallBold">How sure to be of remembering</ThemedText>
          <View style={styles.choices} accessibilityRole="radiogroup" accessibilityLabel="How sure to be of remembering">
            {RETENTION_CHOICES.map((r) => (
              <Chip key={r} label={`${Math.round(r * 100)}%`} selected={scheduling.data?.retention === r} onPress={() => setScheduling.mutate({ retention: r })} />
            ))}
          </View>
          <ThemedText type="caption" themeColor="textSecondary">
            The chance you still remember a card on the day it comes back. 90% is the usual choice. Higher means cards return sooner and you review more; lower means fewer reviews and more forgetting. It applies from your next answers; cards already scheduled keep their dates until you answer them.
          </ThemedText>
          <Button title={`Update scheduled cards to ${Math.round((scheduling.data?.retention ?? 0.9) * 100)}%`} variant="secondary" onPress={updateScheduledCards} loading={applying} />
          <ThemedText type="caption" themeColor="textSecondary">
            Cards you have already studied keep their dates unless you press this. It shows what would change first, and works from how well each card is known, so it can be changed back. It is done on this phone only: after a reinstall or a new phone, press it again.
          </ThemedText>
          {applyNote ? <Banner tone={applyNote.tone} message={applyNote.text} /> : null}
          <View style={styles.switchRow}>
            <ThemedText type="small" style={styles.flex}>
              Spread out reviews
            </ThemedText>
            <Switch accessibilityLabel="Spread out reviews" value={scheduling.data?.spread ?? true} onValueChange={(on) => setScheduling.mutate({ spread: on })} />
          </View>
          <ThemedText type="caption" themeColor="textSecondary">
            Nudges longer waits by a day or so, so cards you learned together do not all come back on the same day. Waits under a few days are never changed.
          </ThemedText>

          <ThemedText type="smallBold">New card order</ThemedText>
          <View style={styles.choices} accessibilityRole="radiogroup" accessibilityLabel="New card order">
            <Chip label="Shuffled" selected={(orderPrefs.data?.newOrder ?? 'shuffled') === 'shuffled'} onPress={() => setOrderPrefs.mutate({ newOrder: 'shuffled' })} />
            <Chip label="In order" selected={orderPrefs.data?.newOrder === 'order'} onPress={() => setOrderPrefs.mutate({ newOrder: 'order' })} />
          </View>
          <ThemedText type="caption" themeColor="textSecondary">
            Shuffled mixes your decks and topics, so related cards are not back to back. It stays the same all day and changes tomorrow. In order follows the order the cards were made, deck by deck.
          </ThemedText>
          <View style={styles.switchRow}>
            <ThemedText type="small" style={styles.flex}>
              Mix new cards in with reviews
            </ThemedText>
            <Switch accessibilityLabel="Mix new cards in with reviews" value={orderPrefs.data?.mixNew ?? true} onValueChange={(on) => setOrderPrefs.mutate({ mixNew: on })} />
          </View>
          <ThemedText type="caption" themeColor="textSecondary">
            On: new cards are spread through your reviews, in proportion to how many of each are left. Off: reviews first, then new cards. Cards you are still learning always come first.
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
