import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { useColorScheme } from '@/hooks/use-color-scheme';
import { AppearanceProvider } from '@/lib/appearance-provider';
import { SettingsProvider } from '@/lib/settings';
import { SyncProvider } from '@/sync/provider';

export default function RootLayout() {
  return (
    <AppearanceProvider>
      <ThemedApp />
    </AppearanceProvider>
  );
}

/** Below the provider, so it can read the colours in use. */
function ThemedApp() {
  const scheme = useColorScheme();
  return (
    <ThemeProvider value={scheme === 'dark' ? DarkTheme : DefaultTheme}>
      <SettingsProvider>
        <SyncProvider>
          <Stack screenOptions={{ headerShadowVisible: false }} />
        </SyncProvider>
        <StatusBar style={scheme === 'dark' ? 'light' : 'dark'} />
      </SettingsProvider>
    </ThemeProvider>
  );
}
