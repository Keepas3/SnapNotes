import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useColorScheme } from 'react-native';

import { SettingsProvider } from '@/lib/settings';
import { SyncProvider } from '@/sync/provider';

export default function RootLayout() {
  const scheme = useColorScheme();
  return (
    <ThemeProvider value={scheme === 'dark' ? DarkTheme : DefaultTheme}>
      <SettingsProvider>
        <SyncProvider>
          <Stack screenOptions={{ headerShadowVisible: false }} />
        </SyncProvider>
        <StatusBar style="auto" />
      </SettingsProvider>
    </ThemeProvider>
  );
}
