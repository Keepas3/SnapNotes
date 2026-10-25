import { Tabs } from 'expo-router/js-tabs';

import { DecksIcon, SettingsIcon, StatsIcon } from '@/components/tab-icons';
import { useTheme } from '@/hooks/use-theme';
import { useSettings } from '@/lib/settings';

/**
 * The three places you move between: your decks (Home), your progress, and settings. Every other screen (a deck, studying, adding notes)
 * is opened above these, so it covers the bar and going back returns to it. Until the server is set up the bar is hidden: the only
 * thing to do is Settings.
 */
export default function TabsLayout() {
  const theme = useTheme();
  const { configured } = useSettings();
  return (
    <Tabs
      screenOptions={{
        headerShadowVisible: false,
        tabBarActiveTintColor: theme.tint,
        tabBarInactiveTintColor: theme.textSecondary,
        tabBarStyle: configured
          ? { backgroundColor: theme.background, borderTopColor: theme.border }
          : { display: 'none' },
        tabBarLabelStyle: { fontSize: 12, fontWeight: '600' },
      }}>
      <Tabs.Screen name="index" options={{ title: 'Decks', headerTitle: 'SnapNotes', tabBarIcon: ({ color }) => <DecksIcon color={color} /> }} />
      <Tabs.Screen name="stats" options={{ title: 'Stats', tabBarIcon: ({ color }) => <StatsIcon color={color} /> }} />
      <Tabs.Screen name="settings" options={{ title: 'Settings', tabBarIcon: ({ color }) => <SettingsIcon color={color} /> }} />
    </Tabs>
  );
}
