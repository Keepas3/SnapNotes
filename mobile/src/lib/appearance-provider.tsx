import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Appearance as NativeAppearance, useColorScheme as useSystemScheme } from 'react-native';

import { parseAppearance, resolveScheme, type Appearance, type Scheme } from './appearance';
import * as storage from './storage';

const KEY = 'snapnotes.appearance';

interface AppearanceValue {
  preference: Appearance;
  /** The colours actually in use. */
  scheme: Scheme;
  setPreference: (next: Appearance) => void;
}

/** Without a provider (a test, a stray import) the app simply follows the phone. */
const AppearanceContext = createContext<AppearanceValue | null>(null);

export function AppearanceProvider({ children }: { children: ReactNode }) {
  const system = useSystemScheme();
  const [preference, setPreferenceState] = useState<Appearance>('system');

  // Read the saved choice once. Until it arrives the app follows the phone, which is also what it does when nothing was saved.
  useEffect(() => {
    let cancelled = false;
    storage.getItem(KEY).then((saved) => {
      if (!cancelled) setPreferenceState(parseAppearance(saved));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Tell the platform too, so what it draws itself (alerts, the keyboard, the system bars) matches the app. "system" hands control back.
  useEffect(() => {
    try {
      NativeAppearance.setColorScheme?.(preference === 'system' ? 'unspecified' : preference);
    } catch {
      /* a platform without this call (older web) just keeps its own look for those parts */
    }
  }, [preference]);

  const setPreference = useCallback((next: Appearance) => {
    setPreferenceState(next);
    storage.setItem(KEY, next).catch(() => {}); // not saving only means the choice is forgotten at the next start
  }, []);

  const value = useMemo<AppearanceValue>(
    () => ({ preference, scheme: resolveScheme(preference, system), setPreference }),
    [preference, system, setPreference],
  );
  return <AppearanceContext.Provider value={value}>{children}</AppearanceContext.Provider>;
}

export function useAppearance(): AppearanceValue {
  const ctx = useContext(AppearanceContext);
  const system = useSystemScheme();
  if (ctx) return ctx;
  return { preference: 'system', scheme: resolveScheme('system', system), setPreference: () => {} };
}
