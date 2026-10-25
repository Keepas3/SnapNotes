import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

import { ApiClient, normalizeBaseUrl } from '@/api/client';

import * as storage from './storage';

const KEY_URL = 'snapnotes.serverUrl';
const KEY_TOKEN = 'snapnotes.token';

/** The server and token a development run was started with, or null (always null in a release build). */
export function devConnection(url: string | undefined, token: string | undefined, isDev: boolean = __DEV__) {
  const base = normalizeBaseUrl(url ?? '');
  const tok = (token ?? '').trim();
  return isDev && base && tok ? { url: base, token: tok } : null;
}

interface SettingsContextValue {
  /** false until the stored values have been read, so screens don't flash the "set up" state */
  ready: boolean;
  baseUrl: string;
  token: string;
  configured: boolean;
  api: ApiClient | null;
  save: (baseUrl: string, token: string) => Promise<void>;
  clear: () => Promise<void>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: 1,
            refetchOnWindowFocus: false,
            staleTime: 5_000,
          },
        },
      }),
  );
  const [ready, setReady] = useState(false);
  const [baseUrl, setBaseUrl] = useState('');
  const [token, setToken] = useState('');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      let [url, tok] = await Promise.all([storage.getItem(KEY_URL), storage.getItem(KEY_TOKEN)]);
      const dev = devConnection(process.env.EXPO_PUBLIC_SNAPNOTES_DEV_URL, process.env.EXPO_PUBLIC_SNAPNOTES_DEV_TOKEN);
      if (dev && (dev.url !== url || dev.token !== tok)) {
        // start.ps1 hands a development build the server and token it just started, so they never go stale
        [url, tok] = [dev.url, dev.token];
        await Promise.all([storage.setItem(KEY_URL, dev.url), storage.setItem(KEY_TOKEN, dev.token)]).catch(() => {});
      }
      if (cancelled) return;
      setBaseUrl(url ?? '');
      setToken(tok ?? '');
      setReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const save = useCallback(
    async (rawUrl: string, rawToken: string) => {
      const url = normalizeBaseUrl(rawUrl);
      if (!url) throw new Error('Enter a server address like http://192.168.1.20:8000');
      const tok = rawToken.trim();
      if (!tok) throw new Error('Enter the access token');
      await Promise.all([storage.setItem(KEY_URL, url), storage.setItem(KEY_TOKEN, tok)]);
      queryClient.clear(); // cached data belongs to the previous server
      setBaseUrl(url);
      setToken(tok);
    },
    [queryClient],
  );

  const clear = useCallback(async () => {
    await Promise.all([storage.deleteItem(KEY_URL), storage.deleteItem(KEY_TOKEN)]);
    queryClient.clear();
    setBaseUrl('');
    setToken('');
  }, [queryClient]);

  const value = useMemo<SettingsContextValue>(() => {
    const configured = Boolean(baseUrl && token);
    return { ready, baseUrl, token, configured, api: configured ? new ApiClient(baseUrl, token) : null, save, clear };
  }, [ready, baseUrl, token, save, clear]);

  return (
    <SettingsContext.Provider value={value}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used inside <SettingsProvider>');
  return ctx;
}

/** For screens that only render once the app is configured (the home screen redirects to Settings otherwise). */
export function useApi(): ApiClient {
  const { api } = useSettings();
  if (!api) throw new Error('The server is not configured yet');
  return api;
}
