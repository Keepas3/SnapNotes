import { useQueryClient } from '@tanstack/react-query';
import * as Crypto from 'expo-crypto';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { AppState } from 'react-native';

import { ApiError } from '@/api/client';
import { openLocalDb } from '@/db/open';
import { useSettings } from '@/lib/settings';
import { LocalStore } from '@/study/store';

import { runSync } from './engine';

export interface SyncStatus {
  syncing: boolean;
  lastSyncedAt: Date | null;
  /** the server could not be reached; studying still works from the data on this device */
  offline: boolean;
  /** a failure that is not just "offline" (rejected token, server error) */
  error: string | null;
}

interface SyncOptions {
  /** Skip if a sync started less recently than this (lets screens ask freely without hammering the server). */
  minIntervalMs?: number;
}

interface SyncContextValue {
  /** null until the on-device database has been opened */
  store: LocalStore | null;
  /** set if the on-device database could not be opened */
  storeError: string | null;
  status: SyncStatus;
  sync: (opts?: SyncOptions) => Promise<void>;
}

const SyncContext = createContext<SyncContextValue | null>(null);

/** Query-key prefix for everything read from the on-device database. */
export const LOCAL_KEY = ['local'] as const;

export function SyncProvider({ children }: { children: ReactNode }) {
  const { api, baseUrl, configured } = useSettings();
  const queryClient = useQueryClient();
  const [store, setStore] = useState<LocalStore | null>(null);
  const [storeError, setStoreError] = useState<string | null>(null);
  const [status, setStatus] = useState<SyncStatus>({ syncing: false, lastSyncedAt: null, offline: false, error: null });

  const inFlight = useRef<Promise<void> | null>(null);
  const rerun = useRef(false);
  const lastStart = useRef(0);

  // Open the on-device database once.
  useEffect(() => {
    let cancelled = false;
    openLocalDb()
      .then((db) => {
        if (!cancelled) setStore(new LocalStore(db, () => Crypto.randomUUID()));
      })
      .catch((e) => {
        if (!cancelled) setStoreError(e instanceof Error ? e.message : 'Could not open the on-device database.');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const sync = useCallback(
    async (opts?: SyncOptions) => {
      if (!store || !api) return;
      if (opts?.minIntervalMs !== undefined && Date.now() - lastStart.current < opts.minIntervalMs) return;
      if (inFlight.current) {
        rerun.current = true; // something changed while syncing: go round once more afterwards
        return inFlight.current;
      }
      lastStart.current = Date.now();
      setStatus((s) => ({ ...s, syncing: true }));

      const run = (async () => {
        try {
          do {
            rerun.current = false;
            await runSync(api, store);
          } while (rerun.current);
          setStatus({ syncing: false, lastSyncedAt: new Date(), offline: false, error: null });
        } catch (e) {
          const offline = e instanceof ApiError && e.isNetwork;
          setStatus((s) => ({ ...s, syncing: false, offline, error: offline ? null : e instanceof Error ? e.message : 'Sync failed.' }));
        } finally {
          inFlight.current = null;
          queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
        }
      })();
      inFlight.current = run;
      return run;
    },
    [store, api, queryClient],
  );

  // When the server this app talks to changes, the cached library belongs to the old one: clear it, then sync.
  useEffect(() => {
    if (!store || !configured) return;
    let cancelled = false;
    (async () => {
      const result = await store.bindServer(baseUrl);
      if (cancelled) return;
      if (result === 'reset') queryClient.invalidateQueries({ queryKey: LOCAL_KEY });
      void sync();
    })();
    return () => {
      cancelled = true;
    };
  }, [store, configured, baseUrl, sync, queryClient]);

  // Catch up whenever the app comes back to the foreground.
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state === 'active') void sync({ minIntervalMs: 15_000 });
    });
    return () => subscription.remove();
  }, [sync]);

  const value = useMemo<SyncContextValue>(() => ({ store, storeError, status, sync }), [store, storeError, status, sync]);
  return <SyncContext.Provider value={value}>{children}</SyncContext.Provider>;
}

export function useSync(): SyncContextValue {
  const ctx = useContext(SyncContext);
  if (!ctx) throw new Error('useSync must be used inside <SyncProvider>');
  return ctx;
}
