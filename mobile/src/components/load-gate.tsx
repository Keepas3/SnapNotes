import type { ReactNode } from 'react';

import { ApiError } from '@/api/client';
import { useSync } from '@/sync/provider';

import { Banner, Button, Centered, EmptyState, Screen, Spinner } from './ui';

/** A warning at the top of a screen whose actions all go to the server, shown while the phone knows the server cannot be reached. */
export function OfflineNotice({ doing }: { doing: string }) {
  const { status } = useSync();
  if (!status.offline) return null;
  return <Banner tone="warning" message={`You are offline. ${doing} needs a connection, so it will not work until you are back online.`} />;
}

/**
 * What a screen that reads from the server shows in place of its content until the content is there: a spinner, a plain "this needs a
 * connection" the moment the phone knows the server cannot be reached, or the error. Returns null once the data has loaded.
 *
 * "Knows" means the last sync could not reach the server, or this load itself failed that way, so a phone that is offline says so at once
 * instead of spinning until the request times out. Once data is on screen it stays there, offline or not. Call it with the other hooks, before
 * any early return, and `return` what it gives when that is not null.
 *
 * @param what the thing being loaded, for the sentence ("this deck's cards")
 */
export function useLoadGate(state: { pending: boolean; error: Error | null | undefined }, what: string, onRetry: () => void): ReactNode {
  const { status, sync } = useSync();
  const unreachable = (state.pending && status.offline) || (state.error instanceof ApiError && state.error.isNetwork);
  if (unreachable) {
    return (
      <Screen>
        <EmptyState title="This needs a connection" message={`Loading ${what} needs your server, which cannot be reached right now. Studying, your decks, your stats and your settings still work without a connection.`}>
          <Button
            title="Try again"
            variant="secondary"
            loading={status.syncing}
            onPress={() => {
              // A sync both finds out whether the server is back and brings the phone up to date; then the load runs again.
              void sync().then(onRetry);
            }}
          />
        </EmptyState>
      </Screen>
    );
  }
  if (state.pending) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }
  if (state.error) {
    return (
      <Screen>
        <Banner message={state.error.message} actionLabel="Try again" onAction={onRetry} />
      </Screen>
    );
  }
  return null;
}
