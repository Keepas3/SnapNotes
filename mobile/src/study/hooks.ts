import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { LOCAL_KEY, useSync } from '@/sync/provider';

import type { DeckSummary, StudyCounts } from './store';

/** Reads from the on-device database. They work offline and refetch after every sync or study answer. */

export function useLocalDecks() {
  const { store } = useSync();
  return useQuery<DeckSummary[]>({
    queryKey: [...LOCAL_KEY, 'decks'],
    queryFn: async () => store!.listDecks(new Date(), await store!.newPerDay()),
    enabled: store !== null,
    staleTime: 0,
  });
}

export function useStudyCounts(deckId: string | null) {
  const { store } = useSync();
  return useQuery<StudyCounts>({
    queryKey: [...LOCAL_KEY, 'counts', deckId],
    queryFn: async () => store!.counts(deckId, new Date(), await store!.newPerDay()),
    enabled: store !== null,
    staleTime: 0,
  });
}

export function usePendingReviews() {
  const { store } = useSync();
  return useQuery<number>({
    queryKey: [...LOCAL_KEY, 'pending'],
    queryFn: () => store!.pendingCount(),
    enabled: store !== null,
    staleTime: 0,
  });
}

export function useNewPerDay() {
  const { store } = useSync();
  return useQuery<number>({
    queryKey: [...LOCAL_KEY, 'newPerDay'],
    queryFn: () => store!.newPerDay(),
    enabled: store !== null,
  });
}

export function useSetNewPerDay() {
  const { store } = useSync();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (n: number) => store!.setNewPerDay(n),
    onSuccess: () => qc.invalidateQueries({ queryKey: LOCAL_KEY }),
  });
}

export function useInvalidateLocal() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: LOCAL_KEY });
}
