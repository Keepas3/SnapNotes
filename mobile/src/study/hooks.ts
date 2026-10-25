import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { LOCAL_KEY, useSync } from '@/sync/provider';

import type { DeckSummary, FolderSummary, StudyCounts, StudyScope } from './store';

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

/** The folders on this phone, A to Z, with how many decks and cards each holds. */
export function useLocalFolders() {
  const { store } = useSync();
  return useQuery<FolderSummary[]>({
    queryKey: [...LOCAL_KEY, 'folders'],
    queryFn: () => store!.listFolders(),
    enabled: store !== null,
    staleTime: 0,
  });
}

/** What is due and new for one deck, a whole folder (`{ folderId }`) or everything (null). */
export function useStudyCounts(scope: StudyScope) {
  const { store } = useSync();
  return useQuery<StudyCounts>({
    queryKey: [...LOCAL_KEY, 'counts', typeof scope === 'object' && scope !== null ? `folder:${scope.folderId}` : scope],
    queryFn: async () => store!.counts(scope, new Date(), await store!.newPerDay()),
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

/** The order the user dragged their decks into, kept on this phone. */
export function useDeckOrder() {
  const { store } = useSync();
  return useQuery<string[]>({
    queryKey: [...LOCAL_KEY, 'deckOrder'],
    queryFn: () => store!.deckOrder(),
    enabled: store !== null,
  });
}

export function useSetDeckOrder() {
  const { store } = useSync();
  const qc = useQueryClient();
  const key = [...LOCAL_KEY, 'deckOrder'];
  return useMutation({
    mutationFn: (ids: string[]) => store!.setDeckOrder(ids),
    // Show the new order at once; the save to the phone's database follows.
    onMutate: (ids) => qc.setQueryData(key, ids),
    onSettled: () => qc.invalidateQueries({ queryKey: key }),
  });
}

/** The folders collapsed on Home, kept on this phone. */
export function useClosedFolders() {
  const { store } = useSync();
  return useQuery<string[]>({
    queryKey: [...LOCAL_KEY, 'closedFolders'],
    queryFn: () => store!.closedFolders(),
    enabled: store !== null,
  });
}

export function useSetClosedFolders() {
  const { store } = useSync();
  const qc = useQueryClient();
  const key = [...LOCAL_KEY, 'closedFolders'];
  return useMutation({
    mutationFn: (ids: string[]) => store!.setClosedFolders(ids),
    onMutate: (ids) => qc.setQueryData(key, ids), // open or close it at once; the save follows
    onSettled: () => qc.invalidateQueries({ queryKey: key }),
  });
}

export function useInvalidateLocal() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: LOCAL_KEY });
}
