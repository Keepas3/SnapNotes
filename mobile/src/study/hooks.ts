import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { LOCAL_KEY, useSync } from '@/sync/provider';

import type { SchedulingOptions } from './scheduler';
import type { CardStat, ChoicePrefs, DeckSummary, FolderSummary, StudyCounts, StudyScope, StudyStats } from './store';

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

/** Your progress: answers per day, remembering, cards by stage and what is due soon. Read again after every answer or sync. */
export function useStudyStats() {
  const { store } = useSync();
  return useQuery<StudyStats>({
    queryKey: [...LOCAL_KEY, 'stats'],
    queryFn: () => store!.stats(new Date()),
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

/** How multiple choice is set up on this phone: on or off, 4 or 5 options, and whether gaps are filled from other cards. */
export function useChoicePrefs() {
  const { store } = useSync();
  return useQuery<ChoicePrefs>({
    queryKey: [...LOCAL_KEY, 'choicePrefs'],
    queryFn: () => store!.choicePrefs(),
    enabled: store !== null,
  });
}

export function useSetChoicePrefs() {
  const { store } = useSync();
  const qc = useQueryClient();
  const key = [...LOCAL_KEY, 'choicePrefs'];
  return useMutation({
    mutationFn: (change: Partial<ChoicePrefs>) => store!.setChoicePrefs(change),
    onMutate: (change) => qc.setQueryData<ChoicePrefs>(key, (old) => ({ enabled: false, count: 4, fillFromDeck: true, ...old, ...change })), // apply at once; the save follows
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

/** How cards are scheduled on this phone: the chance of remembering to aim for, and whether longer intervals are spread. */
export function useSchedulingPrefs() {
  const { store } = useSync();
  return useQuery<SchedulingOptions>({
    queryKey: [...LOCAL_KEY, 'schedulingPrefs'],
    queryFn: () => store!.schedulingPrefs(),
    enabled: store !== null,
  });
}

export function useSetSchedulingPrefs() {
  const { store } = useSync();
  const qc = useQueryClient();
  const key = [...LOCAL_KEY, 'schedulingPrefs'];
  return useMutation({
    mutationFn: (change: Partial<SchedulingOptions>) => store!.setSchedulingPrefs(change),
    onMutate: (change) => qc.setQueryData<SchedulingOptions>(key, (old) => ({ retention: 0.9, spread: true, ...old, ...change })), // apply at once; the save follows
    onSettled: () => qc.invalidateQueries({ queryKey: key }),
  });
}

/** Missed-count and paused state of each card in a deck (what the phone knows; empty until cards are synced). */
export function useCardStats(deckId: string) {
  const { store } = useSync();
  return useQuery<CardStat[]>({
    queryKey: [...LOCAL_KEY, 'cardStats', deckId],
    queryFn: () => store!.cardStats(deckId),
    enabled: store !== null,
    staleTime: 0,
  });
}

/** Pause or resume a card. Everything that counts cards is read again afterwards. */
export function usePauseCard() {
  const { store } = useSync();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ cardId, paused }: { cardId: string; paused: boolean }) => (paused ? store!.pauseCard(cardId) : store!.resumeCard(cardId)),
    onSettled: () => qc.invalidateQueries({ queryKey: LOCAL_KEY }),
  });
}

export function useInvalidateLocal() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: LOCAL_KEY });
}
