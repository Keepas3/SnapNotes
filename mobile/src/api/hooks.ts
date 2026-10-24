import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { isActive } from '@/lib/steps';
import { useApi } from '@/lib/settings';

import type { Card, CardPatch, CardStatus, UploadFile } from './types';

const POLL_MS = 1500;

export const keys = {
  jobs: ['jobs'] as const,
  job: (id: string) => ['job', id] as const,
  evidence: (id: string) => ['evidence', id] as const,
  jobCards: (id: string, status?: CardStatus) => ['jobCards', id, status ?? 'all'] as const,
  decks: ['decks'] as const,
  deck: (id: string) => ['deck', id] as const,
  deckCards: (id: string, status?: CardStatus) => ['deckCards', id, status ?? 'all'] as const,
};

// -- queries -----------------------------------------------------------------------------

export function useJobs() {
  const api = useApi();
  return useQuery({
    queryKey: keys.jobs,
    queryFn: () => api.listJobs(),
    // Keep the list fresh while anything is still being processed.
    refetchInterval: (q) => (q.state.data?.some((j) => isActive(j.status)) ? 3000 : false),
  });
}

export function useJob(id: string) {
  const api = useApi();
  return useQuery({
    queryKey: keys.job(id),
    queryFn: () => api.getJob(id),
    refetchInterval: (q) => (q.state.data && isActive(q.state.data.status) ? POLL_MS : false),
  });
}

export function useEvidence(jobId: string, enabled = true) {
  const api = useApi();
  // Always refetch on mount: a cached copy from before the research phase finished would be empty.
  return useQuery({ queryKey: keys.evidence(jobId), queryFn: () => api.getEvidence(jobId), enabled, staleTime: 0 });
}

export function useJobCards(jobId: string, status?: CardStatus) {
  const api = useApi();
  return useQuery({ queryKey: keys.jobCards(jobId, status), queryFn: () => api.jobCards(jobId, status) });
}

export function useDecks() {
  const api = useApi();
  return useQuery({ queryKey: keys.decks, queryFn: () => api.listDecks() });
}

/** `id` is null while a job has not produced a deck yet; the query stays idle then (an empty id would hit the deck list). */
export function useDeck(id: string | null | undefined) {
  const api = useApi();
  return useQuery({ queryKey: keys.deck(id ?? ''), queryFn: () => api.getDeck(id as string), enabled: Boolean(id) });
}

export function useDeckCards(deckId: string, status?: CardStatus) {
  const api = useApi();
  return useQuery({ queryKey: keys.deckCards(deckId, status), queryFn: () => api.deckCards(deckId, status) });
}

// -- mutations ---------------------------------------------------------------------------

export function useUpload() {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { files: UploadFile[]; autoApprove: boolean }) => api.uploadDocument(v.files, v.autoApprove),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.jobs }),
  });
}

export function useApproveSources(jobId: string) {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (rejectedIds: number[]) => api.approveSources(jobId, rejectedIds),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.job(jobId) });
      qc.invalidateQueries({ queryKey: keys.jobs });
    },
  });
}

export function useRetryJob(jobId: string) {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.retryJob(jobId),
    onSuccess: (job) => {
      qc.setQueryData(keys.job(jobId), job);
      qc.invalidateQueries({ queryKey: keys.jobs });
    },
  });
}

export function useDeleteDocument() {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.jobs });
      qc.invalidateQueries({ queryKey: keys.decks });
    },
  });
}

export function useRenameDeck() {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => api.renameDeck(id, name),
    onSuccess: (deck) => {
      qc.setQueryData(keys.deck(deck.id), deck);
      qc.invalidateQueries({ queryKey: keys.decks });
    },
  });
}

export function useDeleteDeck() {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (deckId: string) => api.deleteDeck(deckId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.decks });
      qc.invalidateQueries({ queryKey: keys.jobs });
    },
  });
}

/** Refresh everything that displays cards or deck counts. */
export function useInvalidateLibrary() {
  const qc = useQueryClient();
  return () => {
    qc.invalidateQueries({ queryKey: keys.decks });
    qc.invalidateQueries({ queryKey: ['deck'] });
    qc.invalidateQueries({ queryKey: ['deckCards'] });
    qc.invalidateQueries({ queryKey: ['jobCards'] });
  };
}

export function usePatchCard() {
  const api = useApi();
  const invalidate = useInvalidateLibrary();
  return useMutation({
    mutationFn: (v: { id: string; patch: CardPatch }): Promise<Card> => api.patchCard(v.id, v.patch),
    onSuccess: invalidate,
  });
}

export function useDeleteCard() {
  const api = useApi();
  const invalidate = useInvalidateLibrary();
  return useMutation({ mutationFn: (id: string) => api.deleteCard(id), onSuccess: invalidate });
}

/** Approve/reject cards. No automatic invalidation: the review screen keeps its own queue while swiping. */
export function useReviewCards() {
  const api = useApi();
  return useMutation({
    mutationFn: (v: { approve?: string[]; reject?: string[] }) => api.reviewCards(v.approve ?? [], v.reject ?? []),
  });
}
