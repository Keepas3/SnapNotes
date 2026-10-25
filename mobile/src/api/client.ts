import { Platform } from 'react-native';

import { readLocalFile } from '@/lib/localFile';

import type {
  Card,
  CardPatch,
  CardStatus,
  Deck,
  EvidenceResponse,
  Folder,
  Job,
  MergeResult,
  NewCard,
  SyncRequest,
  SyncResponse,
  UploadFile,
} from './types';

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }

  /** The request never produced an HTTP response (offline, wrong address, timeout). */
  get isNetwork() {
    return this.status === 0;
  }
}

const DEFAULT_TIMEOUT_MS = 20_000;
const UPLOAD_TIMEOUT_MS = 180_000;

type FetchLike = typeof fetch;

interface RequestOptions {
  json?: unknown;
  form?: FormData;
  query?: Record<string, string | number | undefined>;
  timeoutMs?: number;
}

/** "192.168.1.5:8000" -> "http://192.168.1.5:8000"; strips trailing slashes. Returns null if unusable. */
export function normalizeBaseUrl(raw: string): string | null {
  let url = raw.trim();
  if (!url) return null;
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(url)) url = `http://${url}`;
  if (!/^https?:\/\//i.test(url)) return null;
  try {
    const parsed = new URL(url);
    if (!parsed.hostname) return null;
  } catch {
    return null;
  }
  return url.replace(/\/+$/, '');
}

/**
 * Installed builds (not the Expo Go / dev server) refuse plain-HTTP connections on both platforms, which shows up as a
 * confusing "network request failed". Say so up front instead.
 */
export function insecureUrlProblem(url: string, isDev: boolean): string | null {
  if (isDev || !/^http:\/\//i.test(url)) return null;
  return 'Installed builds only connect over HTTPS. Use the https:// address of your deployed server (see backend/DEPLOY.md).';
}

/** FastAPI errors are `{detail: string}` or `{detail: [{msg, loc}]}`. */
function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: string };
      if (first?.msg) return first.msg;
    }
  }
  return fallback;
}

/** " (the underlying reason)" for a failed request, so "cannot reach the server" is not the only clue. */
function reasonOf(e: unknown): string {
  const text = e instanceof Error ? e.message.replace(/^fetch failed:\s*/i, '').trim() : '';
  return text ? ` (${text.slice(0, 160)})` : '';
}

async function readUpload(f: UploadFile): Promise<Uint8Array> {
  try {
    return await readLocalFile(f.uri);
  } catch (e) {
    console.warn(`Could not read ${f.name} (${f.uri}):`, e);
    throw new ApiError(0, `Could not read ${f.name} from this device${reasonOf(e)}. Choose it again.`);
  }
}

export class ApiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly token: string,
    private readonly fetchImpl: FetchLike = (...args) => fetch(...args),
  ) {}

  private async request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
    const params = Object.entries(opts.query ?? {}).filter(([, v]) => v !== undefined);
    const qs = params.length
      ? `?${params.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&')}`
      : '';

    const headers: Record<string, string> = { Authorization: `Bearer ${this.token}` };
    let body: BodyInit | undefined;
    if (opts.json !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(opts.json);
    } else if (opts.form) {
      body = opts.form; // fetch sets the multipart boundary itself
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    let res: Response;
    try {
      res = await this.fetchImpl(`${this.baseUrl}${path}${qs}`, { method, headers, body, signal: controller.signal });
    } catch (e) {
      const timedOut = e instanceof Error && e.name === 'AbortError';
      throw new ApiError(0, timedOut ? 'The server took too long to respond.' : `Cannot reach the server at ${this.baseUrl}.${reasonOf(e)}`);
    } finally {
      clearTimeout(timer);
    }

    if (res.status === 204) return undefined as T;
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      /* non-JSON body (e.g. proxy error page) */
    }
    if (!res.ok) {
      const fallback = res.status === 401 ? 'The access token was rejected.' : `Request failed (${res.status}).`;
      throw new ApiError(res.status, errorMessage(payload, fallback));
    }
    return payload as T;
  }

  /** Unauthenticated liveness check. */
  async health(): Promise<void> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8_000);
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/health`, { signal: controller.signal });
      if (!res.ok) throw new ApiError(res.status, `The server answered ${res.status}.`);
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(0, `Cannot reach the server at ${this.baseUrl}.${reasonOf(e)}`);
    } finally {
      clearTimeout(timer);
    }
  }

  // -- documents & jobs ---------------------------------------------------------------

  /** `deckId`: add the new cards to that existing deck instead of making a new one. */
  async uploadDocument(files: UploadFile[], autoApprove: boolean, deckId?: string): Promise<{ document_id: string; job_id: string }> {
    const form = new FormData();
    for (const f of files) {
      if (Platform.OS === 'web') {
        const blob = await (await this.fetchImpl(f.uri)).blob();
        form.append('files', blob, f.name);
      } else {
        // Expo's fetch (the global one since SDK 53) cannot send React Native's old `{ uri, name, type }` file part: it
        // throws before anything goes out. It accepts any object with a name, a type and bytes(), so read the file here,
        // which also lets an unreadable file be reported by name instead of as a network failure.
        const bytes = await readUpload(f);
        form.append('files', { name: f.name, type: f.mimeType ?? 'application/octet-stream', bytes: async () => bytes } as unknown as Blob);
      }
    }
    form.append('auto_approve', autoApprove ? 'true' : 'false');
    if (deckId) form.append('deck_id', deckId);
    return this.request('POST', '/documents', { form, timeoutMs: UPLOAD_TIMEOUT_MS });
  }

  deleteDocument(documentId: string): Promise<void> {
    return this.request('DELETE', `/documents/${documentId}`);
  }

  listJobs(limit = 50): Promise<Job[]> {
    return this.request('GET', '/jobs', { query: { limit } });
  }

  getJob(id: string): Promise<Job> {
    return this.request('GET', `/jobs/${id}`);
  }

  getEvidence(jobId: string): Promise<EvidenceResponse> {
    return this.request('GET', `/jobs/${jobId}/evidence`);
  }

  approveSources(jobId: string, rejectedIds: number[]): Promise<{ approved: number; rejected: number }> {
    return this.request('POST', `/jobs/${jobId}/approve`, { json: { rejected_ids: rejectedIds } });
  }

  retryJob(jobId: string): Promise<Job> {
    return this.request('POST', `/jobs/${jobId}/retry`);
  }

  jobCards(jobId: string, status?: CardStatus): Promise<Card[]> {
    return this.request('GET', `/jobs/${jobId}/cards`, { query: { status } });
  }

  // -- decks & cards ------------------------------------------------------------------

  listDecks(): Promise<Deck[]> {
    return this.request('GET', '/decks');
  }

  getDeck(id: string): Promise<Deck> {
    if (!id) throw new Error('getDeck needs a deck id'); // "/decks/" would silently return the deck list
    return this.request('GET', `/decks/${id}`);
  }

  renameDeck(id: string, name: string): Promise<Deck> {
    if (!id) throw new Error('renameDeck needs a deck id');
    return this.request('PATCH', `/decks/${id}`, { json: { name } });
  }

  deleteDeck(id: string): Promise<void> {
    return this.request('DELETE', `/decks/${id}`);
  }

  // -- folders ------------------------------------------------------------------------

  listFolders(): Promise<Folder[]> {
    return this.request('GET', '/folders');
  }

  createFolder(name: string): Promise<Folder> {
    return this.request('POST', '/folders', { json: { name } });
  }

  renameFolder(id: string, name: string): Promise<Folder> {
    if (!id) throw new Error('renameFolder needs a folder id');
    return this.request('PATCH', `/folders/${id}`, { json: { name } });
  }

  /** Removes the folder only; its decks are kept and become unfiled. */
  deleteFolder(id: string): Promise<void> {
    if (!id) throw new Error('deleteFolder needs a folder id');
    return this.request('DELETE', `/folders/${id}`);
  }

  /** File a deck in a folder, or take it out with `null`. */
  setDeckFolder(deckId: string, folderId: string | null): Promise<Deck> {
    if (!deckId) throw new Error('setDeckFolder needs a deck id');
    return this.request('PATCH', `/decks/${deckId}`, { json: { folder_id: folderId } });
  }

  /** Move every card of `fromId` into `intoId` and remove `fromId`. Study history travels with the cards. */
  mergeDecks(intoId: string, fromId: string): Promise<MergeResult> {
    if (!intoId || !fromId) throw new Error('mergeDecks needs two deck ids');
    return this.request('POST', `/decks/${intoId}/merge`, { json: { from_deck_id: fromId } });
  }

  /** Add a card you typed yourself; it is approved at once and reaches the phone on the next sync. */
  createCard(deckId: string, card: NewCard): Promise<Card> {
    if (!deckId) throw new Error('createCard needs a deck id');
    return this.request('POST', `/decks/${deckId}/cards`, { json: card });
  }

  deckCards(deckId: string, status?: CardStatus): Promise<Card[]> {
    return this.request('GET', `/decks/${deckId}/cards`, { query: { status } });
  }

  patchCard(id: string, patch: CardPatch): Promise<Card> {
    return this.request('PATCH', `/cards/${id}`, { json: patch });
  }

  deleteCard(id: string): Promise<void> {
    return this.request('DELETE', `/cards/${id}`);
  }

  reviewCards(approveIds: string[], rejectIds: string[]): Promise<{ changed: number }> {
    return this.request('POST', '/cards/review', { json: { approve_ids: approveIds, reject_ids: rejectIds } });
  }

  /** URL and headers for downloading a deck export. The file is fetched by the platform's downloader, not by `request`. */
  exportRequest(deckId: string, format: 'apkg' | 'csv'): { url: string; headers: Record<string, string> } {
    if (!deckId) throw new Error('exportRequest needs a deck id');
    return {
      url: `${this.baseUrl}/decks/${encodeURIComponent(deckId)}/export?format=${format}`,
      headers: { Authorization: `Bearer ${this.token}` },
    };
  }

  // -- sync ---------------------------------------------------------------------------

  /** Upload review logs and fetch what changed on the server since `req.cursor`. */
  sync(req: SyncRequest): Promise<SyncResponse> {
    return this.request('POST', '/sync', { json: req, timeoutMs: 60_000 });
  }
}
