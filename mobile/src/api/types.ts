/** Mirrors backend/app/api/models.py. Keep the two in sync. */

export type JobStatus =
  | 'queued'
  | 'researching'
  | 'awaiting_approval'
  | 'approved'
  | 'generating'
  | 'completed'
  | 'failed';

export type CardStatus = 'draft' | 'approved' | 'rejected';
export type CardType = 'basic' | 'cloze' | 'definition';
export type Origin = 'notes' | 'web' | 'ai' | 'manual'; // "manual": typed in by the user
export type Difficulty = 'easy' | 'medium' | 'hard';

export interface Understanding {
  main_topic: string;
  subject_area: string;
  subtopics: string[];
  key_terms: string[];
  gaps: string[];
  search_queries: string[];
  /** Statements in the notes that look factually wrong ("claim — correction"); absent from older servers and when the fact check is off. */
  suspect_claims?: string[];
  /** Set when a web search that was wanted did not work, or found nothing; says so in plain words. */
  research_note?: string;
}

export interface Job {
  id: string;
  document_id: string;
  filename: string;
  kind: string;
  status: JobStatus;
  stage: string | null;
  error: string | null;
  auto_approve: boolean;
  cost_usd: number;
  deck_id: string | null;
  /** The deck this upload was told to add its cards to; absent from older servers, and null for an ordinary upload. */
  target_deck_id?: string | null;
  understanding: Understanding | null;
  created_at: string;
  updated_at: string;
}

export interface EvidenceItem {
  id: number;
  claim: string;
  snippet: string;
  url: string;
  title: string;
  approved: boolean;
}

export interface EvidenceResponse {
  understanding: Understanding | null;
  evidence: EvidenceItem[];
}

export interface Card {
  id: string;
  deck_id: string;
  job_id: string | null;
  type: CardType;
  front: string;
  back: string;
  origin: Origin;
  source_ref: string;
  source_snippet: string;
  tags: string[];
  difficulty: Difficulty;
  status: CardStatus;
  atomic: boolean;
  grounded: boolean;
  issue: string;
  duplicate_of: string | null;
  created_at: string;
  updated_at: string;
}

export interface CardPatch {
  front?: string;
  back?: string;
  tags?: string[];
  difficulty?: Difficulty;
  status?: CardStatus;
}

/** A card typed in by hand. A fill-in-the-blank card needs a {{c1::blank}} in its front and may leave the back empty. */
export interface NewCard {
  type: Exclude<CardType, 'definition'>;
  front: string;
  back: string;
}

/** A folder that groups decks (Biology > Bio I, Bio II). `deck_count` counts the decks filed in it. */
export interface Folder {
  id: string;
  name: string;
  deck_count: number;
  created_at: string;
  updated_at: string;
}

export interface Deck {
  id: string;
  name: string;
  topic: string;
  /** The folder this deck is filed in; absent from older servers, null when it is not in one. */
  folder_id?: string | null;
  counts: Record<CardStatus, number>;
  created_at: string;
  updated_at: string;
}

/** What a merge returns: the deck that absorbed the other one, and how many cards moved. */
export interface MergeResult {
  deck: Deck;
  moved: number;
}

// -- sync (mirrors backend/app/api/sync.py) ---------------------------------------------

/** A card's FSRS schedule as computed on the device. `state`: 0 new, 1 learning, 2 review, 3 relearning. */
export interface FsrsState {
  due: string;
  stability: number;
  difficulty: number;
  elapsed_days: number;
  scheduled_days: number;
  learning_steps: number;
  reps: number;
  lapses: number;
  state: number;
  last_review: string | null;
}

export interface SyncLogWire {
  id: string;
  card_id: string;
  rating: number;
  reviewed_at: string;
  elapsed_ms: number;
  state_before: number;
  state_after: FsrsState;
}

export interface SyncRequest {
  device_id: string;
  cursor: number;
  /** The `epoch` from the last response, so the server can tell if this is no longer the database we synced with. */
  epoch?: string | null;
  logs: SyncLogWire[];
  limit?: number;
}

export interface SyncCard {
  id: string;
  deck_id: string;
  type: CardType;
  front: string;
  back: string;
  origin: Origin;
  source_ref: string;
  source_snippet: string;
  status: CardStatus;
  created_at: string;
  updated_at: string;
}

export interface SyncResponse {
  /** Identity of the server's database. */
  epoch: string;
  /**
   * True when the cursor or epoch we sent did not belong to this database (a restored backup, a rebuilt volume).
   * The response is then a full download from zero: discard the cached library and re-upload every review.
   */
  reset: boolean;
  cursor: number;
  has_more: boolean;
  server_time: string;
  accepted: number;
  duplicates: number;
  rejected: { id: string; reason: string }[];
  /** Absent from servers that predate folders. */
  folders?: { id: string; name: string; updated_at: string }[];
  decks: { id: string; name: string; topic: string; folder_id?: string | null; updated_at: string }[];
  cards: SyncCard[];
  states: { card_id: string; last_review: string; state: FsrsState }[];
  deleted: { kind: 'deck' | 'card' | 'folder'; id: string }[];
}

/** A local file chosen for upload (from the document picker, camera or photo library). */
export interface UploadFile {
  uri: string;
  name: string;
  mimeType?: string;
  size?: number;
}
