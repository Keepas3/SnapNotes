import type { UploadFile } from '@/api/types';

/** Must match the server's limits (backend/app/core/config.py and app/ingest/loader.py). */
export const MAX_FILES = 20;
export const MAX_TOTAL_BYTES = 30 * 1024 * 1024;
export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

/**
 * Notes typed or pasted into the box. About 3,000 words: plenty for a lecture or a chapter's key points, and small enough that the AI
 * reads it quickly and for little cost. Longer notes go in as a file, which has no such limit (only the 30 MB one for everything).
 */
export const MAX_TEXT_CHARS = 20_000;
/** A line or two is not enough to make cards from. */
export const MIN_TEXT_CHARS = 20;

const IMAGE_EXT = new Set(['jpg', 'jpeg', 'png', 'gif', 'webp']);
export const ALLOWED_EXT = new Set(['pdf', 'docx', 'pptx', 'md', 'markdown', 'txt', ...IMAGE_EXT]);

export function extOf(name: string): string {
  const i = name.lastIndexOf('.');
  return i < 0 ? '' : name.slice(i + 1).toLowerCase();
}

export function isImage(file: Pick<UploadFile, 'name' | 'mimeType'>): boolean {
  return IMAGE_EXT.has(extOf(file.name)) || (file.mimeType?.startsWith('image/') ?? false);
}

const MIME_EXT: Record<string, string> = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif', 'image/webp': 'webp' };

/** Camera and gallery assets often have no usable file name; the server decides the type from the extension. */
export function photoName(asset: { fileName?: string | null; mimeType?: string | null; uri: string }, index: number, now = Date.now()): string {
  if (asset.fileName && ALLOWED_EXT.has(extOf(asset.fileName))) return asset.fileName;
  const ext = (asset.mimeType && MIME_EXT[asset.mimeType]) || (ALLOWED_EXT.has(extOf(asset.uri)) ? extOf(asset.uri) : 'jpg');
  return `photo-${now}-${index + 1}.${ext}`;
}

/** Why typed notes cannot be sent yet, or null when they can (or when the box is empty, which is fine if files were chosen). */
export function textProblem(text: string): string | null {
  const n = text.trim().length;
  if (n === 0) return null;
  if (n < MIN_TEXT_CHARS) return `Write a little more: at least ${MIN_TEXT_CHARS} characters, so there is something to make cards from.`;
  if (text.length > MAX_TEXT_CHARS) return `The typed notes are ${(text.length - MAX_TEXT_CHARS).toLocaleString('en-US')} characters over the ${MAX_TEXT_CHARS.toLocaleString('en-US')} limit. Shorten them, or save them as a file and choose it instead.`;
  return null;
}

/** A readable file name for typed notes, from their first line ("# Binary search" becomes "Binary-search.txt"). */
export function textNoteName(text: string): string {
  const first = text.split(/\r?\n/).find((line) => line.trim()) ?? '';
  const words = first
    .replace(/^[#>*\-\s]+/, '')
    .replace(/[\/:*?"<>|\u0000-\u001f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 40)
    .trim()
    .replace(/\s/g, '-'); // no spaces: the upload percent-encodes them and the server shows the name as it arrives
  return `${words || 'Typed-notes'}.txt`;
}

/** The typed notes as something that uploads like any other file. */
export function textNoteFile(text: string): UploadFile {
  return { uri: '', text, name: textNoteName(text), mimeType: 'text/plain', size: new TextEncoder().encode(text).length };
}

/** First problem with the chosen files, or null when they can be uploaded. */
export function validateSelection(files: UploadFile[]): string | null {
  if (files.length === 0) return 'Choose at least one file.';
  if (files.length > MAX_FILES) return `Choose at most ${MAX_FILES} files.`;
  for (const f of files) {
    if (!ALLOWED_EXT.has(extOf(f.name))) return `${f.name} is not a supported type (PDF, Word, PowerPoint, Markdown, text or photo).`;
    if (isImage(f) && f.size && f.size > MAX_IMAGE_BYTES) return `${f.name} is over 5 MB. Use a smaller photo.`;
  }
  const total = files.reduce((n, f) => n + (f.size ?? 0), 0);
  if (total > MAX_TOTAL_BYTES) return `These files total ${(total / 1048576).toFixed(0)} MB; the limit is 30 MB.`;
  return null;
}

export function formatBytes(n?: number): string {
  if (!n) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
