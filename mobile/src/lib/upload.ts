import type { UploadFile } from '@/api/types';

/** Must match the server's limits (backend/app/core/config.py and app/ingest/loader.py). */
export const MAX_FILES = 20;
export const MAX_TOTAL_BYTES = 30 * 1024 * 1024;
export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

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
