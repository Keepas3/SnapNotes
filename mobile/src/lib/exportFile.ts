export type ExportFormat = 'apkg' | 'csv';

export const EXPORT_FORMATS: Record<ExportFormat, { label: string; mimeType: string; uti?: string }> = {
  apkg: { label: 'Export to Anki (.apkg)', mimeType: 'application/octet-stream' },
  csv: { label: 'Export as CSV', mimeType: 'text/csv', uti: 'public.comma-separated-values-text' },
};

/** A safe file name for a deck export: letters (any language), digits, spaces, dots, dashes; never a path. */
export function exportFileName(deckName: string, format: ExportFormat): string {
  const stem = deckName
    .replace(/[^\p{L}\p{N} ._-]+/gu, '_')
    .replace(/^[ ._-]+|[ ._-]+$/g, '')
    .slice(0, 60);
  return `${stem || 'deck'}.${format}`;
}

/** Turn whatever the download threw (server error, offline, file system) into a sentence for the user. */
export function describeExportFailure(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/\b409\b/.test(message)) return 'This deck has no approved cards to export yet.';
  if (/\b401\b/.test(message)) return 'The server rejected the access token. Check Settings.';
  if (/\b404\b/.test(message)) return 'That deck no longer exists on the server.';
  if (/network|reach|fetch|connect|timed? ?out/i.test(message)) return 'Could not reach the server. Exports need a connection.';
  return `The export failed: ${message}`;
}
