import { File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';

import type { ApiClient } from '@/api/client';

import { EXPORT_FORMATS, exportFileName, type ExportFormat } from './exportFile';

/**
 * Download a deck export from the server and hand it to the system share sheet, from where it can be opened in
 * Anki (AnkiDroid / AnkiMobile), saved to Files, or sent elsewhere.
 */
export async function exportDeck(api: ApiClient, deckId: string, deckName: string, format: ExportFormat): Promise<'shared' | 'saved'> {
  const { url, headers } = api.exportRequest(deckId, format);
  const target = new File(Paths.cache, exportFileName(deckName, format));
  const file = await File.downloadFileAsync(url, target, { headers, idempotent: true });

  const info = EXPORT_FORMATS[format];
  if (!(await Sharing.isAvailableAsync())) return 'saved';
  await Sharing.shareAsync(file.uri, { mimeType: info.mimeType, UTI: info.uti, dialogTitle: info.label });
  return 'shared';
}
