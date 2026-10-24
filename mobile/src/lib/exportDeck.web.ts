import { ApiError, type ApiClient } from '@/api/client';

import { exportFileName, type ExportFormat } from './exportFile';

/** Browser preview version: fetch the export with the token, then trigger a normal file download. */
export async function exportDeck(api: ApiClient, deckId: string, deckName: string, format: ExportFormat): Promise<'shared' | 'saved'> {
  const { url, headers } = api.exportRequest(deckId, format);
  const res = await fetch(url, { headers });
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, `${res.status}: ${detail}`);
  }

  const objectUrl = URL.createObjectURL(await res.blob());
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = exportFileName(deckName, format);
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 10_000);
  return 'saved';
}
