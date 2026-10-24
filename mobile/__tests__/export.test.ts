import { ApiClient, insecureUrlProblem } from '@/api/client';
import { describeExportFailure, EXPORT_FORMATS, exportFileName } from '@/lib/exportFile';

describe('exportFileName', () => {
  it('uses the deck name with the format as extension', () => {
    expect(exportFileName('Cell biology - week 3', 'apkg')).toBe('Cell biology - week 3.apkg');
    expect(exportFileName('Chemistry', 'csv')).toBe('Chemistry.csv');
  });

  it('keeps letters from any language and replaces everything that could break a path', () => {
    expect(exportFileName('Biología: "células"/week 1', 'apkg')).toBe('Biología_ _células_week 1.apkg');
    expect(exportFileName('日本語のノート', 'csv')).toBe('日本語のノート.csv');
    expect(exportFileName('../../etc/passwd', 'apkg')).not.toMatch(/[\\/]/);
    expect(exportFileName('a\\b:c*d?e', 'csv')).toBe('a_b_c_d_e.csv');
  });

  it('never returns an empty or hidden file name', () => {
    expect(exportFileName('', 'apkg')).toBe('deck.apkg');
    expect(exportFileName('...', 'csv')).toBe('deck.csv');
    expect(exportFileName('  ', 'csv')).toBe('deck.csv');
  });

  it('limits very long names', () => {
    const name = exportFileName('x'.repeat(500), 'apkg');
    expect(name.length).toBeLessThanOrEqual(65);
    expect(name.endsWith('.apkg')).toBe(true);
  });
});

describe('describeExportFailure', () => {
  it.each([
    ['UnableToDownload: HTTP 409', /no approved cards/],
    ['409: This deck has no approved cards to export yet', /no approved cards/],
    ['Request failed (401).', /access token/],
    ['Download failed with status 404', /no longer exists/],
    ['Network request failed', /Could not reach the server/],
    ['The request timed out', /Could not reach the server/],
  ])('%s', (message, expected) => {
    expect(describeExportFailure(new Error(message))).toMatch(expected);
  });

  it('falls back to the underlying message for anything else', () => {
    expect(describeExportFailure(new Error('disk full'))).toBe('The export failed: disk full');
    expect(describeExportFailure('plain string')).toBe('The export failed: plain string');
  });
});

describe('export formats and requests', () => {
  it('describes both formats for the share sheet', () => {
    expect(EXPORT_FORMATS.apkg.label).toMatch(/Anki/);
    expect(EXPORT_FORMATS.csv.mimeType).toBe('text/csv');
  });

  it('builds an authenticated, encoded download request', () => {
    const api = new ApiClient('https://snapnotes.fly.dev', 'secret');
    expect(api.exportRequest('abc123', 'apkg')).toEqual({
      url: 'https://snapnotes.fly.dev/decks/abc123/export?format=apkg',
      headers: { Authorization: 'Bearer secret' },
    });
    expect(api.exportRequest('a/b', 'csv').url).toBe('https://snapnotes.fly.dev/decks/a%2Fb/export?format=csv');
    expect(() => api.exportRequest('', 'csv')).toThrow(/deck id/);
  });
});

describe('insecureUrlProblem', () => {
  it('lets development builds use plain HTTP (Expo Go, emulators, a laptop on the same Wi-Fi)', () => {
    expect(insecureUrlProblem('http://192.168.1.20:8000', true)).toBeNull();
  });

  it('explains why an installed build cannot use plain HTTP', () => {
    expect(insecureUrlProblem('http://192.168.1.20:8000', false)).toMatch(/HTTPS/);
    expect(insecureUrlProblem('HTTP://example.com', false)).toMatch(/DEPLOY\.md/);
  });

  it('accepts HTTPS everywhere', () => {
    expect(insecureUrlProblem('https://snapnotes.fly.dev', false)).toBeNull();
    expect(insecureUrlProblem('https://snapnotes.fly.dev', true)).toBeNull();
  });
});
