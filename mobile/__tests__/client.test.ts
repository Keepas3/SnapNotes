import { ApiClient, ApiError, normalizeBaseUrl } from '@/api/client';

// The phone reads a chosen file from its own storage; here a file "exists" unless its name says otherwise.
jest.mock('@/lib/localFile', () => ({
  readLocalFile: async (uri: string) => {
    if (uri.includes('missing')) throw new Error('no such file');
    return new TextEncoder().encode(`contents of ${uri}`);
  },
}));

type Call = { url: string; init: RequestInit };

function fakeFetch(responses: ({ status?: number; body?: unknown } | Error)[]) {
  const calls: Call[] = [];
  const impl = jest.fn(async (url: string, init: RequestInit = {}) => {
    calls.push({ url, init });
    const next = responses.shift();
    if (next === undefined) throw new Error('unexpected request');
    if (next instanceof Error) throw next;
    const status = next.status ?? 200;
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => {
        if (next.body === undefined) throw new Error('no body');
        return next.body;
      },
    } as unknown as Response;
  });
  return { impl: impl as unknown as typeof fetch, calls };
}

describe('normalizeBaseUrl', () => {
  it.each([
    ['http://192.168.1.5:8000/', 'http://192.168.1.5:8000'],
    ['192.168.1.5:8000', 'http://192.168.1.5:8000'],
    ['  https://api.example.com//  ', 'https://api.example.com'],
    ['localhost:8000', 'http://localhost:8000'],
  ])('%s -> %s', (input, expected) => {
    expect(normalizeBaseUrl(input)).toBe(expected);
  });

  it.each([[''], ['   '], ['ftp://example.com'], ['javascript:alert(1)']])('rejects %p', (input) => {
    expect(normalizeBaseUrl(input)).toBeNull();
  });
});

describe('ApiClient', () => {
  it('sends the bearer token and query string, and returns parsed JSON', async () => {
    const { impl, calls } = fakeFetch([{ body: [{ id: 'c1' }] }]);
    const api = new ApiClient('http://srv', 'secret', impl);

    const cards = await api.jobCards('job 1', 'draft');

    expect(cards).toEqual([{ id: 'c1' }]);
    expect(calls[0].url).toBe('http://srv/jobs/job 1/cards?status=draft');
    expect((calls[0].init.headers as Record<string, string>).Authorization).toBe('Bearer secret');
    expect(calls[0].init.method).toBe('GET');
  });

  it('omits undefined query params', async () => {
    const { impl, calls } = fakeFetch([{ body: [] }]);
    await new ApiClient('http://srv', 't', impl).deckCards('d1');
    expect(calls[0].url).toBe('http://srv/decks/d1/cards');
  });

  it('posts JSON bodies', async () => {
    const { impl, calls } = fakeFetch([{ body: { approved: 2, rejected: 1 } }]);
    const res = await new ApiClient('http://srv', 't', impl).approveSources('j1', [4, 5]);
    expect(res).toEqual({ approved: 2, rejected: 1 });
    expect(calls[0].init.body).toBe(JSON.stringify({ rejected_ids: [4, 5] }));
    expect((calls[0].init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
  });

  it('turns FastAPI error bodies into readable ApiErrors', async () => {
    const { impl } = fakeFetch([
      { status: 409, body: { detail: 'Job is completed, not awaiting_approval' } },
      { status: 422, body: { detail: [{ msg: 'Field required', loc: ['body'] }] } },
      { status: 401, body: {} },
      { status: 502 },
    ]);
    const api = new ApiClient('http://srv', 't', impl);

    await expect(api.approveSources('j', [])).rejects.toMatchObject({ status: 409, message: 'Job is completed, not awaiting_approval' });
    await expect(api.getJob('j')).rejects.toMatchObject({ status: 422, message: 'Field required' });
    await expect(api.getJob('j')).rejects.toMatchObject({ status: 401, message: 'The access token was rejected.' });
    await expect(api.getJob('j')).rejects.toMatchObject({ status: 502, message: 'Request failed (502).' });
  });

  it('renames a deck with a PATCH that carries only the new name', async () => {
    const { impl, calls } = fakeFetch([{ status: 200, body: { id: 'd1', name: 'Biology', topic: '', counts: { draft: 0, approved: 0, rejected: 0 } } }]);
    const deck = await new ApiClient('http://srv', 't', impl).renameDeck('d1', 'Biology');
    expect(deck.name).toBe('Biology');
    expect(calls[0].url).toBe('http://srv/decks/d1');
    expect(calls[0].init.method).toBe('PATCH');
    expect(JSON.parse(calls[0].init.body as string)).toEqual({ name: 'Biology' });
    expect(() => new ApiClient('http://srv', 't', fakeFetch([]).impl).renameDeck('', 'x')).toThrow(/needs a deck id/);
  });

  it('merges one deck into another and says how many cards moved', async () => {
    const deck = { id: 'keep', name: 'Keep', topic: '', counts: { draft: 0, approved: 5, rejected: 0 } };
    const { impl, calls } = fakeFetch([{ status: 200, body: { deck, moved: 3 } }]);
    const api = new ApiClient('http://srv', 't', impl);

    await expect(api.mergeDecks('keep', 'gone')).resolves.toEqual({ deck, moved: 3 });
    expect(calls[0].url).toBe('http://srv/decks/keep/merge');
    expect(calls[0].init.method).toBe('POST');
    expect(JSON.parse(calls[0].init.body as string)).toEqual({ from_deck_id: 'gone' });
    expect(() => api.mergeDecks('', 'gone')).toThrow(/two deck ids/);
    expect(() => api.mergeDecks('keep', '')).toThrow(/two deck ids/);
  });

  it('shows the server\'s plain-words reason when a merge or a typed card is refused', async () => {
    const { impl } = fakeFetch([
      { status: 409, body: { detail: 'A deck cannot be merged into itself' } },
      { status: 422, body: { detail: 'A fill-in-the-blank card needs a blank: wrap the answer like {{c1::answer}}.' } },
    ]);
    const api = new ApiClient('http://srv', 't', impl);
    await expect(api.mergeDecks('a', 'a')).rejects.toMatchObject({ status: 409, message: 'A deck cannot be merged into itself' });
    await expect(api.createCard('d1', { type: 'cloze', front: 'no blank', back: '' })).rejects.toMatchObject({ status: 422, message: expect.stringContaining('needs a blank') });
  });

  it('creates a typed card with a POST to the deck', async () => {
    const { impl, calls } = fakeFetch([{ status: 201, body: { id: 'c9', origin: 'manual', status: 'approved' } }]);
    const made = await new ApiClient('http://srv', 't', impl).createCard('d1', { type: 'basic', front: 'Capital of France?', back: 'Paris' });
    expect(made).toMatchObject({ id: 'c9', origin: 'manual' });
    expect(calls[0].url).toBe('http://srv/decks/d1/cards');
    expect(calls[0].init.method).toBe('POST');
    expect(JSON.parse(calls[0].init.body as string)).toEqual({ type: 'basic', front: 'Capital of France?', back: 'Paris' });
    expect(() => new ApiClient('http://srv', 't', fakeFetch([]).impl).createCard('', { type: 'basic', front: 'a', back: 'b' })).toThrow(/needs a deck id/);
  });

  it('reports an unreachable server as a network error', async () => {
    const { impl } = fakeFetch([new TypeError('Network request failed')]);
    const err = await new ApiClient('http://10.0.0.9:8000', 't', impl).listDecks().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.isNetwork).toBe(true);
    expect(err.message).toContain('http://10.0.0.9:8000');
  });

  it('refuses to fetch a deck without an id (the URL would hit the deck list)', () => {
    expect(() => new ApiClient('http://srv', 't', fakeFetch([]).impl).getDeck('')).toThrow(/needs a deck id/);
  });

  it('returns undefined for 204 responses', async () => {
    const { impl } = fakeFetch([{ status: 204 }]);
    await expect(new ApiClient('http://srv', 't', impl).deleteCard('c1')).resolves.toBeUndefined();
  });

  it('uploads files as multipart with the auto-approve flag', async () => {
    // Jest runs on Node's FormData, which would stringify file-like objects.
    // Swap in a recorder so the test sees exactly what the client appends.
    // (upload-wire.test.ts checks the same upload against Expo's real form converter.)
    type Part = { name: string; type: string; bytes: () => Promise<Uint8Array> };
    const appended: [string, string | Part][] = [];
    const RealFormData = globalThis.FormData;
    globalThis.FormData = class {
      append(key: string, value: string | Part) {
        appended.push([key, value]);
      }
    } as unknown as typeof FormData;

    try {
      const { impl, calls } = fakeFetch([{ status: 202, body: { document_id: 'd', job_id: 'j' } }]);
      const res = await new ApiClient('http://srv', 't', impl).uploadDocument(
        [
          { uri: 'file:///cache/a.pdf', name: 'a.pdf', mimeType: 'application/pdf' },
          { uri: 'file:///cache/p.jpg', name: 'p.jpg' },
        ],
        true,
      );

      expect(res).toEqual({ document_id: 'd', job_id: 'j' });
      expect(appended.map(([key, value]) => [key, typeof value === 'string' ? value : { name: value.name, type: value.type }])).toEqual([
        ['files', { name: 'a.pdf', type: 'application/pdf' }],
        ['files', { name: 'p.jpg', type: 'application/octet-stream' }],
        ['auto_approve', 'true'],
      ]);
      const first = appended[0][1] as Part;
      expect(new TextDecoder().decode(await first.bytes())).toBe('contents of file:///cache/a.pdf');
      // fetch must be left to set the multipart boundary
      expect((calls[0].init.headers as Record<string, string>)['Content-Type']).toBeUndefined();
      expect(calls[0].url).toBe('http://srv/documents');
    } finally {
      globalThis.FormData = RealFormData;
    }
  });

  it('adds the deck id to an upload only when the cards are meant for an existing deck', async () => {
    const appended: [string, unknown][] = [];
    const RealFormData = globalThis.FormData;
    globalThis.FormData = class {
      append(key: string, value: unknown) {
        appended.push([key, value]);
      }
    } as unknown as typeof FormData;

    try {
      const { impl } = fakeFetch([
        { status: 202, body: { document_id: 'd', job_id: 'j' } },
        { status: 202, body: { document_id: 'd', job_id: 'j' } },
      ]);
      const api = new ApiClient('http://srv', 't', impl);
      const file = { uri: 'file:///cache/a.md', name: 'a.md' };

      await api.uploadDocument([file], false, 'deck-7');
      expect(appended.filter(([key]) => key !== 'files')).toEqual([
        ['auto_approve', 'false'],
        ['deck_id', 'deck-7'],
      ]);

      appended.length = 0;
      await api.uploadDocument([file], false);
      expect(appended.map(([key]) => key)).toEqual(['files', 'auto_approve']); // an ordinary upload carries no deck field
    } finally {
      globalThis.FormData = RealFormData;
    }
  });

  it('names a file it cannot read, with the reason, instead of calling it a network failure', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { impl, calls } = fakeFetch([]);
    const upload = new ApiClient('http://srv', 't', impl).uploadDocument([{ uri: 'file:///cache/missing.pdf', name: 'notes.pdf' }], false);
    await expect(upload).rejects.toMatchObject({
      status: 0,
      message: 'Could not read notes.pdf from this device (no such file). Choose it again.',
    });
    expect(calls).toHaveLength(0); // nothing was sent
    expect(warn).toHaveBeenCalled(); // and the details reach the Metro log
    warn.mockRestore();
  });

  it('keeps the underlying reason when a request cannot be made', async () => {
    const { impl } = fakeFetch([new Error('fetch failed: Network request failed')]);
    await expect(new ApiClient('http://srv', 't', impl).getDeck('d1')).rejects.toMatchObject({
      status: 0,
      message: 'Cannot reach the server at http://srv. (Network request failed)',
    });
    await expect(new ApiClient('http://srv', 't', fakeFetch([new Error('')]).impl).getDeck('d1')).rejects.toMatchObject({
      message: 'Cannot reach the server at http://srv.',
    });
  });
});
