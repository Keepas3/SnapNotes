/**
 * What actually goes on the wire. The phone's global fetch is Expo's own (expo/src/winter/fetch), not React Native's, and it
 * turns a FormData into bytes itself. It throws on the old `{ uri, name, type }` file part, which the app used to send, and
 * the app reported that as "cannot reach the server" while nothing had left the phone. So this test pushes the app's real upload
 * through Expo's real converter instead of checking what the app appends.
 */
import { convertFormDataAsync } from 'expo/src/winter/fetch/convertFormData';

import { ApiClient } from '@/api/client';
import { textNoteFile } from '@/lib/upload';

// React Native's own FormData class (it ships no type declarations), and the patch Expo applies to it.
const RNFormData = jest.requireActual('react-native/Libraries/Network/FormData').default;
const { installFormDataPatch } = jest.requireActual('expo/src/winter/FormData');

jest.mock('@/lib/localFile', () => ({
  readLocalFile: async (uri: string) => new TextEncoder().encode(`contents of ${uri}`),
}));

describe('upload as Expo fetch will send it', () => {
  const RealFormData = globalThis.FormData;
  beforeAll(() => {
    // the same patch Expo installs on the phone (expo/src/winter/runtime.native.ts)
    globalThis.FormData = installFormDataPatch(RNFormData) as unknown as typeof FormData;
  });
  afterAll(() => {
    globalThis.FormData = RealFormData;
  });

  it('turns two files and the flag into a well-formed multipart body', async () => {
    let wire = '';
    const impl = jest.fn(async (_url: string, init: RequestInit = {}) => {
      const { body } = await convertFormDataAsync(init.body as FormData, 'TESTBOUNDARY'); // throws on anything Expo cannot send
      wire = new TextDecoder().decode(body);
      return { ok: true, status: 202, json: async () => ({ document_id: 'd', job_id: 'j' }) } as unknown as Response;
    });

    const res = await new ApiClient('http://srv', 't', impl as unknown as typeof fetch).uploadDocument(
      [
        { uri: 'file:///cache/DocumentPicker/cell-biology-sample.md', name: 'cell-biology-sample.md', mimeType: 'text/markdown' },
        { uri: 'file:///cache/ImagePicker/abc.webp', name: 'abc.webp', mimeType: 'image/webp' },
      ],
      false,
    );

    expect(res).toEqual({ document_id: 'd', job_id: 'j' });
    expect(wire).toContain(
      '--TESTBOUNDARY\r\n' +
        'content-disposition: form-data; name="files"; filename="cell-biology-sample.md"\r\n' +
        'content-type: text/markdown\r\n\r\n' +
        'contents of file:///cache/DocumentPicker/cell-biology-sample.md\r\n',
    );
    expect(wire).toContain('filename="abc.webp"\r\ncontent-type: image/webp\r\n\r\ncontents of file:///cache/ImagePicker/abc.webp\r\n');
    expect(wire).toContain('name="auto_approve"\r\n\r\nfalse\r\n');
    expect(wire.endsWith('--TESTBOUNDARY--\r\n')).toBe(true);
  });
});

describe('typed notes as an upload', () => {
  it('goes out as a text file with its own contents, without reading anything from the device', async () => {
    let wire = '';
    const impl = jest.fn(async (_url: string, init: RequestInit = {}) => {
      const RealFormData = globalThis.FormData;
      globalThis.FormData = installFormDataPatch(RNFormData) as unknown as typeof FormData;
      try {
        const { body } = await convertFormDataAsync(init.body as FormData, 'TESTBOUNDARY');
        wire = new TextDecoder().decode(body);
      } finally {
        globalThis.FormData = RealFormData;
      }
      return { ok: true, status: 202, json: async () => ({ document_id: 'd', job_id: 'j' }) } as unknown as Response;
    });
    const RealFormData = globalThis.FormData;
    globalThis.FormData = installFormDataPatch(RNFormData) as unknown as typeof FormData;
    try {
      await new ApiClient('http://srv', 't', impl as unknown as typeof fetch).uploadDocument([textNoteFile('# Binary search\nSplit the range in half.')], false);
    } finally {
      globalThis.FormData = RealFormData;
    }
    expect(wire).toContain('filename="Binary-search.txt"\r\ncontent-type: text/plain\r\n\r\n# Binary search\nSplit the range in half.\r\n');
    expect(wire).not.toContain('contents of'); // the stand-in file reader was never used
  });
});
