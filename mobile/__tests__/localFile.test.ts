import { readLocalFile } from '@/lib/localFile';

/** Just enough of XMLHttpRequest and FileReader to drive readLocalFile: the real ones only exist on the phone. */
class FakeRequest {
  static next: { status?: number; fail?: boolean; response?: unknown } = {};
  static opened: [string, string][] = [];
  responseType = '';
  status = 0;
  response: unknown = null;
  onerror: (() => void) | null = null;
  onload: (() => void) | null = null;
  open(method: string, url: string) {
    FakeRequest.opened.push([method, url]);
  }
  send() {
    const { status = 200, fail = false, response = 'BLOB' } = FakeRequest.next;
    setTimeout(() => {
      if (fail) return this.onerror?.();
      this.status = status;
      this.response = response;
      this.onload?.();
    });
  }
}

class FakeReader {
  static result: unknown = new Uint8Array([104, 105]).buffer;
  static fail = false;
  result: unknown = null;
  onerror: (() => void) | null = null;
  onload: (() => void) | null = null;
  read: unknown = null;
  readAsArrayBuffer(blob: unknown) {
    this.read = blob;
    setTimeout(() => {
      if (FakeReader.fail) return this.onerror?.();
      this.result = FakeReader.result;
      this.onload?.();
    });
  }
}

describe('readLocalFile', () => {
  const g = globalThis as unknown as Record<string, unknown>;
  const real = { XMLHttpRequest: g.XMLHttpRequest, FileReader: g.FileReader };

  beforeEach(() => {
    g.XMLHttpRequest = FakeRequest;
    g.FileReader = FakeReader;
    FakeRequest.next = {};
    FakeRequest.opened = [];
    FakeReader.result = new Uint8Array([104, 105]).buffer;
    FakeReader.fail = false;
  });
  afterAll(() => {
    g.XMLHttpRequest = real.XMLHttpRequest;
    g.FileReader = real.FileReader;
  });

  it('reads the file named by the path into bytes, asking the request for a blob', async () => {
    const bytes = await readLocalFile('file:///cache/DocumentPicker/a.md');
    expect(Array.from(bytes)).toEqual([104, 105]);
    expect(FakeRequest.opened).toEqual([['GET', 'file:///cache/DocumentPicker/a.md']]);
  });

  it('reports a file that cannot be opened', async () => {
    FakeRequest.next = { fail: true };
    await expect(readLocalFile('file:///gone')).rejects.toThrow('the file could not be opened');
    FakeRequest.next = { status: 404 };
    await expect(readLocalFile('file:///gone')).rejects.toThrow('status 404');
  });

  it('reports contents that cannot be read', async () => {
    FakeReader.fail = true;
    await expect(readLocalFile('file:///a')).rejects.toThrow('the file could not be read');
    FakeReader.fail = false;
    FakeReader.result = 'not an ArrayBuffer';
    await expect(readLocalFile('file:///a')).rejects.toThrow('the file could not be read');
  });
});
