/**
 * Reads a file from the phone's own storage (the path a picker or the camera hands back) into memory.
 *
 * This goes through React Native's networking layer, which opens `file://` and `content://` locations itself and hands the
 * result back as a Blob. Expo's file API (`new File(uri).bytes()`) checks every path against its own permission service
 * instead, and inside Expo Go that rejects the document picker's cache folder with "missing read permission".
 */
export function readLocalFile(uri: string): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.responseType = 'blob';
    request.onerror = () => reject(new Error('the file could not be opened'));
    request.onload = () => {
      if (request.status !== 200 && request.status !== 0) {
        reject(new Error(`the file could not be opened (status ${request.status})`));
        return;
      }
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('the file could not be read'));
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) resolve(new Uint8Array(reader.result));
        else reject(new Error('the file could not be read'));
      };
      reader.readAsArrayBuffer(request.response as Blob);
    };
    request.open('GET', uri);
    request.send();
  });
}
