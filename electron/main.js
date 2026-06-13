import { app, BrowserWindow, ipcMain, dialog, protocol, net } from 'electron';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { writeFile } from 'node:fs/promises';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR   = join(__dirname, '..', 'web');

// Must be called before app is ready.
protocol.registerSchemesAsPrivileged([{
  scheme: 'app',
  privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
}]);

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadURL('app://web/index.html');
}

app.whenReady().then(() => {
  // Serve web/ via app:// with COOP+COEP+CORP headers so SharedArrayBuffer is
  // available for ONNX WASM threading. The coi-serviceworker.js in index.html
  // detects SharedArrayBuffer and skips service-worker registration entirely.
  protocol.handle('app', async (request) => {
    const { pathname } = new URL(request.url);
    const fileUrl = pathToFileURL(join(WEB_DIR, pathname)).href;
    const response = await net.fetch(fileUrl);
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Resource-Policy', 'same-origin');
    return new Response(response.body, { status: response.status, headers });
  });

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

ipcMain.handle('save-mask', async (event, { data, filename }) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  const result = await dialog.showSaveDialog(win, {
    defaultPath: filename,
    filters: [{ name: 'NIfTI', extensions: ['nii'] }],
  });
  if (result.canceled) return false;
  await writeFile(result.filePath, Buffer.from(data));
  return true;
});
