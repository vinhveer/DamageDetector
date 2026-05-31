import { app, BrowserWindow, dialog, ipcMain, session } from 'electron';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { labelingDefaults, listRuns, listQueue, commitSession } from './labeling/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const MAX_BASE64_CHARS = 120 * 1024 * 1024;
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']);

const isPlainObject = (value) => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const cleanPath = (value, fieldName) => {
  if (typeof value !== 'string' || value.trim() === '' || value.includes('\0')) {
    throw new Error(`Invalid ${fieldName}`);
  }
  return value.trim();
};

const createWindow = async () => {
  const mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 1040,
    minHeight: 680,
    title: 'Semi-labeling Review',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'hidden',
    trafficLightPosition: { x: 14, y: 14 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: !isDev
    }
  });

  if (isDev) {
    await mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools({ mode: 'detach' });
    return;
  }

  await mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
};

const installSecurityHeaders = () => {
  if (isDev) return;
  const csp = [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: file:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "media-src 'self' data: file:",
    "object-src 'none'",
    "base-uri 'self'",
    "frame-ancestors 'none'"
  ].join('; ');
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp]
      }
    });
  });
};

const listImagesUnder = async (rootPath, recursive) => {
  const root = path.resolve(cleanPath(rootPath, 'rootPath'));
  const out = [];
  const scan = async (dir) => {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (recursive) await scan(fullPath);
        continue;
      }
      if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
        out.push(fullPath);
      }
    }
  };
  await scan(root);
  return out.sort((a, b) => a.localeCompare(b));
};

// ── Generic shell IPC (no feature screens) ──────────────────────────────────
ipcMain.handle('app:get-version', () => app.getVersion());
ipcMain.handle('app:get-downloads-path', () => app.getPath('downloads'));
ipcMain.handle('files:list-images', async (_event, payload = {}) => {
  const options = isPlainObject(payload) ? payload : {};
  return listImagesUnder(options.rootPath, Boolean(options.recursive));
});

// ── Labeling feature IPC ────────────────────────────────────────────────────
ipcMain.handle('labeling:defaults', () => labelingDefaults());
ipcMain.handle('labeling:list-runs', (_event, payload) => listRuns(payload));
ipcMain.handle('labeling:list-queue', (_event, payload) => listQueue(payload));
ipcMain.handle('labeling:commit', (_event, payload) => commitSession(payload));

ipcMain.handle('dialog:browse-path', async (_event, mode) => {
  if (!['file', 'directory', 'file_or_directory', 'files'].includes(mode)) {
    throw new Error('Invalid browse mode');
  }
  const properties = mode === 'directory' ? ['openDirectory'] : ['openFile'];
  if (mode === 'file_or_directory') {
    properties.push('openDirectory');
  }
  if (mode === 'files') {
    properties.push('multiSelections');
  }
  const result = await dialog.showOpenDialog({ properties });
  if (result.canceled || result.filePaths.length === 0) {
    return mode === 'files' ? [] : '';
  }
  return mode === 'files' ? result.filePaths : result.filePaths[0];
});

ipcMain.handle('dialog:save-path', async (_event, opts = {}) => {
  const options = isPlainObject(opts) ? opts : {};
  const result = await dialog.showSaveDialog({
    defaultPath: typeof options.defaultPath === 'string' ? options.defaultPath : 'output.png',
    filters: Array.isArray(options.filters) ? options.filters : [{ name: 'PNG image', extensions: ['png'] }],
  });
  return result.canceled ? '' : (result.filePath || '');
});

ipcMain.handle('saveCroppedImage', async (_event, payload = {}) => {
  try {
    const { srcPath, pngBase64, outputDir } = isPlainObject(payload) ? payload : {};
    const sourcePath = cleanPath(srcPath, 'srcPath');
    const targetDir = path.resolve(cleanPath(outputDir, 'outputDir'));
    const imageBase64 = String(pngBase64 || '');
    if (!/^[a-zA-Z0-9+/=\s]+$/.test(imageBase64) || imageBase64.length > MAX_BASE64_CHARS) {
      throw new Error('Invalid PNG payload');
    }
    await fs.mkdir(targetDir, { recursive: true });

    const ext = path.extname(sourcePath);
    const base = path.basename(sourcePath, ext).replace(/[^a-zA-Z0-9._-]+/g, '_') || 'image';
    const outPath = path.join(targetDir, `${base}_crop.png`);

    const buffer = Buffer.from(imageBase64, 'base64');
    await fs.writeFile(outPath, buffer);

    return { outPath };
  } catch (err) {
    return { error: err?.message || 'Failed to save cropped image' };
  }
});

app.whenReady().then(async () => {
  installSecurityHeaders();
  await createWindow();
  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection in Electron main process:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception in Electron main process:', error);
});
