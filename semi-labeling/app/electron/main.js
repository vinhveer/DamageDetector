import { app, BrowserWindow, dialog, ipcMain, session } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  labelingDefaults, listRuns, listQueue, commitSession,
  getRunResources, listSessions, listSelfTrainingRuns,
  listCleaned, updateCleanedLabel, commitCorrections,
  getSessionDecisions, getSelfTrainingPromotions, getRunMetrics,
  listPrototypeCandidates, latestPrototype,
  runStep, bridgeInfo,
} from './labeling/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);

const isPlainObject = (value) => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

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

// ── Generic shell IPC ───────────────────────────────────────────────────────
ipcMain.handle('app:get-version', () => app.getVersion());
ipcMain.handle('app:get-downloads-path', () => app.getPath('downloads'));

// ── Labeling feature IPC ────────────────────────────────────────────────────
ipcMain.handle('labeling:defaults', () => labelingDefaults());
ipcMain.handle('labeling:list-runs', (_event, payload) => listRuns(payload));
ipcMain.handle('labeling:list-queue', (_event, payload) => listQueue(payload));
ipcMain.handle('labeling:commit', (_event, payload) => commitSession(payload));
ipcMain.handle('labeling:run-resources', (_event, payload) => getRunResources(payload));
ipcMain.handle('labeling:list-sessions', (_event, payload) => listSessions(payload));
ipcMain.handle('labeling:list-selftrain', (_event, payload) => listSelfTrainingRuns(payload));
ipcMain.handle('labeling:bridge-info', () => bridgeInfo());
ipcMain.handle('labeling:list-cleaned', (_event, payload) => listCleaned(payload));
ipcMain.handle('labeling:update-cleaned', (_event, payload) => updateCleanedLabel(payload));
ipcMain.handle('labeling:commit-corrections', (_event, payload) => commitCorrections(payload));
ipcMain.handle('labeling:session-decisions', (_event, payload) => getSessionDecisions(payload));
ipcMain.handle('labeling:selftrain-promotions', (_event, payload) => getSelfTrainingPromotions(payload));
ipcMain.handle('labeling:run-metrics', (_event, payload) => getRunMetrics(payload));
ipcMain.handle('labeling:proto-candidates', (_event, payload) => listPrototypeCandidates(payload));
ipcMain.handle('labeling:latest-prototype', (_event, payload) => latestPrototype(payload));

// Streams step stdout/stderr to the renderer via 'labeling:step-output' events,
// keyed by a caller-supplied jobId. Resolves with the final { code, output }.
ipcMain.handle('labeling:run-step', async (event, payload = {}) => {
  const { step, flags, jobId } = payload || {};
  const sender = event.sender;
  const onData = (chunk) => {
    if (!sender.isDestroyed()) sender.send('labeling:step-output', { jobId, chunk });
  };
  return runStep({ step, flags, onData });
});

// Export cleaned_labels to a YOLO/COCO dataset via the export_dataset tool.
// Parses the tool's final JSON line into a structured result for the renderer.
ipcMain.handle('labeling:export-dataset', async (_event, payload = {}) => {
  const options = isPlainObject(payload) ? payload : {};
  const flags = {
    '--db': String(options.resemiDbPath || ''),
    '--run-id': String(options.runId || ''),
    '--image-root': String(options.imageRootPath || ''),
    '--output-dir': String(options.outputDir || ''),
    '--format': String(options.format || 'both'),
  };
  const res = await runStep({ step: 'export_dataset', flags });
  // find the last non-empty line that parses as JSON
  const lines = String(res.output || '').split('\n').map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    try {
      const parsed = JSON.parse(lines[i]);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch { /* not JSON, keep scanning */ }
  }
  return { error: `Export failed (exit ${res.code})`, output: res.output };
});

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
