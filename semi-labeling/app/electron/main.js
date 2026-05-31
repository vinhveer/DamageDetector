import { app, BrowserWindow, dialog, ipcMain, session } from 'electron';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  dedupGroupsDefaults,
  listDedupGroupMembers,
  listDedupGroups,
  listDedupImageBoxes,
  listDedupImages,
  listDedupRuns
} from './dedup_groups/index.js';
import {
  clusterLabelingDefaults,
  createSession,
  deleteSession,
  getBoxImage,
  getClusterMembers,
  listClusterRuns,
  listClusters,
  listSessions,
  loadSession,
  saveSession,
} from './cluster_labeling/index.js';
import {
  classifierResultsDefaults,
  getApplyRun,
  getTrainingRun,
  listClassifierRuns,
} from './classifier_results/index.js';
import {
  labelReviewDefaults,
  listSubclusterRuns,
  listSubclustersByClass,
  getSubclusterMembers,
  listSuspectRuns,
  listSuspectClusters,
  getSuspectClusterMembers,
  listSessions as listLabelReviewSessions,
  loadSession as loadLabelReviewSession,
  saveSession as saveLabelReviewSession,
  createSession as createLabelReviewSession,
  deleteSession as deleteLabelReviewSession,
} from './label_review/index.js';
import {
  finalReviewDefaults,
  listFinalCsvs,
  listFinalImages,
  getFinalImageBoxes,
  exportFinalToCoco,
} from './final_review/index.js';

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

ipcMain.handle('app:get-version', () => app.getVersion());
ipcMain.handle('app:get-downloads-path', () => app.getPath('downloads'));
ipcMain.handle('files:list-images', async (_event, payload = {}) => {
  const options = isPlainObject(payload) ? payload : {};
  return listImagesUnder(options.rootPath, Boolean(options.recursive));
});

ipcMain.handle('dedup-groups:defaults', dedupGroupsDefaults);
ipcMain.handle('dedup-groups:list-runs', (_event, payload) => listDedupRuns(payload));
ipcMain.handle('dedup-groups:list-groups', (_event, payload) => listDedupGroups(payload));
ipcMain.handle('dedup-groups:list-members', (_event, payload) => listDedupGroupMembers(payload));
ipcMain.handle('dedup-groups:list-images', (_event, payload) => listDedupImages(payload));
ipcMain.handle('dedup-groups:list-image-boxes', (_event, payload) => listDedupImageBoxes(payload));

ipcMain.handle('cluster-labeling:defaults', clusterLabelingDefaults);
ipcMain.handle('cluster-labeling:list-runs', (_event, payload) => listClusterRuns(payload));
ipcMain.handle('cluster-labeling:list-clusters', (_event, payload) => listClusters(payload));
ipcMain.handle('cluster-labeling:get-cluster-members', (_event, payload) => getClusterMembers(payload));
ipcMain.handle('cluster-labeling:get-box-image', (_event, payload) => getBoxImage(payload));
ipcMain.handle('cluster-labeling:list-sessions', (_event, payload) => listSessions(payload));
ipcMain.handle('cluster-labeling:load-session', (_event, payload) => loadSession(payload));
ipcMain.handle('cluster-labeling:save-session', (_event, payload) => saveSession(payload));
ipcMain.handle('cluster-labeling:create-session', (_event, payload) => createSession(payload));
ipcMain.handle('cluster-labeling:delete-session', (_event, payload) => deleteSession(payload));

ipcMain.handle('classifier-results:defaults', classifierResultsDefaults);
ipcMain.handle('classifier-results:list-runs', (_event, payload) => listClassifierRuns(payload));
ipcMain.handle('classifier-results:get-apply', (_event, payload) => getApplyRun(payload));
ipcMain.handle('classifier-results:get-training', (_event, payload) => getTrainingRun(payload));

ipcMain.handle('label-review:defaults', labelReviewDefaults);
ipcMain.handle('label-review:list-runs', (_event, payload) => listSubclusterRuns(payload));
ipcMain.handle('label-review:list-subclusters', (_event, payload) => listSubclustersByClass(payload));
ipcMain.handle('label-review:get-subcluster-members', (_event, payload) => getSubclusterMembers(payload));
ipcMain.handle('label-review:list-suspect-runs', (_event, payload) => listSuspectRuns(payload));
ipcMain.handle('label-review:list-suspect-clusters', (_event, payload) => listSuspectClusters(payload));
ipcMain.handle('label-review:get-suspect-cluster-members', (_event, payload) => getSuspectClusterMembers(payload));
ipcMain.handle('label-review:list-sessions', (_event, payload) => listLabelReviewSessions(payload));
ipcMain.handle('label-review:load-session', (_event, payload) => loadLabelReviewSession(payload));
ipcMain.handle('label-review:save-session', (_event, payload) => saveLabelReviewSession(payload));
ipcMain.handle('label-review:create-session', (_event, payload) => createLabelReviewSession(payload));
ipcMain.handle('label-review:delete-session', (_event, payload) => deleteLabelReviewSession(payload));

ipcMain.handle('final-review:defaults', finalReviewDefaults);
ipcMain.handle('final-review:list-csvs', (_event, payload) => listFinalCsvs(payload));
ipcMain.handle('final-review:list-images', (_event, payload) => listFinalImages(payload));
ipcMain.handle('final-review:get-image-boxes', (_event, payload) => getFinalImageBoxes(payload));
ipcMain.handle('final-review:export-coco', (_event, payload) => exportFinalToCoco(payload));

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
