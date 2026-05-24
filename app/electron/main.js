import { app, BrowserWindow, dialog, ipcMain, session } from 'electron';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
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
const appRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(appRoot, '..');
const workflowsDir = path.join(appRoot, 'workflows');
const sessions = new Map();
const MAX_SUBPROCESS_SESSIONS = 4;
const MAX_SEGMENT_JOBS = 2;
const SEGMENT_TIMEOUT_MS = 5 * 60 * 1000;
const MAX_BASE64_CHARS = 120 * 1024 * 1024;
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']);
let activeSegmentJobs = 0;

const isPlainObject = (value) => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const cleanPath = (value, fieldName) => {
  if (typeof value !== 'string' || value.trim() === '' || value.includes('\0')) {
    throw new Error(`Invalid ${fieldName}`);
  }
  return value.trim();
};

const cleanOptionalPath = (value, fieldName) => {
  if (value == null || value === '') return '';
  return cleanPath(value, fieldName);
};

const validateWorkflowId = (workflowId) => {
  if (typeof workflowId !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(workflowId)) {
    throw new Error('Invalid workflow id');
  }
  return workflowId;
};

const parsePythonJsonResult = (stdout, stderr, code) => {
  const lines = stdout.trim().split('\n').filter(Boolean);
  const resultLine = [...lines].reverse().find((line) => {
    try { const parsed = JSON.parse(line); return !parsed.progress; } catch { return false; }
  });
  if (code !== 0) {
    let errorMsg = stderr.trim() || `Python exited with code ${code}`;
    if (resultLine) {
      try { const parsed = JSON.parse(resultLine); if (parsed.error) errorMsg = parsed.error; } catch { /* ignore */ }
    }
    return { error: errorMsg };
  }
  if (!resultLine) {
    return { error: `No result from Python. stderr: ${stderr.trim()}` };
  }
  try {
    return JSON.parse(resultLine);
  } catch {
    return { error: `Invalid JSON from Python: ${resultLine}` };
  }
};

const pythonEnv = () => ({
  ...process.env,
  PYTHONPATH: [appRoot, repoRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)
});

const createWindow = async () => {
  const mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 1040,
    minHeight: 680,
    title: 'Damage Detector',
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

const readWorkflowMetadata = async () => {
  const entries = await fs.readdir(workflowsDir, { withFileTypes: true });
  const workflows = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name === 'app') {
      continue;
    }
    const metadataPath = path.join(workflowsDir, entry.name, `${entry.name}.json`);
    try {
      const payload = JSON.parse(await fs.readFile(metadataPath, 'utf8'));
      workflows.push({ ...payload, metadataPath });
    } catch {
      continue;
    }
  }
  return workflows.sort((a, b) => String(a.name).localeCompare(String(b.name)));
};

const workflowPython = (payload = {}) => {
  const venvDir = cleanOptionalPath(payload.venvDir, 'venvDir');
  if (payload.useGlobalPython || !venvDir) {
    return process.env.PYTHON || 'python3';
  }
  const binName = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  return path.join(venvDir, binName);
};

const runSegmentPython = async ({ payload, inputPath, moduleName, inputData }) => {
  if (activeSegmentJobs >= MAX_SEGMENT_JOBS) {
    return { error: `Too many segmentation jobs running (${MAX_SEGMENT_JOBS} max)` };
  }

  try {
    await fs.writeFile(inputPath, JSON.stringify(inputData), 'utf8');
  } catch (err) {
    return { error: `Failed to write input file: ${err.message}` };
  }

  activeSegmentJobs += 1;
  const python = workflowPython(payload);
  return new Promise((resolve) => {
    const child = spawn(python, ['-m', moduleName, '--input', inputPath], {
      cwd: repoRoot,
      env: pythonEnv(),
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      activeSegmentJobs -= 1;
      fs.unlink(inputPath).catch(() => {});
      resolve(result);
    };
    const timeout = setTimeout(() => {
      child.kill();
      finish({ error: `Segmentation timed out after ${Math.round(SEGMENT_TIMEOUT_MS / 1000)} seconds` });
    }, SEGMENT_TIMEOUT_MS);

    child.stdout.on('data', (data) => { stdout += data; });
    child.stderr.on('data', (data) => { stderr += data; });
    child.on('error', (error) => finish({ error: error.message }));
    child.on('close', (code) => finish(parsePythonJsonResult(stdout, stderr, code)));
  });
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
ipcMain.handle('workflows:list', readWorkflowMetadata);
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

ipcMain.handle('final-review:defaults', finalReviewDefaults);
ipcMain.handle('final-review:list-csvs', (_event, payload) => listFinalCsvs(payload));
ipcMain.handle('final-review:list-images', (_event, payload) => listFinalImages(payload));
ipcMain.handle('final-review:get-image-boxes', (_event, payload) => getFinalImageBoxes(payload));
ipcMain.handle('final-review:export-coco', (_event, payload) => exportFinalToCoco(payload));
ipcMain.handle('label-review:list-sessions', (_event, payload) => listLabelReviewSessions(payload));
ipcMain.handle('label-review:load-session', (_event, payload) => loadLabelReviewSession(payload));
ipcMain.handle('label-review:save-session', (_event, payload) => saveLabelReviewSession(payload));
ipcMain.handle('label-review:create-session', (_event, payload) => createLabelReviewSession(payload));
ipcMain.handle('label-review:delete-session', (_event, payload) => deleteLabelReviewSession(payload));


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
    // Create output directory if needed
    await fs.mkdir(targetDir, { recursive: true });

    // Derive output filename
    const ext = path.extname(sourcePath);
    const base = path.basename(sourcePath, ext).replace(/[^a-zA-Z0-9._-]+/g, '_') || 'image';
    const outPath = path.join(targetDir, `${base}_crop.png`);

    // Decode base64 and write
    const buffer = Buffer.from(imageBase64, 'base64');
    await fs.writeFile(outPath, buffer);

    return { outPath };
  } catch (err) {
    return { error: err?.message || 'Failed to save cropped image' };
  }
});

ipcMain.handle('workflow:start', async (_event, payload = {}) => {
  try {
    if (!isPlainObject(payload)) throw new Error('Invalid workflow payload');
    if (sessions.size >= MAX_SUBPROCESS_SESSIONS) {
      return { error: `Too many subprocess sessions running (${MAX_SUBPROCESS_SESSIONS} max)` };
    }
    const workflowId = validateWorkflowId(payload.workflowId);
    const values = isPlainObject(payload.values) ? payload.values : {};
    const sessionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const valuesPath = path.join(repoRoot, 'results_workflows', 'sessions', `${workflowId}-${sessionId}.json`);

    await fs.mkdir(path.dirname(valuesPath), { recursive: true });
    await fs.writeFile(valuesPath, JSON.stringify(values, null, 2), 'utf8');

    const python = workflowPython(payload);
    const child = spawn(python, ['-m', 'workflows', 'run', workflowId, '--values-json', valuesPath], {
      cwd: repoRoot,
      env: pythonEnv()
    });
    sessions.set(sessionId, child);
    const send = (type, data) => {
      for (const window of BrowserWindow.getAllWindows()) {
        window.webContents.send('workflow:event', { sessionId, type, data: String(data) });
      }
    };
    send('started', `${python} -m workflows run ${workflowId} --values-json ${valuesPath}`);
    child.stdout.on('data', (data) => send('stdout', data));
    child.stderr.on('data', (data) => send('stderr', data));
    child.on('error', (error) => { sessions.delete(sessionId); send('stderr', error.message); send('closed', '1'); });
    child.on('close', (code) => {
      sessions.delete(sessionId);
      send('closed', String(code));
    });
    return { sessionId };
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('segment:point-sam', async (_event, rawPayload = {}) => {
  const payload = isPlainObject(rawPayload) ? rawPayload : {};
  const inputPath = path.join(os.tmpdir(), `sam-input-${Date.now()}-${Math.random().toString(16).slice(2)}.json`);
  try {
    const inputData = {
      image_path: cleanPath(payload.imagePath, 'imagePath'),
      points: Array.isArray(payload.points) ? payload.points : [],
      labels: Array.isArray(payload.labels) ? payload.labels : [],
      box: Array.isArray(payload.box) ? payload.box : null,
      sam_checkpoint: cleanPath(payload.samCheckpoint, 'samCheckpoint'),
      model_type: typeof payload.modelType === 'string' ? payload.modelType : 'auto',
      device: typeof payload.device === 'string' ? payload.device : 'auto',
      output_dir: path.join(repoRoot, 'results_point_sam'),
    };
    return await runSegmentPython({ payload, inputPath, moduleName: 'segmentation.sam.point_predict', inputData });
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('segment:text-sam', async (_event, rawPayload = {}) => {
  const payload = isPlainObject(rawPayload) ? rawPayload : {};
  const inputPath = path.join(os.tmpdir(), `sam-text-input-${Date.now()}-${Math.random().toString(16).slice(2)}.json`);
  try {
    const inputData = {
      image_path: cleanPath(payload.imagePath, 'imagePath'),
      text_prompt: typeof payload.textPrompt === 'string' ? payload.textPrompt : '',
      sam_checkpoint: cleanPath(payload.samCheckpoint, 'samCheckpoint'),
      gdino_checkpoint: cleanOptionalPath(payload.gdinoCheckpoint, 'gdinoCheckpoint'),
      model_type: typeof payload.modelType === 'string' ? payload.modelType : 'auto',
      device: typeof payload.device === 'string' ? payload.device : 'auto',
      box_threshold: payload.boxThreshold ?? 0.15,
      text_threshold: payload.textThreshold ?? 0.15,
      output_dir: path.join(repoRoot, 'results_text_sam'),
    };
    return await runSegmentPython({ payload, inputPath, moduleName: 'segmentation.sam.text_predict', inputData });
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('workflow:stop', (_event, sessionId) => {
  if (typeof sessionId !== 'string') return false;
  const child = sessions.get(sessionId);
  if (!child) {
    return false;
  }
  child.kill();
  return true;
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
