import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  clearFlagsForCluster,
  clearFlagsForResults,
  listAssignments,
  listClusters,
  listRuns,
  resultViewerDefaults
} from './result_viewer/index.js';
import {
  listPrototypeReviewAssignments,
  listPrototypeReviewRuns,
  listPrototypeReviewScores,
  prototypeReviewDefaults
} from './prototype_review/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const appRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(appRoot, '..');
const workflowsDir = path.join(appRoot, 'workflows');
const sessions = new Map();

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
      webSecurity: false
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
  if (payload.useGlobalPython || !payload.venvDir) {
    return process.env.PYTHON || 'python3';
  }
  const binName = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  return path.join(payload.venvDir, binName);
};

ipcMain.handle('app:get-version', () => app.getVersion());
ipcMain.handle('app:get-downloads-path', () => app.getPath('downloads'));
ipcMain.handle('workflows:list', readWorkflowMetadata);

ipcMain.handle('result-viewer:defaults', resultViewerDefaults);
ipcMain.handle('result-viewer:list-runs', (_event, payload) => listRuns(payload));
ipcMain.handle('result-viewer:list-clusters', (_event, payload) => listClusters(payload));
ipcMain.handle('result-viewer:list-assignments', (_event, payload) => listAssignments(payload));
ipcMain.handle('result-viewer:clear-flags-results', (_event, payload) => clearFlagsForResults(payload));
ipcMain.handle('result-viewer:clear-flags-cluster', (_event, payload) => clearFlagsForCluster(payload));

ipcMain.handle('prototype-review:defaults', prototypeReviewDefaults);
ipcMain.handle('prototype-review:list-runs', (_event, payload) => listPrototypeReviewRuns(payload));
ipcMain.handle('prototype-review:list-scores', (_event, payload) => listPrototypeReviewScores(payload));
ipcMain.handle('prototype-review:list-assignments', (_event, payload) => listPrototypeReviewAssignments(payload));


ipcMain.handle('dialog:browse-path', async (_event, mode) => {
  const properties = mode === 'directory' ? ['openDirectory'] : ['openFile'];
  if (mode === 'file_or_directory') {
    properties.push('openDirectory');
  }
  const result = await dialog.showOpenDialog({ properties });
  if (result.canceled || result.filePaths.length === 0) {
    return '';
  }
  return result.filePaths[0];
});

ipcMain.handle('workflow:start', (_event, payload) => {
  const sessionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const valuesPath = path.join(repoRoot, 'results_workflows', 'sessions', `${payload.workflowId}-${sessionId}.json`);
  fs.mkdir(path.dirname(valuesPath), { recursive: true })
    .then(() => fs.writeFile(valuesPath, JSON.stringify(payload.values || {}, null, 2), 'utf8'))
    .then(() => {
      const python = workflowPython(payload);
      const child = spawn(python, ['-m', 'workflows', 'run', payload.workflowId, '--values-json', valuesPath], {
        cwd: repoRoot,
        env: pythonEnv()
      });
      sessions.set(sessionId, child);
      const send = (type, data) => {
        for (const window of BrowserWindow.getAllWindows()) {
          window.webContents.send('workflow:event', { sessionId, type, data: String(data) });
        }
      };
      send('started', `${python} -m workflows run ${payload.workflowId} --values-json ${valuesPath}`);
      child.stdout.on('data', (data) => send('stdout', data));
      child.stderr.on('data', (data) => send('stderr', data));
      child.on('close', (code) => {
        sessions.delete(sessionId);
        send('closed', String(code));
      });
    })
    .catch((error) => {
      for (const window of BrowserWindow.getAllWindows()) {
        window.webContents.send('workflow:event', { sessionId, type: 'stderr', data: String(error) });
      }
    });
  return { sessionId };
});

ipcMain.handle('workflow:stop', (_event, sessionId) => {
  const child = sessions.get(sessionId);
  if (!child) {
    return false;
  }
  child.kill();
  return true;
});

app.whenReady().then(async () => {
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
