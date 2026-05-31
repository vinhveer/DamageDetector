const { contextBridge, ipcRenderer } = require('electron');

// Generic shell IPC only — feature-screen APIs were removed.
contextBridge.exposeInMainWorld('electronAPI', {
  getAppVersion: () => ipcRenderer.invoke('app:get-version'),
  getDownloadsPath: () => ipcRenderer.invoke('app:get-downloads-path'),
  listImageFiles: (payload) => ipcRenderer.invoke('files:list-images', payload),
  browsePath: (mode) => ipcRenderer.invoke('dialog:browse-path', mode),
  browseFiles: () => ipcRenderer.invoke('dialog:browse-path', 'files'),
  saveFileDialog: (opts) => ipcRenderer.invoke('dialog:save-path', opts),
  saveCroppedImage: (payload) => ipcRenderer.invoke('saveCroppedImage', payload),

  // Labeling feature
  getLabelingDefaults: () => ipcRenderer.invoke('labeling:defaults'),
  listLabelingRuns: (payload) => ipcRenderer.invoke('labeling:list-runs', payload),
  listLabelingQueue: (payload) => ipcRenderer.invoke('labeling:list-queue', payload),
  commitLabeling: (payload) => ipcRenderer.invoke('labeling:commit', payload),
  getRunResources: (payload) => ipcRenderer.invoke('labeling:run-resources', payload),
  listSessions: (payload) => ipcRenderer.invoke('labeling:list-sessions', payload),
  listSelfTrainingRuns: (payload) => ipcRenderer.invoke('labeling:list-selftrain', payload),
  getBridgeInfo: () => ipcRenderer.invoke('labeling:bridge-info'),
  runStep: (payload) => ipcRenderer.invoke('labeling:run-step', payload),
  onStepOutput: (handler) => {
    const listener = (_event, data) => handler(data);
    ipcRenderer.on('labeling:step-output', listener);
    return () => ipcRenderer.removeListener('labeling:step-output', listener);
  },
});
