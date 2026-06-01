const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getAppVersion: () => ipcRenderer.invoke('app:get-version'),
  getDownloadsPath: () => ipcRenderer.invoke('app:get-downloads-path'),
  browsePath: (mode) => ipcRenderer.invoke('dialog:browse-path', mode),
  browseFiles: () => ipcRenderer.invoke('dialog:browse-path', 'files'),
  saveFileDialog: (opts) => ipcRenderer.invoke('dialog:save-path', opts),

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

  // Review loop (R2/R3/R5/R6/R4)
  listCleaned: (payload) => ipcRenderer.invoke('labeling:list-cleaned', payload),
  updateCleanedLabel: (payload) => ipcRenderer.invoke('labeling:update-cleaned', payload),
  commitCorrections: (payload) => ipcRenderer.invoke('labeling:commit-corrections', payload),
  getSessionDecisions: (payload) => ipcRenderer.invoke('labeling:session-decisions', payload),
  getSelfTrainingPromotions: (payload) => ipcRenderer.invoke('labeling:selftrain-promotions', payload),
  getRunMetrics: (payload) => ipcRenderer.invoke('labeling:run-metrics', payload),
  listPrototypeCandidates: (payload) => ipcRenderer.invoke('labeling:proto-candidates', payload),
  latestPrototype: (payload) => ipcRenderer.invoke('labeling:latest-prototype', payload),
  exportDataset: (payload) => ipcRenderer.invoke('labeling:export-dataset', payload),
});
