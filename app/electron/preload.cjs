const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getAppVersion: () => ipcRenderer.invoke('app:get-version'),
  getDownloadsPath: () => ipcRenderer.invoke('app:get-downloads-path'),
  listWorkflows: () => ipcRenderer.invoke('workflows:list'),
  browsePath: (mode) => ipcRenderer.invoke('dialog:browse-path', mode),
  startWorkflow: (payload) => ipcRenderer.invoke('workflow:start', payload),
  stopWorkflow: (sessionId) => ipcRenderer.invoke('workflow:stop', sessionId),
  getResultViewerDefaults: () => ipcRenderer.invoke('result-viewer:defaults'),
  listResultViewerRuns: (payload) => ipcRenderer.invoke('result-viewer:list-runs', payload),
  listResultViewerClusters: (payload) => ipcRenderer.invoke('result-viewer:list-clusters', payload),
  listResultViewerAssignments: (payload) => ipcRenderer.invoke('result-viewer:list-assignments', payload),
  clearResultViewerResultFlags: (payload) => ipcRenderer.invoke('result-viewer:clear-flags-results', payload),
  clearResultViewerClusterFlags: (payload) => ipcRenderer.invoke('result-viewer:clear-flags-cluster', payload),
  onWorkflowEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('workflow:event', listener);
    return () => ipcRenderer.removeListener('workflow:event', listener);
  }
});
