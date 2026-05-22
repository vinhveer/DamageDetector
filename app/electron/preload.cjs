const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getAppVersion: () => ipcRenderer.invoke('app:get-version'),
  getDownloadsPath: () => ipcRenderer.invoke('app:get-downloads-path'),
  listWorkflows: () => ipcRenderer.invoke('workflows:list'),
  listImageFiles: (payload) => ipcRenderer.invoke('files:list-images', payload),
  browsePath: (mode) => ipcRenderer.invoke('dialog:browse-path', mode),
  browseFiles: () => ipcRenderer.invoke('dialog:browse-path', 'files'),
  saveFileDialog: (opts) => ipcRenderer.invoke('dialog:save-path', opts),
  startWorkflow: (payload) => ipcRenderer.invoke('workflow:start', payload),
  stopWorkflow: (sessionId) => ipcRenderer.invoke('workflow:stop', sessionId),
  getResultViewerDefaults: () => ipcRenderer.invoke('result-viewer:defaults'),
  listResultViewerRuns: (payload) => ipcRenderer.invoke('result-viewer:list-runs', payload),
  listResultViewerClusters: (payload) => ipcRenderer.invoke('result-viewer:list-clusters', payload),
  listResultViewerAssignments: (payload) => ipcRenderer.invoke('result-viewer:list-assignments', payload),
  clearResultViewerResultFlags: (payload) => ipcRenderer.invoke('result-viewer:clear-flags-results', payload),
  clearResultViewerClusterFlags: (payload) => ipcRenderer.invoke('result-viewer:clear-flags-cluster', payload),
  getPrototypeReviewDefaults: () => ipcRenderer.invoke('prototype-review:defaults'),
  listPrototypeReviewRuns: (payload) => ipcRenderer.invoke('prototype-review:list-runs', payload),
  listPrototypeReviewScores: (payload) => ipcRenderer.invoke('prototype-review:list-scores', payload),
  listPrototypeReviewAssignments: (payload) => ipcRenderer.invoke('prototype-review:list-assignments', payload),
  listPrototypeReviewAssignmentsBulk: (payload) => ipcRenderer.invoke('prototype-review:list-assignments-bulk', payload),
  markPrototypeAssignmentsAsOutlier: (payload) => ipcRenderer.invoke('prototype-review:mark-outlier', payload),
  setPrototypeAssignmentsLabel: (payload) => ipcRenderer.invoke('prototype-review:set-label', payload),
  listPrototypeVersions: (payload) => ipcRenderer.invoke('prototype-review:list-versions', payload),
  getPrototypeVersionDetail: (payload) => ipcRenderer.invoke('prototype-review:get-version-detail', payload),
  getPrototypeCandidates: (payload) => ipcRenderer.invoke('prototype-review:get-candidates', payload),
  createPrototypeVersion: (payload) => ipcRenderer.invoke('prototype-review:create-version', payload),
  setPrototypeVersionActive: (payload) => ipcRenderer.invoke('prototype-review:set-version-active', payload),
  archivePrototypeVersion: (payload) => ipcRenderer.invoke('prototype-review:archive-version', payload),
  unarchivePrototypeVersion: (payload) => ipcRenderer.invoke('prototype-review:unarchive-version', payload),
  renamePrototypeVersion: (payload) => ipcRenderer.invoke('prototype-review:rename-version', payload),
  onPrototypeReviewJobEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('prototype-review:job-event', listener);
    return () => ipcRenderer.removeListener('prototype-review:job-event', listener);
  },
  saveCroppedImage: (payload) => ipcRenderer.invoke('saveCroppedImage', payload),
  segmentPointSam: (payload) => ipcRenderer.invoke('segment:point-sam', payload),
  segmentTextSam: (payload) => ipcRenderer.invoke('segment:text-sam', payload),
  onWorkflowEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('workflow:event', listener);
    return () => ipcRenderer.removeListener('workflow:event', listener);
  }
});
