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
});
