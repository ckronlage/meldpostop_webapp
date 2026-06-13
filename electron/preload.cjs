'use strict';
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  saveMask: (data, filename) => ipcRenderer.invoke('save-mask', { data, filename }),
});
