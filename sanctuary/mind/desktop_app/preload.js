const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// specific electron APIs without exposing the entire API
contextBridge.exposeInMainWorld(
    'electron',
    {
        store: {
            get: (key) => ipcRenderer.invoke('electron-store-get', key),
            set: (key, value) => ipcRenderer.invoke('electron-store-set', key, value),
            delete: (key) => ipcRenderer.invoke('electron-store-delete', key)
        },
        notifications: {
            send: (message) => ipcRenderer.invoke('send-notification', message)
        },
        sanctuaryAPI: {
            connect: () => ipcRenderer.invoke('sanctuary-connect'),
            disconnect: () => ipcRenderer.invoke('sanctuary-disconnect'),
            send: (message) => ipcRenderer.invoke('sanctuary-send', message)
        }
    }
);