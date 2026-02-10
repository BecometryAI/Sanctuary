const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');
const Store = require('electron-store');

// Initialize electron store
const store = new Store();

function createWindow() {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        title: 'Sanctuary',
        icon: path.join(__dirname, 'assets/icon.png')
    });

    // Load the app
    if (isDev) {
        mainWindow.loadURL('http://localhost:8000');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile('index.html');
    }

    // Handle window state
    let windowState = store.get('windowState', {
        isMaximized: false,
        bounds: { x: undefined, y: undefined, width: 1200, height: 800 }
    });

    if (windowState.bounds) {
        mainWindow.setBounds(windowState.bounds);
    }
    if (windowState.isMaximized) {
        mainWindow.maximize();
    }

    // Save window state on close
    mainWindow.on('close', () => {
        windowState.isMaximized = mainWindow.isMaximized();
        if (!windowState.isMaximized) {
            windowState.bounds = mainWindow.getBounds();
        }
        store.set('windowState', windowState);
    });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

// Quit when all windows are closed.
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});