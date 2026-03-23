const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        title: 'Sanctuary',
        icon: path.join(__dirname, 'assets/icon.png')
    });

    // Always load the local index.html — WebSocket connects separately
    mainWindow.loadFile(path.join(__dirname, 'index.html'));

    // Handle window state persistence via localStorage (no electron-store needed)
    mainWindow.on('close', () => {
        // Window state is handled by Electron defaults
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
