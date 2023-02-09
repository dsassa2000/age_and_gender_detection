// Modules to control application life and create native browser window
const { app, ipcMain,BrowserWindow, Notification } = require("electron");
const exec = require('child_process').exec;
const path = require('path')

var nodeConsole = require('console');
require('@electron/remote/main').initialize()


function createWindow() {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 620,
        height: 500,
        resizable: true,
        webPreferences: {
            enableRemoteModule: true, // turn off remote
            //preload: path.join(__dirname, 'gui_example.js'),
            nodeIntegration: true,
            contextIsolation: false
            
            //contextIsolation: true
                // nodeIntegration: true
        }
    })
    mainWindow.maximize();
    // and load the index.html of the app.
    mainWindow.loadFile('./electron/gui_example.html');

    // Open the DevTools.
    mainWindow.webContents.openDevTools()
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
    createWindow();

    app.on('activate', function() {
        // On macOS it's common to re-create a window in the app when the
        // dock icon is clicked and there are no other windows open.
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    });
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', function() {
    if (process.platform !== 'darwin') app.quit();
});

