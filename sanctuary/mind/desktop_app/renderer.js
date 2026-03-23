// Renderer process — connects to Sanctuary's WebSocket server
let ws = null;
let showInner = true;

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-btn');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const saveSettingsBtn = document.getElementById('save-settings');
const closeSettingsBtn = document.getElementById('close-settings');
const serverAddressInput = document.getElementById('server-address');
const notificationsCheckbox = document.getElementById('notifications-enabled');
const showInnerCheckbox = document.getElementById('show-inner');
const themeSelect = document.getElementById('theme-select');
const clearBtn = document.getElementById('clear-btn');
const exportBtn = document.getElementById('export-btn');

// Initialize settings from localStorage (works in both Electron and browser)
function initializeSettings() {
    const saved = localStorage.getItem('sanctuary-settings');
    if (saved) {
        try {
            const settings = JSON.parse(saved);
            serverAddressInput.value = settings.serverAddress || 'ws://localhost:8765/ws';
            notificationsCheckbox.checked = settings.notifications !== false;
            showInner = settings.showInner !== false;
            if (showInnerCheckbox) showInnerCheckbox.checked = showInner;
            themeSelect.value = settings.theme || 'dark';
            applyTheme(themeSelect.value);
        } catch (e) {
            console.warn('Failed to load settings:', e);
        }
    }
}

function saveSettings() {
    const settings = {
        serverAddress: serverAddressInput.value,
        notifications: notificationsCheckbox.checked,
        showInner: showInnerCheckbox ? showInnerCheckbox.checked : true,
        theme: themeSelect.value
    };
    localStorage.setItem('sanctuary-settings', JSON.stringify(settings));
    showInner = settings.showInner;
}

// WebSocket Connection
function getServerAddress() {
    const saved = localStorage.getItem('sanctuary-settings');
    if (saved) {
        try {
            return JSON.parse(saved).serverAddress || 'ws://localhost:8765/ws';
        } catch (e) { /* use default */ }
    }
    return 'ws://localhost:8765/ws';
}

function connect() {
    const serverAddress = getServerAddress();

    if (ws) {
        ws.close();
    }

    updateStatus('connecting', 'Connecting...');

    try {
        ws = new WebSocket(serverAddress);
    } catch (e) {
        updateStatus('disconnected', 'Connection failed');
        setTimeout(connect, 5000);
        return;
    }

    ws.onopen = () => {
        updateStatus('connected', 'Connected to Sanctuary');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'message') {
                addMessage('sanctuary', data.content);
            } else if (data.type === 'inner' && showInner) {
                addInnerMessage(data.content);
            } else if (data.type === 'status') {
                updateStatus(data.status, data.message);
            } else if (data.type === 'system') {
                console.log('System status:', data.content);
            } else if (data.type === 'error') {
                addSystemMessage(data.content);
            }
        } catch (e) {
            console.warn('Failed to parse message:', e);
        }
    };

    ws.onclose = () => {
        updateStatus('disconnected', 'Connection lost — reconnecting...');
        setTimeout(connect, 5000);
    };

    ws.onerror = (err) => {
        console.warn('WebSocket error:', err);
    };
}

// UI Functions
function updateStatus(status, message) {
    statusIndicator.className = status;
    statusText.textContent = message || status;
}

function addMessage(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Show notification if enabled and window is not focused
    if (sender === 'sanctuary' && notificationsCheckbox.checked && !document.hasFocus()) {
        if (Notification.permission === 'granted') {
            new Notification('Sanctuary', { body: content.substring(0, 200) });
        }
    }
}

function addInnerMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message inner-message';
    messageDiv.textContent = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addSystemMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system-message';
    messageDiv.textContent = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendMessage() {
    const content = messageInput.value.trim();
    if (content && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'message',
            content: content
        }));
        addMessage('user', content);
        messageInput.value = '';
    }
}

function applyTheme(theme) {
    document.body.className = theme;
}

// Event Listeners
sendButton.onclick = sendMessage;
messageInput.onkeypress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
};

settingsBtn.onclick = () => {
    settingsModal.style.display = 'block';
};

closeSettingsBtn.onclick = () => {
    settingsModal.style.display = 'none';
};

saveSettingsBtn.onclick = () => {
    saveSettings();
    applyTheme(themeSelect.value);
    settingsModal.style.display = 'none';
    connect(); // Reconnect with new settings
};

clearBtn.onclick = () => {
    chatContainer.innerHTML = '';
};

exportBtn.onclick = () => {
    const messages = Array.from(chatContainer.children).map(msg => ({
        type: msg.classList.contains('user-message') ? 'user' :
              msg.classList.contains('inner-message') ? 'inner' : 'sanctuary',
        content: msg.textContent,
        timestamp: new Date().toISOString()
    }));

    const blob = new Blob([JSON.stringify(messages, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sanctuary-chat-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
};

// Initialize
initializeSettings();
connect();

// Request notification permission
if (notificationsCheckbox.checked && Notification.permission === 'default') {
    Notification.requestPermission();
}
