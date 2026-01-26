// Renderer process
let ws = null;
const store = window.electron.store;

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
const themeSelect = document.getElementById('theme-select');
const clearBtn = document.getElementById('clear-btn');
const exportBtn = document.getElementById('export-btn');

// Initialize settings from store
async function initializeSettings() {
    const settings = await store.get('settings') || {};
    serverAddressInput.value = settings.serverAddress || 'ws://localhost:8000';
    notificationsCheckbox.checked = settings.notifications !== false;
    themeSelect.value = settings.theme || 'dark';
    applyTheme(themeSelect.value);
}

// WebSocket Connection
async function connect() {
    const settings = await store.get('settings') || {};
    const serverAddress = settings.serverAddress || 'ws://localhost:8000';
    
    if (ws) {
        ws.close();
    }

    ws = new WebSocket(serverAddress);
    
    ws.onopen = () => {
        updateStatus('connected', 'Connected to Lyra');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'message') {
            addMessage('lyra', data.content);
        } else if (data.type === 'status') {
            updateStatus(data.status, data.message);
        }
    };
    
    ws.onclose = () => {
        updateStatus('disconnected', 'Connection lost');
        setTimeout(connect, 5000);
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
    if (sender === 'lyra' && notificationsCheckbox.checked && !document.hasFocus()) {
        new Notification('Message from Lyra', { body: content });
    }
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

saveSettingsBtn.onclick = async () => {
    const settings = {
        serverAddress: serverAddressInput.value,
        notifications: notificationsCheckbox.checked,
        theme: themeSelect.value
    };
    await store.set('settings', settings);
    applyTheme(settings.theme);
    settingsModal.style.display = 'none';
    connect(); // Reconnect with new settings
};

clearBtn.onclick = () => {
    chatContainer.innerHTML = '';
};

exportBtn.onclick = async () => {
    const messages = Array.from(chatContainer.children).map(msg => ({
        type: msg.classList.contains('user-message') ? 'user' : 'lyra',
        content: msg.textContent,
        timestamp: new Date().toISOString()
    }));
    
    const blob = new Blob([JSON.stringify(messages, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lyra-chat-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
};

// Initialize
initializeSettings().then(() => {
    connect();
    
    // Request notification permission if enabled
    if (notificationsCheckbox.checked) {
        Notification.requestPermission();
    }
});