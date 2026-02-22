/**
 * Agentic Helper — Client-side chat logic
 * Handles WebSocket communication, message rendering, and UI interactions.
 */

// ── State ────────────────────────────────────────────────────────────────────
let ws = null;
let isConnected = false;

// ── DOM Elements ─────────────────────────────────────────────────────────────
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusBadge = document.getElementById('status-badge');
const toolList = document.getElementById('tool-list');
const clearChatBtn = document.getElementById('clear-chat');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');

// ── WebSocket Connection ─────────────────────────────────────────────────────
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);

    ws.onopen = () => {
        isConnected = true;
        setStatus('Ready', '');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };

    ws.onclose = () => {
        isConnected = false;
        setStatus('Disconnected', '');
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        setStatus('Error', '');
    };
}

// ── Message Handling ─────────────────────────────────────────────────────────
function handleServerMessage(data) {
    switch (data.type) {
        case 'status':
            if (data.content === 'thinking') {
                setStatus('Thinking...', 'thinking');
                showThinkingIndicator();
            }
            break;

        case 'tool_call':
            setStatus(`Using ${data.content.name}...`, 'tool-running');
            removeThinkingIndicator();
            addToolBadge(data.content.name, JSON.stringify(data.content.args, null, 2));
            break;

        case 'tool_result':
            addToolResult(data.content.name, data.content.result);
            break;

        case 'response':
            removeThinkingIndicator();
            setStatus('Ready', '');
            addMessage('assistant', data.content, data.tools_used || []);
            enableInput();
            break;

        case 'error':
            removeThinkingIndicator();
            setStatus('Error', '');
            addMessage('assistant', `⚠️ ${data.content}`);
            enableInput();
            break;
    }
}

// ── UI Helpers ───────────────────────────────────────────────────────────────
function setStatus(text, className) {
    statusBadge.textContent = text;
    statusBadge.className = 'status-badge' + (className ? ` ${className}` : '');
}

function addMessage(role, content, toolsUsed = []) {
    // Remove welcome message if present
    const welcome = messagesContainer.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatContent(content);

    msg.appendChild(avatar);
    msg.appendChild(contentDiv);
    messagesContainer.appendChild(msg);
    scrollToBottom();
}

function addToolBadge(toolName, args) {
    const badge = document.createElement('div');
    badge.className = 'tool-badge';
    badge.innerHTML = `<strong>${toolName}</strong>`;
    badge.title = args;
    messagesContainer.appendChild(badge);
    scrollToBottom();
}

function addToolResult(toolName, result) {
    const badge = document.createElement('div');
    badge.className = 'tool-badge result';
    const truncated = result.length > 120 ? result.substring(0, 120) + '...' : result;
    badge.innerHTML = `<strong>${toolName}</strong> → ${escapeHtml(truncated)}`;
    badge.title = result;
    messagesContainer.appendChild(badge);
    scrollToBottom();
}

function showThinkingIndicator() {
    removeThinkingIndicator();
    const indicator = document.createElement('div');
    indicator.className = 'thinking-indicator';
    indicator.id = 'thinking';
    indicator.innerHTML = `
        <div class="message-avatar" style="background: var(--accent-gradient); box-shadow: var(--shadow-glow);">🤖</div>
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    messagesContainer.appendChild(indicator);
    scrollToBottom();
}

function removeThinkingIndicator() {
    const indicator = document.getElementById('thinking');
    if (indicator) indicator.remove();
}

function formatContent(text) {
    if (!text) return '';

    // Escape HTML first
    let formatted = escapeHtml(text);

    // Code blocks
    formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

function enableInput() {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
}

function disableInput() {
    sendBtn.disabled = true;
}

// ── Event Listeners ──────────────────────────────────────────────────────────
chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const content = userInput.value.trim();
    if (!content || !isConnected) return;

    addMessage('user', content);
    disableInput();
    userInput.value = '';
    userInput.style.height = 'auto';

    ws.send(JSON.stringify({ content }));
});

// Auto-resize textarea
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
});

// Submit on Enter (Shift+Enter for newline)
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Quick action buttons
document.querySelectorAll('.quick-action').forEach((btn) => {
    btn.addEventListener('click', () => {
        userInput.value = btn.dataset.prompt;
        chatForm.dispatchEvent(new Event('submit'));
    });
});

// Clear chat
clearChatBtn.addEventListener('click', () => {
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🤖</div>
            <h2>Welcome to Agentic Helper</h2>
            <p>I'm an AI assistant with access to tools including a web browser, calculator, and more. Ask me anything!</p>
            <div class="quick-actions">
                <button class="quick-action" data-prompt="What time is it right now?">🕐 Current Time</button>
                <button class="quick-action" data-prompt="Calculate 1234 * 5678">🧮 Calculator</button>
                <button class="quick-action" data-prompt="Navigate to https://news.ycombinator.com and tell me the top 3 stories">🌐 Browse Web</button>
            </div>
        </div>
    `;
    // Reconnect to reset thread
    if (ws) ws.close();
    // Re-attach quick action listeners
    document.querySelectorAll('.quick-action').forEach((btn) => {
        btn.addEventListener('click', () => {
            userInput.value = btn.dataset.prompt;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });
});

// Sidebar toggle (mobile)
sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
});

// Close sidebar on outside click (mobile)
document.addEventListener('click', (e) => {
    if (sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== sidebarToggle) {
        sidebar.classList.remove('open');
    }
});

// ── Initialization ───────────────────────────────────────────────────────────
async function loadTools() {
    try {
        const res = await fetch('/api/tools');
        const tools = await res.json();
        toolList.innerHTML = tools
            .map((t) => `<li class="tool-item" title="${escapeHtml(t.description)}">${t.name}</li>`)
            .join('');
    } catch {
        toolList.innerHTML = '<li class="tool-item">Could not load tools</li>';
    }
}

// Boot
loadTools();
connectWebSocket();
