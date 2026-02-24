/* =============================================================================
   HUMMINGBIRD-LEA - Frontend Application
   Powered by CoDre-X | B & D Servicing LLC
   ============================================================================= */

// =============================================================================
// State
// =============================================================================

const state = {
    token: localStorage.getItem('hummingbird_token'),
    username: localStorage.getItem('hummingbird_username') || 'Guest',
    currentAgent: 'lea',

    // Per-agent storage (prevents cross-talk)
    conversationHistoryByAgent: {
        lea: [],
        chiquis: [],
        grant: [],
    },
    chatByAgent: {
        lea: [],
        chiquis: [],
        grant: [],
    },
    isProcessingByAgent: {
        lea: false,
        chiquis: false,
        grant: false,
    },

    // Stream agent responses (SSE)
    useStreaming: true,
};

// Agent info
const agents = {
    lea: { name: 'Lea', iconType: 'emoji', icon: '🐦', color: '#10b981' },
    chiquis: { name: 'Chiquis', iconType: 'emoji', icon: '🐾', color: '#3b82f6' },
    grant: { name: 'Grant', iconType: 'emoji', icon: '🏛️', color: '#8b5cf6' },
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    loginModal: document.getElementById('login-modal'),
    loginForm: document.getElementById('login-form'),
    loginError: document.getElementById('login-error'),
    mainContent: document.getElementById('main-content'),
    chatMessages: document.getElementById('chat-messages'),
    messageInput: document.getElementById('message-input'),
    sendBtn: document.getElementById('send-btn'),
    typingIndicator: document.getElementById('typing-indicator'),
    connectionStatus: document.getElementById('connection-status'),
    userName: document.getElementById('user-name'),
    logoutBtn: document.getElementById('logout-btn'),
    agentBtns: document.querySelectorAll('.agent-btn'),
};

// =============================================================================
// Authentication
// =============================================================================

async function login(username, password) {
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Login failed');
        }
        
        const data = await response.json();
        state.token = data.access_token;
        state.username = username;
        
        localStorage.setItem('hummingbird_token', data.access_token);
        localStorage.setItem('hummingbird_username', username);
        
        showMainContent();
        loadGreeting();
        
    } catch (error) {
        elements.loginError.textContent = error.message;
    }
}

function logout() {
    state.token = null;
    state.username = 'Guest';
    state.conversationHistory = [];
    
    localStorage.removeItem('hummingbird_token');
    localStorage.removeItem('hummingbird_username');
    
    elements.chatMessages.innerHTML = '';
    showLoginModal();
}

function showLoginModal() {
    elements.loginModal.style.display = 'flex';
    elements.mainContent.style.display = 'none';
    elements.logoutBtn.style.display = 'none';
}

function showMainContent() {
    elements.loginModal.style.display = 'none';
    elements.mainContent.style.display = 'flex';
    elements.userName.textContent = state.username;
    elements.logoutBtn.style.display = 'inline-block';
}


function getAgentKey(agentKey) {
    return agentKey || state.currentAgent;
}

function agentAvatarHTML(agentKey) {
    const a = agents[agentKey];
    if (!a) return '🤖';
    if (a.iconType === 'img') {
        return `<img src="${a.icon}" alt="${a.name}" style="height:24px;width:24px;object-fit:contain;" />`;
    }
    return a.icon || '🤖';
}

function renderChat(agentKey) {
    elements.chatMessages.innerHTML = '';
    const msgs = state.chatByAgent[agentKey] || [];
    for (const m of msgs) {
        renderMessage(agentKey, m.role, m.content, m.name, m.confidence);
    }
    scrollToBottom();
}

function renderMessage(agentKey, role, content, name, confidence = 'high') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    if (role === 'assistant') {
        messageDiv.dataset.agent = agentKey;
    }

    const avatar = role === 'user' ? '👤' : agentAvatarHTML(agentKey);
    const displayName = role === 'user' ? name : (agents[agentKey]?.name || name);

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const formattedContent = formatMessageContent(content);

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-name">${displayName}</span>
                <span class="message-time">${time}</span>
                ${role === 'assistant' ? `<span class="confidence-indicator confidence-${confidence}">${confidence}</span>` : ''}
            </div>
            <div class="message-text">${formattedContent}</div>
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
}

function recordMessage(agentKey, role, content, name, confidence = 'high') {
    if (!state.chatByAgent[agentKey]) state.chatByAgent[agentKey] = [];
    state.chatByAgent[agentKey].push({ role, content, name, confidence });

    if (!state.conversationHistoryByAgent[agentKey]) state.conversationHistoryByAgent[agentKey] = [];
    state.conversationHistoryByAgent[agentKey].push({ role, content });

    // Render immediately only if we are viewing that agent
    if (state.currentAgent === agentKey) {
        renderMessage(agentKey, role, content, name, confidence);
        scrollToBottom();
    }
}

async function streamChat(agentKey, message) {
    // Build history BEFORE adding placeholder assistant message
    const history = (state.conversationHistoryByAgent[agentKey] || []).slice(-20);

    // Create placeholder assistant message
    recordMessage(agentKey, 'assistant', '', agentKey, 'high');

    const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...(state.token && { 'Authorization': `Bearer ${state.token}` }),
        },
        body: JSON.stringify({
            message: message,
            agent: agentKey,
            history: history,
        }),
    });

    if (!res.ok || !res.body) {
        throw new Error(`Streaming failed (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let full = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events separated by blank line
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const ev of events) {
            const lines = ev.split('\n');
            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                let data = line.slice(5);
                if (data.startsWith(' ')) data = data.slice(1);

                if (data === '[DONE]') {
                    _finalizeLastAssistant(agentKey, full);
                    return;
                }

                // Parse backend JSON format: {"t": "chunk"} or {"done": true}
                if (data.startsWith('{')) {
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.done) {
                            _finalizeLastAssistant(agentKey, full);
                            return;
                        }
                        if (parsed.t !== undefined) {
                            data = parsed.t;
                        }
                    } catch (_) {}
                }

                // Also handle JSON-quoted strings: data: "Hello"
                if (data.startsWith('"')) {
                    try { data = JSON.parse(data); } catch (_) {}
                }

                full += data;
                _updateLastAssistant(agentKey, full);
            }
        }
    }

    _finalizeLastAssistant(agentKey, full);
}

function _finalizeLastAssistant(agentKey, content) {
    // Update DOM only if currently viewing that agent
    if (state.currentAgent !== agentKey) return;

    // Find the last assistant message node for this agent
    const nodes = elements.chatMessages.querySelectorAll('.message.assistant');
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (n.dataset.agent === agentKey) {
            const textEl = n.querySelector('.message-text');
            if (textEl) textEl.textContent = content;
            scrollToBottom();
            return;
        }
    }
}

function _updateLastAssistant(agentKey, content) {
    // Update state (last assistant message)
    const msgs = state.chatByAgent[agentKey] || [];
    for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === 'assistant') {
            msgs[i].content = content;
            break;
        }
    }

    // Update history (last assistant entry)
    const hist = state.conversationHistoryByAgent[agentKey] || [];
    for (let i = hist.length - 1; i >= 0; i--) {
        if (hist[i].role === 'assistant') {
            hist[i].content = content;
            break;
        }
    }

    // Update DOM only if currently viewing that agent
    if (state.currentAgent !== agentKey) return;

    // Find the last assistant message node for this agent
    const nodes = elements.chatMessages.querySelectorAll('.message.assistant');
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (n.dataset.agent === agentKey) {
            const textEl = n.querySelector('.message-text');
            if (textEl) textEl.textContent = content;
            scrollToBottom();
            return;
        }
    }
}



function getPlainTextFromElement(el) {
    if (!el) return '';
    // innerText gives us readable text without HTML tags
    return (el.innerText || '').trim();
}

function speakText(text) {
    if (!('speechSynthesis' in window)) {
        alert('TTS not supported in this browser.');
        return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
}

// Delegate click for TTS buttons (works for all future messages)
elements.chatMessages.addEventListener('click', (e) => {
    const btn = e.target.closest && e.target.closest('.tts-btn');
    if (!btn) return;
    const msgEl = btn.closest('.message');
    const textEl = msgEl ? msgEl.querySelector('.message-text') : null;
    const text = getPlainTextFromElement(textEl);
    if (text) speakText(text);
});

// =============================================================================
// Chat Functions
// =============================================================================

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    const agentKey = state.currentAgent;

    if (!message || state.isProcessingByAgent[agentKey]) return;

    state.isProcessingByAgent[agentKey] = true;
    elements.sendBtn.disabled = true;

    // Add user message to UI + store
    recordMessage(agentKey, 'user', message, state.username, 'high');
    elements.messageInput.value = '';
    autoResizeInput();

    // Show typing indicator (only for the currently viewed agent)
    showTypingIndicator(agentKey);

    try {
        const history = (state.conversationHistoryByAgent[agentKey] || []).slice(-20);

        if (state.useStreaming) {
            await streamChat(agentKey, message);
        } else {
            const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(state.token && { 'Authorization': `Bearer ${state.token}` }),
            },
            body: JSON.stringify({
                message: message,
                agent: agentKey,
                history: history,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get response');
        }

        const data = await response.json();

        // Store response under the agent that generated it (and/or the agentKey we sent)
        const responseAgent = data.agent || agentKey;
        recordMessage(responseAgent, 'assistant', data.content, responseAgent, data.confidence || 'high');
        }

    } catch (error) {
        recordMessage(agentKey, 'assistant', `❌ Error: ${error.message}`, agentKey, 'low');
    } finally {
        if (state.currentAgent === agentKey) hideTypingIndicator();
        state.isProcessingByAgent[agentKey] = false;

        // Re-enable send button based on current agent state
        elements.sendBtn.disabled = !!state.isProcessingByAgent[state.currentAgent];
        elements.messageInput.focus();
    }
}

function addMessage(role, content, name, confidence = 'high') {
    // Backwards-compatible wrapper; records to the current agent by default
    const agentKey = state.currentAgent;
    recordMessage(agentKey, role, content, name, confidence);
}

function formatMessageContent(content) {
    // Escape HTML
    let formatted = content
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Format code blocks (```code```)
    formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });
    
    // Format inline code (`code`)
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Format bold (**text**)
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Format italic (*text*)
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Format line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function showTypingIndicator(agentKey) {
    if (agentKey !== state.currentAgent) return;
    const agentName = agents[agentKey]?.name || 'Agent';
    elements.typingIndicator.querySelector('.agent-typing').textContent = agentName;
    elements.typingIndicator.style.display = 'block';
    scrollToBottom();
}

function hideTypingIndicator() {
    elements.typingIndicator.style.display = 'none';
}

// =============================================================================
// Agent Switching
// =============================================================================

function switchAgent(agentName) {
    if (state.currentAgent === agentName) return;

    state.currentAgent = agentName;

    // Update UI
    elements.agentBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.agent === agentName);
    });

    // Render this agent's chat without nuking other agents
    renderChat(agentName);

    // Load greeting only if this agent has no messages yet
    if ((state.chatByAgent[agentName] || []).length === 0) {
        loadGreeting(agentName);
    }

    // Update send button state for this agent
    elements.sendBtn.disabled = !!state.isProcessingByAgent[agentName];
}

async function loadGreeting(agentKey = state.currentAgent) {
    try {
        const response = await fetch(`/api/chat/greeting/${agentKey}`);
        if (response.ok) {
            const data = await response.json();
            const a = data.agent || agentKey;
            recordMessage(a, 'assistant', data.greeting, a, 'high');
        }
    } catch (error) {
        console.error('Failed to load greeting:', error);
    }
}

// =============================================================================
// Health Check
// =============================================================================

async function checkConnection() {
    try {
        const response = await fetch('/api/health/ping');
        const isConnected = response.ok;
        
        elements.connectionStatus.classList.toggle('connected', isConnected);
        elements.connectionStatus.classList.toggle('disconnected', !isConnected);
        elements.connectionStatus.title = isConnected ? 'Connected' : 'Disconnected';
        
    } catch (error) {
        elements.connectionStatus.classList.remove('connected');
        elements.connectionStatus.classList.add('disconnected');
        elements.connectionStatus.title = 'Disconnected';
    }
}

// =============================================================================
// Input Handling
// =============================================================================

function autoResizeInput() {
    const input = elements.messageInput;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
}

// =============================================================================
// Event Listeners
// =============================================================================

// Login form
elements.loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    login(username, password);
});

// Logout button
elements.logoutBtn.addEventListener('click', logout);

// Send button
elements.sendBtn.addEventListener('click', sendMessage);

// Message input
elements.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

elements.messageInput.addEventListener('input', autoResizeInput);

// Agent buttons
elements.agentBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        switchAgent(btn.dataset.agent);
    });
});


function setupMicButton() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const micBtn = document.createElement('button');
    micBtn.id = 'mic-btn';
    micBtn.type = 'button';
    micBtn.title = 'Dictate';
    micBtn.textContent = '🎤';
    micBtn.style.marginRight = '8px';

    // Insert mic button just before Send button
    if (elements.sendBtn && elements.sendBtn.parentNode) {
        elements.sendBtn.parentNode.insertBefore(micBtn, elements.sendBtn);
    }

    const recog = new SpeechRecognition();
    recog.lang = 'en-US';
    recog.interimResults = true;
    recog.continuous = false;

    let listening = false;
    let finalText = '';

    micBtn.addEventListener('click', () => {
        if (listening) {
            try { recog.stop(); } catch (_) {}
            return;
        }
        finalText = '';
        try {
            recog.start();
            listening = true;
            micBtn.textContent = '🛑';
        } catch (e) {
            console.error('Speech recognition start failed', e);
        }
    });

    recog.onresult = (event) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const txt = event.results[i][0].transcript;
            if (event.results[i].isFinal) finalText += txt;
            else interim += txt;
        }
        const base = (elements.messageInput.value || '').split('⟂')[0].trim();
        const combined = (base ? base + ' ' : '') + finalText + (interim ? ' ⟂ ' + interim : '');
        elements.messageInput.value = combined.trim();
        autoResizeInput();
    };

    recog.onend = () => {
        listening = false;
        micBtn.textContent = '🎤';
        elements.messageInput.value = (elements.messageInput.value || '').replace(/\s*⟂\s*.*$/, '').trim();
        autoResizeInput();
    };

    recog.onerror = (e) => {
        listening = false;
        micBtn.textContent = '🎤';
        console.error('Speech recognition error', e);
    };
}


// =============================================================================
// Initialization
// =============================================================================

function init() {
    setupMicButton();
    // Check connection periodically
    checkConnection();
    setInterval(checkConnection, 30000); // Every 30 seconds
    
    // Check if already logged in
    if (state.token) {
        showMainContent();
        loadGreeting();
    } else {
        showLoginModal();
    }
}

// Start the app
init();

// ===== Import / Export wiring =====
(function setupImportExport() {
  const importBtn = document.getElementById('importBtn');
  const exportBtn = document.getElementById('exportBtn');
  const fileInput = document.getElementById('fileInput');

  if (!importBtn || !exportBtn || !fileInput) return;

  importBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    for (const file of files) {
      const form = new FormData();
      form.append('file', file);

      try {
        const res = await fetch('/api/knowledge/upload', {
          method: 'POST',
          body: form
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          console.error('Upload failed:', file.name, data);
          alert(`Upload failed for ${file.name}`);
        } else {
          console.log('Uploaded:', file.name, data);
        }
      } catch (err) {
        console.error('Upload error:', file.name, err);
        alert(`Upload error for ${file.name}`);
      }
    }

    fileInput.value = '';
    alert('Import complete.');
  });

  exportBtn.addEventListener('click', () => {
    const chatEl = document.getElementById('chatMessages') || document.querySelector('.chat-messages') || document.body;
    const text = chatEl.innerText || '';
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `lea_chat_export_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });
})();
