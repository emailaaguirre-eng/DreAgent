/* =============================================================================
   CHIQUIS IDE - JavaScript
   Powered by CoDre-X | B & D Servicing LLC
   ============================================================================= */

// =============================================================================
// State
// =============================================================================

const state = {
    token: localStorage.getItem('hummingbird_token'),
    currentProject: 'default',
    openFiles: new Map(), // path -> { content, modified, model }
    activeFile: null,
    editor: null,
    chatHistory: [],
    selectedCode: null,
    isProcessing: false,
};

// File icons mapping
const fileIcons = {
    '.py': '🐍',
    '.js': '📜',
    '.ts': '📘',
    '.tsx': '⚛️',
    '.jsx': '⚛️',
    '.html': '🌐',
    '.css': '🎨',
    '.json': '📋',
    '.md': '📝',
    '.sql': '🗄️',
    '.sh': '💻',
    '.yaml': '⚙️',
    '.yml': '⚙️',
    'default': '📄',
    'folder': '📁',
    'folder-open': '📂',
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    projectSelect: document.getElementById('project-select'),
    fileTree: document.getElementById('file-tree'),
    editorTabs: document.getElementById('editor-tabs'),
    editorContainer: document.getElementById('editor-container'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    chatContextIndicator: document.getElementById('chat-context-indicator'),
    inlineEditDialog: document.getElementById('inline-edit-dialog'),
    inlineEditInput: document.getElementById('inline-edit-input'),
    cursorPosition: document.getElementById('cursor-position'),
    languageIndicator: document.getElementById('language-indicator'),
    fileStatus: document.getElementById('file-status'),
    aiStatus: document.getElementById('ai-status'),
    connectionStatus: document.getElementById('connection-status'),
};

// =============================================================================
// Monaco Editor Setup
// =============================================================================

function initMonaco() {
    require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }});

    require(['vs/editor/editor.main'], function() {
        // Set dark theme
        monaco.editor.setTheme('vs-dark');

        // Create editor
        state.editor = monaco.editor.create(document.getElementById('monaco-editor'), {
            value: '# Welcome to Chiquis IDE!\n# Open a file from the sidebar to start coding.\n\nprint("Hello, World!")',
            language: 'python',
            theme: 'vs-dark',
            automaticLayout: true,
            fontSize: 14,
            fontFamily: "'Cascadia Code', 'Fira Code', Consolas, monospace",
            minimap: { enabled: true },
            lineNumbers: 'on',
            renderWhitespace: 'selection',
            bracketPairColorization: { enabled: true },
            scrollBeyondLastLine: false,
            wordWrap: 'off',
            tabSize: 4,
            insertSpaces: true,
            formatOnPaste: true,
            suggestOnTriggerCharacters: true,
        });

        // Cursor position tracking
        state.editor.onDidChangeCursorPosition((e) => {
            const pos = e.position;
            elements.cursorPosition.textContent = `Ln ${pos.lineNumber}, Col ${pos.column}`;
        });

        // Content change tracking
        state.editor.onDidChangeModelContent(() => {
            if (state.activeFile) {
                const fileData = state.openFiles.get(state.activeFile);
                if (fileData) {
                    fileData.modified = true;
                    updateTabModifiedState(state.activeFile, true);
                }
            }
        });

        // Selection change for chat context
        state.editor.onDidChangeCursorSelection((e) => {
            const selection = state.editor.getSelection();
            if (!selection.isEmpty()) {
                state.selectedCode = state.editor.getModel().getValueInRange(selection);
                elements.chatContextIndicator.style.display = 'flex';
            } else {
                state.selectedCode = null;
                elements.chatContextIndicator.style.display = 'none';
            }
        });

        // Register Cmd+K / Ctrl+K for inline edit
        state.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyK, () => {
            showInlineEditDialog();
        });

        // Register Cmd+S / Ctrl+S for save
        state.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
            saveCurrentFile();
        });

        // Setup AI autocomplete
        setupAIAutocomplete();

        console.log('Monaco Editor initialized');
    });
}

// =============================================================================
// AI Autocomplete
// =============================================================================

function setupAIAutocomplete() {
    // Register completion provider for all languages
    const languages = ['python', 'javascript', 'typescript', 'html', 'css', 'json'];

    languages.forEach(lang => {
        monaco.languages.registerCompletionItemProvider(lang, {
            triggerCharacters: ['.', '(', '[', '{', ' ', '\n'],
            provideCompletionItems: async (model, position) => {
                // Only trigger on specific conditions
                const lineContent = model.getLineContent(position.lineNumber);
                const textBeforeCursor = lineContent.substring(0, position.column - 1);

                // Don't trigger in comments or strings (basic check)
                if (textBeforeCursor.includes('#') || textBeforeCursor.includes('//')) {
                    return { suggestions: [] };
                }

                try {
                    const code = model.getValue();
                    const offset = model.getOffsetAt(position);

                    const response = await fetchAPI('/api/ide/completion', {
                        method: 'POST',
                        body: JSON.stringify({
                            code: code,
                            cursor_position: offset,
                            file_path: state.activeFile,
                            language: lang,
                            max_tokens: 100,
                        }),
                    });

                    if (response.completion && response.completion.trim()) {
                        const completion = response.completion;

                        return {
                            suggestions: [{
                                label: completion.split('\n')[0].substring(0, 50) + '...',
                                kind: monaco.languages.CompletionItemKind.Snippet,
                                insertText: completion,
                                detail: '✨ Chiquis AI',
                                documentation: 'AI-generated completion',
                                range: {
                                    startLineNumber: position.lineNumber,
                                    startColumn: position.column,
                                    endLineNumber: position.lineNumber,
                                    endColumn: position.column,
                                },
                            }],
                        };
                    }
                } catch (error) {
                    console.error('AI autocomplete error:', error);
                }

                return { suggestions: [] };
            },
        });
    });
}

// =============================================================================
// Inline Edit (Cmd+K)
// =============================================================================

function showInlineEditDialog() {
    const selection = state.editor.getSelection();
    let selectedText = '';

    if (!selection.isEmpty()) {
        selectedText = state.editor.getModel().getValueInRange(selection);
    } else {
        // Get current line
        const position = state.editor.getPosition();
        selectedText = state.editor.getModel().getLineContent(position.lineNumber);
    }

    if (!selectedText.trim()) {
        return;
    }

    elements.inlineEditDialog.style.display = 'block';
    elements.inlineEditInput.value = '';
    elements.inlineEditInput.focus();

    // Store selection info
    elements.inlineEditInput.dataset.selection = JSON.stringify({
        text: selectedText,
        range: selection.isEmpty() ? null : {
            startLine: selection.startLineNumber,
            startCol: selection.startColumn,
            endLine: selection.endLineNumber,
            endCol: selection.endColumn,
        },
    });
}

function hideInlineEditDialog() {
    elements.inlineEditDialog.style.display = 'none';
    elements.inlineEditInput.value = '';
    state.editor.focus();
}

async function applyInlineEdit() {
    const instruction = elements.inlineEditInput.value.trim();
    if (!instruction) return;

    const selectionData = JSON.parse(elements.inlineEditInput.dataset.selection || '{}');
    if (!selectionData.text) return;

    hideInlineEditDialog();
    elements.aiStatus.textContent = 'AI: Editing...';

    try {
        const model = state.editor.getModel();
        const code = selectionData.text;

        // Get context
        const fullCode = model.getValue();
        const cursorOffset = model.getOffsetAt(state.editor.getPosition());
        const contextBefore = fullCode.substring(Math.max(0, cursorOffset - 500), cursorOffset);
        const contextAfter = fullCode.substring(cursorOffset, cursorOffset + 500);

        const response = await fetchAPI('/api/ide/inline-edit', {
            method: 'POST',
            body: JSON.stringify({
                code: code,
                instruction: instruction,
                context_before: contextBefore,
                context_after: contextAfter,
                file_path: state.activeFile,
                language: model.getLanguageId(),
            }),
        });

        if (response.edited_code) {
            // Apply the edit
            const selection = state.editor.getSelection();
            state.editor.executeEdits('inline-edit', [{
                range: selection.isEmpty()
                    ? new monaco.Range(
                        state.editor.getPosition().lineNumber, 1,
                        state.editor.getPosition().lineNumber + 1, 1
                    )
                    : selection,
                text: response.edited_code + (selection.isEmpty() ? '\n' : ''),
            }]);

            // Show feedback
            addChatMessage('assistant', `✨ Applied edit: ${response.explanation}`);
        }

    } catch (error) {
        console.error('Inline edit error:', error);
        addChatMessage('assistant', `❌ Edit failed: ${error.message}`);
    } finally {
        elements.aiStatus.textContent = 'AI: Ready';
    }
}

// =============================================================================
// File Management
// =============================================================================

async function loadFileTree() {
    try {
        const data = await fetchAPI(`/api/ide/files/tree?project=${state.currentProject}`);
        renderFileTree(data.tree, elements.fileTree, '');
    } catch (error) {
        console.error('Failed to load file tree:', error);
        elements.fileTree.innerHTML = '<div style="padding: 12px; color: #f14c4c;">Failed to load files</div>';
    }
}

function renderFileTree(nodes, container, parentPath) {
    container.innerHTML = '';

    nodes.forEach(node => {
        const itemDiv = document.createElement('div');
        itemDiv.className = `file-tree-item ${node.type}`;
        itemDiv.dataset.path = node.path;
        itemDiv.dataset.type = node.type;

        const ext = node.name.includes('.') ? '.' + node.name.split('.').pop() : '';
        const icon = node.type === 'directory'
            ? fileIcons['folder']
            : (fileIcons[ext] || fileIcons['default']);

        itemDiv.innerHTML = `
            <span class="icon">${icon}</span>
            <span class="name">${node.name}</span>
        `;

        if (node.type === 'file') {
            itemDiv.addEventListener('click', () => openFile(node.path));
        } else if (node.type === 'directory' && node.children) {
            const childrenDiv = document.createElement('div');
            childrenDiv.className = 'file-tree-children';
            renderFileTree(node.children, childrenDiv, node.path);

            itemDiv.addEventListener('click', (e) => {
                e.stopPropagation();
                itemDiv.classList.toggle('collapsed');
                const iconSpan = itemDiv.querySelector('.icon');
                iconSpan.textContent = itemDiv.classList.contains('collapsed')
                    ? fileIcons['folder']
                    : fileIcons['folder-open'];
            });

            container.appendChild(itemDiv);
            container.appendChild(childrenDiv);
            return;
        }

        container.appendChild(itemDiv);
    });
}

async function openFile(path) {
    // Check if already open
    if (state.openFiles.has(path)) {
        switchToFile(path);
        return;
    }

    try {
        elements.fileStatus.textContent = 'Loading...';
        const data = await fetchAPI(`/api/ide/files/read?path=${encodeURIComponent(path)}&project=${state.currentProject}`);

        // Store file data
        state.openFiles.set(path, {
            content: data.content,
            language: data.language,
            modified: false,
        });

        // Create tab
        createTab(path, data.language);

        // Switch to file
        switchToFile(path);

        elements.fileStatus.textContent = 'Ready';

    } catch (error) {
        console.error('Failed to open file:', error);
        elements.fileStatus.textContent = 'Error';
    }
}

function switchToFile(path) {
    const fileData = state.openFiles.get(path);
    if (!fileData) return;

    state.activeFile = path;

    // Update editor content
    const model = monaco.editor.createModel(fileData.content, fileData.language);
    state.editor.setModel(model);

    // Update tab state
    document.querySelectorAll('.editor-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.path === path);
    });

    // Update language indicator
    elements.languageIndicator.textContent = fileData.language.charAt(0).toUpperCase() + fileData.language.slice(1);

    // Update file tree selection
    document.querySelectorAll('.file-tree-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.path === path);
    });
}

function createTab(path, language) {
    const tab = document.createElement('div');
    tab.className = 'editor-tab active';
    tab.dataset.path = path;

    const fileName = path.split('/').pop();
    const ext = fileName.includes('.') ? '.' + fileName.split('.').pop() : '';
    const icon = fileIcons[ext] || fileIcons['default'];

    tab.innerHTML = `
        <span class="tab-icon">${icon}</span>
        <span class="tab-name">${fileName}</span>
        <button class="tab-close" title="Close">×</button>
    `;

    tab.addEventListener('click', (e) => {
        if (!e.target.classList.contains('tab-close')) {
            switchToFile(path);
        }
    });

    tab.querySelector('.tab-close').addEventListener('click', (e) => {
        e.stopPropagation();
        closeFile(path);
    });

    // Deactivate other tabs
    document.querySelectorAll('.editor-tab').forEach(t => t.classList.remove('active'));

    elements.editorTabs.appendChild(tab);
}

function updateTabModifiedState(path, modified) {
    const tab = document.querySelector(`.editor-tab[data-path="${path}"]`);
    if (tab) {
        tab.classList.toggle('modified', modified);
    }
}

async function saveCurrentFile() {
    if (!state.activeFile) return;

    const fileData = state.openFiles.get(state.activeFile);
    if (!fileData) return;

    try {
        elements.fileStatus.textContent = 'Saving...';

        const content = state.editor.getValue();

        await fetchAPI(`/api/ide/files/write?project=${state.currentProject}`, {
            method: 'POST',
            body: JSON.stringify({
                path: state.activeFile,
                content: content,
            }),
        });

        fileData.content = content;
        fileData.modified = false;
        updateTabModifiedState(state.activeFile, false);

        elements.fileStatus.textContent = 'Saved';
        setTimeout(() => {
            elements.fileStatus.textContent = 'Ready';
        }, 2000);

    } catch (error) {
        console.error('Failed to save file:', error);
        elements.fileStatus.textContent = 'Save failed';
    }
}

function closeFile(path) {
    const fileData = state.openFiles.get(path);
    if (fileData && fileData.modified) {
        if (!confirm('File has unsaved changes. Close anyway?')) {
            return;
        }
    }

    // Remove from state
    state.openFiles.delete(path);

    // Remove tab
    const tab = document.querySelector(`.editor-tab[data-path="${path}"]`);
    if (tab) tab.remove();

    // Switch to another file if this was active
    if (state.activeFile === path) {
        const remainingFiles = Array.from(state.openFiles.keys());
        if (remainingFiles.length > 0) {
            switchToFile(remainingFiles[remainingFiles.length - 1]);
        } else {
            state.activeFile = null;
            state.editor.setValue('// No file open');
            elements.languageIndicator.textContent = 'Plain Text';
        }
    }
}

// =============================================================================
// Chat with Chiquis
// =============================================================================

async function sendChatMessage() {
    const message = elements.chatInput.value.trim();
    if (!message || state.isProcessing) return;

    state.isProcessing = true;
    elements.chatSendBtn.disabled = true;

    // Add user message
    addChatMessage('user', message);
    elements.chatInput.value = '';

    // Show thinking
    elements.aiStatus.textContent = 'AI: Thinking...';

    try {
        const response = await fetchAPI(`/api/ide/chat?project=${state.currentProject}`, {
            method: 'POST',
            body: JSON.stringify({
                message: message,
                code_context: state.selectedCode,
                file_path: state.activeFile,
                language: state.editor ? state.editor.getModel()?.getLanguageId() : null,
                history: state.chatHistory.slice(-10),
                use_rag: true,
            }),
        });

        addChatMessage('assistant', response.response);

        // Store in history
        state.chatHistory.push({ role: 'user', content: message });
        state.chatHistory.push({ role: 'assistant', content: response.response });

        // Show referenced files
        if (response.referenced_files && response.referenced_files.length > 0) {
            addChatMessage('assistant', `📎 Referenced: ${response.referenced_files.join(', ')}`, true);
        }

    } catch (error) {
        console.error('Chat error:', error);
        addChatMessage('assistant', `❌ Error: ${error.message}`);
    } finally {
        state.isProcessing = false;
        elements.chatSendBtn.disabled = false;
        elements.aiStatus.textContent = 'AI: Ready';
    }
}

function addChatMessage(role, content, isSystem = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    if (isSystem) {
        messageDiv.style.fontSize = '11px';
        messageDiv.style.opacity = '0.7';
    }

    // Format content
    let formattedContent = content
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Format code blocks
    formattedContent = formattedContent.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Format inline code
    formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Format line breaks
    formattedContent = formattedContent.replace(/\n/g, '<br>');

    const name = role === 'user' ? 'You' : 'Chiquis';
    messageDiv.innerHTML = `
        <div class="message-content">
            <strong>${name}:</strong> ${formattedContent}
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// =============================================================================
// Project Management
// =============================================================================

async function loadProjects() {
    try {
        const data = await fetchAPI('/api/ide/projects');
        elements.projectSelect.innerHTML = '';

        data.projects.forEach(project => {
            const option = document.createElement('option');
            option.value = project.name;
            option.textContent = project.name;
            elements.projectSelect.appendChild(option);
        });

        if (data.projects.length === 0) {
            const option = document.createElement('option');
            option.value = 'default';
            option.textContent = 'default';
            elements.projectSelect.appendChild(option);
        }

    } catch (error) {
        console.error('Failed to load projects:', error);
    }
}

async function createProject(name) {
    try {
        await fetchAPI(`/api/ide/projects/create?name=${encodeURIComponent(name)}`, {
            method: 'POST',
        });

        await loadProjects();
        state.currentProject = name;
        elements.projectSelect.value = name;
        await loadFileTree();

    } catch (error) {
        console.error('Failed to create project:', error);
        alert('Failed to create project: ' + error.message);
    }
}

// =============================================================================
// API Helper
// =============================================================================

async function fetchAPI(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...(state.token && { 'Authorization': `Bearer ${state.token}` }),
        ...options.headers,
    };

    const response = await fetch(url, {
        ...options,
        headers,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(error.detail || 'Request failed');
    }

    return response.json();
}

// =============================================================================
// Event Listeners
// =============================================================================

// Chat input
elements.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

elements.chatSendBtn.addEventListener('click', sendChatMessage);

// Clear chat context
document.getElementById('clear-context-btn').addEventListener('click', () => {
    state.selectedCode = null;
    elements.chatContextIndicator.style.display = 'none';
});

// Inline edit
elements.inlineEditInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        applyInlineEdit();
    } else if (e.key === 'Escape') {
        hideInlineEditDialog();
    }
});

// Project selector
elements.projectSelect.addEventListener('change', async (e) => {
    state.currentProject = e.target.value;
    state.openFiles.clear();
    elements.editorTabs.innerHTML = '';
    state.activeFile = null;
    await loadFileTree();
});

// New project
document.getElementById('new-project-btn').addEventListener('click', () => {
    document.getElementById('new-project-modal').style.display = 'flex';
    document.getElementById('new-project-name').value = '';
    document.getElementById('new-project-name').focus();
});

document.getElementById('new-project-cancel').addEventListener('click', () => {
    document.getElementById('new-project-modal').style.display = 'none';
});

document.getElementById('new-project-create').addEventListener('click', async () => {
    const name = document.getElementById('new-project-name').value.trim();
    if (name) {
        await createProject(name);
        document.getElementById('new-project-modal').style.display = 'none';
    }
});

// New file
document.getElementById('new-file-btn').addEventListener('click', () => {
    document.getElementById('new-item-modal').style.display = 'flex';
    document.getElementById('new-item-title').textContent = 'New File';
    document.getElementById('new-item-name').value = '';
    document.getElementById('new-item-name').placeholder = 'filename.py';
    document.getElementById('new-item-name').dataset.type = 'file';
    document.getElementById('new-item-name').focus();
});

// New folder
document.getElementById('new-folder-btn').addEventListener('click', () => {
    document.getElementById('new-item-modal').style.display = 'flex';
    document.getElementById('new-item-title').textContent = 'New Folder';
    document.getElementById('new-item-name').value = '';
    document.getElementById('new-item-name').placeholder = 'folder-name';
    document.getElementById('new-item-name').dataset.type = 'folder';
    document.getElementById('new-item-name').focus();
});

document.getElementById('new-item-cancel').addEventListener('click', () => {
    document.getElementById('new-item-modal').style.display = 'none';
});

document.getElementById('new-item-create').addEventListener('click', async () => {
    const name = document.getElementById('new-item-name').value.trim();
    const type = document.getElementById('new-item-name').dataset.type;

    if (name) {
        try {
            if (type === 'folder') {
                await fetchAPI(`/api/ide/files/create-folder?path=${encodeURIComponent(name)}&project=${state.currentProject}`, {
                    method: 'POST',
                });
            } else {
                await fetchAPI(`/api/ide/files/write?project=${state.currentProject}`, {
                    method: 'POST',
                    body: JSON.stringify({
                        path: name,
                        content: name.endsWith('.py') ? '# New file\n' : '',
                    }),
                });
            }

            await loadFileTree();
            document.getElementById('new-item-modal').style.display = 'none';

            if (type === 'file') {
                openFile(name);
            }

        } catch (error) {
            alert('Failed to create: ' + error.message);
        }
    }
});

// Refresh
document.getElementById('refresh-btn').addEventListener('click', loadFileTree);

// Toggle chat
document.getElementById('toggle-chat-btn').addEventListener('click', () => {
    document.getElementById('chat-sidebar').classList.toggle('collapsed');
});

// Connection check
async function checkConnection() {
    try {
        const response = await fetch('/api/health/ping');
        const isConnected = response.ok;
        elements.connectionStatus.classList.toggle('connected', isConnected);
        elements.connectionStatus.classList.toggle('disconnected', !isConnected);
    } catch {
        elements.connectionStatus.classList.remove('connected');
        elements.connectionStatus.classList.add('disconnected');
    }
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
    // Check authentication
    if (!state.token) {
        window.location.href = '/';
        return;
    }

    // Initialize Monaco Editor
    initMonaco();

    // Load projects and file tree
    await loadProjects();
    await loadFileTree();

    // Check connection
    checkConnection();
    setInterval(checkConnection, 30000);

    console.log('Chiquis IDE initialized');
}

// Start
init();
