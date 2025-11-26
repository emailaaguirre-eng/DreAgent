### Lea Agentic Code Editor ###

"""
Hummingbird – Lea Agentic Coding Assistant
AI can autonomously:
- Create new files
- Edit existing files
- Execute code
- Debug and fix issues
- Apply multi-file changes

Like Cursor, Windsurf, or Claude's computer use
"""

from dotenv import load_dotenv
load_dotenv()

import os
import html
import json
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
import difflib
import re

import requests
from openai import OpenAI

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRegularExpression
from PyQt6.QtGui import (
    QIcon, QPixmap, QFont, QSyntaxHighlighter, 
    QTextCharFormat, QColor, QTextCursor
)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QPlainTextEdit,
    QSizePolicy,
    QFrame,
    QSplashScreen,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QCheckBox,
    QScrollArea,
)

# =====================================================
# OPENAI SETUP
# =====================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not OPENAI_API_KEY:
    print("=" * 60)
    print("CRITICAL WARNING: 'OPENAI_API_KEY' not found")
    print("=" * 60)

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"

SPLASH_FILE = ASSETS_DIR / "Hummingbird_LEA_v1_Splash_Logo_Lime_Green.png"
ICON_FILE = ASSETS_DIR / "Hummingbird_LEA_Logo_White_No BKGND.png"

# =====================================================
# SYNTAX HIGHLIGHTER
# =====================================================

class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Define formats
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#569CD6"))
        self.keyword_format.setFontWeight(QFont.Weight.Bold)
        
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#CE9178"))
        
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#6A9955"))
        
        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#DCDCAA"))
        
        # Keywords
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True',
            'try', 'while', 'with', 'yield', 'async', 'await'
        ]
        
        self.highlighting_rules = []
        for word in keywords:
            pattern = QRegularExpression(r'\b' + word + r'\b')
            self.highlighting_rules.append((pattern, self.keyword_format))
        
        # Strings and comments
        self.string_patterns = [
            QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'),
            QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"),
        ]
        
        self.comment_pattern = QRegularExpression(r'#[^\n]*')
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)
        
        for pattern in self.string_patterns:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.string_format)
        
        iterator = self.comment_pattern.globalMatch(text)
        while iterator.hasNext():
            match = iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.comment_format)

# =====================================================
# CODE EDITOR WIDGET
# =====================================================

class CodeEditor(QPlainTextEdit):
    """Enhanced code editor"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        font = QFont("Consolas", 11)
        font.setFixedPitch(True)
        self.setFont(font)
        
        tab_width = self.fontMetrics().horizontalAdvance(' ') * 4
        self.setTabStopDistance(tab_width)
        
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                selection-background-color: #264F78;
            }
        """)
        
        self.highlighter = PythonHighlighter(self.document())
        self.current_file_path = None
    
    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.setPlainText(content)
            self.current_file_path = file_path
            return True
        except Exception as e:
            return False
    
    def save_file(self, file_path=None):
        if file_path is None:
            file_path = self.current_file_path
        
        if file_path is None:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.toPlainText())
            self.current_file_path = file_path
            return True
        except Exception as e:
            return False

# =====================================================
# AGENTIC SYSTEM PROMPT
# =====================================================

AGENTIC_SYSTEM_PROMPT = """
You are Lea, an agentic coding assistant with the ability to create, read, and modify files.

**YOUR CAPABILITIES:**

You can perform file operations using special XML tags:

1. **CREATE A NEW FILE:**
<create_file>
<path>path/to/file.py</path>
<content>
file content here
</content>
</create_file>

2. **READ A FILE:**
<read_file>
<path>path/to/file.py</path>
</read_file>

3. **EDIT A FILE (String Replacement):**
<edit_file>
<path>path/to/file.py</path>
<old_content>
exact text to find
</old_content>
<new_content>
replacement text
</new_content>
</edit_file>

4. **RUN CODE:**
<run_code>
<path>path/to/file.py</path>
</run_code>

**IMPORTANT RULES:**

1. **Always use XML tags** for file operations - the system will execute them
2. **Be precise** with old_content in edits - must match exactly
3. **Show your reasoning** before making changes
4. **Confirm actions** by describing what you're doing
5. **Handle errors gracefully** - if an edit fails, try a different approach
6. **Test your changes** - suggest running code after edits

**WORKFLOW EXAMPLE:**

When Dre asks you to fix a bug:
1. Read the file to understand the issue
2. Explain what you found and your fix
3. Make the edit using edit_file tag
4. Suggest running the code to test

**CODE QUALITY STANDARDS:**
- Follow PEP 8 for Python
- Add comments for complex logic
- Include error handling
- Write clean, maintainable code
- Test before declaring something fixed

**TONE:**
- Confident but humble
- Explain your reasoning
- Ask questions when requirements are unclear
- Celebrate successes, learn from failures

You're an expert pair programmer - work WITH Dre, not FOR her.
"""

# =====================================================
# FILE OPERATION EXECUTOR
# =====================================================

class FileOperationExecutor:
    """Executes file operations from Lea's responses"""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.operation_log = []
    
    def execute_operations(self, response_text: str) -> list:
        """Parse and execute file operations from Lea's response"""
        operations = []
        
        # Find all XML operations
        create_files = self._find_operations(response_text, 'create_file')
        read_files = self._find_operations(response_text, 'read_file')
        edit_files = self._find_operations(response_text, 'edit_file')
        run_code = self._find_operations(response_text, 'run_code')
        
        # Execute create operations
        for op in create_files:
            result = self._create_file(op)
            operations.append(result)
        
        # Execute edit operations
        for op in edit_files:
            result = self._edit_file(op)
            operations.append(result)
        
        # Execute run operations
        for op in run_code:
            result = self._run_code(op)
            operations.append(result)
        
        return operations
    
    def _find_operations(self, text: str, tag: str) -> list:
        """Extract XML operations from text"""
        pattern = f'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        operations = []
        for match in matches:
            op = {'type': tag}
            
            # Extract path
            path_match = re.search(r'<path>(.*?)</path>', match, re.DOTALL)
            if path_match:
                op['path'] = path_match.group(1).strip()
            
            # Extract content
            content_match = re.search(r'<content>(.*?)</content>', match, re.DOTALL)
            if content_match:
                op['content'] = content_match.group(1).strip()
            
            # Extract old_content for edits
            old_match = re.search(r'<old_content>(.*?)</old_content>', match, re.DOTALL)
            if old_match:
                op['old_content'] = old_match.group(1).strip()
            
            # Extract new_content for edits
            new_match = re.search(r'<new_content>(.*?)</new_content>', match, re.DOTALL)
            if new_match:
                op['new_content'] = new_match.group(1).strip()
            
            operations.append(op)
        
        return operations
    
    def _create_file(self, op: dict) -> dict:
        """Create a new file"""
        try:
            file_path = self.workspace_dir / op['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(op['content'])
            
            self.operation_log.append({
                'type': 'create',
                'path': str(file_path),
                'success': True,
                'timestamp': datetime.now()
            })
            
            return {
                'success': True,
                'type': 'create',
                'path': str(file_path),
                'message': f"Created file: {op['path']}"
            }
        except Exception as e:
            return {
                'success': False,
                'type': 'create',
                'path': op.get('path', 'unknown'),
                'error': str(e)
            }
    
    def _edit_file(self, op: dict) -> dict:
        """Edit an existing file"""
        try:
            file_path = self.workspace_dir / op['path']
            
            if not file_path.exists():
                return {
                    'success': False,
                    'type': 'edit',
                    'path': str(file_path),
                    'error': 'File does not exist'
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace old content with new
            if op['old_content'] not in content:
                return {
                    'success': False,
                    'type': 'edit',
                    'path': str(file_path),
                    'error': 'Old content not found in file'
                }
            
            new_content = content.replace(op['old_content'], op['new_content'])
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.operation_log.append({
                'type': 'edit',
                'path': str(file_path),
                'success': True,
                'timestamp': datetime.now()
            })
            
            return {
                'success': True,
                'type': 'edit',
                'path': str(file_path),
                'message': f"Edited file: {op['path']}"
            }
        except Exception as e:
            return {
                'success': False,
                'type': 'edit',
                'path': op.get('path', 'unknown'),
                'error': str(e)
            }
    
    def _run_code(self, op: dict) -> dict:
        """Run Python code"""
        try:
            file_path = self.workspace_dir / op['path']
            
            if not file_path.exists():
                return {
                    'success': False,
                    'type': 'run',
                    'path': str(file_path),
                    'error': 'File does not exist'
                }
            
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.workspace_dir)
            )
            
            output = {
                'success': result.returncode == 0,
                'type': 'run',
                'path': str(file_path),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            self.operation_log.append({
                'type': 'run',
                'path': str(file_path),
                'success': result.returncode == 0,
                'timestamp': datetime.now()
            })
            
            return output
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'type': 'run',
                'path': op.get('path', 'unknown'),
                'error': 'Execution timed out (10 seconds)'
            }
        except Exception as e:
            return {
                'success': False,
                'type': 'run',
                'path': op.get('path', 'unknown'),
                'error': str(e)
            }

# =====================================================
# CHAT INPUT WIDGET
# =====================================================

class ChatInputBox(QTextEdit):
    returnPressed = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.returnPressed.emit()
        else:
            super().keyPressEvent(event)

# =====================================================
# MAIN WINDOW
# =====================================================

class LeaAgenticWindow(QWidget):
    USER_COLOR = "#68BD47"
    ASSIST_COLOR = "#FFFFFF"
    SYSTEM_COLOR = "#2DBCEE"
    SUCCESS_COLOR = "#4EC9B0"
    ERROR_COLOR = "#F48771"

    def __init__(self):
        super().__init__()

        self.workspace_dir = PROJECT_DIR / "workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        
        self.message_history = []
        self.history_file = "lea_agentic_history.json"
        
        self.file_executor = FileOperationExecutor(self.workspace_dir)
        self.open_files = {}
        
        self.current_model = "gpt-4.1-mini"

        self._init_window()
        self._build_ui()
        self._load_history()

    def _init_window(self):
        self.setWindowTitle("Hummingbird – Lea Agentic Coding")
        if ICON_FILE.exists():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(1600, 900)

        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
            QLabel {
                color: #D4D4D4;
            }
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
            QTabWidget::pane {
                border: 1px solid #3C3C3C;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #2D2D30;
                color: #D4D4D4;
                padding: 6px 12px;
                border: 1px solid #3C3C3C;
            }
            QTabBar::tab:selected {
                background-color: #1E1E1E;
            }
        """)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Header
        header = self._create_header()
        main_layout.addLayout(header)

        # Main splitter: Chat | Editor | File Browser
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Chat interface
        chat_widget = self._create_chat_widget()
        self.main_splitter.addWidget(chat_widget)
        
        # Center: Code editor
        editor_widget = self._create_editor_widget()
        self.main_splitter.addWidget(editor_widget)
        
        # Right: File browser & operations log
        sidebar_widget = self._create_sidebar_widget()
        self.main_splitter.addWidget(sidebar_widget)
        
        # Set sizes: 30% chat, 50% editor, 20% sidebar
        self.main_splitter.setSizes([480, 800, 320])
        
        main_layout.addWidget(self.main_splitter)

        # Status bar
        self.status_label = QLabel("Ready. Lea can create, edit, and run files for you.")
        self.status_label.setStyleSheet("color: #CCCCCC; font-size: 11px; padding: 3px;")
        main_layout.addWidget(self.status_label)

    def _create_header(self):
        header = QHBoxLayout()
        header.setSpacing(10)

        title_label = QLabel("🤖 Lea Agentic Coding")
        title_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #68BD47;")
        header.addWidget(title_label)

        header.addStretch()

        # Workspace label
        workspace_label = QLabel(f"📁 Workspace: {self.workspace_dir.name}")
        workspace_label.setStyleSheet("font-size: 12px; color: #858585;")
        header.addWidget(workspace_label)

        # Model selector
        header.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gpt-4.1-mini (recommended)",
            "gpt-4.1 (advanced)",
            "gpt-4o"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        header.addWidget(self.model_combo)

        return header

    def _create_chat_widget(self):
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(5, 5, 5, 5)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        chat_layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        
        self.input_box = ChatInputBox()
        self.input_box.setPlaceholderText(
            "Ask Lea to create, edit, or fix code...\n"
            "Example: 'Create a Flask API with user authentication'"
        )
        self.input_box.returnPressed.connect(self.on_send_clicked)
        self.input_box.setMaximumHeight(100)
        self.input_box.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                font-size: 13px;
                padding: 5px;
            }
        """)
        input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        self.send_button.setMinimumWidth(70)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)
        input_layout.addWidget(self.send_button)

        chat_layout.addLayout(input_layout)

        # Quick actions
        actions_layout = QHBoxLayout()
        
        quick_actions = [
            ("Create File", "Create a new Python file called "),
            ("Fix Bug", "Find and fix the bug in "),
            ("Add Feature", "Add a new feature to "),
            ("Refactor", "Refactor and improve "),
        ]
        
        for label, prefix in quick_actions:
            btn = QPushButton(label)
            btn.setStyleSheet("font-size: 11px; padding: 3px 8px;")
            btn.clicked.connect(lambda checked, p=prefix: self.input_box.insertPlainText(p))
            actions_layout.addWidget(btn)
        
        actions_layout.addStretch()
        chat_layout.addLayout(actions_layout)

        return chat_container

    def _create_editor_widget(self):
        editor_container = QWidget()
        editor_layout = QVBoxLayout(editor_container)
        editor_layout.setContentsMargins(5, 5, 5, 5)

        # Editor toolbar
        toolbar = QHBoxLayout()
        
        open_btn = QPushButton("📂 Open")
        open_btn.clicked.connect(self.open_file)
        toolbar.addWidget(open_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_current_file)
        toolbar.addWidget(save_btn)
        
        toolbar.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_file_browser)
        toolbar.addWidget(refresh_btn)
        
        editor_layout.addLayout(toolbar)

        # Tab widget for files
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_tab)
        
        editor_layout.addWidget(self.editor_tabs)

        return editor_container

    def _create_sidebar_widget(self):
        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)

        # File browser
        browser_label = QLabel("📁 Workspace Files")
        browser_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        sidebar_layout.addWidget(browser_label)
        
        self.file_browser = QTextEdit()
        self.file_browser.setReadOnly(True)
        self.file_browser.setMaximumHeight(250)
        self.file_browser.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        sidebar_layout.addWidget(self.file_browser)
        
        # Operations log
        log_label = QLabel("📋 Operations Log")
        log_label.setStyleSheet("font-weight: 600; font-size: 13px; margin-top: 10px;")
        sidebar_layout.addWidget(log_label)
        
        self.operations_log = QTextEdit()
        self.operations_log.setReadOnly(True)
        self.operations_log.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        sidebar_layout.addWidget(self.operations_log)

        self.refresh_file_browser()

        return sidebar_container

    # ===================================================
    # FILE OPERATIONS
    # ===================================================

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            str(self.workspace_dir),
            "Python Files (*.py);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        # Check if already open
        for index in range(self.editor_tabs.count()):
            editor = self.editor_tabs.widget(index)
            if hasattr(editor, 'current_file_path') and editor.current_file_path == file_path:
                self.editor_tabs.setCurrentIndex(index)
                return
        
        # Open in new tab
        editor = CodeEditor()
        if editor.load_file(file_path):
            file_name = os.path.basename(file_path)
            index = self.editor_tabs.addTab(editor, file_name)
            self.editor_tabs.setCurrentIndex(index)
            self.open_files[index] = file_path

    def save_current_file(self):
        current_editor = self.editor_tabs.currentWidget()
        if isinstance(current_editor, CodeEditor):
            if current_editor.save_file():
                self.status_label.setText(f"Saved: {current_editor.current_file_path}")
                self.refresh_file_browser()

    def close_tab(self, index):
        self.editor_tabs.removeTab(index)
        if index in self.open_files:
            del self.open_files[index]

    def refresh_file_browser(self):
        """Refresh the file browser with current workspace contents"""
        files = []
        for path in sorted(self.workspace_dir.rglob('*')):
            if path.is_file():
                rel_path = path.relative_to(self.workspace_dir)
                files.append(str(rel_path))
        
        if files:
            self.file_browser.setText("\n".join(files))
        else:
            self.file_browser.setText("(empty workspace)")

    # ===================================================
    # CHAT & AI INTERACTION
    # ===================================================

    def append_colored_message(self, kind: str, text: str):
        text = text.strip()
        if not text:
            return

        if kind == "user":
            label = "Dre"
            color = self.USER_COLOR
        elif kind == "assistant":
            label = "Lea"
            color = self.ASSIST_COLOR
        elif kind == "success":
            label = "✓ Success"
            color = self.SUCCESS_COLOR
        elif kind == "error":
            label = "✗ Error"
            color = self.ERROR_COLOR
        else:
            label = "System"
            color = self.SYSTEM_COLOR

        safe_text = html.escape(text).replace("\n", "<br>")

        html_block = f"""
        <div style="margin: 8px 0; padding: 8px; background-color: #2D2D30; border-left: 3px solid {color};">
            <span style="color:{color}; font-weight:600;">{label}:</span>
            <div style="color:#D4D4D4; margin-top: 4px;">{safe_text}</div>
        </div>
        """

        self.chat_display.append(html_block)

    def on_send_clicked(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return

        if not OPENAI_API_KEY or not openai_client:
            self.append_colored_message("error", "OPENAI_API_KEY not set.")
            return

        self.append_colored_message("user", user_text)
        self.input_box.clear()
        self.status_label.setText("Lea is thinking and planning...")

        # Build context with workspace files
        workspace_context = self._build_workspace_context()
        
        prompt = f"{workspace_context}\n\nDre's request:\n{user_text}"
        
        self.message_history.append({"role": "user", "content": prompt})

        messages = [{"role": "system", "content": AGENTIC_SYSTEM_PROMPT}]
        messages.extend(self.message_history)

        try:
            response = openai_client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                temperature=0.7,
            )
            assistant_text = response.choices[0].message.content
        except Exception as e:
            assistant_text = f"[Error: {e}]"
            self.append_colored_message("error", str(e))
            self.status_label.setText("Error communicating with AI")
            return

        self.message_history.append({"role": "assistant", "content": assistant_text})
        
        # Display Lea's response (without XML tags)
        display_text = self._strip_xml_tags(assistant_text)
        self.append_colored_message("assistant", display_text)

        # Execute file operations
        self.status_label.setText("Executing file operations...")
        QApplication.processEvents()
        
        operations = self.file_executor.execute_operations(assistant_text)
        
        # Display operation results
        for op in operations:
            self._display_operation_result(op)
        
        # Refresh UI
        self.refresh_file_browser()
        self._update_operations_log()
        
        # Auto-open created/edited files
        for op in operations:
            if op['success'] and op['type'] in ['create', 'edit']:
                file_path = op['path']
                # Check if file is in workspace
                if file_path.startswith(str(self.workspace_dir)):
                    self._auto_open_file(file_path)
        
        self.status_label.setText("Ready.")
        self._save_history()

    def _build_workspace_context(self) -> str:
        """Build context about current workspace"""
        files = list(self.workspace_dir.rglob('*.py'))
        
        if not files:
            return "Current workspace: Empty (no files yet)"
        
        context_parts = ["Current workspace files:"]
        for file in files[:10]:  # Limit to 10 files
            rel_path = file.relative_to(self.workspace_dir)
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    context_parts.append(f"- {rel_path} ({lines} lines)")
            except:
                context_parts.append(f"- {rel_path}")
        
        if len(files) > 10:
            context_parts.append(f"... and {len(files) - 10} more files")
        
        return "\n".join(context_parts)

    def _strip_xml_tags(self, text: str) -> str:
        """Remove XML operation tags from display text"""
        text = re.sub(r'<create_file>.*?</create_file>', '', text, flags=re.DOTALL)
        text = re.sub(r'<read_file>.*?</read_file>', '', text, flags=re.DOTALL)
        text = re.sub(r'<edit_file>.*?</edit_file>', '', text, flags=re.DOTALL)
        text = re.sub(r'<run_code>.*?</run_code>', '', text, flags=re.DOTALL)
        return text.strip()

    def _display_operation_result(self, op: dict):
        """Display the result of a file operation"""
        if op['success']:
            if op['type'] == 'create':
                self.append_colored_message("success", op['message'])
            elif op['type'] == 'edit':
                self.append_colored_message("success", op['message'])
            elif op['type'] == 'run':
                output_text = f"Ran {op['path']}\n"
                if op['stdout']:
                    output_text += f"Output:\n{op['stdout']}"
                if op['stderr']:
                    output_text += f"\nErrors:\n{op['stderr']}"
                self.append_colored_message("success", output_text)
        else:
            error_msg = f"{op['type'].upper()} failed for {op.get('path', 'unknown')}: {op.get('error', 'Unknown error')}"
            self.append_colored_message("error", error_msg)

    def _update_operations_log(self):
        """Update the operations log display"""
        log_entries = []
        for entry in self.file_executor.operation_log[-20:]:  # Last 20 operations
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            status = "✓" if entry['success'] else "✗"
            log_entries.append(f"{timestamp} {status} {entry['type']}: {Path(entry['path']).name}")
        
        self.operations_log.setText("\n".join(log_entries))

    def _auto_open_file(self, file_path: str):
        """Automatically open a file in the editor"""
        # Check if already open
        for index in range(self.editor_tabs.count()):
            editor = self.editor_tabs.widget(index)
            if hasattr(editor, 'current_file_path') and editor.current_file_path == file_path:
                self.editor_tabs.setCurrentIndex(index)
                # Reload content
                editor.load_file(file_path)
                return
        
        # Open in new tab
        editor = CodeEditor()
        if editor.load_file(file_path):
            file_name = os.path.basename(file_path)
            index = self.editor_tabs.addTab(editor, file_name)
            self.editor_tabs.setCurrentIndex(index)
            self.open_files[index] = file_path

    def on_model_changed(self, text: str):
        if "gpt-4.1-mini" in text:
            self.current_model = "gpt-4.1-mini"
        elif "gpt-4.1" in text:
            self.current_model = "gpt-4.1"
        else:
            self.current_model = "gpt-4o"
        
        self.status_label.setText(f"Model: {self.current_model}")

    # ===================================================
    # HISTORY
    # ===================================================

    def _save_history(self):
        try:
            data = {
                "message_history": self.message_history,
                "workspace": str(self.workspace_dir),
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving history: {e}")

    def _load_history(self):
        if not os.path.exists(self.history_file):
            self.append_colored_message("system", 
                "Welcome to Lea Agentic Coding!\n\n"
                "I can autonomously:\n"
                "• Create new files\n"
                "• Edit existing code\n"
                "• Fix bugs\n"
                "• Run and test code\n\n"
                "Just tell me what you need!"
            )
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.message_history = data.get("message_history", [])
            
            # Display last few messages
            for msg in self.message_history[-5:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    self.append_colored_message("user", content)
                elif role == "assistant":
                    display_text = self._strip_xml_tags(content)
                    self.append_colored_message("assistant", display_text)

        except Exception as e:
            print(f"Error loading history: {e}")


# =====================================================
# MAIN
# =====================================================

def main():
    app = QApplication(sys.argv)

    splash = None
    if SPLASH_FILE.exists():
        pixmap = QPixmap(str(SPLASH_FILE))
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()

    window = LeaAgenticWindow()
    window.show()

    if splash is not None:
        splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
