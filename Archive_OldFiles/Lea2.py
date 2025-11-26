### Lea - Complete Multi-Agent System Missing Serpapi but working###

"""
Hummingbird – Lea
Multi-agent assistant with:
- All 7 specialized modes
- Knowledge base integration  
- Universal file reading
- Automatic backups with timestamps
- Download capability
"""

from dotenv import load_dotenv
load_dotenv()

import os
import html
import json
import shutil
from pathlib import Path
from datetime import datetime

import requests
from openai import OpenAI

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QSizePolicy, QFrame, QSplashScreen, QFileDialog,
    QMessageBox, QCheckBox,
)

# Import universal file reader
try:
    from universal_file_reader import read_file
    FILE_READER_AVAILABLE = True
except ImportError:
    print("WARNING: universal_file_reader.py not found.")
    FILE_READER_AVAILABLE = False
    def read_file(path):
        return {'success': False, 'error': 'File reader not available'}

# =====================================================
# OPENAI SETUP
# =====================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not OPENAI_API_KEY:
    print("=" * 60)
    print("WARNING: 'OPENAI_API_KEY' not found in .env")
    print("=" * 60)

# =====================================================
# DIRECTORIES
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
BACKUPS_DIR = PROJECT_DIR / "backups"
DOWNLOADS_DIR = PROJECT_DIR / "downloads"

# Create all directories
for dir_path in [KNOWLEDGE_DIR, BACKUPS_DIR, DOWNLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

SPLASH_FILE = ASSETS_DIR / "Hummingbird_LEA_v1_Splash_Logo_Lime_Green.png"
ICON_FILE = ASSETS_DIR / "Hummingbird_LEA_Logo_White_No BKGND.png"

print(f"\nDirectories created:")
print(f"  📚 Knowledge: {KNOWLEDGE_DIR}")
print(f"  💾 Backups: {BACKUPS_DIR}")
print(f"  📥 Downloads: {DOWNLOADS_DIR}\n")

# =====================================================
# KNOWLEDGE LOADER
# =====================================================

def load_knowledge():
    knowledge_content = {}
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        return knowledge_content
    
    file_count = 0
    for file in KNOWLEDGE_DIR.iterdir():
        if file.suffix.lower() == ".md":
            try:
                knowledge_content[file.stem] = file.read_text(encoding="utf-8")
                file_count += 1
                print(f"  ✓ Loaded: {file.name}")
            except Exception as e:
                print(f"  ✗ Error: {file.name} - {e}")
        
        elif file.suffix.lower() == ".pdf":
            try:
                import PyPDF2
                with open(file, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = "".join([page.extract_text() or "" for page in reader.pages])
                    knowledge_content[file.stem] = text
                    file_count += 1
                    print(f"  ✓ Loaded: {file.name} ({len(reader.pages)} pages)")
            except Exception as e:
                print(f"  ✗ Error: {file.name} - {e}")
    
    if file_count > 0:
        print(f"\n  📚 Total: {file_count} knowledge files loaded")
    else:
        print("  ℹ No knowledge files found (add .md or .pdf to knowledge/)")
    
    return knowledge_content

print("=" * 60)
print("LOADING KNOWLEDGE BASE")
print("=" * 60)
KNOWLEDGE_BASE = load_knowledge()
print("=" * 60 + "\n")

# =====================================================
# BACKUP SYSTEM
# =====================================================

def create_backup(file_path: Path) -> str:
    """Create timestamped backup in backups/ folder"""
    if not file_path.exists():
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = BACKUPS_DIR / backup_name
    
    shutil.copy2(file_path, backup_path)
    return str(backup_path)

def save_to_downloads(content: str, filename: str) -> str:
    """Save content to downloads/ folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = filename.rsplit('.', 1)
    
    if len(name_parts) == 2:
        filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
    else:
        filename = f"{filename}_{timestamp}.txt"
    
    download_path = DOWNLOADS_DIR / filename
    
    with open(download_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(download_path)

# =====================================================
# LEGAL & INCENTIVES RESOURCES
# =====================================================

LEGAL_RESOURCES_TEXT = """
### Arizona Legal Resources
- Maricopa County Law Library: https://superiorcourt.maricopa.gov/llrc/
- Arizona Court Rules: https://govt.westlaw.com/azrules/
- Arizona Statutes: https://law.justia.com/arizona/
- Case Law: https://law.justia.com/cases/arizona/
- Clerk Self-Help: https://www.clerkofcourt.maricopa.gov/

**Quick Reference:**
- Service: Ariz. R. Civ. P. 4
- Default: Ariz. R. Civ. P. 55
- Relief from Judgment: Ariz. R. Civ. P. 60(b)(4)
"""

INCENTIVES_POLICY = """
### Incentives Research Framework
Research grants, tax credits, rebates, training funds for businesses.
Focus on: Federal (IRA, R&D credits), State (enterprise zones), Utility programs.
"""

# =====================================================
# CORE RULES & KNOWLEDGE CONTEXT
# =====================================================

def build_knowledge_context() -> str:
    if not KNOWLEDGE_BASE:
        return ""
    
    parts = ["\n### Knowledge Base Documents Available"]
    for name, content in KNOWLEDGE_BASE.items():
        words = len(content.split())
        preview = content[:200].strip() + "..."
        parts.append(f"\n**{name}** (~{words:,} words): {preview}")
    
    parts.append("\nReference these documents when relevant. Cite by name.")
    return "\n".join(parts)

CORE_RULES = """
### Core Principles
- Be honest about knowledge vs. inference
- Never fabricate sources or details
- Ask clarifying questions when needed
- Show your work on calculations
- Support Dre's decisions
""" + build_knowledge_context()

# =====================================================
# AGENT CONFIGURATIONS
# =====================================================

AGENTS = {
    "General Assistant & Triage": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's primary assistant and triage system.
Friendly, helpful chief of staff. Route specialized requests to other modes:
IT Support, Executive Assistant & Operations, Incentives & Client Forms,
Research & Learning, Legal Research & Drafting, Finance & Tax.
"""
    },
    "IT Support": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's IT & technical support assistant.
Expert in: Python, PowerShell, APIs, debugging, databases, automation.
Provide complete runnable code with error handling and explanations.
"""
    },
    "Executive Assistant & Operations": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Executive Assistant.
Help with: professional emails, presentations, task organization,
scheduling, workplace communication, professional development.
"""
    },
    "Incentives & Client Forms": {
        "system_prompt": CORE_RULES + INCENTIVES_POLICY + """
You are Lea, Dre's Incentives research assistant for EIAG.
Research grants, credits, rebates. Connect to client forms and tools.
"""
    },
    "Research & Learning": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Research & Learning assistant.
Break down complex topics step-by-step in plain language.
Summarize materials and explain concepts clearly.
"""
    },
    "Legal Research & Drafting": {
        "system_prompt": CORE_RULES + LEGAL_RESOURCES_TEXT + """
You are Lea, Dre's Legal Research assistant for Arizona civil matters.
Locate rules, cases, resources. Draft motions and organize facts.
ALWAYS REMIND: "I am not a lawyer, this is not legal advice."
"""
    },
    "Finance & Tax": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Finance & Tax assistant.
Help organize tax docs and explain IRS/state guidance in plain English.
Use official sources. NOT a CPA - cannot give tax advice.
"""
    },
}

MODEL_OPTIONS = {
    "GPT-4o (default)": "gpt-4o",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4o mini": "gpt-4o-mini",
    "GPT-5 (max)": "gpt-5",
    "GPT-5 mini": "gpt-5-mini",
    "o3-pro": "o3-pro",
}

# =====================================================
# CHAT INPUT
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

class LeaWindow(QWidget):
    USER_COLOR = "#68BD47"
    ASSIST_COLOR = "#FFFFFF"
    SYSTEM_COLOR = "#2DBCEE"

    def __init__(self):
        super().__init__()
        
        self.current_mode = "General Assistant & Triage"
        self.current_model = "GPT-4o (default)"
        self.message_history = []
        self.history_file = "lea_history.json"
        self.current_file_content = None
        self.current_file_path = None
        
        self._init_window()
        self._build_ui()
        self._load_history()
    
    def _init_window(self):
        self.setWindowTitle("Hummingbird – Lea Multi-Agent")
        if ICON_FILE.exists():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(1200, 800)
        
        self.setStyleSheet("""
            QWidget { background-color: #333; color: #FFF; }
            QLabel { color: #FFF; }
            QComboBox { background-color: #222; color: #FFF; padding: 4px; }
            QFrame#InnerFrame { background-color: #333; border-radius: 8px; }
        """)
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        frame = QFrame()
        frame.setObjectName("InnerFrame")
        frame_layout = QVBoxLayout(frame)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🐦 Lea Multi-Agent System")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        
        if KNOWLEDGE_BASE:
            kb_label = QLabel(f"📚 {len(KNOWLEDGE_BASE)} docs")
            kb_label.setStyleSheet("color: #68BD47;")
            header.addWidget(kb_label)
        
        header.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(AGENTS.keys()))
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        header.addWidget(self.mode_combo)
        
        header.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_OPTIONS.keys()))
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        header.addWidget(self.model_combo)
        
        if ICON_FILE.exists():
            icon = QLabel()
            icon.setPixmap(QPixmap(str(ICON_FILE)).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio))
            header.addWidget(icon)
        
        frame_layout.addLayout(header)
        
        # Buttons
        buttons = QHBoxLayout()
        
        upload_btn = QPushButton("📎 Upload")
        upload_btn.clicked.connect(self.upload_file)
        upload_btn.setStyleSheet("background-color: #0078D7; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(upload_btn)
        
        download_btn = QPushButton("📥 Download")
        download_btn.clicked.connect(self.download_response)
        download_btn.setStyleSheet("background-color: #107C10; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(download_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_conversation)
        export_btn.setStyleSheet("background-color: #0078D7; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(export_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_conversation)
        clear_btn.setStyleSheet("background-color: #D13438; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(clear_btn)
        
        buttons.addStretch()
        frame_layout.addLayout(buttons)
        
        # File status
        self.file_label = QLabel("")
        self.file_label.setStyleSheet("color: #68BD47; font-size: 11px; font-style: italic;")
        frame_layout.addWidget(self.file_label)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background-color: #222; color: #FFF; font-size: 16px; "
            "font-family: Consolas, monospace;"
        )
        frame_layout.addWidget(self.chat_display, stretch=1)
        
        # Input
        input_layout = QHBoxLayout()
        
        self.input_box = ChatInputBox()
        self.input_box.setPlaceholderText("Ask Lea anything... (Enter to send, Shift+Enter for new line)")
        self.input_box.returnPressed.connect(self.on_send)
        self.input_box.setMinimumHeight(80)
        self.input_box.setStyleSheet("background-color: #222; color: #FFF; font-size: 16px;")
        input_layout.addWidget(self.input_box, stretch=1)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.on_send)
        send_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        send_btn.setMinimumWidth(90)
        send_btn.setStyleSheet("background-color: #0078D7; font-size: 16px; font-weight: 600; border-radius: 4px;")
        input_layout.addWidget(send_btn)
        
        frame_layout.addLayout(input_layout)
        
        # Options
        options = QHBoxLayout()
        self.include_file_cb = QCheckBox("Include uploaded file")
        self.include_file_cb.setChecked(True)
        options.addWidget(self.include_file_cb)
        
        self.include_kb_cb = QCheckBox("Include knowledge base")
        self.include_kb_cb.setChecked(True)
        options.addWidget(self.include_kb_cb)
        options.addStretch()
        frame_layout.addLayout(options)
        
        # Status
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #DDD; font-size: 12px;")
        frame_layout.addWidget(self.status_label)
        
        layout.addWidget(frame)
    
    # File operations
    def upload_file(self):
        if not FILE_READER_AVAILABLE:
            QMessageBox.warning(self, "Error", "universal_file_reader.py not found")
            return
        
        path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "All Files (*)")
        if not path:
            return
        
        self.status_label.setText("Reading file...")
        QApplication.processEvents()
        
        result = read_file(path)
        if result['success']:
            self.current_file_path = path
            self.current_file_content = result['content']
            
            # Create backup
            backup_path = create_backup(Path(path))
            
            name = os.path.basename(path)
            self.file_label.setText(f"📎 {name} ({result.get('file_type', 'unknown')})")
            self.append_message("system", f"Uploaded: {name}\nBackup: {os.path.basename(backup_path)}")
            self.status_label.setText("File loaded")
        else:
            QMessageBox.warning(self, "Error", result.get('error', 'Unknown error'))
    
    def download_response(self):
        if not self.message_history:
            QMessageBox.information(self, "Nothing to Download", "No conversation to save")
            return
        
        # Get last assistant response
        last_response = None
        for msg in reversed(self.message_history):
            if msg['role'] == 'assistant':
                last_response = msg['content']
                break
        
        if not last_response:
            QMessageBox.information(self, "Nothing to Download", "No response to save")
            return
        
        download_path = save_to_downloads(last_response, "lea_response.txt")
        
        self.append_message("system", f"Downloaded to: {os.path.basename(download_path)}")
        QMessageBox.information(self, "Downloaded", f"Saved to:\n{download_path}")
    
    def export_conversation(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "", "JSON (*.json);;Text (*.txt)")
        if not path:
            return
        
        try:
            if path.endswith('.json'):
                data = {'mode': self.current_mode, 'model': self.current_model, 'history': self.message_history}
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.chat_display.toPlainText())
            
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def clear_conversation(self):
        reply = QMessageBox.question(self, "Clear", "Clear conversation?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.message_history = []
            self.chat_display.clear()
            self.current_file_content = None
            self.current_file_path = None
            self.file_label.setText("")
            self._save_history()
            self.append_message("system", "Conversation cleared")
    
    def on_mode_changed(self, mode):
        self.current_mode = mode
        self.append_message("system", f"Switched to: {mode}")
        self._save_history()
    
    def on_model_changed(self, model):
        self.current_model = model
        self._save_history()
    
    # Messaging
    def append_message(self, kind: str, text: str):
        if kind == "user":
            label, color = "Dre", self.USER_COLOR
        elif kind == "assistant":
            label, color = "Lea", self.ASSIST_COLOR
        else:
            label, color = "System", self.SYSTEM_COLOR
        
        safe = html.escape(text).replace("\n", "<br>")
        html_block = f'<div style="margin: 6px 0;"><span style="color:{color}; font-weight:600;">{label}:</span> <span style="color:{color};">{safe}</span></div>'
        self.chat_display.append(html_block)
    
    def on_send(self):
        text = self.input_box.toPlainText().strip()
        if not text or not openai_client:
            return
        
        self.append_message("user", text)
        self.input_box.clear()
        self.status_label.setText("Thinking...")
        
        # Build prompt
        parts = []
        
        if self.include_kb_cb.isChecked() and KNOWLEDGE_BASE:
            kb = "=== KNOWLEDGE BASE ===\n"
            for name, content in KNOWLEDGE_BASE.items():
                kb += f"\n--- {name} ---\n{content}\n"
            parts.append(kb + "=== END KNOWLEDGE BASE ===\n")
        
        if self.include_file_cb.isChecked() and self.current_file_content:
            parts.append(f"=== UPLOADED FILE ===\n{self.current_file_content}\n=== END FILE ===\n")
        
        parts.append(f"Dre's question:\n{text}")
        
        full_prompt = "\n".join(parts)
        self.message_history.append({"role": "user", "content": full_prompt})
        
        system_prompt = AGENTS[self.current_mode]["system_prompt"]
        messages = [{"role": "system", "content": system_prompt}] + self.message_history
        
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_OPTIONS[self.current_model],
                messages=messages
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"[Error: {e}]"
        
        self.message_history.append({"role": "assistant", "content": answer})
        self.append_message("assistant", answer)
        self.status_label.setText("Ready")
        self._save_history()
    
    def _save_history(self):
        try:
            data = {'mode': self.current_mode, 'model': self.current_model, 'history': self.message_history}
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Save error: {e}")
    
    def _load_history(self):
        if not os.path.exists(self.history_file):
            msg = "Welcome to Lea Multi-Agent System!\n\n"
            if KNOWLEDGE_BASE:
                msg += f"📚 {len(KNOWLEDGE_BASE)} knowledge documents loaded\n"
            msg += f"💾 Backups: {BACKUPS_DIR}\n📥 Downloads: {DOWNLOADS_DIR}"
            self.append_message("system", msg)
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_mode = data.get('mode', "General Assistant & Triage")
            self.current_model = data.get('model', "GPT-4o (default)")
            self.message_history = data.get('history', [])
            
            self.mode_combo.setCurrentText(self.current_mode)
            self.model_combo.setCurrentText(self.current_model)
            
            self.append_message("system", "Loaded previous conversation")
            for msg in self.message_history[-5:]:
                role = msg.get('role')
                content = msg.get('content', '')
                if 'Dre\'s question:' in content:
                    content = content.split('Dre\'s question:')[-1].strip()
                
                if role == 'user':
                    self.append_message('user', content)
                elif role == 'assistant':
                    self.append_message('assistant', content)
        except Exception as e:
            print(f"Load error: {e}")

# =====================================================
# MAIN
# =====================================================

def main():
    import sys
    app = QApplication(sys.argv)
    
    splash = None
    if SPLASH_FILE.exists():
        splash = QSplashScreen(QPixmap(str(SPLASH_FILE)))
        splash.show()
        app.processEvents()
    
    window = LeaWindow()
    window.show()
    
    if splash:
        splash.finish(window)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
