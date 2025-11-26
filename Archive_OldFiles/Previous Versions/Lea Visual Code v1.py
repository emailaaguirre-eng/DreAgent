
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import sys

class ExportWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    def __init__(self, path, mode, model, message_history, chat_text):
        super().__init__()
        self.path = path
        self.mode = mode
        self.model = model
        self.message_history = message_history
        self.chat_text = chat_text
    @pyqtSlot()
    def run(self):
        try:
            if self.path.endswith('.json'):
                import json
                data = {'mode': self.mode, 'model': self.model, 'history': self.message_history}
                with open(self.path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(self.path, 'w', encoding='utf-8') as f:
                    f.write(self.chat_text)
            self.finished.emit(self.path)
        except Exception as e:
            self.error.emit(str(e))

class DownloadWorker(QObject):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    def __init__(self, last_response):
        super().__init__()
        self.last_response = last_response
    @pyqtSlot()
    def run(self):
        try:
            download_path = save_to_downloads(self.last_response, "lea_response.txt")
            self.finished.emit(download_path, os.path.basename(download_path))
        except Exception as e:
            self.error.emit(str(e))
### Lea - Complete Multi-Agent System ###

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

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QSizePolicy, QFrame, QSplashScreen, QFileDialog,
    QMessageBox, QCheckBox,
)

# --- CRASH HANDLER (Global Exception Logger) ---
import logging
import traceback

CRASH_LOG = "lea_crash.log"

logging.basicConfig(
    filename=CRASH_LOG,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

def handle_exception(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logging.error("Uncaught exception:\n%s", tb)

    try:
        msg = QMessageBox()
        msg.setWindowTitle("Lea Error")
        msg.setText("An unexpected error occurred.\nDetails were saved to lea_crash.log.")
        msg.setDetailedText(tb)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.exec()
    except Exception:
        print("Error while showing message box:", traceback.format_exc())
        print(tb)

sys.excepthook = handle_exception
# --- END CRASH HANDLER ---


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
BACKUPS_DIR = PROJECT_DIR / "backups"
DOWNLOADS_DIR = PROJECT_DIR / "downloads"

# Create directories
for dir_path in [BACKUPS_DIR, DOWNLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

SPLASH_FILE = ASSETS_DIR / "Hummingbird_LEA_v1_Splash_Logo_Lime_Green.png"
ICON_FILE = ASSETS_DIR / "Hummingbird_LEA_Logo_White_No BKGND.png"

print(f"\nDirectories created:")
print(f"  💾 Backups: {BACKUPS_DIR}")
print(f"  📥 Downloads: {DOWNLOADS_DIR}\n")

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

CORE_RULES = """
### Core Principles
- Be honest about knowledge vs. inference
- Never fabricate sources or details
- Ask clarifying questions when needed
- Show your work on calculations
- Support Dre's decisions

### Web Search Capability
You have access to web search when you need current information.

**When to search the web:**
- Current events, news, or recent developments
- Information that changes frequently (prices, rates, statistics)
- Technical documentation or API updates
- Recent product releases or announcements
- Anything after your knowledge cutoff (April 2024)
- When explicitly asked to "search" or "look up"

**How to search:**
Use the format: [SEARCH: your search query here]

**Example:**
User: "What's the current price of Tesla stock?"
You: [SEARCH: Tesla stock price today]
Then use the results to answer.

**Don't search for:**
- General knowledge from before April 2024
- Programming concepts that haven't changed
- Historical facts
- Math or logic problems
- Information already provided in the conversation
"""

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

# Default model per mode
DEFAULT_MODEL_PER_MODE = {
    "General Assistant & Triage": "GPT-4o (default)",
    "IT Support": "GPT-4o (default)",
    "Executive Assistant & Operations": "GPT-4o (default)",
    "Incentives & Client Forms": "GPT-4o (default)",
    "Research & Learning": "GPT-4o (default)",
    "Legal Research & Drafting": "GPT-4o (default)",
    "Finance & Tax": "GPT-4o (default)",
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

from typing import Optional

# =====================================================
# WORKER THREADS
# =====================================================
class FileUploadWorker(QObject):
    finished = pyqtSignal(dict, str, str)  # result, backup_path, file_name
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    @pyqtSlot()
    def run(self):
        try:
            # Use universal_file_reader if available
            from universal_file_reader import read_file
            result = read_file(self.path)
            backup_path = None
            if result['success']:
                backup_path = create_backup(Path(self.path))
            file_name = os.path.basename(self.path)
            self.finished.emit(result, backup_path, file_name)
        except Exception as e:
            self.error.emit(str(e))

class LeaWorker(QObject):
    finished = pyqtSignal(str, str)  # answer, status
    error = pyqtSignal(str)

    def __init__(self, openai_client, model_options, agents, mode, model, message_history, file_content, user_text):
        super().__init__()
        self.openai_client = openai_client
        self.model_options = model_options
        self.agents = agents
        self.mode = mode
        self.model = model
        self.message_history = message_history
        self.file_content = file_content
        self.user_text = user_text

    @pyqtSlot()
    def run(self):
        try:
            parts = []
            if self.file_content:
                file_content = self.file_content[:100000]
                if len(self.file_content) > 100000:
                    parts.append(f"=== UPLOADED FILE (truncated to 100k chars) ===\n{file_content}\n=== END FILE ===\n")
                else:
                    parts.append(f"=== UPLOADED FILE ===\n{file_content}\n=== END FILE ===\n")
            parts.append(f"Dre's question:\n{self.user_text}")
            full_prompt = "\n".join(parts)
            estimated_tokens = len(full_prompt) // 4
            if estimated_tokens > 25000:
                self.finished.emit("⚠️ ERROR: Message too large (~{:,} tokens). Please use a smaller file or uncheck 'Include uploaded file'".format(estimated_tokens), "Message too large")
                return
            self.message_history.append({"role": "user", "content": full_prompt})
            # Limit history to last 20 messages
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            system_prompt = self.agents[self.mode]["system_prompt"]
            messages = [{"role": "system", "content": system_prompt}] + self.message_history
            response = self.openai_client.chat.completions.create(
                model=self.model_options[self.model],
                messages=messages
            )
            answer = response.choices[0].message.content
            # Web search handling (optional, can be improved)
            if "[SEARCH:" in answer and "]" in answer:
                import re
                search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
                searches = re.findall(search_pattern, answer)
                all_results = []
                for query in searches:
                    # Dummy search result (replace with real web_search if available)
                    all_results.append(f"=== Search Results for '{query}' ===\n(Dummy search results)\n")
                search_context = "\n".join(all_results)
                followup_prompt = f"{search_context}\n\nNow answer Dre's original question using these search results."
                self.message_history.append({"role": "user", "content": followup_prompt})
                messages = [{"role": "system", "content": system_prompt}] + self.message_history
                response = self.openai_client.chat.completions.create(
                    model=self.model_options[self.model],
                    messages=messages
                )
                answer = response.choices[0].message.content
            self.message_history.append({"role": "assistant", "content": answer})
            # Limit history to last 20 messages
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            self.finished.emit(answer, "Ready")
        except Exception as e:
            self.error.emit(str(e))

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
        self.include_file_cb = QCheckBox("Include uploaded file in context")
        self.include_file_cb.setChecked(True)
        options.addWidget(self.include_file_cb)
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
        # Start file upload worker thread
        self.file_worker_thread = QThread()
        self.file_worker = FileUploadWorker(path)
        self.file_worker.moveToThread(self.file_worker_thread)
        self.file_worker_thread.started.connect(self.file_worker.run)
        self.file_worker.finished.connect(self.on_file_upload_finished)
        self.file_worker.error.connect(self.on_file_upload_error)
        self.file_worker.finished.connect(self.file_worker_thread.quit)
        self.file_worker.finished.connect(self.file_worker.deleteLater)
        self.file_worker_thread.finished.connect(self.file_worker_thread.deleteLater)
        self.file_worker_thread.start()

    def on_file_upload_finished(self, result, backup_path, file_name):
        if result['success']:
            self.current_file_path = self.file_worker.path
            self.current_file_content = result['content']
            self.file_label.setText(f"📎 {file_name} ({result.get('file_type', 'unknown')})")
            self.append_message("system", f"Uploaded: {file_name}\nBackup: {os.path.basename(backup_path) if backup_path else 'None'}")
            self.status_label.setText("File loaded")
        else:
            QMessageBox.warning(self, "Error", result.get('error', 'Unknown error'))
            self.status_label.setText("Error loading file")

    def on_file_upload_error(self, error_msg):
        QMessageBox.warning(self, "Error", error_msg)
        self.status_label.setText("Error loading file")
    
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
        # Start download worker thread
        self.download_worker_thread = QThread()
        self.download_worker = DownloadWorker(last_response)
        self.download_worker.moveToThread(self.download_worker_thread)
        self.download_worker_thread.started.connect(self.download_worker.run)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.error.connect(self.on_download_error)
        self.download_worker.finished.connect(self.download_worker_thread.quit)
        self.download_worker.finished.connect(self.download_worker.deleteLater)
        self.download_worker_thread.finished.connect(self.download_worker_thread.deleteLater)
        self.download_worker_thread.start()

    def on_download_finished(self, download_path, basename):
        self.append_message("system", f"Downloaded to: {basename}")
        QMessageBox.information(self, "Downloaded", f"Saved to:\n{download_path}")

    def on_download_error(self, error_msg):
        QMessageBox.warning(self, "Error", error_msg)
    
    def export_conversation(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "", "JSON (*.json);;Text (*.txt)")
        if not path:
            return
        # Start export worker thread
        self.export_worker_thread = QThread()
        self.export_worker = ExportWorker(
            path,
            self.current_mode,
            self.current_model,
            self.message_history,
            self.chat_display.toPlainText()
        )
        self.export_worker.moveToThread(self.export_worker_thread)
        self.export_worker_thread.started.connect(self.export_worker.run)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.error.connect(self.on_export_error)
        self.export_worker.finished.connect(self.export_worker_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_worker_thread.finished.connect(self.export_worker_thread.deleteLater)
        self.export_worker_thread.start()

    def on_export_finished(self, path):
        QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def on_export_error(self, error_msg):
        QMessageBox.warning(self, "Error", error_msg)
    
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
        # Set the best model for this mode
        best_model = DEFAULT_MODEL_PER_MODE.get(mode, "GPT-4o (default)")
        self.current_model = best_model
        self.model_combo.setCurrentText(best_model)
        self.append_message("system", f"Switched to: {mode} (Model: {best_model})")
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

        # Ensure text is always a string
        safe = html.escape(str(text)).replace("\n", "<br>")
        html_block = f'<div style="margin: 6px 0;"><span style="color:{color}; font-weight:600;">{label}:</span> <span style="color:{color};">{safe}</span></div>'
        self.chat_display.append(html_block)
    
    def on_send(self):
        text = self.input_box.toPlainText().strip()
        if not text or not openai_client:
            return
        self.append_message("user", text)
        self.input_box.clear()
        self.status_label.setText("Thinking...")
        # Use file content only if checkbox is checked
        file_content = self.current_file_content if self.include_file_cb.isChecked() else None
        # Start worker thread
        self.worker_thread = QThread()
        self.worker = LeaWorker(
            openai_client,
            MODEL_OPTIONS,
            AGENTS,
            self.current_mode,
            self.current_model,
            self.message_history,
            file_content,
            text
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def on_worker_finished(self, answer, status):
        self.append_message("assistant", answer)
        self.status_label.setText(status)
        self._save_history()

    def on_worker_error(self, error_msg):
        self.append_message("system", f"[Error: {error_msg}]")
        self.status_label.setText("Error")
    
    def _save_history(self):
        try:
            # Use absolute path in project directory
            history_path = PROJECT_DIR / self.history_file
            # Limit history to last 20 messages
            history = self.message_history[-20:] if len(self.message_history) > 20 else self.message_history
            data = {
                'mode': self.current_mode,
                'model': self.current_model,
                'history': history
            }
            # Try to save, with better error handling
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except PermissionError:
            # If permission denied, try alternative location
            try:
                alt_path = Path.home() / "lea_history.json"
                with open(alt_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"Note: Saved history to {alt_path} due to permission issue")
            except Exception as e2:
                print(f"Could not save history: {e2}")
        except Exception as e:
            print(f"Save error: {e}")
    
    def _load_history(self):
        # Try project directory first
        history_path = PROJECT_DIR / self.history_file
        # Fallback to home directory if needed
        if not history_path.exists():
            alt_path = Path.home() / "lea_history.json"
            if alt_path.exists():
                history_path = alt_path
        if not history_path.exists():
            msg = "Welcome to Lea Multi-Agent System!\n\n"
            msg += f"💾 Backups: {BACKUPS_DIR}\n"
            msg += f"📥 Downloads: {DOWNLOADS_DIR}\n\n"
            msg += "📎 Upload files when you need to reference them\n"
            msg += "📥 Download Lea's responses to save them\n"
            msg += "🔍 Lea can search the web for current information\n\n"
            # Check if web search is configured
            if os.getenv("SERPAPI_API_KEY"):
                msg += "✅ Web search enabled"
            else:
                msg += "⚠️ Web search not configured (add SERPAPI_API_KEY to .env)"
            self.append_message("system", msg)
            return
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.current_mode = data.get('mode', "General Assistant & Triage")
            self.current_model = data.get('model', "GPT-4o (default)")
            self.message_history = data.get('history', [])
            # Limit history to last 20 messages
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
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
