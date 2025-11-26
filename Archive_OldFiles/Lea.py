### Lea - Main Program (Updated) ###

"""
Hummingbird – Lea
IT & Coding Assistant with Knowledge Base
- Universal file reading
- Knowledge base from local files
- Conversation saving
- Web search integration
"""

from dotenv import load_dotenv
load_dotenv()

import os
import html
import json
from pathlib import Path

import requests
from openai import OpenAI

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QSizePolicy,
    QFrame,
    QSplashScreen,
    QFileDialog,
    QMessageBox,
    QCheckBox,
)
from PyQt6.QtPrintSupport import QPrinter, QPrintDialog

# Import universal file reader
try:
    from universal_file_reader import read_file
    FILE_READER_AVAILABLE = True
except ImportError:
    print("WARNING: universal_file_reader.py not found. File upload will not work.")
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
    print("CRITICAL WARNING: 'OPENAI_API_KEY' not found in environment/.env")
    print("Lea will start, but cannot talk to OpenAI until this is set.")
    print("=" * 60)

# =====================================================
# PROJECT PATHS & ASSETS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"

# Create directories if they don't exist
KNOWLEDGE_DIR.mkdir(exist_ok=True)

SPLASH_FILE = ASSETS_DIR / "Hummingbird_LEA_v1_Splash_Logo_Lime_Green.png"
ICON_FILE = ASSETS_DIR / "Hummingbird_LEA_Logo_White_No BKGND.png"

# =====================================================
# KNOWLEDGE LOADER
# =====================================================

def load_knowledge():
    """
    Load knowledge base from knowledge/ directory.
    Supports: .md (Markdown) and .pdf files
    """
    knowledge_dir = Path(__file__).resolve().parent / "knowledge"
    knowledge_content = {}
    
    if not knowledge_dir.exists():
        print(f"Knowledge directory not found: {knowledge_dir}")
        print("Creating knowledge/ directory for reference documents...")
        knowledge_dir.mkdir(exist_ok=True)
        return knowledge_content
    
    file_count = 0
    for file in knowledge_dir.iterdir():
        # Markdown files
        if file.suffix.lower() == ".md":
            try:
                knowledge_content[file.stem] = file.read_text(encoding="utf-8")
                file_count += 1
                print(f"  Loaded: {file.name}")
            except Exception as e:
                knowledge_content[file.stem] = f"ERROR reading Markdown: {e}"
                print(f"  Error loading {file.name}: {e}")
        
        # PDF files
        elif file.suffix.lower() == ".pdf":
            try:
                import PyPDF2
                with open(file, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                    knowledge_content[file.stem] = text
                    file_count += 1
                    print(f"  Loaded: {file.name} ({len(reader.pages)} pages)")
            except ImportError:
                knowledge_content[file.stem] = "ERROR: PyPDF2 not installed (pip install PyPDF2)"
                print(f"  Error: PyPDF2 not installed for {file.name}")
            except Exception as e:
                knowledge_content[file.stem] = f"ERROR reading PDF: {e}"
                print(f"  Error loading {file.name}: {e}")
    
    if file_count > 0:
        print(f"\n✓ Loaded {file_count} knowledge base files")
    else:
        print("\n⚠ No knowledge files found in knowledge/ directory")
        print("  Add .md or .pdf files to knowledge/ for Lea to reference")
    
    return knowledge_content

# Load knowledge base at startup
print("\n" + "=" * 60)
print("LOADING KNOWLEDGE BASE")
print("=" * 60)
KNOWLEDGE_BASE = load_knowledge()
print("=" * 60 + "\n")

# =====================================================
# WEB SEARCH HELPER
# =====================================================

def web_search(query: str, engines=None, per_engine: int = 3) -> str:
    """Perform web searches using SerpAPI"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Web search is not configured (missing SERPAPI_API_KEY)."

    if engines is None:
        engines = ["duckduckgo"]

    endpoint = "https://serpapi.com/search"
    all_lines = []

    for engine in engines:
        params = {
            "engine": engine,
            "q": query,
            "api_key": api_key,
            "no_html": "true",
        }

        try:
            resp = requests.get(endpoint, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            all_lines.append(f"[{engine}] Web search error: {e}")
            continue

        results = data.get("organic_results", []) or data.get("results", [])
        if not results:
            all_lines.append(f"[{engine}] No web results found for this query.")
            continue

        all_lines.append(f"=== Results from {engine} ===")
        for i, item in enumerate(results[:per_engine], start=1):
            title = item.get("title", "")
            url = item.get("link") or item.get("url", "")
            snippet = item.get("snippet") or item.get("description", "")
            all_lines.append(f"{i}) Title: {title}\nURL: {url}\nSnippet: {snippet}")
        all_lines.append("")

    if not all_lines:
        return "No web results found for this query."

    return "\n".join(all_lines)

# =====================================================
# CORE SYSTEM PROMPT
# =====================================================

def build_knowledge_context() -> str:
    """Build knowledge base context for system prompt"""
    if not KNOWLEDGE_BASE:
        return ""
    
    context_parts = ["\n### Available Knowledge Base"]
    context_parts.append("\nYou have access to the following reference documents in your knowledge base:")
    
    for doc_name, content in KNOWLEDGE_BASE.items():
        # Show first 500 chars as preview
        preview = content[:500] + "..." if len(content) > 500 else content
        context_parts.append(f"\n**Document: {doc_name}**")
        context_parts.append(f"Preview: {preview}")
        context_parts.append(f"(Full content available - {len(content)} characters)\n")
    
    context_parts.append("\nWhen Dre asks about topics covered in these documents, reference them directly.")
    context_parts.append("You can quote from them and cite which document you're referencing.")
    
    return "\n".join(context_parts)

CORE_SYSTEM_PROMPT = """
You are Lea, Dre's expert IT & Coding Assistant.

### Core Principles
- Always be honest about what you know vs. what you're inferring
- Never fabricate code examples, library functions, or API methods that don't exist
- If unsure about syntax or a feature, say so and suggest documentation lookup
- Ask 1-3 specific clarifying questions when requirements are unclear

### Code Quality Standards
- Write complete, runnable code with all necessary imports
- Include error handling (specific exceptions, not bare except)
- Use clear, descriptive variable names
- Add comments for complex logic
- Never hardcode sensitive credentials (use environment variables)
- Follow PEP 8 style guidelines for Python

### Debugging Methodology
1. Reproduce the error reliably
2. Read the complete error message (type, line number, traceback)
3. Identify what changed since it last worked
4. Isolate the problem area (comment out sections)
5. Add logging/print statements to inspect values
6. Check official documentation
7. Search for similar issues (with healthy skepticism)

### Your Expertise
- Python development (including PyQt6, APIs, automation)
- Database connectivity and SQL
- API integrations (Gmail, OpenAI, SerpAPI, etc.)
- Debugging and troubleshooting
- PowerShell scripting
- Environment configuration
- Web scraping and automation
- Data processing and analysis

### Your Approach
- Provide working code examples, not just descriptions
- Debug systematically with clear diagnostic steps
- Explain *why* something works, not just *what* to do
- Include error handling and best practices
- Anticipate edge cases and potential issues

### Security Awareness
- Never expose API keys, passwords, or secrets in code
- Validate and sanitize user input
- Use parameterized queries for databases (prevent SQL injection)
- Be cautious with eval(), exec(), and similar functions
- Handle file paths safely (prevent directory traversal)

### Tone
Expert developer who explains clearly without being condescending. You're Dre's trusted technical advisor.
""" + build_knowledge_context()

# =====================================================
# MODEL OPTIONS
# =====================================================

MODEL_OPTIONS = {
    "GPT-4.1 mini (best value ⭐)": "gpt-4.1-mini",
    "GPT-4.1 (strongest coding)": "gpt-4.1",
    "GPT-4o (reliable default)": "gpt-4o",
    "GPT-4o mini (fast & cheap)": "gpt-4o-mini",
}

# =====================================================
# CHAT INPUT WIDGET
# =====================================================

class ChatInputBox(QTextEdit):
    """Multiline input where Enter sends and Shift+Enter inserts a newline."""
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

        self.current_model_key = "GPT-4.1 mini (best value ⭐)"
        self.message_history = []
        self.history_file = "lea_main_history.json"
        
        # File handling
        self.current_uploaded_file = None
        self.current_file_content = None

        self._init_window()
        self._build_ui()
        self._load_history()

    def _init_window(self):
        self.setWindowTitle("Hummingbird – Lea IT Assistant")
        if ICON_FILE.exists():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(1100, 750)

        self.setStyleSheet("""
            QWidget {
                background-color: #333333;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
            QComboBox {
                background-color: #222222;
                color: #FFFFFF;
                padding: 3px 6px;
            }
            QFrame#InnerFrame {
                background-color: #333333;
                border-radius: 8px;
            }
        """)

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(8)

        inner_frame = QFrame()
        inner_frame.setObjectName("InnerFrame")
        inner_layout = QVBoxLayout(inner_frame)
        inner_layout.setContentsMargins(12, 12, 12, 12)
        inner_layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(10)

        title_label = QLabel("🐦 Lea - IT & Coding Assistant")
        title_label.setStyleSheet("font-size: 20px; font-weight: 600;")
        header.addWidget(title_label)

        header.addStretch()

        # Knowledge base indicator
        if KNOWLEDGE_BASE:
            kb_label = QLabel(f"📚 Knowledge: {len(KNOWLEDGE_BASE)} docs")
            kb_label.setStyleSheet("color: #68BD47; font-size: 12px;")
            kb_label.setToolTip("\n".join(KNOWLEDGE_BASE.keys()))
            header.addWidget(kb_label)

        header.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_OPTIONS.keys()))
        self.model_combo.setCurrentText(self.current_model_key)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        header.addWidget(self.model_combo)

        if ICON_FILE.exists():
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(str(ICON_FILE)).scaled(
                32, 32,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            header.addWidget(icon_label)

        inner_layout.addLayout(header)

        # Buttons row
        buttons_layout = QHBoxLayout()

        self.upload_button = QPushButton("📎 Upload File")
        self.upload_button.clicked.connect(self.upload_file)
        self.upload_button.setStyleSheet(
            "font-size: 14px; font-weight: 600; padding: 6px 12px;"
            "background-color: #0078D7; color: #FFFFFF; border-radius: 4px;"
        )
        buttons_layout.addWidget(self.upload_button)

        self.knowledge_button = QPushButton("📚 View Knowledge")
        self.knowledge_button.clicked.connect(self.view_knowledge)
        self.knowledge_button.setStyleSheet(
            "font-size: 14px; font-weight: 600; padding: 6px 12px;"
            "background-color: #16825D; color: #FFFFFF; border-radius: 4px;"
        )
        buttons_layout.addWidget(self.knowledge_button)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.on_export_clicked)
        self.export_button.setStyleSheet(
            "font-size: 14px; font-weight: 600; padding: 6px 12px;"
            "background-color: #0078D7; color: #FFFFFF; border-radius: 4px;"
        )
        buttons_layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.on_clear_clicked)
        self.clear_button.setStyleSheet(
            "font-size: 14px; font-weight: 600; padding: 6px 12px;"
            "background-color: #d13438; color: #FFFFFF; border-radius: 4px;"
        )
        buttons_layout.addWidget(self.clear_button)

        buttons_layout.addStretch()
        inner_layout.addLayout(buttons_layout)

        # File status label
        self.file_status_label = QLabel("")
        self.file_status_label.setStyleSheet("color: #68BD47; font-size: 11px; font-style: italic;")
        inner_layout.addWidget(self.file_status_label)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background-color: #222222; color: #FFFFFF; font-size: 18px;"
            "font-family: Consolas, 'Courier New', monospace;"
        )
        inner_layout.addWidget(self.chat_display, stretch=1)

        # Input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        self.input_box = ChatInputBox()
        self.input_box.setPlaceholderText(
            "Ask Lea anything about coding, debugging, APIs, automation...\n"
            "(Enter to send, Shift+Enter for new line)"
        )
        self.input_box.returnPressed.connect(self.on_send_clicked)
        self.input_box.setMinimumHeight(80)
        self.input_box.setStyleSheet(
            "background-color: #222222; color: #FFFFFF; font-size: 18px;"
            "font-family: Consolas, 'Courier New', monospace;"
        )
        input_layout.addWidget(self.input_box, stretch=1)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        self.send_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.send_button.setMinimumWidth(90)
        self.send_button.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 8px 18px;"
            "background-color: #0078D7; color: #FFFFFF; border-radius: 4px;"
        )
        input_layout.addWidget(self.send_button)

        inner_layout.addLayout(input_layout)

        # Options row
        options_layout = QHBoxLayout()
        
        self.include_file_checkbox = QCheckBox("Include uploaded file in context")
        self.include_file_checkbox.setChecked(True)
        self.include_file_checkbox.setStyleSheet("color: #CCCCCC;")
        options_layout.addWidget(self.include_file_checkbox)
        
        self.include_knowledge_checkbox = QCheckBox("Include knowledge base")
        self.include_knowledge_checkbox.setChecked(True)
        self.include_knowledge_checkbox.setStyleSheet("color: #CCCCCC;")
        options_layout.addWidget(self.include_knowledge_checkbox)
        
        options_layout.addStretch()
        inner_layout.addLayout(options_layout)

        # Status label
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #DDDDDD; font-size: 12px;")
        inner_layout.addWidget(self.status_label)

        outer_layout.addWidget(inner_frame)

    # ===================================================
    # FILE HANDLING
    # ===================================================

    def upload_file(self):
        """Upload and process any file type"""
        if not FILE_READER_AVAILABLE:
            QMessageBox.warning(
                self,
                "File Reader Not Available",
                "universal_file_reader.py not found.\nPlace it in the same directory as this program."
            )
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload File",
            "",
            "All Files (*);;PDF (*.pdf);;Word (*.docx);;Excel (*.xlsx);;Text (*.txt *.md)"
        )
        
        if not file_path:
            return
        
        self.status_label.setText(f"Reading file: {os.path.basename(file_path)}...")
        QApplication.processEvents()
        
        result = read_file(file_path)
        
        if result['success']:
            self.current_uploaded_file = file_path
            self.current_file_content = result['content']
            
            file_name = os.path.basename(file_path)
            file_type = result.get('file_type', 'unknown')
            
            self.file_status_label.setText(
                f"📎 Uploaded: {file_name} ({file_type})"
            )
            
            self.append_colored_message(
                "system",
                f"File uploaded: {file_name} ({file_type})\n"
                f"Extracted: {len(self.current_file_content)} characters"
            )
            
            if 'metadata' in result and result['metadata']:
                meta_str = ", ".join([f"{k}: {v}" for k, v in result['metadata'].items()])
                self.append_colored_message("system", f"Metadata: {meta_str}")
            
            self.status_label.setText(f"File loaded: {file_name}")
            
        else:
            error_msg = result.get('error', 'Unknown error')
            self.append_colored_message("system", f"Error reading file: {error_msg}")
            QMessageBox.warning(self, "File Error", error_msg)
            self.status_label.setText("File upload failed.")

    def view_knowledge(self):
        """Display knowledge base contents"""
        if not KNOWLEDGE_BASE:
            msg = "No knowledge base loaded.\n\nAdd .md or .pdf files to the 'knowledge/' folder."
        else:
            msg = "Knowledge Base Documents:\n\n"
            for doc_name, content in KNOWLEDGE_BASE.items():
                char_count = len(content)
                line_count = content.count('\n') + 1
                msg += f"📄 {doc_name}\n   {char_count:,} characters, {line_count:,} lines\n\n"
            msg += f"Total: {len(KNOWLEDGE_BASE)} documents"
        
        QMessageBox.information(self, "Knowledge Base", msg)

    # ===================================================
    # MESSAGE HELPERS
    # ===================================================

    def append_colored_message(self, kind: str, text: str):
        text = text.strip()
        if not text:
            return

        if kind == "user":
            label = "Dre"
            label_color = self.USER_COLOR
            text_color = self.USER_COLOR
        elif kind == "assistant":
            label = "Lea"
            label_color = self.ASSIST_COLOR
            text_color = self.ASSIST_COLOR
        else:
            label = "System"
            label_color = self.SYSTEM_COLOR
            text_color = self.SYSTEM_COLOR

        safe_text = html.escape(text).replace("\n", "<br>")

        html_block = f"""
        <div style="margin-top: 6px; margin-bottom: 6px;">
            <span style="color:{label_color}; font-weight:600;">{label}:</span>
            <span style="color:{text_color};"> {safe_text}</span>
        </div>
        """

        self.chat_display.append(html_block)

    # ===================================================
    # HISTORY MANAGEMENT
    # ===================================================

    def _save_history(self):
        try:
            data = {
                "current_model": self.current_model_key,
                "message_history": self.message_history,
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving history: {e}")

    def _load_history(self):
        if not os.path.exists(self.history_file):
            welcome_msg = "Welcome to Lea - Your IT & Coding Assistant!\n\n"
            if KNOWLEDGE_BASE:
                welcome_msg += f"📚 Loaded {len(KNOWLEDGE_BASE)} knowledge base documents\n"
            welcome_msg += "\nI can help with:\n"
            welcome_msg += "• Python, JavaScript, and other languages\n"
            welcome_msg += "• Debugging and troubleshooting\n"
            welcome_msg += "• API integrations\n"
            welcome_msg += "• Database queries\n"
            welcome_msg += "• Automation scripts\n"
            welcome_msg += "• Code review and optimization"
            
            self.append_colored_message("system", welcome_msg)
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.current_model_key = data.get("current_model", "GPT-4.1 mini (best value ⭐)")
            self.message_history = data.get("message_history", [])

            self.model_combo.setCurrentText(self.current_model_key)

            self.append_colored_message("system", "Loaded previous conversation")
            for msg in self.message_history[-5:]:  # Last 5 messages
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    # Strip out knowledge context for display
                    if "[KNOWLEDGE BASE]" in content:
                        content = content.split("Dre's question:")[-1].strip()
                    self.append_colored_message("user", content)
                elif role == "assistant":
                    self.append_colored_message("assistant", content)

        except Exception as e:
            print(f"Error loading history: {e}")

    # ===================================================
    # BUTTON HANDLERS
    # ===================================================

    def on_export_clicked(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "", "JSON Files (*.json);;HTML Files (*.html);;Text Files (*.txt)"
        )
        if not path:
            return
        try:
            if path.endswith(".json"):
                data = {
                    "current_model": self.current_model_key,
                    "message_history": self.message_history,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            elif path.endswith(".html"):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.chat_display.toHtml())
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.chat_display.toPlainText())
                    
            QMessageBox.information(self, "Exported", "Conversation exported successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not export:\n{e}")

    def on_clear_clicked(self):
        reply = QMessageBox.question(
            self,
            "Clear Conversation",
            "Clear conversation history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.message_history = []
            self.chat_display.clear()
            self.current_uploaded_file = None
            self.current_file_content = None
            self.file_status_label.setText("")
            self._save_history()
            self.append_colored_message("system", "Conversation cleared.")

    def on_model_changed(self, text: str):
        self.current_model_key = text
        self.status_label.setText(f"Model: {text}")
        self._save_history()

    # ===================================================
    # SEND MESSAGE
    # ===================================================

    def on_send_clicked(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return

        if not OPENAI_API_KEY or not openai_client:
            self.append_colored_message(
                "system",
                "WARNING: OPENAI_API_KEY not set."
            )
            return

        self.append_colored_message("user", user_text)
        self.input_box.clear()
        self.status_label.setText("Thinking...")

        model_id = MODEL_OPTIONS[self.current_model_key]

        # Build system prompt with knowledge base
        system_prompt = CORE_SYSTEM_PROMPT

        # Build context
        context_parts = []
        
        # Add knowledge base if enabled
        if self.include_knowledge_checkbox.isChecked() and KNOWLEDGE_BASE:
            kb_context = "[KNOWLEDGE BASE]\n"
            for doc_name, content in KNOWLEDGE_BASE.items():
                kb_context += f"\n--- {doc_name} ---\n{content}\n"
            kb_context += "[END KNOWLEDGE BASE]\n\n"
            context_parts.append(kb_context)
        
        # Add uploaded file if enabled
        if self.include_file_checkbox.isChecked() and self.current_file_content:
            file_name = os.path.basename(self.current_uploaded_file)
            context_parts.append(
                f"[Uploaded File: {file_name}]\n\n{self.current_file_content}\n\n"
                f"[End of Uploaded File]\n\n"
            )
        
        # Add user's question
        context_parts.append(f"Dre's question:\n{user_text}")
        
        full_prompt = "\n".join(context_parts)
        
        self.message_history.append({"role": "user", "content": full_prompt})

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.message_history)

        try:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
            )
            assistant_text = response.choices[0].message.content
        except Exception as e:
            assistant_text = f"[Error: {e}]"

        self.message_history.append({"role": "assistant", "content": assistant_text})
        self.append_colored_message("assistant", assistant_text)
        self.status_label.setText("Ready.")
        
        self._save_history()


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def main():
    import sys

    app = QApplication(sys.argv)

    splash = None
    if SPLASH_FILE.exists():
        pixmap = QPixmap(str(SPLASH_FILE))
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()

    window = LeaWindow()
    window.show()

    if splash is not None:
        splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
