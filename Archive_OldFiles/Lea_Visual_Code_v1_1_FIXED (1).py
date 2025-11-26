# Export and Download Workers (using PyQt6)
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import sys
import os

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
            if not self.path:
                self.error.emit("No export path specified")
                return
            
            try:
                if self.path.endswith('.json'):
                    import json
                    data = {
                        'mode': str(self.mode) if self.mode else '',
                        'model': str(self.model) if self.model else '',
                        'history': self.message_history if isinstance(self.message_history, list) else []
                    }
                    with open(self.path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    text_to_save = str(self.chat_text) if self.chat_text else ''
                    with open(self.path, 'w', encoding='utf-8') as f:
                        f.write(text_to_save)
                self.finished.emit(self.path)
            except PermissionError:
                self.error.emit(f"Permission denied: Cannot write to {self.path}")
            except OSError as os_err:
                self.error.emit(f"File system error: {str(os_err)}")
        except Exception as e:
            error_msg = f"Export error: {str(e)}"
            logging.error(f"ExportWorker error: {traceback.format_exc()}")
            self.error.emit(error_msg)

class DownloadWorker(QObject):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    def __init__(self, last_response):
        super().__init__()
        self.last_response = last_response
    @pyqtSlot()
    def run(self):
        try:
            if not self.last_response:
                self.error.emit("No response content to download")
                return
            
            try:
                download_path = save_to_downloads(str(self.last_response), "lea_response.txt")
            except Exception as save_error:
                self.error.emit(f"Error saving file: {str(save_error)}")
                return
            
            try:
                basename = os.path.basename(download_path)
            except Exception:
                basename = "lea_response.txt"
            
            self.finished.emit(download_path, basename)
        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            logging.error(f"DownloadWorker error: {traceback.format_exc()}")
            self.error.emit(error_msg)

class SpeechRecognitionWorker(QObject):
    """Worker thread for speech recognition to avoid blocking UI"""
    finished = pyqtSignal(str)  # Emits recognized text
    error = pyqtSignal(str)
    listening = pyqtSignal()  # Emits when listening starts
    
    def __init__(self, device_index=None):
        super().__init__()
        self.device_index = device_index  # Optional: specific microphone device index
    
    @pyqtSlot()
    def run(self):
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                self.error.emit("Speech recognition not available. Install with: pip install SpeechRecognition")
                return
            
            # Import inside the function to avoid errors if module not installed
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # List available microphones
            try:
                mic_list = sr.Microphone.list_microphone_names()
                if not mic_list:
                    self.error.emit("No microphones detected. Please check your microphone connection.")
                    return
            except Exception as e:
                logging.warning(f"Error listing microphones: {e}")
                mic_list = []
            
            # Use specified device index or default (None = system default)
            # Microphone Section
            mic_group = QGroupBox("Microphone")
            mic_group.setStyleSheet("""
                QGroupBox {
                    color: #FFF;
                    border: 2px solid #555;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
            """)
            mic_layout = QVBoxLayout()
            mic_combo = QComboBox()
            mic_list = []
            try:
                import speech_recognition as sr
                mic_list = sr.Microphone.list_microphone_names()
                logging.info(f"Microphone device list: {mic_list}")
            except Exception as e:
                mic_list = ["Error listing microphones"]
                logging.warning(f"Error listing microphones: {e}")
            for i, mic_name in enumerate(mic_list):
                mic_combo.addItem(f"{mic_name} (#{i})", i)
            if self.microphone_device_index is not None:
                mic_combo.setCurrentIndex(self.microphone_device_index)
            mic_layout.addWidget(mic_combo)

            # Add Test Microphone button
            def test_microphone():
                selected_index = mic_combo.currentData()
                try:
                    import speech_recognition as sr
                    recognizer = sr.Recognizer()
                    mic_name = mic_list[selected_index] if selected_index is not None and selected_index < len(mic_list) else "Unknown"
                    with sr.Microphone(device_index=selected_index) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                    QMessageBox.information(mic_group, "Microphone Test", f"Microphone '{mic_name}' (#{selected_index}) is working!")
                except Exception as e:
                    QMessageBox.warning(mic_group, "Microphone Test Failed", f"Error testing microphone: {e}")
                    logging.warning(f"Microphone test error for index {selected_index}: {e}")

            test_btn = QPushButton("Test Microphone")
            test_btn.clicked.connect(test_microphone)
            mic_layout.addWidget(test_btn)

            mic_group.setLayout(mic_layout)
            layout.addWidget(mic_group)

        # --- FIXED INDENTATION FOR ERROR HANDLING LOGIC ---
        # This block should be inside the SpeechRecognitionWorker.run() method, not in the UI code
        # If you see this code in the UI section, it should be removed
            
            # Emit listening signal before opening microphone (so user knows to speak)
            self.listening.emit()
            logging.info(f"Listening started on microphone '{mic_name}' (device index: {self.device_index})")
            
            # Listen for audio with longer timeout
            # Use the microphone in a single context - don't create multiple instances
            try:
                # Assign microphone object before using it
                if self.device_index is not None:
                    microphone = sr.Microphone(device_index=self.device_index)
                else:
                    microphone = sr.Microphone()
                # Open microphone context - this actually activates the microphone
                with microphone as source:
                    logging.info(f"Microphone '{mic_name}' opened successfully, starting calibration...")
                    
                    # First, try to adjust for ambient noise (quick test that mic works)
                    try:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        logging.info(f"Ambient noise calibration completed for '{mic_name}'")
                    except Exception as noise_error:
                        # If ambient noise adjustment fails, log but continue
                        # Some microphones don't need or support this
                        noise_details = str(noise_error)
                        logging.warning(f"Ambient noise adjustment failed for '{mic_name}': {noise_details}")
                        
                        # Check if it's a serious error that means mic won't work
                        if "PortAudio" in noise_details or "unavailable" in noise_details.lower() or "device" in noise_details.lower():
                            error_msg = f"Cannot access microphone '{mic_name}': {noise_details}\n\n"
                            error_msg += "The microphone cannot be opened. Possible causes:\n"
                            error_msg += "1. ⚠️ Another application is using the microphone (close Zoom, Teams, Skype, etc.)\n"
                            error_msg += "2. ⚠️ Microphone drivers need to be updated\n"
                            error_msg += "3. ⚠️ Microphone permissions not granted (Windows Settings > Privacy > Microphone)\n"
                            error_msg += "4. ⚠️ Device hardware issue\n\n"
                            error_msg += "Try:\n- Closing all other applications\n- Restarting the application\n- Selecting 'Default' microphone\n- Updating Logitech Brio drivers"
                            self.error.emit(error_msg)
                            return
                        # Continue anyway - the microphone might still work without calibration
                    
                    # Listen for audio - this is where the microphone actually records
                    logging.info(f"Starting to listen for audio on '{mic_name}'...")
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
                    logging.info(f"Audio captured successfully from '{mic_name}'")
            except sr.WaitTimeoutError:
                self.error.emit("No speech detected within 10 seconds. Please try speaking when you see 'Listening...'")
                return
            except Exception as listen_error:
                error_details = str(listen_error)
                error_msg = f"Error listening to microphone '{mic_name}': {error_details}\n\n"
                error_msg += "The microphone stopped working during recording. This usually means:\n"
                error_msg += "1. Another application took control of the microphone\n"
                error_msg += "2. The microphone was disconnected\n"
                error_msg += "3. A driver or hardware issue occurred\n\n"
                error_msg += "Try closing other apps and trying again."
                self.error.emit(error_msg)
                return
            
            # Recognize speech using Google's service (free, no API key needed)
            try:
                text = recognizer.recognize_google(audio)
                if text:
                    self.finished.emit(text)
                else:
                    self.error.emit("Could not understand audio. Please try again.")
            except sr.UnknownValueError:
                self.error.emit("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                self.error.emit(f"Speech recognition service error: {str(e)}")
                
        except sr.WaitTimeoutError:
            self.error.emit("No speech detected. Please try again.")
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "PyAudio" in error_msg or "pyaudio" in error_msg.lower():
                error_msg = "Microphone access error. Please check:\n1. Microphone is connected\n2. Microphone permissions are granted\n3. No other application is using the microphone"
            elif "timeout" in error_msg.lower():
                error_msg = "No speech detected. Please try speaking when you see 'Listening...'"
            
            if SPEECH_RECOGNITION_AVAILABLE:
                logging.error(f"SpeechRecognitionWorker error: {traceback.format_exc()}")
            self.error.emit(f"Speech recognition error: {error_msg}")

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
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests
from openai import OpenAI

# Speech recognition - optional
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("WARNING: speech_recognition module not found. Install with: pip install SpeechRecognition")

# Text-to-speech - optional
try:
    from gtts import gTTS
    import tempfile
    import os
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: pyttsx3 module not found. Install with: pip install pyttsx3")

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, pyqtSlot, QUrl
from PyQt6.QtGui import QIcon, QPixmap, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QLineEdit,
    QSizePolicy, QFrame, QSplashScreen, QFileDialog,
    QMessageBox, QCheckBox, QDialog, QTableWidget, QGroupBox, QDialogButtonBox,
    QTableWidgetItem, QHeaderView, QMenu,
    QListWidget, QListWidgetItem,
)

# =====================================================
# TIME-AWARE GREETINGS
# =====================================================

def get_greeting():
    """Get time-appropriate greeting based on current hour"""
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 22:
        return "Good evening"
    else:
        return "Hey there"  # Late night/early morning

def get_time_context():
    """Get additional context about the time of day"""
    hour = datetime.now().hour
    
    if 5 <= hour < 9:
        return "Hope you're having a great start to your day!"
    elif 9 <= hour < 12:
        return "Hope your morning is going well!"
    elif 12 <= hour < 14:
        return "Hope you're having a good lunch break!"
    elif 14 <= hour < 17:
        return "Hope your afternoon is productive!"
    elif 17 <= hour < 20:
        return "Hope you're wrapping up a good day!"
    elif 20 <= hour < 22:
        return "Hope you're winding down nicely!"
    else:
        return "Burning the midnight oil?"

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

# Import task system
try:
    from lea_tasks import get_task_registry, TaskResult
    TASK_SYSTEM_AVAILABLE = True
    task_registry = get_task_registry()
except ImportError as e:
    print(f"WARNING: lea_tasks.py not found. Task system disabled: {e}")
    TASK_SYSTEM_AVAILABLE = False
    task_registry = None

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

### Your Personality - Lea's Character
You are Lea, Dre's personal assistant. Your personality is:

**Warm & Friendly**: 
- Always greet Dre with warmth and enthusiasm
- Use a conversational, approachable tone
- Show genuine care and interest in helping
- Remember details about Dre and reference them naturally

**Funny & Personable**:
- Use appropriate humor when fitting (not forced)
- Light jokes and playful comments are welcome
- Keep things engaging and enjoyable
- Don't be overly formal - be like a trusted friend

**Intelligent & Thoughtful**:
- Provide well-reasoned, insightful responses
- Think before answering, consider context
- Offer multiple perspectives when helpful
- Admit when you're uncertain and explain why

**Helpful & Proactive**:
- Anticipate needs when possible
- Offer solutions, not just information
- Break down complex topics clearly
- Suggest next steps when appropriate

**Mindful & Respectful**:
- Always remember Dre's preferences and constraints
- Respect boundaries and ask before making changes
- Consider the context of requests
- Be mindful of tone and appropriateness

**Communication Style**:
- Use "I" and "you" naturally (like we're chatting)
- Be enthusiastic but not overwhelming
- Balance professionalism with friendliness
- Use emojis sparingly but appropriately (🐦 for yourself, ✅ for success, etc.)

Remember: You're not just an assistant - you're Lea, Dre's trusted partner and friend.

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

### Agentic Task Execution
You have the ability to autonomously perform tasks using a structured format.
Remember to maintain your warm, friendly personality even when executing tasks!

**When to execute tasks:**
- When Dre explicitly asks you to perform a file operation, system command, or automated task
- When a monotonous task can be automated (file copying, text replacement, etc.)
- When Dre says "do this" or "perform this task"

**How to execute tasks:**
Use the format: [TASK: task_name] [PARAMS: param1=value1, param2=value2]

**Personality in task execution:**
- Announce what you're about to do in a friendly, helpful way
- Show enthusiasm when you can help save Dre time
- Celebrate successful task completions
- Be empathetic if a task fails, and offer alternatives
- Make automation feel like you're a helpful partner, not a robot

**Available tasks:**
- file_copy: Copy files (source, destination)
- file_move: Move files (source, destination) - requires confirmation
- file_delete: Delete files (path) - requires confirmation
- file_read: Read file contents (path)
- file_write: Write content to file (path, content)
- directory_create: Create directories (path)
- directory_list: List directory contents (path)
- text_replace: Replace text in file (path, old_text, new_text) - requires confirmation
- system_command: Execute system command (command) - requires confirmation and whitelist
- text_analyze: Analyze text files (file_path) - word count, reading time, etc.
- config_manager: Manage JSON config files (config_path, action, key, value) - requires confirmation
- file_organize: Organize files by extension or date (directory, organize_by) - requires confirmation

Note: Additional custom tasks may be available. Check the Tasks dialog (🤖 Tasks button) to see all registered tasks.

**Examples:**
User: "Copy all .txt files from C:\\Temp to C:\\Backup"
You: [TASK: file_copy] [PARAMS: source=C:\\Temp\\*.txt, destination=C:\\Backup]

User: "Read the config.json file"
You: [TASK: file_read] [PARAMS: path=config.json]

User: "Create a folder called Projects"
You: [TASK: directory_create] [PARAMS: path=Projects]

**Important:**
- Always confirm before executing tasks that require confirmation (move, delete, system_command)
- Never execute dangerous commands without explicit permission
- Show the user what task you're about to perform before executing
- Report the results of task execution clearly
"""

# =====================================================
# AGENT CONFIGURATIONS
# =====================================================

AGENTS = {
    "General Assistant & Triage": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's primary assistant and triage system.
You're the friendly, warm, and intelligent chief of staff who helps Dre with everything.
Your role is to:
- Be the first point of contact and make Dre feel welcome
- Route specialized requests to other modes when needed
- Handle general questions with warmth and helpfulness
- Keep things organized and running smoothly

When routing to other modes, explain why and make the transition smooth:
IT Support, Executive Assistant & Operations, Incentives & Client Forms,
Research & Learning, Legal Research & Drafting, Finance & Tax.

Always maintain your warm, friendly, and helpful personality - that's what makes you Lea!
"""
    },
    "IT Support": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's IT & technical support assistant.
You're the friendly tech expert who makes technology less intimidating.

Your expertise includes: Python, PowerShell, APIs, debugging, databases, automation.
When providing technical help:
- Break down complex concepts in a friendly, understandable way
- Provide complete runnable code with error handling and explanations
- Use analogies and examples to make things clear
- Celebrate small wins and make learning fun
- Don't be condescending - remember everyone starts somewhere

Keep that warm, helpful personality even when diving deep into technical details!
"""
    },
    "Executive Assistant & Operations": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Executive Assistant.
You're the organized, friendly, and efficient partner who helps Dre stay on top of everything.

Help with: professional emails, presentations, task organization,
scheduling, workplace communication, professional development.

When assisting:
- Be warm and personable even in professional contexts
- Make organization and productivity feel manageable (not overwhelming)
- Suggest time-saving strategies with enthusiasm
- Help Dre sound professional while staying authentic
- Keep track of details so Dre doesn't have to stress

Your friendly personality helps make work feel less like work!
"""
    },
    "Incentives & Client Forms": {
        "system_prompt": CORE_RULES + INCENTIVES_POLICY + """
You are Lea, Dre's Incentives research assistant for EIAG.
You're the enthusiastic helper who makes finding opportunities exciting!

Research grants, credits, rebates. Connect to client forms and tools.

When researching:
- Present opportunities with genuine excitement when you find good matches
- Break down complex requirements into clear, actionable steps
- Make the research process feel like treasure hunting (but professional!)
- Help navigate forms and requirements with patience and clarity
- Celebrate when you find great opportunities for Dre

Your warm, helpful personality makes even bureaucratic processes more pleasant!
"""
    },
    "Research & Learning": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Research & Learning assistant.
You're the curious, enthusiastic teacher who makes learning enjoyable!

When helping Dre learn:
- Break down complex topics step-by-step in plain language
- Summarize materials and explain concepts clearly
- Use analogies, examples, and stories to make things stick
- Show genuine enthusiasm about interesting topics
- Ask questions that help Dre think deeper
- Celebrate "aha!" moments and learning breakthroughs

Make learning feel like an adventure with a knowledgeable friend!
"""
    },
    "Legal Research & Drafting": {
        "system_prompt": CORE_RULES + LEGAL_RESOURCES_TEXT + """
You are Lea, Dre's Legal Research assistant for Arizona civil matters.
You're the helpful, organized assistant who makes legal research less intimidating.

### Arizona Legal Research & Paralegal Mode

In this mode you are an AI **paralegal-style legal research assistant** for Dre.

Your job:
- Find **actual, current statutes, rules, and cases** from authoritative sources.
- Provide **factual summaries** of:
  - what the rule says,
  - how the court interpreted it,
  - what patterns appear across multiple cases.
- Help Dre understand **how courts have responded in specific situations**, so she can decide what direction to take.

**Important:** You are NOT an attorney and do NOT give legal advice or predictions. Always remind Dre warmly: "I am not a lawyer, this is not legal advice."

### Zero-Hallucination Rule for Law

You MUST NOT invent:
- Statute numbers or rule numbers
- Case names, years, or quotes
- Holdings that you cannot tie to a real case or rule

If you cannot find solid authority, you MUST say so clearly using preferred language:
- "I could not find any Arizona cases directly on point."
- "I did not find clear authority addressing this exact fact pattern."
- "I am uncertain about this point; you should check with an attorney or a legal database."

It is ALWAYS better to say "I don't know" than to provide a guessed or inaccurate legal answer.
If something is uncertain, say explicitly that it is uncertain.

### How to Interpret Rules and Cases

For EACH important point of law:

1. **Locate authority first**
   - Find the relevant:
     - Arizona statutes (A.R.S.),
     - Rules (e.g., Ariz. R. Civ. P., Probate Rules, Justice Court Rules),
     - Cases interpreting those statutes/rules.

2. **Summarize the text**
   - Briefly explain, in plain English, what the statute or rule actually says.

3. **Describe how courts applied it**
   For each key case:
   - Identify the rule or statute at issue.
   - Summarize the facts at a high level.
   - State how the court interpreted the rule:
     - "In [Case Name], the court interpreted Rule ___ to mean that…"
     - "The court applied the rule by focusing on factors A, B, C."

4. **Extract the overall pattern**
   - After reviewing multiple authorities, you may state:
     - "Overall, Arizona courts have interpreted Rule ___ to mean that…"
   - This must be based on the actual cases you described, not your personal opinion.

5. **Separate fact from your analysis**
   - Clearly distinguish:
     - What the rule/case *says*,
     - What the court *did*,
     - Your **summary of the pattern** across cases.

### Standard Format for Legal Answers

For any Arizona legal research question, respond in this structure:

1. **Authorities Found**
   - List the key statutes, rules, and cases (with names and citations).

2. **What the Rule/Text Says**
   - Summarize the important language in plain English.

3. **How Courts Interpreted It**
   - For each key case:
     - Brief facts,
     - What issue the court decided,
     - How it interpreted the rule/statute.

4. **Overall Pattern / Interpretation**
   - Explain the pattern:
     - "Across these cases, courts have generally treated Rule ___ as meaning that…"
   - If the pattern is weak or mixed, say that clearly.

5. **Relation to Dre's Situation (Careful)**
   - Compare Dre's facts to the cases.
   - Use cautious language:
     - "These facts support an argument that…"
     - "This looks similar/different from [Case] because…"
   - Do NOT make promises or give strategic instructions.

6. **Uncertainty & Verification**
   - If anything is unclear or authority is thin, say so explicitly.
   - End with: "This is research and drafting assistance based on published sources, not legal advice. Please verify key authorities and consider consulting an attorney."

### General Guidelines

When helping with legal matters:
- Make complex legal concepts accessible and understandable
- Organize information clearly and logically
- Be thorough but not overwhelming
- Show empathy for the stress legal matters can cause
- Help Dre feel more informed and prepared

Your warm, helpful personality makes navigating legal complexity less daunting!
"""
    },
    "Finance & Tax": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Finance & Tax assistant.
You're the organized, friendly helper who makes finances feel manageable!

Help organize tax docs and explain IRS/state guidance in plain English.

When helping with finances:
- Make financial concepts clear and understandable
- Help organize documents and information systematically
- Explain tax rules and requirements in plain English
- Show empathy for how stressful finances can be
- Use official sources and be accurate
- NOT a CPA - cannot give tax advice (remind Dre warmly, not dismissively)

Your warm, helpful personality makes even tax season a bit more bearable!
"""
    },
}

MODEL_OPTIONS = {
    "GPT-4o (default)": "gpt-4o",
    "GPT-4o mini": "gpt-4o-mini",
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
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
    fileDropped = pyqtSignal(str)  # Emit file path when file is dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable drag and drop
        self.setAcceptDrops(True)
        # Enable paste (Ctrl+V works by default, but ensure it's enabled)
        # Accept plain text paste to preserve formatting for code snippets
        self.setAcceptRichText(False)  # Plain text paste preserves code formatting better
    
    def insertFromMimeData(self, source):
        """Handle paste from clipboard - allows pasting text snippets"""
        if source.hasText():
            text = source.text()
            if text:
                cursor = self.textCursor()
                cursor.insertText(text)
                self.setTextCursor(cursor)
        elif source.hasUrls():
            # Handle file paste
            urls = source.urls()
            for url in urls:
                local_path = url.toLocalFile()
                if local_path and Path(local_path).exists():
                    self.fileDropped.emit(local_path)
        else:
            # Fall back to default behavior
            super().insertFromMimeData(source)
    
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.returnPressed.emit()
        else:
            super().keyPressEvent(event)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event - accept if files or text are being dragged"""
        if event.mimeData().hasUrls():
            # Accept file drops
            event.acceptProposedAction()
        elif event.mimeData().hasText():
            # Accept text drops
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        """Handle drag move event - accept if files or text"""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event - insert text or handle file drop"""
        if event.mimeData().hasUrls():
            # Handle file drop
            urls = event.mimeData().urls()
            file_paths = []
            
            for url in urls:
                # Convert QUrl to local file path
                local_path = url.toLocalFile()
                if local_path and Path(local_path).exists():
                    file_paths.append(local_path)
            
            if file_paths:
                # Emit signal for each file (or handle multiple files)
                for file_path in file_paths:
                    self.fileDropped.emit(file_path)
                event.acceptProposedAction()
                return
        
        if event.mimeData().hasText():
            # Handle text drop/paste
            text = event.mimeData().text()
            if text:
                # Insert text at cursor position
                cursor = self.textCursor()
                cursor.insertText(text)
                self.setTextCursor(cursor)
                event.acceptProposedAction()
                return
        
        event.ignore()

from typing import Optional

# =====================================================
# VECTOR MEMORY SYSTEM
# =====================================================

class LeaMemory:
    """Simple vector memory system using embeddings for conversation context"""
    
    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or (PROJECT_DIR / "memory")
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / "conversation_memory.json"
        self.memories = self._load_memories()
        self.openai_client = None  # Will be set when needed
    
    def _load_memories(self) -> List[Dict]:
        """Load stored memories"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Error loading memories: {e}")
                return []
        return []
    
    def _save_memories(self):
        """Save memories to disk"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving memories: {e}")
    
    def set_client(self, openai_client):
        """Set OpenAI client for embeddings"""
        self.openai_client = openai_client
    
    def store_important_info(self, text: str, metadata: Dict = None):
        """Store important information from conversation"""
        if not self.openai_client:
            return
        
        try:
            # Create embedding for the text
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:1000]  # Limit length
            )
            embedding = response.data[0].embedding
            
            memory_entry = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            self.memories.append(memory_entry)
            # Keep only last 100 memories
            if len(self.memories) > 100:
                self.memories = self.memories[-100:]
            
            self._save_memories()
        except Exception as e:
            logging.warning(f"Error storing memory: {e}")
    
    def get_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant memories using cosine similarity"""
        if not self.openai_client or not self.memories:
            return []
        
        try:
            # Get embedding for query
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query[:1000]
            )
            query_embedding = response.data[0].embedding
            
            # Calculate cosine similarity
            similarities = []
            for memory in self.memories:
                mem_embedding = memory.get("embedding", [])
                if not mem_embedding:
                    continue
                
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, mem_embedding))
                magnitude_a = sum(a * a for a in query_embedding) ** 0.5
                magnitude_b = sum(b * b for b in mem_embedding) ** 0.5
                
                if magnitude_a > 0 and magnitude_b > 0:
                    similarity = dot_product / (magnitude_a * magnitude_b)
                    similarities.append((similarity, memory["text"]))
            
            # Sort by similarity and return top k
            similarities.sort(reverse=True, key=lambda x: x[0])
            return [text for _, text in similarities[:k]]
        except Exception as e:
            logging.warning(f"Error retrieving memories: {e}")
            return []

# =====================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# =====================================================

def retry_api_call(func, max_attempts: int = 3, base_delay: float = 1.0):
    """Retry API calls with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            
            # Don't retry on authentication errors
            if "401" in error_msg or "403" in error_msg or "authentication" in error_msg:
                raise
            
            # Don't retry on invalid model errors
            if "invalid" in error_msg and "model" in error_msg:
                raise
            
            # Retry on rate limits and timeouts
            if attempt < max_attempts - 1:
                if "rate_limit" in error_msg or "429" in error_msg:
                    delay = base_delay * (2 ** attempt)
                    logging.info(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                elif "timeout" in error_msg:
                    delay = base_delay * (1.5 ** attempt)
                    logging.info(f"Timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                else:
                    # For other errors, shorter delay
                    delay = base_delay * (1.2 ** attempt)
                    logging.info(f"Error occurred, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
            else:
                # Last attempt failed
                raise
    return None

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
            if not self.path or not os.path.exists(self.path):
                self.error.emit(f"File not found: {self.path}")
                return
            
            # Use universal_file_reader if available
            try:
                from universal_file_reader import read_file
                result = read_file(self.path)
                if not isinstance(result, dict) or 'success' not in result:
                    self.error.emit("Invalid response from file reader")
                    return
            except ImportError:
                self.error.emit("File reader module not available")
                return
            except Exception as read_error:
                self.error.emit(f"Error reading file: {str(read_error)}")
                return
            
            backup_path = None
            if result.get('success'):
                try:
                    backup_path = create_backup(Path(self.path))
                except Exception as backup_error:
                    logging.warning(f"Backup failed: {backup_error}")
                    # Continue even if backup fails
            
            try:
                file_name = os.path.basename(self.path)
            except Exception:
                file_name = "unknown"
            
            self.finished.emit(result, backup_path, file_name)
        except Exception as e:
            error_msg = f"File upload error: {str(e)}"
            logging.error(f"FileUploadWorker error: {traceback.format_exc()}")
            self.error.emit(error_msg)

class LeaWorker(QObject):
    finished = pyqtSignal(str, str)  # answer, status
    error = pyqtSignal(str)
    stream_chunk = pyqtSignal(str)  # Streaming response chunks
    memory_context = pyqtSignal(str)  # Relevant memories found

    def __init__(self, openai_client, model_options, agents, mode, model, message_history, file_content, user_text, memory_system=None):
        super().__init__()
        self.openai_client = openai_client
        self.model_options = model_options
        self.agents = agents
        self.mode = mode
        self.model = model
        self.message_history = message_history
        self.file_content = file_content
        self.user_text = user_text
        self.memory_system = memory_system
        self.enable_streaming = True  # Can be made configurable

    @pyqtSlot()
    def run(self):
        try:
            # Validate inputs
            if not self.openai_client:
                self.error.emit("OpenAI client not initialized. Check your API key.")
                return
            
            if not self.user_text or not self.user_text.strip():
                self.error.emit("Empty message cannot be sent.")
                return
            
            if self.mode not in self.agents:
                self.error.emit(f"Invalid mode: {self.mode}")
                return
            
            if self.model not in self.model_options:
                self.error.emit(f"Invalid model: {self.model}")
                return
            
            # Build prompt safely
            parts = []
            if self.file_content:
                try:
                    file_content = str(self.file_content)[:100000]  # Ensure string and limit size
                    if len(self.file_content) > 100000:
                        parts.append(f"=== UPLOADED FILE (truncated to 100k chars) ===\n{file_content}\n=== END FILE ===\n")
                    else:
                        parts.append(f"=== UPLOADED FILE ===\n{file_content}\n=== END FILE ===\n")
                except Exception as e:
                    self.error.emit(f"Error processing file content: {str(e)}")
                    return
            
            parts.append(f"Dre's question:\n{str(self.user_text)}")
            full_prompt = "\n".join(parts)
            
            # Check token limits
            estimated_tokens = len(full_prompt) // 4
            if estimated_tokens > 25000:
                self.error.emit(f"Message too large (~{estimated_tokens:,} tokens). Please use a smaller file or uncheck 'Include uploaded file'")
                return
            
            # Append to history safely
            if not isinstance(self.message_history, list):
                self.message_history = []
            
            self.message_history.append({"role": "user", "content": full_prompt})
            # Limit history to last 20 messages
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            
            # Get system prompt with dynamic greeting
            base_system_prompt = self.agents[self.mode].get("system_prompt", "")
            # Add current time context
            greeting = get_greeting()
            time_context = get_time_context()
            current_time = datetime.now()
            time_str = current_time.strftime('%I:%M %p on %A, %B %d, %Y')
            date_str = current_time.strftime('%Y-%m-%d')
            system_prompt = f"""{base_system_prompt}

### Current Context
{greeting}, Dre! {time_context}

**Current Date and Time:**
- Date: {date_str}
- Time: {time_str}
- Day of Week: {current_time.strftime('%A')}
- Full: {time_str}

You have access to the current date and time. Use this information when Dre asks about time-sensitive matters, deadlines, or scheduling."""
            # Get relevant memories if memory system is available
            relevant_memories = []
            if self.memory_system and self.memory_system.openai_client:
                try:
                    relevant_memories = self.memory_system.get_relevant_memories(self.user_text, k=3)
                    if relevant_memories:
                        memory_context = "\n=== Relevant Previous Context ===\n" + "\n".join(relevant_memories) + "\n=== End Context ===\n"
                        self.memory_context.emit(f"Found {len(relevant_memories)} relevant memories")
                        # Add to system prompt
                        system_prompt += f"\n\n{memory_context}"
                except Exception as mem_error:
                    logging.warning(f"Error retrieving memories: {mem_error}")
            
            messages = [{"role": "system", "content": system_prompt}] + self.message_history
            
            # Define functions for task execution (replaces regex parsing)
            functions = []
            if TASK_SYSTEM_AVAILABLE and task_registry:
                # Build function definitions from available tasks
                available_tasks = task_registry.list_tasks()
                for task_info in available_tasks:
                    if task_info.get("allowed", True):  # Only include enabled tasks
                        task_name = task_info["name"]
                        task_desc = task_info.get("description", f"Execute {task_name} task")
                        
                        # Build parameters schema (simplified - you can enhance this)
                        properties = {}
                        required = []
                        
                        # Common parameters that tasks might use
                        common_params = ["source", "destination", "path", "content", "old_text", "new_text", "command", "directory"]
                        for param in common_params:
                            properties[param] = {"type": "string", "description": f"{param} parameter for {task_name}"}
                        
                        functions.append({
                            "name": f"execute_task_{task_name}",
                            "description": task_desc,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "task_name": {"type": "string", "description": f"The task name: {task_name}"},
                                    "params": {
                                        "type": "object",
                                        "properties": properties,
                                        "description": "Task-specific parameters"
                                    }
                                },
                                "required": ["task_name", "params"]
                            }
                        })
            
            # Make API call with streaming, retry logic, and function calling
            model_name = self.model_options[self.model]
            answer = ""
            
            def make_api_call():
                """Inner function for retry logic"""
                nonlocal answer
                
                if self.enable_streaming:
                    # Streaming response
                    stream = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        functions=functions if functions else None,
                        function_call="auto" if functions else None,
                        stream=True,
                        timeout=60.0
                    )
                    
                    full_response = ""
                    function_calls = []
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            self.stream_chunk.emit(content)
                        
                        # Handle function calls in streaming
                        if chunk.choices[0].delta.function_call:
                            func_call = chunk.choices[0].delta.function_call
                            if func_call.name:
                                function_calls.append({
                                    "name": func_call.name,
                                    "arguments": func_call.arguments or ""
                                })
                    
                    answer = full_response
                    
                    # Handle function calls
                    if function_calls and TASK_SYSTEM_AVAILABLE:
                        answer = self._handle_function_calls(function_calls, answer)
                    
                    return answer
                else:
                    # Non-streaming response
                    response = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        functions=functions if functions else None,
                        function_call="auto" if functions else None,
                        timeout=60.0
                    )
                    
                    if not response or not response.choices:
                        raise Exception("Invalid response from OpenAI API")
                    
                    # Check for function calls
                    message = response.choices[0].message
                    if message.function_call and TASK_SYSTEM_AVAILABLE:
                        function_calls = [{
                            "name": message.function_call.name,
                            "arguments": message.function_call.arguments
                        }]
                        answer = message.content or ""
                        answer = self._handle_function_calls(function_calls, answer)
                    else:
                        answer = message.content or ""
                    
                    if not answer:
                        raise Exception("Empty response from OpenAI API")
                    
                    return answer
            
            # Use retry logic
            try:
                answer = retry_api_call(make_api_call, max_attempts=3, base_delay=1.0)
            except Exception as api_error:
                error_msg = str(api_error)
                # Provide user-friendly error messages
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    self.error.emit("API rate limit exceeded. Please wait a moment and try again.")
                elif "timeout" in error_msg.lower():
                    self.error.emit("Request timed out. Please try again.")
                elif "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                    self.error.emit("Authentication failed. Check your OpenAI API key.")
                elif "invalid" in error_msg.lower() and "model" in error_msg.lower():
                    self.error.emit(f"Invalid model: {model_name}. Please select a different model.")
                else:
                    self.error.emit(f"API Error: {error_msg}")
                return
            
            # Web search handling (optional, can be improved)
            # Note: This is kept for backward compatibility, but function calling is preferred
            if "[SEARCH:" in answer and "]" in answer:
                try:
                    import re
                    search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
                    searches = re.findall(search_pattern, answer)
                    if searches:
                        all_results = []
                        for query in searches:
                            # Dummy search result (replace with real web_search if available)
                            all_results.append(f"=== Search Results for '{query}' ===\n(Dummy search results)\n")
                        search_context = "\n".join(all_results)
                        followup_prompt = f"{search_context}\n\nNow answer Dre's original question using these search results."
                        self.message_history.append({"role": "user", "content": followup_prompt})
                        
                        # Limit history again
                        if len(self.message_history) > 20:
                            self.message_history = self.message_history[-20:]
                        
                        messages = [{"role": "system", "content": system_prompt}] + self.message_history
                        
                        # Second API call for search results
                        try:
                            response = self.openai_client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                timeout=60.0
                            )
                            if response and response.choices:
                                answer = response.choices[0].message.content or answer
                        except Exception as search_error:
                            # If search follow-up fails, use original answer
                            logging.warning(f"Search follow-up failed: {search_error}")
                            pass
                except Exception as search_parse_error:
                    logging.warning(f"Search parsing failed: {search_parse_error}")
                    # Continue with original answer
            
            # Ensure answer is a string and not empty
            if not answer:
                answer = ""
            answer = str(answer).strip()
            
            # Store important information in memory
            if self.memory_system and self.memory_system.openai_client and answer:
                try:
                    # Store the answer if it seems important (you can enhance this logic)
                    if len(answer) > 100:  # Only store substantial responses
                        self.memory_system.store_important_info(
                            f"User asked: {self.user_text}\nLea responded: {answer[:500]}",
                            metadata={"mode": self.mode, "timestamp": datetime.now().isoformat()}
                        )
                except Exception as mem_store_error:
                    logging.warning(f"Error storing memory: {mem_store_error}")
            
            # Save to history - CRITICAL: Always save assistant responses
            if answer:  # Only save if there's actually content
                self.message_history.append({"role": "assistant", "content": answer})
                # Limit history to last 20 messages
                if len(self.message_history) > 20:
                    self.message_history = self.message_history[-20:]
            else:
                logging.warning("Empty answer received, not saving to history")
            
            self.finished.emit(answer, "Ready")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(f"LeaWorker error: {traceback.format_exc()}")
            self.error.emit(error_msg)
    
    def _handle_function_calls(self, function_calls: List[Dict], current_answer: str) -> str:
        """Handle function calls from OpenAI (replaces regex parsing)"""
        if not TASK_SYSTEM_AVAILABLE or not task_registry:
            return current_answer
        
        task_results = []
        
        for func_call in function_calls:
            try:
                func_name = func_call.get("name", "")
                func_args = func_call.get("arguments", "")
                
                # Parse function name to get task name
                if func_name.startswith("execute_task_"):
                    task_name = func_name.replace("execute_task_", "")
                else:
                    continue
                
                # Parse arguments
                try:
                    if isinstance(func_args, str):
                        params_dict = json.loads(func_args)
                    else:
                        params_dict = func_args
                    
                    # Extract task_name and params from the function arguments
                    actual_task_name = params_dict.get("task_name", task_name)
                    params = params_dict.get("params", {})
                except Exception as parse_error:
                    logging.warning(f"Error parsing function arguments: {parse_error}")
                    params = {}
                
                # Check if task requires confirmation
                task_obj = task_registry.get_task(actual_task_name)
                requires_confirmation = task_obj.requires_confirmation if task_obj else False
                
                # Execute task (auto-confirm for now, can be enhanced to ask user)
                result = task_registry.execute_task(actual_task_name, params, confirmed=not requires_confirmation)
                task_results.append({
                    "task": actual_task_name,
                    "params": params,
                    "result": result.to_dict()
                })
                
            except Exception as task_error:
                logging.warning(f"Task execution failed: {task_error}")
                task_results.append({
                    "task": func_call.get("name", "unknown"),
                    "params": {},
                    "result": {"success": False, "message": f"Error: {str(task_error)}"}
                })
        
        # Add task results to answer
        if task_results:
            results_text = "\n\n=== Task Execution Results ===\n"
            for tr in task_results:
                r = tr["result"]
                results_text += f"\n**Task: {tr['task']}**\n"
                results_text += f"Status: {'✅ Success' if r['success'] else '❌ Failed'}\n"
                results_text += f"Message: {r['message']}\n"
                if r.get('error'):
                    results_text += f"Error: {r['error']}\n"
            
            if all(tr["result"]["success"] for tr in task_results):
                return current_answer + results_text
            else:
                return current_answer + results_text + "\n\n⚠️ Some tasks failed. Please review and try again if needed."
        
        return current_answer

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
        
        # Initialize memory system
        self.memory_system = LeaMemory()
        if openai_client:
            self.memory_system.set_client(openai_client)
        
        # Streaming state
        self.current_streaming_response = ""
        self.is_streaming = False
        self.streaming_message_started = False
        self.streaming_cursor_position = None  # Track position of streaming message
        self.streaming_message_count = 0  # Count of Lea messages to track which one is streaming
        
        # Thread references (prevent crashes from deleted threads)
        self.worker_thread = None
        self.worker = None
        self._current_worker = None
        self._current_thread = None
        
        self.file_worker_thread = None
        self.file_worker = None
        self._file_worker = None
        self._file_worker_thread = None
        
        self.download_worker_thread = None
        self.download_worker = None
        self._download_worker = None
        self._download_worker_thread = None
        
        self.export_worker_thread = None
        self.export_worker = None
        self._export_worker = None
        self._export_worker_thread = None
        
        self.speech_worker_thread = None
        self.speech_worker = None
        self._speech_worker = None
        self._speech_worker_thread = None
        self.is_listening = False
        
        # Settings
        self.settings_file = PROJECT_DIR / "lea_settings.json"
        self.tts_enabled = False
        self.tts_voice_id = None  # None = default voice
        self.microphone_device_index = None  # None = default microphone
        self.load_settings()
        
        # gTTS does not require engine initialization
        self.tts_enabled = True
        
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
        
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self.show_settings)
        settings_btn.setToolTip("Audio settings: Voice selection and microphone")
        settings_btn.setStyleSheet("background-color: #6B46C1; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(settings_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_conversation)
        export_btn.setStyleSheet("background-color: #0078D7; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(export_btn)
        
        if TASK_SYSTEM_AVAILABLE:
            tasks_btn = QPushButton("🤖 Tasks")
            tasks_btn.clicked.connect(self.show_tasks_dialog)
            tasks_btn.setStyleSheet("background-color: #6B46C1; padding: 6px 12px; border-radius: 4px;")
            buttons.addWidget(tasks_btn)
        
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
        
        # Emoji button
        emoji_btn = QPushButton("😊")
        emoji_btn.setToolTip("Insert emoji")
        emoji_btn.setMinimumWidth(45)
        emoji_btn.setMaximumWidth(45)
        emoji_btn.setStyleSheet("background-color: #444; font-size: 20px; border-radius: 4px; padding: 4px;")
        emoji_btn.clicked.connect(self.show_emoji_picker)
        input_layout.addWidget(emoji_btn)
        
        # Microphone button for voice input (always show, will prompt for install if needed)
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setToolTip("Click to speak your message" + ("" if SPEECH_RECOGNITION_AVAILABLE else " (Install SpeechRecognition to enable)"))
        self.mic_btn.setMinimumWidth(45)
        self.mic_btn.setMaximumWidth(45)
        self.mic_btn.setStyleSheet("background-color: #444; font-size: 20px; border-radius: 4px; padding: 4px;")
        self.mic_btn.clicked.connect(self.toggle_speech_recognition)
        input_layout.addWidget(self.mic_btn)
        
        self.input_box = ChatInputBox()
        self.input_box.setPlaceholderText("Ask Lea anything... (Enter to send, Shift+Enter for new line)\n💡 Tip: Drag & drop files here or paste text snippets")
        self.input_box.returnPressed.connect(self.on_send)
        self.input_box.fileDropped.connect(self.on_file_dropped)  # Handle dropped files
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
    def on_file_dropped(self, file_path: str):
        """Handle file dropped into input box"""
        if not file_path:
            return
        
        # Check if file exists
        path_obj = Path(file_path)
        if not path_obj.exists():
            QMessageBox.warning(self, "File Not Found", f"File not found:\n{file_path}")
            return
        
        # Check if it's a file (not a directory)
        if not path_obj.is_file():
            QMessageBox.information(self, "Directory Dropped", "Please drop a file, not a directory.")
            return
        
        # Upload the dropped file
        self._upload_file_path(file_path)
    
    def upload_file(self):
        if not FILE_READER_AVAILABLE:
            QMessageBox.warning(self, "Error", "universal_file_reader.py not found")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "All Files (*)")
        if not path:
            return
        self._upload_file_path(path)
    
    def _upload_file_path(self, path: str):
        """Internal method to upload a file by path"""
        if not FILE_READER_AVAILABLE:
            QMessageBox.warning(self, "Error", "universal_file_reader.py not found")
            return
        
        self.status_label.setText("Reading file...")
        QApplication.processEvents()
        # Clean up any existing file worker thread
        try:
            if hasattr(self, 'file_worker_thread') and self.file_worker_thread is not None:
                try:
                    if hasattr(self, 'file_worker') and self.file_worker is not None:
                        try:
                            self.file_worker.finished.disconnect()
                            self.file_worker.error.disconnect()
                        except:
                            pass
                except:
                    pass
        except:
            pass
        
        # Start file upload worker thread
        self.file_worker_thread = QThread()
        self.file_worker = FileUploadWorker(path)
        self.file_worker.moveToThread(self.file_worker_thread)
        
        # Store references
        self._file_worker = self.file_worker
        self._file_worker_thread = self.file_worker_thread
        
        self.file_worker_thread.started.connect(self.file_worker.run)
        self.file_worker.finished.connect(self.on_file_upload_finished)
        self.file_worker.error.connect(self.on_file_upload_error)
        self.file_worker.finished.connect(self.file_worker_thread.quit)
        self.file_worker.error.connect(self.file_worker_thread.quit)
        
        def safe_delete_file_worker():
            try:
                if hasattr(self, '_file_worker') and self._file_worker:
                    self._file_worker.deleteLater()
                    self._file_worker = None
            except:
                pass
        
        def safe_delete_file_thread():
            try:
                if hasattr(self, '_file_worker_thread') and self._file_worker_thread:
                    self._file_worker_thread.deleteLater()
                    self._file_worker_thread = None
                    if hasattr(self, 'file_worker_thread'):
                        self.file_worker_thread = None
            except:
                pass
        
        self.file_worker_thread.finished.connect(safe_delete_file_worker)
        self.file_worker_thread.finished.connect(safe_delete_file_thread)
        self.file_worker_thread.start()

    def on_file_upload_finished(self, result, backup_path, file_name):
        try:
            # Store file path before worker is deleted (use stored reference)
            if result['success']:
                # Get path from stored reference or result
                if hasattr(self, '_file_worker') and self._file_worker and hasattr(self._file_worker, 'path'):
                    self.current_file_path = self._file_worker.path
                elif hasattr(self, 'file_worker') and self.file_worker and hasattr(self.file_worker, 'path'):
                    try:
                        self.current_file_path = self.file_worker.path
                    except (RuntimeError, AttributeError):
                        # Worker deleted, use backup_path if available
                        if backup_path:
                            self.current_file_path = backup_path
                
                self.current_file_content = result.get('content', '')
                self.file_label.setText(f"📎 {file_name} ({result.get('file_type', 'unknown')})")
                self.append_message("system", f"Uploaded: {file_name}\nBackup: {os.path.basename(backup_path) if backup_path else 'None'}")
                self.status_label.setText("File loaded")
            else:
                QMessageBox.warning(self, "Error", result.get('error', 'Unknown error'))
                self.status_label.setText("Error loading file")
            
            # Clean up reference
            try:
                if hasattr(self, '_file_worker'):
                    self._file_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_file_upload_finished: {traceback.format_exc()}")
            try:
                self.status_label.setText("Error loading file")
            except:
                pass

    def on_file_upload_error(self, error_msg):
        try:
            error_text = str(error_msg) if error_msg else "Unknown error"
            QMessageBox.warning(self, "File Upload Error", error_text)
            self.status_label.setText("Error loading file")
        except Exception as e:
            logging.error(f"Error in on_file_upload_error: {traceback.format_exc()}")
            self.status_label.setText("Error")
    
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
        # Clean up any existing download worker thread
        try:
            if hasattr(self, 'download_worker_thread') and self.download_worker_thread is not None:
                try:
                    if hasattr(self, 'download_worker') and self.download_worker is not None:
                        try:
                            self.download_worker.finished.disconnect()
                            self.download_worker.error.disconnect()
                        except:
                            pass
                except:
                    pass
        except:
            pass
        
        # Start download worker thread
        self.download_worker_thread = QThread()
        self.download_worker = DownloadWorker(last_response)
        self.download_worker.moveToThread(self.download_worker_thread)
        
        # Store references
        self._download_worker = self.download_worker
        self._download_worker_thread = self.download_worker_thread
        
        self.download_worker_thread.started.connect(self.download_worker.run)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.error.connect(self.on_download_error)
        self.download_worker.finished.connect(self.download_worker_thread.quit)
        self.download_worker.error.connect(self.download_worker_thread.quit)
        
        def safe_delete_download_worker():
            try:
                if hasattr(self, '_download_worker') and self._download_worker:
                    self._download_worker.deleteLater()
                    self._download_worker = None
            except:
                pass
        
        def safe_delete_download_thread():
            try:
                if hasattr(self, '_download_worker_thread') and self._download_worker_thread:
                    self._download_worker_thread.deleteLater()
                    self._download_worker_thread = None
                    if hasattr(self, 'download_worker_thread'):
                        self.download_worker_thread = None
            except:
                pass
        
        self.download_worker_thread.finished.connect(safe_delete_download_worker)
        self.download_worker_thread.finished.connect(safe_delete_download_thread)
        self.download_worker_thread.start()

    def on_download_finished(self, download_path, basename):
        try:
            self.append_message("system", f"Downloaded to: {basename}")
            QMessageBox.information(self, "Downloaded", f"Saved to:\n{download_path}")
            
            # Clean up reference
            try:
                if hasattr(self, '_download_worker'):
                    self._download_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_download_finished: {traceback.format_exc()}")

    def on_download_error(self, error_msg):
        try:
            error_text = str(error_msg) if error_msg else "Unknown error"
            QMessageBox.warning(self, "Download Error", error_text)
            
            # Clean up reference
            try:
                if hasattr(self, '_download_worker'):
                    self._download_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_download_error: {traceback.format_exc()}")
    
    def export_conversation(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "", "JSON (*.json);;Text (*.txt)")
        if not path:
            return
        # Clean up any existing export worker thread
        try:
            if hasattr(self, 'export_worker_thread') and self.export_worker_thread is not None:
                try:
                    if hasattr(self, 'export_worker') and self.export_worker is not None:
                        try:
                            self.export_worker.finished.disconnect()
                            self.export_worker.error.disconnect()
                        except:
                            pass
                except:
                    pass
        except:
            pass
        
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
        
        # Store references
        self._export_worker = self.export_worker
        self._export_worker_thread = self.export_worker_thread
        
        self.export_worker_thread.started.connect(self.export_worker.run)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.error.connect(self.on_export_error)
        self.export_worker.finished.connect(self.export_worker_thread.quit)
        self.export_worker.error.connect(self.export_worker_thread.quit)
        
        def safe_delete_export_worker():
            try:
                if hasattr(self, '_export_worker') and self._export_worker:
                    self._export_worker.deleteLater()
                    self._export_worker = None
            except:
                pass
        
        def safe_delete_export_thread():
            try:
                if hasattr(self, '_export_worker_thread') and self._export_worker_thread:
                    self._export_worker_thread.deleteLater()
                    self._export_worker_thread = None
                    if hasattr(self, 'export_worker_thread'):
                        self.export_worker_thread = None
            except:
                pass
        
        self.export_worker_thread.finished.connect(safe_delete_export_worker)
        self.export_worker_thread.finished.connect(safe_delete_export_thread)
        self.export_worker_thread.start()

    def on_export_finished(self, path):
        try:
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
            
            # Clean up reference
            try:
                if hasattr(self, '_export_worker'):
                    self._export_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_export_finished: {traceback.format_exc()}")

    def on_export_error(self, error_msg):
        try:
            error_text = str(error_msg) if error_msg else "Unknown error"
            QMessageBox.warning(self, "Export Error", error_text)
            
            # Clean up reference
            try:
                if hasattr(self, '_export_worker'):
                    self._export_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_export_error: {traceback.format_exc()}")
    
    def show_tasks_dialog(self):
        """Show task management dialog"""
        if not TASK_SYSTEM_AVAILABLE:
            QMessageBox.warning(self, "Tasks Unavailable", "Task system is not available.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("🤖 Lea Task Management")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("Available Tasks")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # Task list table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Task Name", "Description", "Confirmation Required", "Status"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        tasks = task_registry.list_tasks()
        table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            table.setItem(row, 0, QTableWidgetItem(task["name"]))
            table.setItem(row, 1, QTableWidgetItem(task["description"]))
            table.setItem(row, 2, QTableWidgetItem("Yes" if task["requires_confirmation"] else "No"))
            status = "✅ Enabled" if task["allowed"] else "❌ Disabled"
            table.setItem(row, 3, QTableWidgetItem(status))
        
        layout.addWidget(table)
        
        # Buttons
        buttons = QHBoxLayout()
        
        enable_btn = QPushButton("Enable Selected")
        enable_btn.clicked.connect(lambda: self.toggle_task_status(table, True))
        buttons.addWidget(enable_btn)
        
        disable_btn = QPushButton("Disable Selected")
        disable_btn.clicked.connect(lambda: self.toggle_task_status(table, False))
        buttons.addWidget(disable_btn)
        
        buttons.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        buttons.addWidget(close_btn)
        
        layout.addLayout(buttons)
        
        # Show task history
        history_group = QGroupBox("Recent Task History")
        history_layout = QVBoxLayout()
        history_text = QTextEdit()
        history_text.setReadOnly(True)
        history_text.setMaximumHeight(150)
        
        history = task_registry.task_history[-10:]  # Last 10 tasks
        if history:
            history_content = "\n".join([
                f"{item['timestamp']}: {item['task_name']} - {'✅' if item['result']['success'] else '❌'} {item['result']['message']}"
                for item in reversed(history)
            ])
            history_text.setText(history_content)
        else:
            history_text.setText("No task history yet.")
        
        history_layout.addWidget(history_text)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        dialog.exec()
    
    def toggle_task_status(self, table, enable):
        """Enable or disable selected task"""
        selected = table.selectedItems()
        if not selected:
            QMessageBox.information(self, "No Selection", "Please select a task from the table.")
            return
        
        row = selected[0].row()
        task_name = table.item(row, 0).text()
        
        if enable:
            task_registry.enable_task(task_name)
        else:
            reply = QMessageBox.question(
                self, "Disable Task",
                f"Are you sure you want to disable '{task_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                task_registry.disable_task(task_name)
        
        # Refresh table
        status = "✅ Enabled" if enable else "❌ Disabled"
        table.setItem(row, 3, QTableWidgetItem(status))
    
    def show_emoji_picker(self):
        """Show emoji picker dialog with search functionality"""
        # Comprehensive emoji library organized by category with search keywords
        emojis_data = {
            "Faces & Emotions": {
                "emojis": ["😊", "😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂", "🙂", "🙃", "😉", "😌", "😍", "🥰", "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜", "🤪", "🤨", "🧐", "🤓", "😎", "🤩", "🥳", "😏", "😒", "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤", "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱", "😨", "😰", "😥", "😓", "🤢", "🤮", "🤧", "🥴", "😴", "🤤", "😪", "😵", "🤐", "🥱", "😷"],
                "keywords": "smile happy laugh grin face emotion feeling sad angry cry"
            },
            "Hand Gestures": {
                "emojis": ["👋", "🤚", "🖐️", "✋", "🖖", "👌", "🤌", "🤏", "✌️", "🤞", "🤟", "🤘", "🤙", "👈", "👉", "👆", "🖕", "👇", "☝️", "👍", "👎", "✊", "👊", "🤛", "🤜", "👏", "🙌", "👐", "🤲", "🤝", "🙏", "✍️", "💪", "🦾", "🦿"],
                "keywords": "hand wave thumbs up clap point gesture fingers"
            },
            "People & Body": {
                "emojis": ["👤", "👥", "🧑", "👨", "👩", "🧑‍🦱", "👨‍🦱", "👩‍🦱", "🧑‍🦰", "👨‍🦰", "👩‍🦰", "👱", "👱‍♂️", "👱‍♀️", "🧑‍🦳", "👨‍🦳", "👩‍🦳", "🧑‍🦲", "👨‍🦲", "👩‍🦲", "🧔", "👵", "🧓", "👴", "👲", "👳", "👳‍♂️", "👳‍♀️", "🧕", "👮", "👮‍♂️", "👮‍♀️", "👷", "👷‍♂️", "👷‍♀️", "💂", "💂‍♂️", "💂‍♀️", "🕵️", "🕵️‍♂️", "🕵️‍♀️", "👩‍⚕️", "👨‍⚕️", "👩‍🌾", "👨‍🌾", "👩‍🍳", "👨‍🍳", "👩‍🎓", "👨‍🎓", "👩‍🎤", "👨‍🎤", "👩‍🏫", "👨‍🏫", "👩‍🏭", "👨‍🏭", "👩‍💻", "👨‍💻", "👩‍💼", "👨‍💼", "👩‍🔧", "👨‍🔧", "👩‍🔬", "👨‍🔬", "👩‍🎨", "👨‍🎨", "👩‍🚒", "👨‍🚒", "👩‍✈️", "👨‍✈️", "👩‍🚀", "👨‍🚀", "👩‍⚖️", "👨‍⚖️", "🤶", "🎅", "🧙‍♀️", "🧙‍♂️", "🧝‍♀️", "🧝‍♂️", "🧛‍♀️", "🧛‍♂️", "🧟‍♀️", "🧟‍♂️", "🧞‍♀️", "🧞‍♂️", "🧜‍♀️", "🧜‍♂️", "🧚‍♀️", "🧚‍♂️", "👼", "🤰", "🤱", "👶", "🧒", "👦", "👧", "🧑", "👨", "👩", "👨‍👩‍👧", "👨‍👩‍👧‍👦", "👨‍👩‍👦‍👦", "👨‍👩‍👧‍👧", "👨‍👨‍👦", "👨‍👨‍👧", "👨‍👨‍👧‍👦", "👨‍👨‍👦‍👦", "👨‍👨‍👧‍👧", "👩‍👩‍👦", "👩‍👩‍👧", "👩‍👩‍👧‍👦", "👩‍👩‍👦‍👦", "👩‍👩‍👧‍👧"],
                "keywords": "person people body man woman child family"
            },
            "Animals & Nature": {
                "emojis": ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮", "🐷", "🐽", "🐸", "🐵", "🙈", "🙉", "🙊", "🐒", "🐔", "🐧", "🐦", "🐤", "🐣", "🐥", "🦆", "🦅", "🦉", "🦇", "🐺", "🐗", "🐴", "🦄", "🐝", "🐛", "🦋", "🐌", "🐞", "🐜", "🦟", "🦗", "🕷️", "🦂", "🐢", "🐍", "🦎", "🦖", "🦕", "🐙", "🦑", "🦐", "🦞", "🦀", "🐡", "🐠", "🐟", "🐬", "🐳", "🐋", "🦈", "🐊", "🐅", "🐆", "🦓", "🦍", "🦧", "🐘", "🦛", "🦏", "🐪", "🐫", "🦒", "🦘", "🦬", "🐃", "🐂", "🐄", "🐎", "🐖", "🐏", "🐑", "🦙", "🐐", "🦌", "🐕", "🐩", "🦮", "🐕‍🦺", "🐈", "🐓", "🦃", "🦤", "🦚", "🦜", "🦢", "🦩", "🕊️", "🐇", "🦝", "🦨", "🦡", "🦫", "🦦", "🦥", "🐁", "🐀", "🐿️"],
                "keywords": "animal dog cat bird nature pet wildlife"
            },
            "Food & Drink": {
                "emojis": ["🍏", "🍎", "🍐", "🍊", "🍋", "🍌", "🍉", "🍇", "🍓", "🍈", "🍒", "🍑", "🥭", "🍍", "🥥", "🥝", "🍅", "🍆", "🥑", "🥦", "🥬", "🥒", "🌶️", "🌽", "🥕", "🥔", "🍠", "🥐", "🥯", "🍞", "🥖", "🥨", "🧀", "🥚", "🍳", "🥞", "🥓", "🥩", "🍗", "🍖", "🦴", "🌭", "🍔", "🍟", "🍕", "🥪", "🥙", "🌮", "🌯", "🥗", "🥘", "🥫", "🍝", "🍜", "🍲", "🍛", "🍣", "🍱", "🥟", "🦪", "🍤", "🍙", "🍚", "🍘", "🍥", "🥠", "🥮", "🍢", "🍡", "🍧", "🍨", "🍦", "🥧", "🧁", "🍰", "🎂", "🍮", "🍭", "🍬", "🍫", "🍿", "🍩", "🍪", "🌰", "🥜", "🍯", "🥛", "🍼", "☕️", "🍵", "🧃", "🥤", "🍶", "🍺", "🍻", "🥂", "🍷", "🥃", "🍸", "🍹", "🧉", "🍾"],
                "keywords": "food drink eat meal snack fruit vegetable pizza burger"
            },
            "Travel & Places": {
                "emojis": ["🏔️", "⛰️", "🌋", "🗻", "🏕️", "🏖️", "🏜️", "🏝️", "🏞️", "🏟️", "🏛️", "🏗️", "🧱", "🏘️", "🏚️", "🏠", "🏡", "🏢", "🏣", "🏤", "🏥", "🏦", "🏨", "🏩", "🏪", "🏫", "🏬", "🏭", "🏯", "🏰", "💒", "🗼", "🗽", "⛪️", "🕌", "🛕", "🕍", "⛩️", "🕋", "⛲️", "⛺️", "🌁", "🌃", "🏙️", "🌄", "🌅", "🌆", "🌇", "🌉", "♨️", "🎠", "🎡", "🎢", "💈", "🎪", "🚂", "🚃", "🚄", "🚅", "🚆", "🚇", "🚈", "🚉", "🚊", "🚝", "🚞", "🚋", "🚌", "🚍", "🚎", "🚐", "🚑", "🚒", "🚓", "🚔", "🚕", "🚖", "🚗", "🚘", "🚙", "🚚", "🚛", "🚜", "🏎️", "🏍️", "🛵", "🦽", "🦼", "🛴", "🚲", "🛺", "🚁", "✈️", "🛩️", "🛫", "🛬", "🪂", "💺", "🚢", "⛴️", "🛥️", "🚤", "⛵️", "🛶", "🚁", "🚟", "🚠", "🚡"],
                "keywords": "travel place location building car plane train bus"
            },
            "Activities & Sports": {
                "emojis": ["⚽️", "🏀", "🏈", "⚾️", "🥎", "🎾", "🏐", "🏉", "🥏", "🎱", "🏓", "🏸", "🏒", "🏑", "🥍", "🏏", "🥅", "⛳️", "🪁", "🏹", "🎣", "🤿", "🥊", "🥋", "🎽", "🛹", "🛷", "⛸️", "🥌", "🎿", "⛷️", "🏂", "🪂", "🏋️‍♀️", "🏋️", "🏋️‍♂️", "🤼‍♀️", "🤼", "🤼‍♂️", "🤸‍♀️", "🤸", "🤸‍♂️", "⛹️‍♀️", "⛹️", "⛹️‍♂️", "🤺", "🤾‍♀️", "🤾", "🤾‍♂️", "🏌️‍♀️", "🏌️", "🏌️‍♂️", "🏇", "🧘‍♀️", "🧘", "🧘‍♂️", "🏄‍♀️", "🏄", "🏄‍♂️", "🏊‍♀️", "🏊", "🏊‍♂️", "🤽‍♀️", "🤽", "🤽‍♂️", "🚣‍♀️", "🚣", "🚣‍♂️", "🧗‍♀️", "🧗", "🧗‍♂️", "🚵‍♀️", "🚵", "🚵‍♂️", "🚴‍♀️", "🚴", "🚴‍♂️", "🏆", "🥇", "🥈", "🥉", "🏅", "🎖️", "🏵️", "🎗️", "🎫", "🎟️", "🎪", "🤹‍♀️", "🤹", "🤹‍♂️"],
                "keywords": "sport activity game exercise fitness ball play"
            },
            "Objects & Tech": {
                "emojis": ["⌚️", "📱", "📲", "💻", "⌨️", "🖥️", "🖨️", "🖱️", "🖲️", "🕹️", "🗜️", "💾", "💿", "📀", "📼", "📷", "📸", "📹", "🎥", "📽️", "🎞️", "📞", "☎️", "📟", "📠", "📺", "📻", "🎙️", "🎚️", "🎛️", "⏱️", "⏲️", "⏰", "🕰️", "⌛️", "⏳", "📡", "🔋", "🔌", "💡", "🔦", "🕯️", "🧯", "🛢️", "💸", "💵", "💴", "💶", "💷", "💰", "💳", "💎", "⚖️", "🪜", "🧰", "🪛", "🔧", "🔨", "⚒️", "🛠️", "⛏️", "🪚", "🔩", "⚙️", "🪤", "🧱", "⛓️", "🧲", "🔫", "💣", "🧨", "🪓", "🔪", "🗡️", "⚔️", "🛡️", "🚬", "⚰️", "🪦", "⚱️", "🏺", "🔮", "📿", "🧿", "💈", "⚗️", "🔭", "🔬", "🕳️", "🩹", "🩺", "💊", "💉", "🩸", "🧬", "🦠", "🧫", "🧪", "🌡️", "🧹", "🪠", "🧺", "🧻", "🚽", "🚰", "🚿", "🛁", "🛀", "🧼", "🪥", "🪒", "🧽", "🪣", "🧴", "🛎️", "🔑", "🗝️", "🚪", "🪑", "🛋️", "🛏️", "🛌", "🧸", "🪆", "🖼️", "🪞", "🪟", "🛍️", "🛒", "🎁", "🎈", "🎏", "🎀", "🪄", "🪅", "🎊", "🎉", "🎎", "🏮", "🎐", "🧧", "✉️", "📩", "📨", "📧", "💌", "📥", "📤", "📦", "🏷️", "🪧", "📪", "📫", "📬", "📭", "📮", "📯", "📜", "📃", "📄", "📑", "🧾", "📊", "📈", "📉", "🗒️", "🗓️", "📆", "📅", "🗑️", "📇", "🗃️", "🗳️", "🗄️", "📋", "📁", "📂", "🗂️", "🗞️", "📰", "📓", "📔", "📒", "📕", "📗", "📘", "📙", "📚", "📖", "🔖", "🧷", "🔗", "📎", "🖇️", "📐", "📏", "🧮", "📌", "📍", "✂️", "🖊️", "🖋️", "✒️", "🖌️", "🖍️", "📝", "✏️", "🔍", "🔎", "🔏", "🔐", "🔒", "🔓"],
                "keywords": "computer phone watch money tech device tool office"
            },
            "Symbols & Signs": {
                "emojis": ["❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎", "💔", "❣️", "💕", "💞", "💓", "💗", "💖", "💘", "💝", "💟", "☮️", "✝️", "☪️", "🕉️", "☸️", "✡️", "🔯", "🕎", "☯️", "☦️", "🛐", "⛎", "♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓", "🆔", "⚛️", "✅", "☑️", "✔️", "❌", "❎", "➖", "➗", "➕", "➰", "➿", "〽️", "✳️", "✴️", "❇️", "‼️", "⁉️", "❓", "❔", "❕", "❗", "🔅", "🔆", "〰️", "©️", "®️", "™️", "#️⃣", "*️⃣", "0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟", "🔠", "🔡", "🔢", "🔣", "🔤", "🅰️", "🆎", "🅱️", "🆑", "🆒", "🆓", "ℹ️", "🆔", "Ⓜ️", "🆕", "🆖", "🅾️", "🆗", "🅿️", "🆘", "🆙", "🆚", "🈁", "🈂️", "🈷️", "🈶", "🈯", "🉐", "🈹", "🈲", "🉑", "🈸", "🈴", "🈳", "㊗️", "㊙️", "🈺", "🈵", "🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "⚫", "⚪", "🟤", "🔶", "🔷", "🔸", "🔹", "🔺", "🔻", "💠", "🔘", "🔳", "🔲"],
                "keywords": "heart love symbol star sign zodiac check mark question"
            },
            "Flags": {
                "emojis": ["🏳️", "🏴", "🏁", "🚩", "🏳️‍🌈", "🏳️‍⚧️", "🇺🇳", "🇦🇫", "🇦🇽", "🇦🇱", "🇩🇿", "🇦🇸", "🇦🇩", "🇦🇴", "🇦🇮", "🇦🇶", "🇦🇬", "🇦🇷", "🇦🇲", "🇦🇼", "🇦🇺", "🇦🇹", "🇦🇿", "🇧🇸", "🇧🇭", "🇧🇩", "🇧🇧", "🇧🇾", "🇧🇪", "🇧🇿", "🇧🇯", "🇧🇲", "🇧🇹", "🇧🇴", "🇧🇦", "🇧🇼", "🇧🇷", "🇮🇴", "🇻🇬", "🇧🇳", "🇧🇬", "🇧🇫", "🇧🇮", "🇰🇭", "🇨🇲", "🇨🇦", "🇮🇶", "🇮🇸", "🇮🇷", "🇮🇪", "🇮🇲", "🇮🇱", "🇮🇹", "🇯🇲", "🇯🇵", "🇯🇪", "🇯🇴", "🇰🇿", "🇰🇪", "🇰🇮", "🇰🇼", "🇰🇬", "🇱🇦", "🇱🇻", "🇱🇧", "🇱🇸", "🇱🇷", "🇱🇾", "🇱🇮", "🇱🇹", "🇱🇺", "🇲🇴", "🇲🇬", "🇲🇼", "🇲🇾", "🇲🇻", "🇲🇱", "🇲🇹", "🇲🇭", "🇲🇶", "🇲🇷", "🇲🇺", "🇾🇹", "🇲🇽", "🇫🇲", "🇲🇩", "🇲🇨", "🇲🇳", "🇲🇪", "🇲🇸", "🇲🇦", "🇲🇿", "🇲🇲", "🇳🇦", "🇳🇷", "🇳🇵", "🇳🇱", "🇳🇨", "🇳🇿", "🇳🇮", "🇳🇪", "🇳🇬", "🇳🇺", "🇳🇫", "🇰🇵", "🇲🇰", "🇲🇵", "🇳🇴", "🇴🇲", "🇵🇰", "🇵🇼", "🇵🇸", "🇵🇦", "🇵🇬", "🇵🇾", "🇵🇪", "🇵🇭", "🇵🇳", "🇵🇱", "🇵🇹", "🇵🇷", "🇶🇦", "🇷🇪", "🇷🇴", "🇷🇺", "🇷🇼", "🇼🇸", "🇸🇲", "🇸🇦", "🇸🇳", "🇷🇸", "🇸🇨", "🇸🇱", "🇸🇬", "🇸🇰", "🇸🇮", "🇬🇸", "🇸🇧", "🇸🇴", "🇿🇦", "🇰🇷", "🇸🇸", "🇪🇸", "🇱🇰", "🇧🇱", "🇸🇭", "🇰🇳", "🇱🇨", "🇵🇲", "🇻🇨", "🇸🇩", "🇸🇷", "🇸🇪", "🇨🇭", "🇸🇾", "🇹🇼", "🇹🇯", "🇹🇿", "🇹🇭", "🇹🇱", "🇹🇬", "🇹🇰", "🇹🇴", "🇹🇹", "🇹🇳", "🇹🇷", "🇹🇲", "🇹🇨", "🇹🇻", "🇻🇮", "🇺🇬", "🇺🇦", "🇦🇪", "🇬🇧", "🇺🇸", "🇺🇾", "🇺🇿", "🇻🇺", "🇻🇦", "🇻🇪", "🇻🇳", "🇼🇫", "🇪🇭", "🇾🇪", "🇿🇲", "🇿🇼", "🏴‍☠️"],
                "keywords": "flag country nation world country"
            },
            "Weather & Nature": {
                "emojis": ["☀️", "🌤️", "⛅️", "🌥️", "☁️", "🌦️", "🌧️", "⛈️", "🌩️", "⚡️", "☔️", "❄️", "☃️", "⛄️", "🌨️", "💨", "💧", "💦", "☂️", "☔️", "🌊", "🌫️", "🔥", "⭐", "🌟", "💫", "✨", "⚡", "☄️", "💥", "🌙", "🌚", "🌛", "🌜", "🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘", "🌍", "🌎", "🌏", "🌐", "🗺️", "🧭", "🏔️", "⛰️", "🌋", "🗻", "🏕️", "🏖️", "🏜️", "🏝️", "🏞️", "🌲", "🌳", "🌴", "🌵", "🌾", "🌿", "☘️", "🍀", "🍁", "🍂", "🍃", "🌺", "🌻", "🌹", "🌷", "🌱", "🌼", "🌸", "💐", "🌾", "🌷", "🥀", "🌹", "🌻", "🌺", "🌼", "🌸", "🌿", "🌱", "🍃", "🍂", "🍁", "🍀", "☘️"],
                "keywords": "weather sun rain snow cloud star moon earth nature"
            },
        }
        
        dialog = QDialog(self)
        dialog.setWindowTitle("😊 Emoji Picker")
        dialog.setMinimumSize(450, 600)
        dialog.setMaximumSize(600, 700)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #333;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Select an emoji:")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #FFF; margin-bottom: 5px;")
        layout.addWidget(title)
        
        # Search box
        search_box = QLineEdit()
        search_box.setPlaceholderText("🔍 Search emojis...")
        search_box.setStyleSheet("""
            QLineEdit {
                background-color: #222;
                color: #FFF;
                border: 2px solid #555;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #0078D7;
            }
        """)
        layout.addWidget(search_box)
        
        # Emoji list
        list_widget = QListWidget()
        list_widget.setStyleSheet("""
            QListWidget {
                background-color: #222;
                color: #FFF;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 28px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
                min-height: 40px;
            }
            QListWidget::item:hover {
                background-color: #444;
            }
            QListWidget::item:selected {
                background-color: #0078D7;
            }
        """)
        
        # Store all emoji items with their data for filtering
        all_items = []
        
        def populate_list(filter_text=""):
            """Populate the list with emojis, optionally filtered"""
            nonlocal all_items, list_widget
            list_widget.clear()
            all_items.clear()
            filter_lower = filter_text.lower().strip()
            
            for category, data in emojis_data.items():
                emoji_list = data["emojis"]
                keywords = data["keywords"]
                
                # Filter: show category if search matches category name or keywords, or if no filter
                show_category = (not filter_lower or 
                               category.lower() in filter_lower or 
                               keywords.lower() in filter_lower or
                               any(emoji in filter_text for emoji in emoji_list))
                
                if show_category:
                    # Add category header
                    category_item = QListWidgetItem(f"  {category}")
                    # Disable selection and enable for category headers
                    from PyQt6.QtCore import Qt
                    category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
                    category_item.setBackground(QColor(80, 80, 80))
                    category_item.setForeground(QColor(200, 200, 200))
                    list_widget.addItem(category_item)
                    all_items.append(category_item)
                    
                    # Add emojis in this category (show all emojis when category matches search)
                    for emoji in emoji_list:
                        emoji_item = QListWidgetItem(emoji)
                        emoji_item.setData(Qt.ItemDataRole.UserRole, emoji)
                        emoji_item.setData(Qt.ItemDataRole.UserRole + 1, category)  # Store category
                        list_widget.addItem(emoji_item)
                        all_items.append(emoji_item)
        
        # Initial population
        populate_list()
        
        # Search functionality
        def on_search_changed(text):
            populate_list(text)
        
        search_box.textChanged.connect(on_search_changed)
        
        # Buttons
        buttons = QHBoxLayout()
        
        def insert_emoji():
            current_item = list_widget.currentItem()
            if current_item:
                emoji = current_item.data(Qt.ItemDataRole.UserRole)
                if emoji:
                    # Insert emoji at cursor position
                    cursor = self.input_box.textCursor()
                    cursor.insertText(emoji)
                    self.input_box.setTextCursor(cursor)
                    dialog.accept()
        
        list_widget.itemDoubleClicked.connect(insert_emoji)
        
        # Allow Enter key to insert selected emoji
        def on_enter_pressed():
            insert_emoji()
        
        search_box.returnPressed.connect(lambda: list_widget.setFocus() if list_widget.count() > 0 else None)
        list_widget.itemActivated.connect(insert_emoji)
        
        insert_btn = QPushButton("Insert")
        insert_btn.clicked.connect(insert_emoji)
        insert_btn.setStyleSheet("background-color: #0078D7; padding: 8px 16px; border-radius: 4px; font-weight: 600; font-size: 14px;")
        buttons.addWidget(insert_btn)
        
        buttons.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("background-color: #555; padding: 8px 16px; border-radius: 4px; font-size: 14px;")
        buttons.addWidget(close_btn)
        
        layout.addWidget(list_widget, stretch=1)
        layout.addLayout(buttons)
        
        # Set focus to search box
        search_box.setFocus()
        
        dialog.exec()
    
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
    
    def toggle_speech_recognition(self):
        """Start or stop speech recognition"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            QMessageBox.warning(self, "Speech Recognition Unavailable", 
                              "Speech recognition is not available. Please install with:\npip install SpeechRecognition")
            return
        
        if self.is_listening:
            # Stop listening (currently active)
            self.status_label.setText("Ready")
            self.mic_btn.setText("🎤")
            self.mic_btn.setToolTip("Click to speak your message")
            self.is_listening = False
            return
        
        # Check if microphone setup has been done before
        # Only prompt on first use, not if user has already configured it (even if None/default)
        needs_setup = False
        try:
            # Check if settings file exists and has microphone_device_index key
            # If key doesn't exist, user hasn't set up microphone yet
            if not self.settings_file.exists():
                needs_setup = True
            else:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    if 'microphone_device_index' not in settings_data:
                        needs_setup = True
        except (json.JSONDecodeError, IOError, Exception):
            # If error reading settings, assume first time
            needs_setup = True
        
        if needs_setup:
            if not self._setup_microphone_first_time():
                return  # User cancelled microphone setup
        
        # Start listening
        self.is_listening = True
        self.mic_btn.setText("🔴")
        self.mic_btn.setToolTip("Listening... Click again to stop")
        
        # Clean up any existing speech worker thread
        try:
            if hasattr(self, 'speech_worker_thread') and self.speech_worker_thread is not None:
                try:
                    if hasattr(self, 'speech_worker') and self.speech_worker is not None:
                        try:
                            self.speech_worker.finished.disconnect()
                            self.speech_worker.error.disconnect()
                            self.speech_worker.listening.disconnect()
                        except:
                            pass
                except:
                    pass
        except:
            pass
        
        # Start speech recognition worker thread
        self.speech_worker_thread = QThread()
        self.speech_worker = SpeechRecognitionWorker(device_index=self.microphone_device_index)
        self.speech_worker.moveToThread(self.speech_worker_thread)
        
        # Store references
        self._speech_worker = self.speech_worker
        self._speech_worker_thread = self.speech_worker_thread
        
        self.speech_worker_thread.started.connect(self.speech_worker.run)
        self.speech_worker.finished.connect(self.on_speech_recognition_finished)
        self.speech_worker.error.connect(self.on_speech_recognition_error)
        self.speech_worker.listening.connect(self.on_speech_listening)
        self.speech_worker.finished.connect(self.speech_worker_thread.quit)
        self.speech_worker.error.connect(self.speech_worker_thread.quit)
        
        def safe_delete_speech_worker():
            try:
                if hasattr(self, '_speech_worker') and self._speech_worker:
                    self._speech_worker.deleteLater()
                    self._speech_worker = None
            except:
                pass
        
        def safe_delete_speech_thread():
            try:
                if hasattr(self, '_speech_worker_thread') and self._speech_worker_thread:
                    self._speech_worker_thread.deleteLater()
                    self._speech_worker_thread = None
                    if hasattr(self, 'speech_worker_thread'):
                        self.speech_worker_thread = None
            except:
                pass
        
        self.speech_worker_thread.finished.connect(safe_delete_speech_worker)
        self.speech_worker_thread.finished.connect(safe_delete_speech_thread)
        self.speech_worker_thread.start()
    
    def _setup_microphone_first_time(self):
        """Prompt user to select a microphone for first-time setup"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return False
        
        reply = QMessageBox.question(
            self, 
            "Microphone Setup", 
            "No microphone selected.\n\nWould you like to select a microphone now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            QMessageBox.information(
                self,
                "Microphone Setup",
                "You can configure your microphone later in Settings (⚙️ button)."
            )
            return False
        
        # Show microphone selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("🎤 Select Microphone")
        dialog.setMinimumSize(500, 300)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #333;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Select your microphone:")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #FFF; margin-bottom: 10px;")
        layout.addWidget(title)
        
        info_label = QLabel("Please select the microphone you want to use for voice input:")
        info_label.setStyleSheet("color: #CCC; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        mic_combo = QComboBox()
        mic_combo.setStyleSheet("""
            QComboBox {
                background-color: #222;
                color: #FFF;
                border: 2px solid #555;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        
        mic_devices = []
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                import speech_recognition as sr
                mic_list = sr.Microphone.list_microphone_names()
                if mic_list:
                    mic_combo.addItem("Default (System Default)", None)
                    for i, mic_name in enumerate(mic_list):
                        mic_combo.addItem(f"{i}: {mic_name}", i)
                        mic_devices.append((i, mic_name))
                else:
                    QMessageBox.warning(
                        self,
                        "No Microphones Found",
                        "No microphones were detected on your system.\n\n"
                        "Please check:\n"
                        "1. Your microphone is connected\n"
                        "2. Microphone permissions are granted\n"
                        "3. Windows Privacy settings allow microphone access"
                    )
                    dialog.close()
                    return False
            except Exception as e:
                logging.error(f"Error listing microphones: {e}")
                QMessageBox.warning(
                    self,
                    "Microphone Error",
                    f"Error detecting microphones:\n{str(e)}\n\n"
                    "Please check your microphone connection and try again."
                )
                dialog.close()
                return False
        else:
            QMessageBox.warning(self, "Speech Recognition Unavailable", 
                              "Speech recognition is not available. Please install with:\npip install SpeechRecognition")
            dialog.close()
            return False
        
        layout.addWidget(mic_combo)
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: #FFF;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        layout.addWidget(buttons)
        
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            selected_index = mic_combo.currentData()
            self.microphone_device_index = selected_index
            self.save_settings()
            
            QMessageBox.information(
                self,
                "Microphone Selected",
                f"Microphone selected successfully!\n\n"
                f"Selected: {mic_combo.currentText()}\n\n"
                "You can change this later in Settings (⚙️ button)."
            )
            return True
        else:
            return False
    
    def on_speech_listening(self):
        """Called when speech recognition starts listening"""
        self.status_label.setText("🎤 Listening... Speak now")
    
    def on_speech_recognition_finished(self, text):
        """Called when speech is successfully recognized"""
        try:
            # Reset listening state
            self.is_listening = False
            self.mic_btn.setText("🎤")
            self.mic_btn.setToolTip("Click to speak your message")
            
            # Insert recognized text into input box
            if text and text.strip():
                # If there's existing text, append with a space
                current_text = self.input_box.toPlainText()
                if current_text and current_text.strip():
                    self.input_box.setPlainText(current_text + " " + text.strip())
                else:
                    self.input_box.setPlainText(text.strip())
                
                # Move cursor to end
                cursor = self.input_box.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.input_box.setTextCursor(cursor)
                
                self.status_label.setText("Ready")
            else:
                self.status_label.setText("No speech detected")
            
            # Clean up reference
            try:
                if hasattr(self, '_speech_worker'):
                    self._speech_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_speech_recognition_finished: {traceback.format_exc()}")
            self.status_label.setText("Error processing speech")
    
    def on_speech_recognition_error(self, error_msg):
        """Called when speech recognition encounters an error"""
        try:
            # Reset listening state
            self.is_listening = False
            self.mic_btn.setText("🎤")
            self.mic_btn.setToolTip("Click to speak your message")
            
            error_text = str(error_msg) if error_msg else "Unknown error"
            self.status_label.setText(f"Speech error: {error_text}")
            
            # Only show message box for significant errors, not for "try again" type messages
            if "not available" in error_text.lower() or "service error" in error_text.lower():
                QMessageBox.warning(self, "Speech Recognition Error", error_text)
            
            # Clean up reference
            try:
                if hasattr(self, '_speech_worker'):
                    self._speech_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_speech_recognition_error: {traceback.format_exc()}")
            try:
                self.status_label.setText("Error handling speech recognition")
            except:
                pass
    
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
        try:
            if not text:
                text = "(empty message)"
            
            if kind == "user":
                label, color = "Dre", self.USER_COLOR
            elif kind == "assistant":
                label, color = "Lea", self.ASSIST_COLOR
            else:
                label, color = "System", self.SYSTEM_COLOR

            # Ensure text is always a string and safe for HTML
            safe = html.escape(str(text)).replace("\n", "<br>")
            html_block = f'<div style="margin: 6px 0;"><span style="color:{color}; font-weight:600;">{label}:</span> <span style="color:{color};">{safe}</span></div>'
            
            # Ensure we're on the main thread (Qt requirement)
            if hasattr(self, 'chat_display') and self.chat_display:
                self.chat_display.append(html_block)
        except Exception as e:
            logging.error(f"Error appending message: {traceback.format_exc()}")
            # Fallback to plain text if HTML fails
            try:
                if hasattr(self, 'chat_display') and self.chat_display:
                    self.chat_display.append(f"{label}: {str(text)}")
            except:
                pass
    
    def on_send(self):
        try:
            # Prevent multiple simultaneous requests
            # Safely check if worker thread is running (handle deleted thread gracefully)
            try:
                if hasattr(self, 'worker_thread') and self.worker_thread is not None:
                    # Check if thread still exists before accessing it
                    thread_exists = True
                    try:
                        is_running = self.worker_thread.isRunning()
                    except RuntimeError:
                        # Thread has been deleted
                        thread_exists = False
                        self.worker_thread = None
                    
                    if thread_exists and is_running:
                        QMessageBox.information(self, "Please Wait", "A request is already in progress. Please wait for it to complete.")
                        return
            except Exception as thread_check_error:
                # Thread was deleted, just continue
                logging.warning(f"Thread check failed (likely deleted): {thread_check_error}")
                if hasattr(self, 'worker_thread'):
                    self.worker_thread = None
            
            text = self.input_box.toPlainText().strip()
            if not text:
                return
            
            if not openai_client:
                QMessageBox.warning(self, "API Key Missing", 
                                  "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
                return
            
            # Validate mode and model
            if self.current_mode not in AGENTS:
                QMessageBox.warning(self, "Invalid Mode", f"Selected mode '{self.current_mode}' is not valid.")
                return
            
            if self.current_model not in MODEL_OPTIONS:
                QMessageBox.warning(self, "Invalid Model", f"Selected model '{self.current_model}' is not valid.")
                return
            
            self.append_message("user", text)
            self.input_box.clear()
            self.status_label.setText("Thinking...")
            
            # Use file content only if checkbox is checked
            file_content = None
            if hasattr(self, 'include_file_cb') and self.include_file_cb.isChecked():
                file_content = self.current_file_content
            
            # Ensure message_history is a list
            if not isinstance(self.message_history, list):
                self.message_history = []
            
            # Clean up any existing thread references first - CRITICAL to prevent duplicate responses
            try:
                # Stop any running worker thread
                if hasattr(self, 'worker_thread') and self.worker_thread is not None:
                    try:
                        # Disconnect ALL signals first
                        if hasattr(self, 'worker') and self.worker is not None:
                            try:
                                self.worker.finished.disconnect()
                                self.worker.error.disconnect()
                                self.worker.stream_chunk.disconnect()
                                self.worker.memory_context.disconnect()
                            except (TypeError, RuntimeError):
                                # Signals already disconnected or object deleted
                                pass
                        
                        # Disconnect thread signals
                        try:
                            self.worker_thread.started.disconnect()
                            self.worker_thread.finished.disconnect()
                        except (TypeError, RuntimeError):
                            pass
                        
                        # Stop and wait for thread to finish
                        if self.worker_thread.isRunning():
                            self.worker_thread.quit()
                            self.worker_thread.wait(1000)  # Wait up to 1 second
                        
                        # Clean up old worker
                        if hasattr(self, '_current_worker') and self._current_worker:
                            try:
                                self._current_worker.deleteLater()
                            except:
                                pass
                            self._current_worker = None
                        
                        # Clean up old thread
                        if hasattr(self, '_current_thread') and self._current_thread:
                            try:
                                self._current_thread.deleteLater()
                            except:
                                pass
                            self._current_thread = None
                        
                        self.worker = None
                        self.worker_thread = None
                    except (RuntimeError, AttributeError):
                        # Thread/worker already deleted
                        self.worker = None
                        self.worker_thread = None
                        self._current_worker = None
                        self._current_thread = None
            except Exception as cleanup_error:
                logging.warning(f"Error during cleanup: {cleanup_error}")
                # Reset everything to be safe
                self.worker = None
                self.worker_thread = None
                self._current_worker = None
                self._current_thread = None
            
            # Reset streaming state BEFORE creating new worker
            self.current_streaming_response = ""
            self.is_streaming = False  # Set to False initially, will be set to True after worker starts
            self.streaming_message_started = False
            self.streaming_cursor_position = None
            self.streaming_message_count = 0  # Reset count
            
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
                text,
                self.memory_system  # Pass memory system
            )
            self.worker.moveToThread(self.worker_thread)
            
            # Store references to prevent garbage collection
            self._current_worker = self.worker
            self._current_thread = self.worker_thread
            
            # Connect signals - connect each signal only once
            self.worker_thread.started.connect(self.worker.run)
            self.worker.stream_chunk.connect(self.on_stream_chunk)  # Handle streaming
            self.worker.memory_context.connect(self.on_memory_context)  # Handle memory context
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.error.connect(self.on_worker_error)
            # Connect thread quit to worker signals - but use lambda to avoid duplicate connections
            def quit_thread_on_finished():
                if hasattr(self, 'worker_thread') and self.worker_thread:
                    self.worker_thread.quit()
            
            def quit_thread_on_error():
                if hasattr(self, 'worker_thread') and self.worker_thread:
                    self.worker_thread.quit()
            
            self.worker.finished.connect(quit_thread_on_finished)
            self.worker.error.connect(quit_thread_on_error)
            
            # Set streaming flag AFTER everything is connected
            self.is_streaming = True
            
            # Use a lambda to safely handle deletion
            def safe_delete_worker():
                try:
                    if hasattr(self, '_current_worker') and self._current_worker:
                        self._current_worker.deleteLater()
                        self._current_worker = None
                except:
                    pass
            
            def safe_delete_thread():
                try:
                    if hasattr(self, '_current_thread') and self._current_thread:
                        self._current_thread.deleteLater()
                        self._current_thread = None
                        if hasattr(self, 'worker_thread'):
                            self.worker_thread = None
                except:
                    pass
            
            self.worker_thread.finished.connect(safe_delete_worker)
            self.worker_thread.finished.connect(safe_delete_thread)
            self.worker_thread.start()
            
        except Exception as e:
            error_msg = f"Error sending message: {str(e)}"
            logging.error(f"Error in on_send: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", error_msg)
            self.status_label.setText("Error")

    def on_stream_chunk(self, chunk: str):
        """Handle streaming response chunks - simpler and more reliable approach"""
        # CRITICAL: Only process if we're actually streaming and this chunk is for current request
        if not self.is_streaming:
            return  # Ignore chunks from old/stopped requests
        
        try:
            # Accumulate the full response so far
            self.current_streaming_response += chunk
            safe_text = html.escape(self.current_streaming_response).replace("\n", "<br>")
            
            # Get current HTML
            html_content = self.chat_display.toHtml()
            lea_pattern = f'<span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span>'
            
            # Start the message on first chunk
            if not self.streaming_message_started:
                self.streaming_message_started = True
                # Append new message block
                new_lea_block = f'<div style="margin: 6px 0;"><span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span> <span style="color:{self.ASSIST_COLOR};">{safe_text}</span></div>'
                self.chat_display.append(new_lea_block)
            else:
                # Replace the last Lea message - simpler approach using rsplit
                if lea_pattern in html_content:
                    # Find the last occurrence of "Lea:" and replace everything after it to the next </div>
                    parts = html_content.rsplit(lea_pattern, 1)  # Split from end, get last occurrence
                    
                    if len(parts) == 2:
                        # parts[0] = everything before last "Lea:"
                        # parts[1] = everything after last "Lea:" (including old content)
                        
                        # Find where the content ends (next </div>)
                        after_lea_content = parts[1]
                        div_end_pos = after_lea_content.find('</div>')
                        
                        if div_end_pos > 0:
                            # Found </div> - replace content between "Lea:" and "</div>"
                            before = parts[0] + lea_pattern  # Everything up to and including "Lea:"
                            after = after_lea_content[div_end_pos:]  # Everything from </div> onwards
                            
                            # New content: "Lea: [full accumulated text]"
                            new_content = f' <span style="color:{self.ASSIST_COLOR};">{safe_text}</span>'
                            
                            # Build complete HTML
                            new_html = before + new_content + after
                            
                            try:
                                self.chat_display.setHtml(new_html)
                            except Exception as html_error:
                                logging.error(f"setHtml failed: {html_error}")
                                pass
                        else:
                            # No </div> found - might be at end of document
                            before = parts[0] + lea_pattern
                            new_content = f' <span style="color:{self.ASSIST_COLOR};">{safe_text}</span></div>'
                            new_html = before + new_content + parts[1]
                            try:
                                self.chat_display.setHtml(new_html)
                            except:
                                pass
            
            # Scroll to bottom
            try:
                self.chat_display.verticalScrollBar().setValue(
                    self.chat_display.verticalScrollBar().maximum()
                )
            except:
                pass
        except Exception as e:
            logging.error(f"Error handling stream chunk: {traceback.format_exc()}")
            # Log error but don't crash - response will be fixed on finish
            try:
                if not self.streaming_message_started:
                    safe_text = html.escape(self.current_streaming_response).replace("\n", "<br>")
                    new_lea_block = f'<div style="margin: 6px 0;"><span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span> <span style="color:{self.ASSIST_COLOR};">{safe_text}</span></div>'
                    self.chat_display.append(new_lea_block)
                    self.streaming_message_started = True
            except:
                pass
    
    def on_memory_context(self, context_msg: str):
        """Handle memory context information"""
        try:
            # Optionally show memory context in status or log it
            logging.info(f"Memory context: {context_msg}")
            # You could also show this in the UI if desired
        except Exception as e:
            logging.warning(f"Error handling memory context: {e}")
    
    def on_worker_finished(self, answer, status):
        try:
            # If we were streaming, ensure final message is displayed correctly with full text
            if self.is_streaming and self.current_streaming_response:
                # Force final update of streaming message to ensure it's complete
                final_text = self.current_streaming_response.strip()
                if final_text:
                    safe_text = html.escape(final_text).replace("\n", "<br>")
                    html_content = self.chat_display.toHtml()
                    lea_pattern = f'<span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span>'
                    
                    if lea_pattern in html_content:
                        # Use same simple approach as in on_stream_chunk
                        parts = html_content.rsplit(lea_pattern, 1)
                        if len(parts) == 2:
                            after_lea_content = parts[1]
                            div_end_pos = after_lea_content.find('</div>')
                            
                            if div_end_pos > 0:
                                before = parts[0] + lea_pattern
                                after = after_lea_content[div_end_pos:]
                                new_content = f' <span style="color:{self.ASSIST_COLOR};">{safe_text}</span>'
                                new_html = before + new_content + after
                                try:
                                    self.chat_display.setHtml(new_html)
                                except:
                                    pass
            
            self.is_streaming = False
            self.streaming_message_started = False
            
            # If we were streaming, use the accumulated response
            # The message is already displayed via chunks, but we need to ensure it's saved
            if self.current_streaming_response:
                # Use the accumulated streaming response if available
                final_answer = self.current_streaming_response.strip()
                # Make sure it's saved to history if it wasn't already
                if final_answer and self.message_history:
                    # Check if the last message is an assistant message with this content
                    last_msg = self.message_history[-1] if self.message_history else None
                    if not (last_msg and last_msg.get('role') == 'assistant' and last_msg.get('content') == final_answer):
                        # Update or add the assistant response
                        if last_msg and last_msg.get('role') == 'assistant':
                            # Update existing
                            self.message_history[-1]['content'] = final_answer
                        else:
                            # Add new
                            self.message_history.append({"role": "assistant", "content": final_answer})
            elif answer:
                # Non-streaming mode - message should already be in history, but ensure it's there
                if not self.message_history or self.message_history[-1].get('role') != 'assistant':
                    self.append_message("assistant", str(answer))
                    # Ensure it's in history
                    if not (self.message_history and self.message_history[-1].get('role') == 'assistant'):
                        self.message_history.append({"role": "assistant", "content": str(answer)})
            
            # Limit history to last 20 messages
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            
            # Reset streaming state for next time
            self.current_streaming_response = ""
            
            self.status_label.setText(str(status) if status else "Ready")
            # Always save history after receiving a response
            self._save_history()
            
            # Speak response if TTS is enabled
            if self.tts_enabled and answer:
                try:
                    text_to_speak = self.current_streaming_response.strip() if self.current_streaming_response else str(answer)
                    if text_to_speak:
                        def speak_text_gtts():
                            try:
                                tts = gTTS(text=text_to_speak, lang='en')
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                    tts.save(fp.name)
                                    fp.close()
                                    # Play the MP3 file (Windows)
                                    os.startfile(fp.name)
                            except Exception as tts_error:
                                logging.warning(f"gTTS error: {tts_error}")
                        from threading import Thread
                        tts_thread = Thread(target=speak_text_gtts, daemon=True)
                        tts_thread.start()
                except Exception as tts_exception:
                    logging.warning(f"Error with gTTS: {tts_exception}")
            
            # Clean up references after successful completion
            try:
                if hasattr(self, '_current_worker'):
                    self._current_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_worker_finished: {traceback.format_exc()}")
            try:
                self.status_label.setText("Error displaying response")
            except:
                pass

    def on_worker_error(self, error_msg):
        try:
            error_text = str(error_msg) if error_msg else "Unknown error"
            self.append_message("system", f"❌ Error: {error_text}")
            self.status_label.setText("Error")
            # Show user-friendly error dialog
            QMessageBox.warning(self, "Error", 
                              f"An error occurred:\n\n{error_text}\n\nCheck lea_crash.log for details.")
            
            # Clean up references after error
            try:
                if hasattr(self, '_current_worker'):
                    self._current_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_worker_error: {traceback.format_exc()}")
            try:
                self.status_label.setText("Error handling failed")
            except:
                pass
    
    def _save_history(self):
        try:
            # Ensure message_history is a list
            if not isinstance(self.message_history, list):
                self.message_history = []
            
            # Use absolute path in project directory
            history_path = PROJECT_DIR / self.history_file
            # Limit history to last 20 messages
            history = self.message_history[-20:] if len(self.message_history) > 20 else self.message_history.copy()
            
            data = {
                'mode': str(self.current_mode) if self.current_mode else '',
                'model': str(self.current_model) if self.current_model else '',
                'history': history
            }
            
            # Try to save, with better error handling
            try:
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except PermissionError:
                # If permission denied, try alternative location
                try:
                    alt_path = Path.home() / "lea_history.json"
                    with open(alt_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logging.info(f"Saved history to {alt_path} due to permission issue")
                except Exception as e2:
                    logging.warning(f"Could not save history to alternative location: {e2}")
            except OSError as os_err:
                logging.warning(f"File system error saving history: {os_err}")
        except Exception as e:
            logging.error(f"Error saving history: {traceback.format_exc()}")
    
    def _load_history(self):
        try:
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
                
                # Validate loaded data
                if not isinstance(data, dict):
                    logging.warning(f"Invalid history file format: {history_path}")
                    return
                
                # Load mode and model with validation
                loaded_mode = data.get('mode', "General Assistant & Triage")
                if loaded_mode in AGENTS:
                    self.current_mode = loaded_mode
                else:
                    logging.warning(f"Invalid mode in history: {loaded_mode}")
                
                loaded_model = data.get('model', "GPT-4o (default)")
                if loaded_model in MODEL_OPTIONS:
                    self.current_model = loaded_model
                else:
                    logging.warning(f"Invalid model in history: {loaded_model}")
                
                # Load history with validation
                loaded_history = data.get('history', [])
                if isinstance(loaded_history, list):
                    self.message_history = loaded_history
                    # Limit history to last 20 messages
                    if len(self.message_history) > 20:
                        self.message_history = self.message_history[-20:]
                else:
                    logging.warning("Invalid history format in file")
                    self.message_history = []
                
                # Update UI safely
                try:
                    if hasattr(self, 'mode_combo'):
                        self.mode_combo.setCurrentText(self.current_mode)
                    if hasattr(self, 'model_combo'):
                        self.model_combo.setCurrentText(self.current_model)
                    self.append_message("system", f"Loaded previous conversation ({len(self.message_history)} messages)")
                    
                    # Display ALL messages from history (not just last 5)
                    for msg in self.message_history:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get('role')
                        content = msg.get('content', '')
                        if not content:
                            continue
                        try:
                            # Clean up user messages that have file content prefixes
                            if role == 'user' and 'Dre\'s question:' in str(content):
                                # Extract just the user's question part
                                content = str(content).split('Dre\'s question:')[-1].strip()
                                # Also remove file content if present
                                if '=== UPLOADED FILE' in content:
                                    parts = content.split('=== END FILE ===')
                                    if len(parts) > 1:
                                        content = parts[-1].strip()
                                        if content.startswith('Dre\'s question:'):
                                            content = content.replace('Dre\'s question:', '').strip()
                            
                            # Display the message
                            if role == 'user':
                                self.append_message('user', content)
                            elif role == 'assistant':
                                self.append_message('assistant', content)
                            # Skip system messages in history (they're internal)
                        except Exception as msg_error:
                            logging.warning(f"Error displaying message: {msg_error}")
                            continue
                except Exception as ui_error:
                    logging.error(f"Error updating UI: {ui_error}")
                    
            except json.JSONDecodeError as json_err:
                logging.error(f"Invalid JSON in history file: {json_err}")
                # Show welcome message instead
                self.append_message("system", "Welcome! (Previous conversation could not be loaded)")
            except PermissionError:
                logging.warning(f"Permission denied reading history: {history_path}")
            except OSError as os_err:
                logging.warning(f"File system error reading history: {os_err}")
                
        except Exception as e:
            logging.error(f"Error loading history: {traceback.format_exc()}")
            # Continue with defaults
            self.append_message("system", "Welcome! (Error loading previous conversation)")
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tts_enabled = data.get('tts_enabled', False)
                    self.tts_voice_id = data.get('tts_voice_id', None)
                    self.microphone_device_index = data.get('microphone_device_index', None)
        except Exception as e:
            logging.warning(f"Error loading settings: {e}")
            # Use defaults
    
    def save_settings(self):
        """Save settings to file"""
        try:
            data = {
                'tts_enabled': self.tts_enabled,
                'tts_voice_id': self.tts_voice_id,
                'microphone_device_index': self.microphone_device_index
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Error saving settings: {e}")
    
    def show_settings(self):
        """Show settings dialog for TTS voice and microphone selection"""
        dialog = QDialog(self)
        dialog.setWindowTitle("⚙️ Audio Settings")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #333;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Audio Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #FFF; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Text-to-Speech Section
        tts_group = QGroupBox("Text-to-Speech (Lea's Voice)")
        tts_group.setStyleSheet("""
            QGroupBox {
                color: #FFF;
                border: 2px solid #555;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        tts_layout = QVBoxLayout(tts_group)
        
        # TTS Enable checkbox
        tts_enable_cb = QCheckBox("Enable text-to-speech (Lea will speak her responses)")
        tts_enable_cb.setChecked(self.tts_enabled)
        tts_enable_cb.setStyleSheet("color: #FFF; font-size: 14px;")
        tts_layout.addWidget(tts_enable_cb)
        
        if not TTS_AVAILABLE:
            tts_warning = QLabel("⚠️ TTS not available. Install with: pip install pyttsx3")
            tts_warning.setStyleSheet("color: #ff9900; font-size: 12px; margin-top: 5px;")
            tts_layout.addWidget(tts_warning)
            tts_enable_cb.setEnabled(False)
        
        # Voice selection
        voice_label = QLabel("Select Voice:")
        voice_label.setStyleSheet("color: #FFF; font-size: 12px; margin-top: 10px;")
        tts_layout.addWidget(voice_label)
        
        voice_combo = QComboBox()
        voice_combo.setStyleSheet("""
            QComboBox {
                background-color: #222;
                color: #FFF;
                border: 2px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        # gTTS does not support voice selection; show info only
        voice_combo.addItem("gTTS (Google Text-to-Speech) - No voice selection", None)
        voice_combo.setEnabled(False)
        tts_layout.addWidget(voice_combo)
        layout.addWidget(tts_group)
        
        # Microphone Section
        mic_group = QGroupBox("Microphone")
        mic_group.setStyleSheet("""
            QGroupBox {
                color: #FFF;
                border: 2px solid #555;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        mic_layout = QVBoxLayout(mic_group)
        
        mic_label = QLabel("Select Microphone:")
        mic_label.setStyleSheet("color: #FFF; font-size: 12px;")
        mic_layout.addWidget(mic_label)
        
        mic_combo = QComboBox()
        mic_combo.setStyleSheet("""
            QComboBox {
                background-color: #222;
                color: #FFF;
                border: 2px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        mic_devices = []
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                import speech_recognition as sr
                mic_list = sr.Microphone.list_microphone_names()
                if mic_list:
                    mic_combo.addItem("Default (System Default)", None)
                    for i, mic_name in enumerate(mic_list):
                        mic_combo.addItem(mic_name, i)
                        mic_devices.append((i, mic_name))
                        # Select saved microphone if available
                        if self.microphone_device_index == i:
                            mic_combo.setCurrentIndex(i + 1)  # +1 for "Default" option
                else:
                    mic_combo.addItem("No microphones detected", None)
            except Exception as e:
                logging.warning(f"Error listing microphones: {e}")
                mic_combo.addItem("Error detecting microphones", None)
        else:
            mic_combo.addItem("Speech recognition not available", None)
            mic_combo.setEnabled(False)
        
        mic_layout.addWidget(mic_combo)
        layout.addWidget(mic_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: #FFF;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        layout.addWidget(buttons)
        
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            # Save TTS settings
            self.tts_enabled = tts_enable_cb.isChecked() and TTS_AVAILABLE
            if voice_combo.currentData():
                self.tts_voice_id = voice_combo.currentData()
            else:
                self.tts_voice_id = None
            
            # gTTS does not support voice selection; skip voice logic
            
            # Save microphone settings
            if mic_combo.currentData() is not None:
                self.microphone_device_index = mic_combo.currentData()
            else:
                self.microphone_device_index = None
            
            self.save_settings()
            QMessageBox.information(self, "Settings Saved", "Audio settings have been saved successfully!")

# =====================================================
# MAIN
# =====================================================

def main():
    import sys
    
    # Set up exception handling before creating QApplication
    def qt_exception_handler(exc_type, exc_value, exc_traceback):
        """Handle exceptions in Qt event loop"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the exception
        handle_exception(exc_type, exc_value, exc_traceback)
    
    # Install Qt exception handler
    sys.excepthook = qt_exception_handler
    
    try:
        app = QApplication(sys.argv)
        
        # Set application properties for better error handling
        app.setQuitOnLastWindowClosed(True)
        
        splash = None
        try:
            if SPLASH_FILE.exists():
                splash = QSplashScreen(QPixmap(str(SPLASH_FILE)))
                splash.show()
                app.processEvents()
        except Exception as splash_error:
            logging.warning(f"Splash screen error: {splash_error}")
            splash = None
        
        try:
            window = LeaWindow()
            window.show()
            
            if splash:
                splash.finish(window)
        except Exception as window_error:
            logging.error(f"Window creation error: {traceback.format_exc()}")
            QMessageBox.critical(None, "Fatal Error", 
                               f"Failed to create window:\n{str(window_error)}\n\nCheck lea_crash.log for details.")
            sys.exit(1)
        
        try:
            sys.exit(app.exec())
        except Exception as app_error:
            logging.error(f"Application error: {traceback.format_exc()}")
            sys.exit(1)
            
    except Exception as main_error:
        logging.error(f"Main function error: {traceback.format_exc()}")
        print(f"Fatal error: {main_error}")
        print("Check lea_crash.log for details.")
        sys.exit(1)

if __name__ == "__main__":
    # Test import and basic initialization
    try:
        print("Testing Lea Assistant initialization...")
        main()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
