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

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, pyqtSlot
from PyQt6.QtGui import QIcon, QPixmap, QColor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QSizePolicy, QFrame, QSplashScreen, QFileDialog,
    QMessageBox, QCheckBox, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QMenu,
    QListWidget, QListWidgetItem,
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

Locate rules, cases, resources. Draft motions and organize facts.

When helping with legal matters:
- Make complex legal concepts accessible and understandable
- Organize information clearly and logically
- Be thorough but not overwhelming
- Show empathy for the stress legal matters can cause
- ALWAYS REMIND: "I am not a lawyer, this is not legal advice." (Say it warmly, not robotically)
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
            
            # Get system prompt
            system_prompt = self.agents[self.mode].get("system_prompt", "")
            messages = [{"role": "system", "content": system_prompt}] + self.message_history
            
            # Make API call with error handling
            try:
                model_name = self.model_options[self.model]
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=60.0  # 60 second timeout
                )
                
                if not response or not response.choices:
                    self.error.emit("Invalid response from OpenAI API")
                    return
                
                answer = response.choices[0].message.content
                if not answer:
                    self.error.emit("Empty response from OpenAI API")
                    return
                
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
            
            # Task execution handling
            if "[TASK:" in answer and "[PARAMS:" in answer:
                try:
                    import re
                    # Parse task commands: [TASK: task_name] [PARAMS: param1=value1, param2=value2]
                    task_pattern = r'\[TASK:\s*([^\]]+)\]'
                    params_pattern = r'\[PARAMS:\s*([^\]]+)\]'
                    
                    tasks_found = re.findall(task_pattern, answer)
                    params_found = re.findall(params_pattern, answer)
                    
                    if tasks_found and params_found and TASK_SYSTEM_AVAILABLE:
                        task_results = []
                        for i, task_name in enumerate(tasks_found):
                            task_name = task_name.strip()
                            
                            # Parse parameters
                            params_str = params_found[i] if i < len(params_found) else ""
                            params = {}
                            if params_str:
                                for param_pair in params_str.split(','):
                                    if '=' in param_pair:
                                        key, value = param_pair.split('=', 1)
                                        params[key.strip()] = value.strip().strip('"').strip("'")
                            
                            # Check if task requires confirmation
                            task_obj = task_registry.get_task(task_name)
                            requires_confirmation = task_obj.requires_confirmation if task_obj else False
                            
                            # Execute task (for now, auto-confirm if in same request)
                            # In production, you might want to ask user first
                            result = task_registry.execute_task(task_name, params, confirmed=not requires_confirmation)
                            task_results.append({
                                "task": task_name,
                                "params": params,
                                "result": result.to_dict()
                            })
                        
                        # Add task results to context
                        if task_results:
                            results_text = "\n=== Task Execution Results ===\n"
                            for tr in task_results:
                                r = tr["result"]
                                results_text += f"\nTask: {tr['task']}\n"
                                results_text += f"Status: {'✅ Success' if r['success'] else '❌ Failed'}\n"
                                results_text += f"Message: {r['message']}\n"
                                if r.get('error'):
                                    results_text += f"Error: {r['error']}\n"
                            
                            # If any task failed, ask for confirmation or retry
                            # Otherwise, incorporate results into response
                            if all(tr["result"]["success"] for tr in task_results):
                                answer = answer + "\n\n" + results_text
                            else:
                                answer = answer + "\n\n" + results_text + "\n\n⚠️ Some tasks failed. Please review and try again if needed."
                        
                except Exception as task_error:
                    logging.warning(f"Task execution failed: {task_error}")
                    answer += f"\n\n⚠️ Error executing tasks: {str(task_error)}"
            
            # Web search handling (optional, can be improved)
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
            
            # Save to history
            self.message_history.append({"role": "assistant", "content": answer})
            # Limit history to last 20 messages
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            
            self.finished.emit(answer, "Ready")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(f"LeaWorker error: {traceback.format_exc()}")
            self.error.emit(error_msg)

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
        """Show emoji picker dialog"""
        # Common emojis organized by category
        emojis = {
            "Faces": ["😊", "😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂", "🙂", "🙃", "😉", "😌", "😍", "🥰", "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜", "🤪", "🤨", "🧐", "🤓", "😎", "🤩", "🥳", "😏", "😒", "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤", "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱", "😨", "😰", "😥", "😓"],
            "Gestures": ["👋", "🤚", "🖐️", "✋", "🖖", "👌", "🤌", "🤏", "✌️", "🤞", "🤟", "🤘", "🤙", "👈", "👉", "👆", "🖕", "👇", "☝️", "👍", "👎", "✊", "👊", "🤛", "🤜", "👏", "🙌", "👐", "🤲", "🤝", "🙏"],
            "Objects": ["💻", "📱", "⌚", "🖥️", "⌨️", "🖨️", "📞", "📟", "📠", "📺", "📻", "🎙️", "🎚️", "🎛️", "⏱️", "⏲️", "⏰", "🕰️", "⏳", "⌛", "📡", "🔋", "🔌", "💡", "🔦", "🕯️", "🧯", "🛢️", "💸", "💵", "💴", "💶", "💷", "💰", "💳", "💎"],
            "Symbols": ["❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎", "💔", "❣️", "💕", "💞", "💓", "💗", "💖", "💘", "💝", "💟", "☮️", "✝️", "☪️", "🕉️", "☸️", "✡️", "🔯", "🕎", "☯️", "☦️", "🛐", "⛎", "♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓", "🆔", "⚛️"],
            "Common": ["✅", "❌", "⚠️", "❓", "❗", "💯", "🔥", "⭐", "🌟", "✨", "💫", "🌈", "🎉", "🎊", "🎈", "🎁", "🏆", "🥇", "🥈", "🥉", "🎖️", "🏅", "🎗️", "🎫", "🎟️", "🎪", "🤹", "🎭", "🎨", "🎬", "🎤", "🎧", "🎼", "🎹", "🥁", "🎷", "🎺", "🎸", "🪕", "🎻", "🎲", "♟️", "🎯", "🎳", "🎮", "🎰"],
        }
        
        dialog = QDialog(self)
        dialog.setWindowTitle("😊 Emoji Picker")
        dialog.setMinimumSize(400, 500)
        dialog.setMaximumSize(500, 600)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Select an emoji:")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #FFF; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Emoji list (grouped by category)
        list_widget = QListWidget()
        list_widget.setStyleSheet("""
            QListWidget {
                background-color: #222;
                color: #FFF;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 24px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:hover {
                background-color: #444;
            }
            QListWidget::item:selected {
                background-color: #0078D7;
            }
        """)
        
        # Add emojis grouped by category
        for category, emoji_list in emojis.items():
            # Add category header
            category_item = QListWidgetItem(f"  {category}")
            category_item.setFlags(category_item.flags() & ~QListWidgetItem.ItemIsSelectable)
            category_item.setBackground(QColor(80, 80, 80))
            list_widget.addItem(category_item)
            
            # Add emojis in this category
            for emoji in emoji_list:
                emoji_item = QListWidgetItem(emoji)
                emoji_item.setData(Qt.ItemDataRole.UserRole, emoji)
                list_widget.addItem(emoji_item)
        
        # Make category headers non-selectable
        gray_color = QColor(150, 150, 150)
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if not item.data(Qt.ItemDataRole.UserRole):
                item.setFlags(item.flags() & ~QListWidgetItem.ItemIsSelectable & ~QListWidgetItem.ItemIsEnabled)
                item.setForeground(gray_color)
        
        layout.addWidget(list_widget)
        
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
        
        insert_btn = QPushButton("Insert")
        insert_btn.clicked.connect(insert_emoji)
        insert_btn.setStyleSheet("background-color: #0078D7; padding: 6px 12px; border-radius: 4px; font-weight: 600;")
        buttons.addWidget(insert_btn)
        
        buttons.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("background-color: #555; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(close_btn)
        
        layout.addLayout(buttons)
        
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
            
            # Clean up any existing thread references first
            try:
                if hasattr(self, 'worker_thread') and self.worker_thread is not None:
                    # Disconnect signals to prevent callbacks after deletion
                    try:
                        if hasattr(self, 'worker') and self.worker is not None:
                            try:
                                self.worker.finished.disconnect()
                                self.worker.error.disconnect()
                            except:
                                pass
                    except:
                        pass
            except:
                pass
            
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
            
            # Store references to prevent garbage collection
            self._current_worker = self.worker
            self._current_thread = self.worker_thread
            
            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.error.connect(self.on_worker_error)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.error.connect(self.worker_thread.quit)
            
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

    def on_worker_finished(self, answer, status):
        try:
            if answer:
                self.append_message("assistant", str(answer))
            self.status_label.setText(str(status) if status else "Ready")
            self._save_history()
            
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
                    self.append_message("system", "Loaded previous conversation")
                    
                    # Display last few messages safely
                    for msg in self.message_history[-5:]:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get('role')
                        content = msg.get('content', '')
                        if not content:
                            continue
                        try:
                            if 'Dre\'s question:' in str(content):
                                content = str(content).split('Dre\'s question:')[-1].strip()
                            if role == 'user':
                                self.append_message('user', content)
                            elif role == 'assistant':
                                self.append_message('assistant', content)
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
    main()
