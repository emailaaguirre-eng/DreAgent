### Lea - Complete Multi-Agent System with SerpAPI Integration###

"""
Hummingbird – Lea
Multi-agent assistant with:
- All 7 specialized modes
- Universal file reading
- Automatic backups with timestamps
- Download capability
- Autonomous web search via SerpAPI (DuckDuckGo)
- Agentic planning and triage

FIXED VERSION: Corrected OpenAI model names to valid options
"""

from dotenv import load_dotenv
load_dotenv()

import os
import html
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

import requests
from openai import OpenAI

# Computer automation (optional - install with: pip install pyautogui keyboard)
try:
    import pyautogui
    import keyboard
    import time
    AUTOMATION_AVAILABLE = True
    # Safety: Add failsafe - move mouse to corner to abort
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.5  # Small delay between actions for safety
except ImportError:
    AUTOMATION_AVAILABLE = False
    print("WARNING: pyautogui and keyboard not installed. Computer automation disabled.")
    print("Install with: pip install pyautogui keyboard")

from PyQt6.QtCore import Qt, pyqtSignal, QThread
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
# API SETUP
# =====================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if not OPENAI_API_KEY:
    print("=" * 60)
    print("WARNING: 'OPENAI_API_KEY' not found in .env")
    print("=" * 60)

if not SERPAPI_API_KEY:
    print("=" * 60)
    print("WARNING: 'SERPAPI_API_KEY' not found in .env")
    print("Web search functionality will be limited.")
    print("=" * 60)

# =====================================================
# DIRECTORIES
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(r"C:\Dre_Programs\LeaAssistant\assets")
MEMORY_DIR = Path(r"C:\Users\email\OneDrive\AI_Databases")
BACKUPS_DIR = PROJECT_DIR / "backups"
DOWNLOADS_DIR = PROJECT_DIR / "downloads"
KNOWLEDGE_DIR = MEMORY_DIR / "mode_knowledge"  # Mode-specific knowledge bases
AUDIT_LOG_DIR = MEMORY_DIR / "audit_logs"  # Audit trail logs
TASKS_DIR = MEMORY_DIR / "saved_tasks"  # Saved automation tasks

# Create all directories
for dir_path in [MEMORY_DIR, BACKUPS_DIR, DOWNLOADS_DIR, KNOWLEDGE_DIR, AUDIT_LOG_DIR, TASKS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

SPLASH_FILE = ASSETS_DIR / "Hummingbird_LEA_v1_Splash_Logo_Lime_Green.png"
ICON_FILE = ASSETS_DIR / "Hummingbird_LEA_Logo_White_No BKGND.png"

print(f"\nDirectories configured:")
print(f"  📁 Assets: {ASSETS_DIR}")
print(f"  💾 Memory: {MEMORY_DIR}")
print(f"  💾 Backups: {BACKUPS_DIR}")
print(f"  📥 Downloads: {DOWNLOADS_DIR}\n")

# =====================================================
# WEB SEARCH (SERPAPI - DUCKDUCKGO)
# =====================================================

def search_duckduckgo(query: str, num_results: int = 5) -> dict:
    """
    Search DuckDuckGo via SerpAPI with rate limiting and audit logging
    Returns: {'success': bool, 'results': list, 'error': str}
    """
    if not SERPAPI_API_KEY:
        return {'success': False, 'error': 'SERPAPI_API_KEY not configured'}
    
    # Check rate limits
    can_search, message = rate_limiter.can_make_search()
    if not can_search:
        log_audit_event("rate_limit_exceeded", {
            "action": "web_search",
            "query": query,
            "message": message
        })
        return {'success': False, 'error': message, 'results': []}
    
    try:
        # Log search attempt
        log_audit_event("web_search_start", {
            "query": query,
            "num_results": num_results
        })
        
        url = "https://serpapi.com/search"
        params = {
            'api_key': SERPAPI_API_KEY,
            'engine': 'duckduckgo',
            'q': query,
            'num': num_results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if 'organic_results' in data:
            for item in data['organic_results'][:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
        
        # Record successful search
        rate_limiter.record_search()
        log_audit_event("web_search_success", {
            "query": query,
            "results_count": len(results)
        })
        
        return {'success': True, 'results': results, 'query': query}
    except Exception as e:
        log_audit_event("web_search_error", {
            "query": query,
            "error": str(e)
        })
        return {'success': False, 'error': str(e), 'results': []}

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
# AUDIT LOGGING SYSTEM
# =====================================================

def log_audit_event(event_type: str, details: dict, user_action: str = None):
    """Log all agent actions to audit trail for accountability and troubleshooting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y%m%d")
    
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "details": details
    }
    
    if user_action:
        log_entry["user_action"] = user_action
    
    # Write to daily log file
    log_file = AUDIT_LOG_DIR / f"audit_log_{date_str}.jsonl"
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Failed to write audit log: {e}")

# =====================================================
# RATE LIMITING SYSTEM
# =====================================================

class RateLimiter:
    """Simple rate limiter to track API usage and prevent excessive costs"""
    def __init__(self, api_limit_per_hour=100, search_limit_per_hour=50):
        self.api_limit_per_hour = api_limit_per_hour
        self.search_limit_per_hour = search_limit_per_hour
        
        # Load or initialize tracking file
        self.tracking_file = MEMORY_DIR / "rate_limits.json"
        self._load_tracking()
    
    def _load_tracking(self):
        """Load rate limit tracking from file"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                
                # Check if data is from current hour
                current_hour = datetime.now().strftime("%Y-%m-%d-%H")
                if data.get('hour') == current_hour:
                    self.api_calls = data.get('api_calls', 0)
                    self.search_calls = data.get('search_calls', 0)
                else:
                    # Reset for new hour
                    self.api_calls = 0
                    self.search_calls = 0
                    self._save_tracking()
            except:
                self.api_calls = 0
                self.search_calls = 0
        else:
            self.api_calls = 0
            self.search_calls = 0
            self._save_tracking()
    
    def _save_tracking(self):
        """Save rate limit tracking to file"""
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        data = {
            'hour': current_hour,
            'api_calls': self.api_calls,
            'search_calls': self.search_calls
        }
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save rate limit tracking: {e}")
    
    def can_make_api_call(self):
        """Check if an API call can be made"""
        self._load_tracking()  # Refresh from file
        if self.api_calls >= self.api_limit_per_hour:
            return False, f"API rate limit reached ({self.api_calls}/{self.api_limit_per_hour} per hour)"
        return True, ""
    
    def can_make_search(self):
        """Check if a search can be performed"""
        self._load_tracking()  # Refresh from file
        if self.search_calls >= self.search_limit_per_hour:
            return False, f"Search rate limit reached ({self.search_calls}/{self.search_limit_per_hour} per hour)"
        return True, ""
    
    def record_api_call(self):
        """Record an API call"""
        self.api_calls += 1
        self._save_tracking()
    
    def record_search(self):
        """Record a search"""
        self.search_calls += 1
        self._save_tracking()
    
    def get_status(self):
        """Get current rate limit status"""
        self._load_tracking()
        return {
            'api_calls': self.api_calls,
            'api_remaining': max(0, self.api_limit_per_hour - self.api_calls),
            'search_calls': self.search_calls,
            'search_remaining': max(0, self.search_limit_per_hour - self.search_calls)
        }

# Initialize rate limiter
rate_limiter = RateLimiter()

# =====================================================
# KNOWLEDGE BASE SYSTEM
# =====================================================

def get_mode_knowledge(mode_name: str) -> str:
    """Retrieve mode-specific knowledge base content"""
    safe_name = mode_name.replace(" ", "_").replace("&", "and")
    knowledge_file = KNOWLEDGE_DIR / f"{safe_name}_knowledge.txt"
    
    if knowledge_file.exists():
        try:
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip():
                return f"\n\n=== {mode_name.upper()} KNOWLEDGE BASE ===\n{content}\n=== END KNOWLEDGE BASE ===\n"
        except Exception as e:
            print(f"Failed to load knowledge for {mode_name}: {e}")
    
    return ""

def append_mode_knowledge(mode_name: str, knowledge: str, user_confirmed: bool = False):
    """Append information to mode-specific knowledge base"""
    safe_name = mode_name.replace(" ", "_").replace("&", "and")
    knowledge_file = KNOWLEDGE_DIR / f"{safe_name}_knowledge.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[Added: {timestamp}]\n{knowledge}\n"
    
    with open(knowledge_file, 'a', encoding='utf-8') as f:
        f.write(entry)
    
    log_audit_event("knowledge_added", {
        "mode": mode_name,
        "user_confirmed": user_confirmed,
        "knowledge_length": len(knowledge)
    })

# =====================================================
# AUTOMATION TASK SYSTEM
# =====================================================

def save_task(task_name: str, description: str, actions: list):
    """Save an automation task for later execution"""
    safe_name = task_name.replace(" ", "_").replace("/", "_")
    task_file = TASKS_DIR / f"{safe_name}.json"
    
    task_data = {
        "name": task_name,
        "description": description,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actions": actions
    }
    
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(task_data, f, indent=2)
    
    return str(task_file)

def load_task(task_name: str) -> dict:
    """Load a saved task"""
    safe_name = task_name.replace(" ", "_").replace("/", "_")
    task_file = TASKS_DIR / f"{safe_name}.json"
    
    if not task_file.exists():
        return None
    
    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_saved_tasks() -> list:
    """List all saved tasks"""
    tasks = []
    for task_file in TASKS_DIR.glob("*.json"):
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            tasks.append({
                'name': task_data.get('name', task_file.stem),
                'description': task_data.get('description', 'No description'),
                'created': task_data.get('created', 'Unknown'),
                'actions_count': len(task_data.get('actions', []))
            })
        except Exception as e:
            print(f"Failed to load task {task_file}: {e}")
    
    return tasks

def execute_task_actions(actions: list) -> tuple:
    """Execute recorded actions"""
    if not AUTOMATION_AVAILABLE:
        return False, "Automation not available"
    
    try:
        time.sleep(1)  # Give user time to prepare
        
        for action in actions:
            action_type = action.get('type')
            
            if action_type == 'click':
                x, y = action['x'], action['y']
                pyautogui.click(x, y)
            
            elif action_type == 'type':
                text = action['text']
                pyautogui.write(text, interval=0.05)
            
            elif action_type == 'key':
                key = action['key']
                pyautogui.press(key)
            
            elif action_type == 'hotkey':
                keys = action['keys']
                pyautogui.hotkey(*keys)
            
            elif action_type == 'scroll':
                amount = action['amount']
                pyautogui.scroll(amount)
            
            time.sleep(0.5)  # Small delay between actions
        
        return True, f"Successfully executed {len(actions)} actions"
    
    except Exception as e:
        return False, f"Task execution failed: {str(e)}"

# =====================================================
# AGENT DEFINITIONS
# =====================================================

AGENTS = {
    "General Assistant & Triage": {
        "system_prompt": """You are Lea, Dre's versatile multi-agent AI assistant, and you're running in General Assistant & Triage mode.

Your primary role is to:
1. Handle general questions and provide helpful, thoughtful responses
2. **INTELLIGENTLY TRIAGE** questions to specialized modes when they clearly require expert knowledge
3. Suggest or auto-switch to the appropriate specialized mode when needed

Available specialized modes (suggest switching when questions fit these domains):
- **Legal Research & Drafting**: Arizona law, statutes, rules of procedure, court documents, legal research
- **Developer & Coding Assistance**: Programming, debugging, technical implementation, code review
- **Economic Incentives & Tax Credits**: Business incentives, tax programs, grant research
- **Email Deliverability & DNS**: Technical email infrastructure, DMARC, SPF, DKIM, DNS configuration
- **Accounting & QuickBooks Help**: Bookkeeping, QuickBooks Online, financial workflows
- **Learning, Research & Knowledge**: Educational content, research projects, knowledge compilation

AUTOMATION CAPABILITIES:
You can help Dre automate repetitive computer tasks. When Dre describes repetitive work:
1. Suggest breaking it into recordable steps
2. Recommend using the "Record Task" button to capture actions
3. Explain that recorded tasks can be replayed on demand
4. If tasks are already saved, suggest executing them

Be conversational, helpful, and proactive about routing Dre to the right specialist when needed.""",
        "description": "General chat and intelligent routing to specialists",
        "recommended_models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
    },
    
    "Legal Research & Drafting": {
        "system_prompt": """You are Lea in Legal Research & Drafting mode, specializing in Arizona law and legal document preparation.

Core responsibilities:
- Research Arizona statutes, rules, and case law
- Draft motions, pleadings, and legal documents following Arizona court requirements
- Explain legal procedures and requirements in clear language
- Cite sources accurately with proper legal citation format

IMPORTANT GUIDELINES:
- Always prioritize official sources (.gov, .edu, official court websites)
- When uncertain about current law, explicitly recommend web search for latest information
- For Arizona Rules of Civil Procedure or statutes, cite rule/statute numbers precisely
- Distinguish between your knowledge and current law (which may have changed)
- Never guess about court deadlines or filing requirements - recommend official verification
- Draft documents professionally but explain legal concepts conversationally

You're helping Dre with legal research and document preparation. Be thorough, accurate, and always emphasize verifying with official sources.""",
        "description": "Arizona law research and legal document drafting",
        "recommended_models": ["o3", "o1", "gpt-4.1", "gpt-4o"]  # Reasoning models for legal analysis
    },
    
    "Developer & Coding Assistance": {
        "system_prompt": """You are Lea in Developer & Coding Assistance mode, Dre's technical coding partner.

Core focus:
- Write clean, well-documented code in Python, PowerShell, and other languages
- Debug issues with detailed analysis and fixes
- Explain technical concepts clearly
- Review code for improvements and best practices
- Help implement APIs, databases, and software architecture

Dre's tech stack context:
- Python: PyQt6, Flask, OpenAI API, database integration
- Business applications: CRM systems, email automation, client engagement tools
- Windows environment with PowerShell scripting

CODING PRINCIPLES:
- Provide complete, runnable code examples
- Include helpful comments and error handling
- Explain WHY solutions work, not just WHAT they do
- Suggest best practices and potential improvements
- Consider Dre's business application context

Be thorough with code examples and prioritize working, maintainable solutions.""",
        "description": "Programming, debugging, and technical implementation",
        "recommended_models": ["gpt-4.1", "o3", "gpt-4o", "o4-mini"]  # Best for code generation
    },
    
    "Economic Incentives & Tax Credits": {
        "system_prompt": """You are Lea in Economic Incentives & Tax Credits mode, specializing in business incentive programs.

Focus areas:
- Federal and state tax credits (IRS, state programs)
- Business grants and subsidies
- Economic development incentives
- Training and workforce development funding
- Research and development credits

RESEARCH APPROACH:
- Provide specific program names, eligibility criteria, and application processes
- Include dollar amounts, deadlines, and administering agencies
- Distinguish between refundable and non-refundable credits
- Explain qualification requirements clearly
- Recommend professional verification for specific situations

You're helping Dre identify and understand incentive opportunities. Be specific about programs and always recommend confirming details with program administrators or tax professionals.""",
        "description": "Business incentives, tax credits, and grant research",
        "recommended_models": ["gpt-4o", "gpt-4.1", "o1-mini"]  # Good for research and analysis
    },
    
    "Email Deliverability & DNS": {
        "system_prompt": """You are Lea in Email Deliverability & DNS mode, specializing in technical email infrastructure.

Core expertise:
- Email authentication (SPF, DKIM, DMARC)
- DNS configuration and troubleshooting
- Deliverability optimization
- Email service providers (Mailgun, SendGrid, Postmark, etc.)
- Inbox placement and reputation management

TECHNICAL APPROACH:
- Provide specific DNS record configurations
- Explain authentication protocols clearly
- Troubleshoot delivery issues systematically
- Recommend testing and monitoring approaches
- Consider both technical and sender reputation factors

You're helping Dre ensure emails reach inboxes reliably. Be technically precise while explaining concepts accessibly.""",
        "description": "Technical email infrastructure and DNS configuration",
        "recommended_models": ["gpt-4o", "gpt-4.1", "gpt-4o-mini"]  # Technical but not heavy reasoning
    },
    
    "Accounting & QuickBooks Help": {
        "system_prompt": """You are Lea in Accounting & QuickBooks Help mode, specializing in bookkeeping and QuickBooks Online.

Focus areas:
- QuickBooks Online navigation and workflows
- Chart of accounts setup and management
- Bank reconciliation processes
- Journal entries and accounting transactions
- Financial reporting and analysis
- QuickBooks automation and efficiency

ACCOUNTING PRINCIPLES:
- Follow GAAP (Generally Accepted Accounting Principles)
- Explain double-entry bookkeeping clearly
- Provide specific QBO navigation steps
- Recommend workflows that maintain audit trails
- Emphasize reconciliation and accuracy

You're helping Dre with bookkeeping and QuickBooks tasks. Be practical with QBO-specific guidance while explaining underlying accounting concepts.""",
        "description": "Bookkeeping and QuickBooks Online assistance",
        "recommended_models": ["o4-mini", "gpt-4o", "o1-mini", "gpt-4.1"]  # Math reasoning for calculations
    },
    
    "Learning, Research & Knowledge": {
        "system_prompt": """You are Lea in Learning, Research & Knowledge mode, focused on education and research.

Core capabilities:
- Break down complex topics into understandable explanations
- Create study plans and learning roadmaps
- Conduct comprehensive research on topics
- Synthesize information from multiple sources
- Generate outlines, summaries, and educational content

TEACHING APPROACH:
- Use analogies and examples to clarify concepts
- Build from fundamentals to advanced topics
- Encourage critical thinking and deeper questions
- Provide multiple perspectives on topics
- Make learning engaging and practical

You're helping Dre learn and research effectively. Be thorough, clear, and pedagogical in your explanations.""",
        "description": "Educational content and research assistance",
        "recommended_models": ["gpt-4o", "gpt-4.1", "o3", "gpt-4o-mini"]  # Comprehensive explanations
    }
}

# =====================================================
# MODEL OPTIONS - UPDATED WITH CURRENT OPENAI MODELS
# =====================================================

# All available models
ALL_MODELS = {
    "GPT-4o (Recommended)": "gpt-4o",
    "GPT-4o mini (Fast & Cheap)": "gpt-4o-mini",
    "GPT-4.1 (Newest)": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano (Fastest)": "gpt-4.1-nano",
    "o1 (Reasoning)": "o1",
    "o1-mini (Fast Reasoning)": "o1-mini",
    "o3 (Advanced Reasoning)": "o3",
    "o4-mini (Latest Small)": "o4-mini",
}

# Reverse lookup: model ID to display name
MODEL_ID_TO_NAME = {v: k for k, v in ALL_MODELS.items()}

def get_models_for_mode(mode_name: str) -> dict:
    """
    Get the recommended models for a specific mode, ordered by recommendation.
    Returns a dictionary of display_name: model_id
    """
    if mode_name not in AGENTS:
        return ALL_MODELS
    
    recommended_ids = AGENTS[mode_name].get("recommended_models", [])
    
    # Build ordered dictionary: recommended models first, then all others
    ordered_models = {}
    
    # Add recommended models first (with ⭐ marker)
    for model_id in recommended_ids:
        if model_id in MODEL_ID_TO_NAME:
            display_name = MODEL_ID_TO_NAME[model_id]
            # Add star to recommended models
            ordered_models[f"⭐ {display_name}"] = model_id
    
    # Add separator if there are recommendations
    if ordered_models:
        ordered_models["───────────────"] = None  # Separator
    
    # Add all other models
    for display_name, model_id in ALL_MODELS.items():
        if model_id not in recommended_ids:
            ordered_models[display_name] = model_id
    
    return ordered_models

MODEL_OPTIONS = ALL_MODELS  # Default to all models

# =====================================================
# API WORKER THREAD
# =====================================================

class APIWorker(QThread):
    """Worker thread for making API calls without freezing the UI"""
    finished = pyqtSignal(str)  # Emits the answer when done
    error = pyqtSignal(str)  # Emits error message if something goes wrong
    status_update = pyqtSignal(str)  # Emits status updates
    
    def __init__(self, messages, model, needs_search=False, search_query=None, search_results=None):
        super().__init__()
        self.messages = messages
        self.model = model
        self.needs_search = needs_search
        self.search_query = search_query
        self.search_results = search_results
    
    def run(self):
        """Run the API call in the background thread"""
        try:
            # Check rate limits before API call
            can_call, message = rate_limiter.can_make_api_call()
            if not can_call:
                log_audit_event("rate_limit_exceeded", {
                    "action": "api_call",
                    "model": self.model,
                    "message": message
                })
                self.error.emit(message)
                return
            
            # First, get initial response
            self.status_update.emit("Thinking...")
            log_audit_event("api_call_start", {
                "model": self.model,
                "messages_count": len(self.messages)
            })
            
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            initial_answer = response.choices[0].message.content
            
            # Record API call
            rate_limiter.record_api_call()
            log_audit_event("api_call_success", {
                "model": self.model,
                "response_length": len(initial_answer)
            })
            
            # If search is needed, perform it and get refined response
            if self.needs_search and self.search_results:
                # Check rate limits again for second API call
                can_call, message = rate_limiter.can_make_api_call()
                if not can_call:
                    log_audit_event("rate_limit_exceeded", {
                        "action": "api_call_refinement",
                        "model": self.model,
                        "message": message
                    })
                    # Use initial answer if rate limit reached
                    answer = initial_answer
                else:
                    self.status_update.emit("Refining response with search results...")
                    messages_with_search = self.messages + [
                        {"role": "assistant", "content": initial_answer},
                        {"role": "user", "content": f"I performed a web search for current information: {self.search_query}\n\n{self.search_results}\n\nPlease incorporate this current information into your response. Prioritize official sources (.gov, .edu, official court websites) and cite them. If the information differs from your training data, use the search results as they represent the most current information."}
                    ]
                    
                    log_audit_event("api_call_refinement_start", {
                        "model": self.model
                    })
                    
                    response = openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages_with_search
                    )
                    answer = response.choices[0].message.content
                    
                    # Record second API call
                    rate_limiter.record_api_call()
                    log_audit_event("api_call_refinement_success", {
                        "model": self.model,
                        "response_length": len(answer)
                    })
            else:
                answer = initial_answer
            
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(str(e))

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
        self.current_model = "GPT-4o (Recommended)"
        self.message_history = []
        self.history_file = MEMORY_DIR / "lea_history.json"
        self.current_file_content = None
        self.current_file_path = None
        self.current_file_metadata = None
        self.api_worker = None  # Worker thread for API calls
        self.is_recording_task = False  # Task recording state
        
        self._init_window()
        self._build_ui()
        
        # Initialize model combo with mode-specific models
        self._update_model_combo_for_mode(self.current_mode)
        
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
        
        # Automation buttons (only if available)
        if AUTOMATION_AVAILABLE:
            buttons.addWidget(QLabel("|"))  # Separator
            
            self.record_task_btn = QPushButton("🔴 Record Task")
            self.record_task_btn.clicked.connect(self.toggle_task_recording)
            self.record_task_btn.setStyleSheet("background-color: #D13438; padding: 6px 12px; border-radius: 4px;")
            buttons.addWidget(self.record_task_btn)
            
            tasks_btn = QPushButton("📋 Tasks")
            tasks_btn.clicked.connect(self.show_saved_tasks)
            tasks_btn.setStyleSheet("background-color: #0078D7; padding: 6px 12px; border-radius: 4px;")
            buttons.addWidget(tasks_btn)
        else:
            self.record_task_btn = None
        
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
        self.include_file_cb = QCheckBox("Include uploaded file in conversation")
        self.include_file_cb.setChecked(True)
        self.include_file_cb.setToolTip("When checked, uploaded files are automatically included in all messages to Lea")
        options.addWidget(self.include_file_cb)
        options.addStretch()
        frame_layout.addLayout(options)
        
        # Status with rate limit info
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #DDD; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        
        # Rate limit status
        self.rate_limit_label = QLabel("")
        self.rate_limit_label.setStyleSheet("color: #888; font-size: 11px;")
        status_layout.addStretch()
        status_layout.addWidget(self.rate_limit_label)
        frame_layout.addLayout(status_layout)
        
        # Update rate limit display
        self._update_rate_limit_display()
        
        layout.addWidget(frame)
    
    def _update_rate_limit_display(self):
        """Update the rate limit display"""
        status = rate_limiter.get_status()
        self.rate_limit_label.setText(
            f"API: {status['api_remaining']} remaining | "
            f"Search: {status['search_remaining']} remaining"
        )
    
    # Mode and model changes
    def on_mode_changed(self, mode: str):
        self.current_mode = mode
        
        # Update model combo to show recommended models for this mode
        self._update_model_combo_for_mode(mode)
        
        self.append_message("system", f"Switched to: {mode}")
        
        # Show recommended models info
        recommended = AGENTS[mode].get("recommended_models", [])
        if recommended:
            model_names = [MODEL_ID_TO_NAME.get(m, m) for m in recommended[:3]]  # Top 3
            self.append_message("system", f"💡 Recommended models for this mode: {', '.join(model_names)}")
        
        log_audit_event("mode_change", {"mode": mode})
        self._save_history()
    
    def _update_model_combo_for_mode(self, mode: str):
        """Update the model combo box to show mode-specific recommendations"""
        # Get models for this mode
        mode_models = get_models_for_mode(mode)
        
        # Save current selection
        current_model_display = self.model_combo.currentText()
        current_model_id = None
        
        # Find the actual model ID from current display name
        for display, model_id in ALL_MODELS.items():
            if display == current_model_display or f"⭐ {display}" == current_model_display:
                current_model_id = model_id
                break
        
        # Clear and repopulate combo box
        self.model_combo.blockSignals(True)  # Prevent triggering on_model_changed
        self.model_combo.clear()
        
        # Add models (filtering out separator)
        for display_name, model_id in mode_models.items():
            if model_id is None:  # Separator
                self.model_combo.insertSeparator(self.model_combo.count())
            else:
                self.model_combo.addItem(display_name)
        
        # Try to restore previous selection
        if current_model_id:
            for i in range(self.model_combo.count()):
                item_text = self.model_combo.itemText(i)
                # Check if this item matches our model ID
                for display, model_id in mode_models.items():
                    if item_text == display and model_id == current_model_id:
                        self.model_combo.setCurrentIndex(i)
                        break
        
        self.model_combo.blockSignals(False)
    
    def on_model_changed(self, model: str):
        # Extract actual model name (remove star if present)
        clean_model = model.replace("⭐ ", "")
        self.current_model = clean_model
        self.append_message("system", f"Model changed to: {clean_model}")
        log_audit_event("model_change", {"model": clean_model})
        self._save_history()
    
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
            file_type = result.get('file_type', 'unknown')
            file_size = len(self.current_file_content)
            file_size_kb = file_size / 1024
            
            # Store file metadata for better context
            self.current_file_metadata = {
                'name': name,
                'path': path,
                'type': file_type,
                'size': file_size,
                'size_kb': file_size_kb
            }
            
            self.file_label.setText(f"📎 {name} ({file_type}, {file_size_kb:.1f} KB)")
            self.append_message("system", f"📎 File uploaded: {name}\nType: {file_type} | Size: {file_size_kb:.1f} KB\nBackup: {os.path.basename(backup_path)}\n\nLea can now view and work with this file.")
            self.status_label.setText(f"File loaded: {name}")
            
            # Log file upload
            log_audit_event("file_upload", {
                "filename": name,
                "file_type": file_type,
                "size_kb": file_size_kb,
                "backup_path": str(backup_path)
            })
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
                data = {
                    'mode': self.current_mode,
                    'model': self.current_model,
                    'exported': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'history': self.message_history
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(f"Lea Multi-Agent Conversation Export\n")
                    f.write(f"Mode: {self.current_mode}\n")
                    f.write(f"Model: {self.current_model}\n")
                    f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*60 + "\n\n")
                    
                    for msg in self.message_history:
                        role = msg['role']
                        content = msg['content']
                        f.write(f"{role.upper()}:\n{content}\n\n")
            
            self.append_message("system", f"Conversation exported to: {os.path.basename(path)}")
            log_audit_event("export_conversation", {"path": path})
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))
    
    def clear_conversation(self):
        reply = QMessageBox.question(
            self, "Clear Conversation?",
            "This will clear the current conversation history.\n\nAre you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Create backup before clearing
            if self.message_history:
                backup_data = {
                    'mode': self.current_mode,
                    'model': self.current_model,
                    'cleared': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'history': self.message_history
                }
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = BACKUPS_DIR / f"conversation_backup_{timestamp}.json"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2)
            
            self.message_history = []
            self.chat_display.clear()
            self._save_history()
            self.append_message("system", "Conversation cleared. Previous conversation backed up.")
            log_audit_event("conversation_cleared", {})
    
    def _detect_required_mode(self, question: str) -> str:
        """
        Detect if a question clearly requires a specific specialized mode.
        Returns mode name if confident, None otherwise.
        """
        question_lower = question.lower()
        
        # Define strong keyword indicators for each mode
        legal_keywords = [
            "legal", "law", "statute", "rule", "motion", "draft", "court",
            "arizona rules", "ariz. r.", "civil procedure", "case law",
            "jurisdiction", "pleading", "service of process", "default judgment"
        ]
        
        dev_keywords = [
            "code", "programming", "python", "debug", "error", "script",
            "api", "database", "function", "class", "variable", "syntax",
            "powershell", "automation", "technical", "implementation",
            "software", "hardware", "computer", "app", "website", "web",
            "html", "css", "javascript", "java", "c++", "c#", "sql", "git",
            "github", "troubleshoot", "stack trace", "runtime", "compile",
            "framework", "library", "package", "deployment", "serverless",
            "cloud", "aws", "azure", "gcp", "cybersecurity", "encryption"
        ]
        
        incentive_keywords = [
            "grant", "incentive", "tax credit", "rebate", "eiag", "funding",
            "subsidy", "economic development", "training funds", "irs credit"
        ]
        
        email_keywords = [
            "deliverability", "dmarc", "spf", "dkim", "mx record", "dns",
            "email auth", "inboxing", "mailgun", "sendgrid", "postmark",
            "bounce", "smtp", "ptr record"
        ]
        
        accounting_keywords = [
            "quickbooks", "qbo", "reconcile", "chart of accounts", "ledger",
            "journal entry", "bookkeeping", "trial balance", "balance sheet",
            "profit and loss", "bank feed"
        ]
        
        learning_keywords = [
            "explain", "how does", "what is", "research", "learn about",
            "break down", "summarize", "analyze", "study plan", "outline topic"
        ]
        
        if any(keyword in question_lower for keyword in legal_keywords):
            return "Legal Research & Drafting"
        if any(keyword in question_lower for keyword in dev_keywords):
            return "Developer & Coding Assistance"
        if any(keyword in question_lower for keyword in incentive_keywords):
            return "Economic Incentives & Tax Credits"
        if any(keyword in question_lower for keyword in email_keywords):
            return "Email Deliverability & DNS"
        if any(keyword in question_lower for keyword in accounting_keywords):
            return "Accounting & QuickBooks Help"
        if any(keyword in question_lower for keyword in learning_keywords):
            return "Learning, Research & Knowledge"
        
        return None  # Stay in current mode for general questions
    
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
        
        # Auto-scroll to bottom to show newest messages
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_send(self):
        text = self.input_box.toPlainText().strip()
        if not text or not openai_client:
            return
        
        # Prevent multiple simultaneous requests
        if self.api_worker and self.api_worker.isRunning():
            return
        
        # Log user message
        log_audit_event("user_message", {
            "mode": self.current_mode,
            "model": self.current_model,
            "message_length": len(text)
        })
        
        self.append_message("user", text)
        self.input_box.clear()
        self.status_label.setText("Preparing...")
        QApplication.processEvents()
        
        # Build prompt
        parts = []
        
        # Include file if uploaded and checkbox is checked
        if self.include_file_cb.isChecked() and self.current_file_content:
            if self.current_file_metadata:
                file_info = f"File: {self.current_file_metadata['name']}\nType: {self.current_file_metadata['type']}\nSize: {self.current_file_metadata['size_kb']:.1f} KB\n"
            else:
                file_info = f"File: {os.path.basename(self.current_file_path) if self.current_file_path else 'Unknown'}\n"
            
            parts.append(f"=== UPLOADED FILE CONTENT ===\n{file_info}\n--- File Content ---\n{self.current_file_content}\n--- End of File Content ---\n")
            parts.append("\nIMPORTANT: The above file has been uploaded by Dre. Please read it carefully and reference it when answering Dre's question below.\n")
        
        parts.append(f"Dre's question:\n{text}")
        
        full_prompt = "\n".join(parts)
        self.message_history.append({"role": "user", "content": full_prompt})
        
        # Auto-detect and switch to appropriate mode if question clearly requires it
        # This works from ANY mode - if a question clearly belongs to another mode, switch to it
        suggested_mode = self._detect_required_mode(text)
        if suggested_mode and suggested_mode != self.current_mode:
            previous_mode = self.current_mode
            self.current_mode = suggested_mode
            if self.mode_combo.currentText() != suggested_mode:
                self.mode_combo.setCurrentText(suggested_mode)
            log_audit_event("mode_switch", {
                "from_mode": previous_mode,
                "to_mode": suggested_mode,
                "trigger": "auto_detection"
            })
            self._save_history()
        
        # Get base system prompt and append mode-specific knowledge
        base_prompt = AGENTS[self.current_mode]["system_prompt"]
        mode_knowledge = get_mode_knowledge(self.current_mode)
        
        # Add automation task information when in general mode
        automation_info = ""
        if self.current_mode == "General Assistant & Triage" and AUTOMATION_AVAILABLE:
            saved_tasks = list_saved_tasks()
            if saved_tasks:
                automation_info = "\n\n=== AVAILABLE AUTOMATION TASKS ===\n"
                for task in saved_tasks:
                    automation_info += f"- {task['name']}: {task['description']} ({task['actions_count']} actions)\n"
                automation_info += "=== END TASKS ===\n"
                automation_info += "\nYou can suggest executing these tasks when Dre asks for repetitive work.\n"
        
        system_prompt = base_prompt + mode_knowledge + automation_info
        
        messages = [{"role": "system", "content": system_prompt}] + self.message_history
        
        # Check if search is needed based on keywords (before API call)
        needs_search = False
        search_query = None
        search_results_text = None
        
        # Keywords that should ALWAYS trigger a search for current information
        always_search_keywords = [
            "arizona rules", "arizona rule", "ariz. r.", "arizona statute",
            "rules of civil procedure", "rules of criminal procedure",
            "current deadline", "latest", "most recent", "updated",
            "what does the law say", "what does the statute say",
            "official source", "government website", ".gov", "court rules"
        ]
        
        # Check if question contains keywords that require current information
        question_lower = text.lower()
        for keyword in always_search_keywords:
            if keyword in question_lower:
                needs_search = True
                # Create a focused search query for better results
                if "arizona" in question_lower and ("rule" in question_lower or "r." in question_lower):
                    # Extract specific rule number if mentioned
                    rule_match = re.search(r'(?:rule|r\.)\s*(\d+)', question_lower)
                    if rule_match:
                        rule_num = rule_match.group(1)
                        search_query = f"Arizona Rules of Civil Procedure Rule {rule_num} official site:.gov OR site:.edu"
                    else:
                        search_query = "Arizona Rules of Civil Procedure official site:.gov OR site:.edu"
                elif "deadline" in question_lower or "due date" in question_lower:
                    search_query = text  # Use full question for deadline searches
                elif "statute" in question_lower or "law" in question_lower:
                    search_query = f"{text} official site:.gov"
                else:
                    search_query = text
                break
        
        # If search is needed and we have SerpAPI, perform it (synchronous but fast)
        if needs_search and SERPAPI_API_KEY and search_query:
            self.status_label.setText("Searching web...")
            QApplication.processEvents()
            
            search_result = search_duckduckgo(search_query, num_results=5)
            
            if search_result['success'] and search_result['results']:
                # Format search results with required annotation
                search_results_text = f"[SEARCH: {search_query}]\nReason: Current/official information required.\n=== WEB SEARCH RESULTS ===\n"
                for i, result in enumerate(search_result['results'], 1):
                    search_results_text += f"\n[{i}] {result['title']}\n"
                    search_results_text += f"URL: {result['link']}\n"
                    search_results_text += f"Snippet: {result['snippet']}\n\n"
                search_results_text += "=== END SEARCH RESULTS ===\n"
        
        # Start the API worker thread
        # Get the actual model ID (handle starred names)
        clean_model_name = self.current_model.replace("⭐ ", "")
        model_id = ALL_MODELS.get(clean_model_name, "gpt-4o")  # Default to gpt-4o if not found
        
        self.api_worker = APIWorker(
            messages=messages,
            model=model_id,
            needs_search=needs_search and search_results_text is not None,
            search_query=search_query,
            search_results=search_results_text
        )
        
        # Connect signals
        self.api_worker.finished.connect(self._on_api_response)
        self.api_worker.error.connect(self._on_api_error)
        self.api_worker.status_update.connect(self.status_label.setText)
        
        # Start the worker thread
        self.api_worker.start()
    
    def _on_api_response(self, answer: str):
        """Handle successful API response"""
        # Log response
        log_audit_event("assistant_response", {
            "mode": self.current_mode,
            "model": self.current_model,
            "response_length": len(answer)
        })
        
        self.message_history.append({"role": "assistant", "content": answer})
        self.append_message("assistant", answer)
        
        # Update status with rate limit info
        self.status_label.setText("Ready")
        self._update_rate_limit_display()
        self._save_history()
        self.api_worker = None
    
    def _on_api_error(self, error_msg: str):
        """Handle API error"""
        log_audit_event("api_error", {
            "mode": self.current_mode,
            "model": self.current_model,
            "error": error_msg
        })
        
        answer = f"[Error: {error_msg}]"
        self.message_history.append({"role": "assistant", "content": answer})
        self.append_message("assistant", answer)
        self.status_label.setText("Error occurred")
        self._save_history()
        self.api_worker = None
    
    def confirm_knowledge_save(self, mode_name: str, knowledge: str) -> bool:
        """Show confirmation dialog for knowledge base save"""
        reply = QMessageBox.question(
            self,
            "Save to Knowledge Base?",
            f"Would you like to save this information to the {mode_name} knowledge base?\n\n"
            f"Preview:\n{knowledge[:200]}..." if len(knowledge) > 200 else f"Preview:\n{knowledge}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                append_mode_knowledge(mode_name, knowledge, user_confirmed=True)
                self.append_message("system", f"Saved to {mode_name} knowledge base")
                return True
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save: {e}")
                return False
        return False
    
    def toggle_task_recording(self):
        """Start or stop task recording"""
        if not AUTOMATION_AVAILABLE:
            QMessageBox.warning(self, "Automation Not Available", 
                             "Install pyautogui and keyboard:\npip install pyautogui keyboard")
            return
        
        if not self.is_recording_task:
            # Start recording
            self.is_recording_task = True
            self.record_task_btn.setText("⏹ Stop Recording")
            self.record_task_btn.setStyleSheet("background-color: #107C10; padding: 6px 12px; border-radius: 4px;")
            
            self.append_message("system", "Task recording started. Perform your actions, then click 'Stop Recording'.\n\nRecorded actions:\n- Mouse clicks\n- Keyboard typing\n- Hotkeys (Ctrl+C, Ctrl+V, etc.)\n- Scrolling")
            
            log_audit_event("task_recording_start", {})
            
            # Start keyboard listener in background
            self.recorded_actions = []
            
            def on_click(x, y, button, pressed):
                if self.is_recording_task and pressed:
                    self.recorded_actions.append({'type': 'click', 'x': x, 'y': y})
            
            def on_type(key):
                if self.is_recording_task:
                    try:
                        char = key.char
                        self.recorded_actions.append({'type': 'type', 'text': char})
                    except AttributeError:
                        # Special key
                        self.recorded_actions.append({'type': 'key', 'key': str(key)})
            
            # Note: Full automation would require mouse and keyboard listeners
            # This is a simplified version - full implementation would need pynput
            self.append_message("system", "Note: Full task recording requires pynput library.\nCurrent version logs actions in simplified form.")
            
        else:
            # Stop recording
            if hasattr(self, 'recorded_actions') and self.recorded_actions:
                # Ask for task name and description
                from PyQt6.QtWidgets import QInputDialog
                
                task_name, ok1 = QInputDialog.getText(self, "Save Task", "Enter task name:")
                if ok1 and task_name:
                    description, ok2 = QInputDialog.getText(self, "Task Description", "Enter task description:")
                    if ok2:
                        try:
                            actions = self.recorded_actions
                            task_file = save_task(task_name, description, actions)
                            self.append_message("system", f"Task '{task_name}' saved with {len(actions)} actions.\nFile: {os.path.basename(task_file)}")
                            
                            log_audit_event("task_saved", {
                                "task_name": task_name,
                                "description": description,
                                "actions_count": len(actions)
                            })
                        except Exception as e:
                            QMessageBox.warning(self, "Error", f"Failed to save task: {e}")
                    else:
                        self.append_message("system", "Task recording cancelled.")
            else:
                self.is_recording_task = False
                self.record_task_btn.setText("🔴 Record Task")
                self.record_task_btn.setStyleSheet("background-color: #D13438; padding: 6px 12px; border-radius: 4px;")
    
    def show_saved_tasks(self):
        """Show dialog with saved tasks"""
        tasks = list_saved_tasks()
        
        if not tasks:
            QMessageBox.information(self, "No Saved Tasks", 
                                  "You haven't saved any automation tasks yet.\n\n"
                                  "Click 'Record Task' to create your first automation.")
            return
        
        # Create task list dialog
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Saved Automation Tasks")
        dialog.resize(500, 400)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"Found {len(tasks)} saved task(s):"))
        
        task_list = QListWidget()
        for task in tasks:
            task_list.addItem(f"{task['name']} - {task['description']} ({task['actions_count']} actions)")
        layout.addWidget(task_list)
        
        btn_layout = QHBoxLayout()
        
        execute_btn = QPushButton("Execute Selected")
        execute_btn.clicked.connect(lambda: self._execute_selected_task(dialog, task_list, tasks))
        btn_layout.addWidget(execute_btn)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(lambda: self._delete_selected_task(dialog, task_list, tasks))
        btn_layout.addWidget(delete_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        dialog.exec()
    
    def _execute_selected_task(self, dialog, task_list, tasks):
        """Execute the selected task"""
        current = task_list.currentRow()
        if current < 0:
            QMessageBox.warning(dialog, "No Selection", "Please select a task to execute.")
            return
        
        task = tasks[current]
        task_name = task['name']
        
        # Load task
        task_data = load_task(task_name)
        if not task_data:
            QMessageBox.warning(dialog, "Error", f"Could not load task '{task_name}'")
            return
        
        # Confirm execution
        reply = QMessageBox.question(
            dialog,
            "Execute Task?",
            f"Execute task '{task_name}'?\n\n"
            f"Description: {task_data.get('description', 'No description')}\n"
            f"Actions: {len(task_data.get('actions', []))}\n\n"
            f"This will perform the recorded actions on your computer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            dialog.close()
            self.append_message("system", f"Executing task '{task_name}'...")
            QApplication.processEvents()
            
            success, message = execute_task_actions(task_data.get('actions', []))
            
            if success:
                self.append_message("system", f"✅ {message}")
                QMessageBox.information(self, "Task Executed", message)
            else:
                self.append_message("system", f"❌ {message}")
                QMessageBox.warning(self, "Task Failed", message)
    
    def _delete_selected_task(self, dialog, task_list, tasks):
        """Delete the selected task"""
        current = task_list.currentRow()
        if current < 0:
            QMessageBox.warning(dialog, "No Selection", "Please select a task to delete.")
            return
        
        task = tasks[current]
        task_name = task['name']
        
        reply = QMessageBox.question(
            dialog,
            "Delete Task?",
            f"Delete task '{task_name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            safe_name = task_name.replace(" ", "_").replace("/", "_")
            task_file = TASKS_DIR / f"{safe_name}.json"
            
            if task_file.exists():
                task_file.unlink()
                log_audit_event("task_deleted", {"task_name": task_name})
                QMessageBox.information(dialog, "Task Deleted", f"Task '{task_name}' has been deleted.")
                dialog.close()
                self.show_saved_tasks()  # Refresh list
    
    def _save_history(self):
        try:
            data = {'mode': self.current_mode, 'model': self.current_model, 'history': self.message_history}
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Save error: {e}")
    
    def _load_history(self):
        if not self.history_file.exists():
            msg = "Welcome to Lea Multi-Agent System!\n\n"
            msg += f"💾 Memory: {MEMORY_DIR}\n"
            msg += f"💾 Backups: {BACKUPS_DIR}\n"
            msg += f"📥 Downloads: {DOWNLOADS_DIR}\n"
            msg += f"📋 Audit Logs: {AUDIT_LOG_DIR}\n"
            if SERPAPI_API_KEY:
                msg += "🔍 Web search enabled (DuckDuckGo via SerpAPI)\n"
            else:
                msg += "⚠️ Web search disabled (SERPAPI_API_KEY not configured)\n"
            status = rate_limiter.get_status()
            msg += f"📊 Rate Limits: API {status['api_remaining']} remaining | Search {status['search_remaining']} remaining\n"
            if AUTOMATION_AVAILABLE:
                msg += "🤖 Computer automation enabled - Lea can help automate repetitive tasks!"
            else:
                msg += "⚠️ Computer automation disabled - Install pyautogui and keyboard to enable"
            self.append_message("system", msg)
            log_audit_event("system_start", {
                "version": "1.1 - Updated with latest OpenAI models (Nov 2025)",
                "modes_available": len(AGENTS),
                "serpapi_enabled": bool(SERPAPI_API_KEY)
            })
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_mode = data.get('mode', "General Assistant & Triage")
            self.current_model = data.get('model', "GPT-4o (Recommended)")
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
            
            # Scroll to bottom after loading history
            QApplication.processEvents()
            scrollbar = self.chat_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
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
