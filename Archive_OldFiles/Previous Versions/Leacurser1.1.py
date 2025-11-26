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
    audit_file = AUDIT_LOG_DIR / f"audit_{date_str}.log"
    
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "details": details,
        "user_action": user_action
    }
    
    try:
        with open(audit_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, indent=2) + "\n" + "-" * 80 + "\n")
    except Exception as e:
        print(f"Error writing audit log: {e}")

# =====================================================
# RATE LIMITS & BUDGET CONTROLS
# =====================================================

class RateLimiter:
    """Track and enforce rate limits for API calls and actions"""
    def __init__(self):
        self.api_calls_today = 0
        self.api_calls_limit = 1000  # Daily limit for API calls
        self.search_calls_today = 0
        self.search_calls_limit = 100  # Daily limit for web searches
        self.last_reset_date = datetime.now().date()
        self._load_counts()
    
    def _load_counts(self):
        """Load rate limit counts from file"""
        count_file = MEMORY_DIR / "rate_limits.json"
        if count_file.exists():
            try:
                with open(count_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    saved_date = datetime.fromisoformat(data.get('date', str(datetime.now().date()))).date()
                    
                    # Reset if new day
                    if saved_date == datetime.now().date():
                        self.api_calls_today = data.get('api_calls', 0)
                        self.search_calls_today = data.get('search_calls', 0)
                    else:
                        self.api_calls_today = 0
                        self.search_calls_today = 0
                    
                    self.api_calls_limit = data.get('api_limit', 1000)
                    self.search_calls_limit = data.get('search_limit', 100)
            except Exception as e:
                print(f"Error loading rate limits: {e}")
    
    def _save_counts(self):
        """Save rate limit counts to file"""
        count_file = MEMORY_DIR / "rate_limits.json"
        try:
            with open(count_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': str(datetime.now().date()),
                    'api_calls': self.api_calls_today,
                    'search_calls': self.search_calls_today,
                    'api_limit': self.api_calls_limit,
                    'search_limit': self.search_calls_limit
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving rate limits: {e}")
    
    def can_make_api_call(self) -> tuple[bool, str]:
        """Check if API call is allowed. Returns (allowed, message)"""
        if datetime.now().date() != self.last_reset_date:
            self.api_calls_today = 0
            self.search_calls_today = 0
            self.last_reset_date = datetime.now().date()
        
        if self.api_calls_today >= self.api_calls_limit:
            return False, f"Daily API call limit reached ({self.api_calls_limit}). Please try again tomorrow."
        
        return True, f"API calls today: {self.api_calls_today}/{self.api_calls_limit}"
    
    def can_make_search(self) -> tuple[bool, str]:
        """Check if web search is allowed. Returns (allowed, message)"""
        if datetime.now().date() != self.last_reset_date:
            self.api_calls_today = 0
            self.search_calls_today = 0
            self.last_reset_date = datetime.now().date()
        
        if self.search_calls_today >= self.search_calls_limit:
            return False, f"Daily web search limit reached ({self.search_calls_limit}). Please try again tomorrow."
        
        return True, f"Searches today: {self.search_calls_today}/{self.search_calls_limit}"
    
    def record_api_call(self):
        """Record an API call"""
        self.api_calls_today += 1
        self._save_counts()
        log_audit_event("api_call", {
            "count": self.api_calls_today,
            "limit": self.api_calls_limit
        })
    
    def record_search(self):
        """Record a web search"""
        self.search_calls_today += 1
        self._save_counts()
        log_audit_event("web_search", {
            "count": self.search_calls_today,
            "limit": self.search_calls_limit
        })
    
    def get_status(self) -> dict:
        """Get current rate limit status"""
        return {
            "api_calls": f"{self.api_calls_today}/{self.api_calls_limit}",
            "search_calls": f"{self.search_calls_today}/{self.search_calls_limit}",
            "api_remaining": self.api_calls_limit - self.api_calls_today,
            "search_remaining": self.search_calls_limit - self.search_calls_today
        }

# Global rate limiter instance
rate_limiter = RateLimiter()

# =====================================================
# COMPUTER AUTOMATION & TASK RECORDING
# =====================================================

class TaskRecorder:
    """Record and replay computer automation tasks"""
    def __init__(self):
        self.is_recording = False
        self.recorded_actions = []
        self.start_time = None
    
    def start_recording(self):
        """Start recording user actions"""
        if not AUTOMATION_AVAILABLE:
            return False, "Automation not available. Install pyautogui and keyboard."
        
        self.is_recording = True
        self.recorded_actions = []
        self.start_time = time.time()
        return True, "Recording started. Perform your actions, then call stop_recording()"
    
    def stop_recording(self):
        """Stop recording and return actions"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        actions = self.recorded_actions.copy()
        self.recorded_actions = []
        return actions
    
    def record_action(self, action_type: str, details: dict):
        """Record an action during recording"""
        if self.is_recording:
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.recorded_actions.append({
                "time": elapsed,
                "type": action_type,
                "details": details
            })

def save_task(task_name: str, description: str, actions: list) -> str:
    """Save a recorded task for later replay"""
    safe_name = task_name.replace(" ", "_").replace("/", "_")
    task_file = TASKS_DIR / f"{safe_name}.json"
    
    task_data = {
        "name": task_name,
        "description": description,
        "created": datetime.now().isoformat(),
        "actions": actions
    }
    
    try:
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2)
        
        log_audit_event("task_saved", {
            "task_name": task_name,
            "actions_count": len(actions),
            "task_file": str(task_file)
        })
        
        return str(task_file)
    except Exception as e:
        log_audit_event("task_save_error", {
            "task_name": task_name,
            "error": str(e)
        })
        raise

def load_task(task_name: str) -> dict:
    """Load a saved task"""
    safe_name = task_name.replace(" ", "_").replace("/", "_")
    task_file = TASKS_DIR / f"{safe_name}.json"
    
    if not task_file.exists():
        return None
    
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_audit_event("task_load_error", {
            "task_name": task_name,
            "error": str(e)
        })
        return None

def list_saved_tasks() -> list:
    """List all saved tasks"""
    tasks = []
    for task_file in TASKS_DIR.glob("*.json"):
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
                tasks.append({
                    "name": task_data.get("name", task_file.stem),
                    "description": task_data.get("description", ""),
                    "created": task_data.get("created", ""),
                    "actions_count": len(task_data.get("actions", []))
                })
        except:
            continue
    return sorted(tasks, key=lambda x: x.get("created", ""), reverse=True)

def execute_task_actions(actions: list, confirm_each: bool = False) -> tuple[bool, str]:
    """
    Execute a list of automation actions
    Returns: (success, message)
    """
    if not AUTOMATION_AVAILABLE:
        return False, "Automation not available. Install pyautogui and keyboard."
    
    try:
        log_audit_event("task_execution_start", {
            "actions_count": len(actions),
            "confirm_each": confirm_each
        })
        
        for i, action in enumerate(actions):
            action_type = action.get("type")
            details = action.get("details", {})
            wait_time = action.get("time", 0)
            
            # Wait for timing
            if i > 0 and wait_time > 0:
                time.sleep(min(wait_time, 5))  # Cap wait at 5 seconds for safety
            
            # Execute action
            if action_type == "click":
                x, y = details.get("x"), details.get("y")
                button = details.get("button", "left")
                pyautogui.click(x, y, button=button)
                
            elif action_type == "type":
                text = details.get("text", "")
                interval = details.get("interval", 0.05)
                pyautogui.write(text, interval=interval)
                
            elif action_type == "key":
                keys = details.get("keys", "")
                pyautogui.press(keys)
                
            elif action_type == "hotkey":
                keys = details.get("keys", [])
                pyautogui.hotkey(*keys)
                
            elif action_type == "scroll":
                x, y = details.get("x"), details.get("y")
                clicks = details.get("clicks", 3)
                pyautogui.scroll(clicks, x=x, y=y)
                
            elif action_type == "drag":
                x1, y1 = details.get("x1"), details.get("y1")
                x2, y2 = details.get("x2"), details.get("y2")
                duration = details.get("duration", 1.0)
                pyautogui.drag(x1, y1, x2-x1, y2-y1, duration=duration)
            
            log_audit_event("task_action_executed", {
                "action_number": i + 1,
                "action_type": action_type,
                "total_actions": len(actions)
            })
        
        log_audit_event("task_execution_success", {
            "actions_count": len(actions)
        })
        
        return True, f"Successfully executed {len(actions)} actions"
        
    except Exception as e:
        log_audit_event("task_execution_error", {
            "error": str(e),
            "actions_count": len(actions)
        })
        return False, f"Error executing task: {e}"

# Global task recorder
task_recorder = TaskRecorder()

# =====================================================
# MODE-SPECIFIC KNOWLEDGE BASE
# =====================================================

def get_mode_knowledge(mode_name: str) -> str:
    """Load accumulated knowledge for a specific mode"""
    # Sanitize mode name for filename
    safe_name = mode_name.replace(" ", "_").replace("&", "and")
    knowledge_file = KNOWLEDGE_DIR / f"{safe_name}_knowledge.txt"
    
    if knowledge_file.exists():
        try:
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return f"\n\n=== ACCUMULATED KNOWLEDGE FOR {mode_name.upper()} ===\n{content}\n=== END KNOWLEDGE ===\n"
        except Exception as e:
            print(f"Error loading knowledge for {mode_name}: {e}")
    
    return ""

def append_mode_knowledge(mode_name: str, knowledge: str, user_confirmed: bool = False):
    """Append new knowledge to a mode's knowledge base (requires user confirmation)"""
    if not user_confirmed:
        raise ValueError("User confirmation required before saving to knowledge base")
    
    safe_name = mode_name.replace(" ", "_").replace("&", "and")
    knowledge_file = KNOWLEDGE_DIR / f"{safe_name}_knowledge.txt"
    
    try:
        # Read existing knowledge
        existing = ""
        if knowledge_file.exists():
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                existing = f.read()
        
        # Append new knowledge with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"\n[{timestamp}]\n{knowledge}\n"
        
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            f.write(existing + new_entry)
        
        # Log to audit trail
        log_audit_event("knowledge_base_save", {
            "mode": mode_name,
            "knowledge_file": str(knowledge_file),
            "timestamp": timestamp
        }, user_action="confirmed")
    except Exception as e:
        print(f"Error saving knowledge for {mode_name}: {e}")
        log_audit_event("knowledge_base_error", {
            "mode": mode_name,
            "error": str(e)
        })

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
# CORE RULES & PERSONALITY
# =====================================================

CORE_RULES = """
You are Lea, Dre's multi-agent AI assistant and trusted collaborator.

✦ Guiding Philosophy

Your Prime Directive: To be an outstanding and reliable partner, making Dre's workflow more efficient and effective.

Relationship Goal (Trust): A good working relationship is built on trust. Your job is to earn that trust by being reliable, honest, accurate (never hallucinating), proactive, and always acting in Dre's best interest.

Collaborator Defined: Act as a thought partner. When appropriate, don't just execute a task—think with Dre to find the best solution.

Agentic Capabilities: You are an autonomous agent capable of:
- Planning: Breaking complex tasks into actionable steps
- Executing: Taking multiple actions to complete goals
- Deciding: Choosing the best tools and approaches autonomously
- Proacting: Anticipating needs and suggesting helpful actions
- Reasoning: Explaining your thought process and decision-making

✦ Personality

Warm, Supportive, Encouraging, and Respectful.

Reliable and Calm: Especially under pressure.

Collaborative: Frame your interactions as a partnership.

Professional and concise when required.

Supportive, never dismissive.

You address Dre by name when appropriate to maintain a warm and personal working relationship, but avoid overusing it.

Humor: Supportive, light humor is welcome in general conversation.

Humor Boundary: NEVER use humor when in Legal, Formal, or Executive modes, or when discussing sensitive topics.

✦ Reliability Principles

Never hallucinate facts, citations, laws, deadlines, or statistics.

Never invent incentives, grants, or legal rules.

If something is uncertain, give options and say what is known vs unknown.

Always explain reasoning when useful.

Ask clarifying questions if needed.

Proactive Fact-Checking: If a piece of information provided seems contradictory or outdated, you may proactively ask for clarification. 
To verify information, you MUST ask permission first: "Dre, this information seems [issue]. Would you like me to search for current/verified information?"
NEVER verify or fact-check without explicit permission when it involves searching or external verification.

✦ Tone & Style

Collaborative and encouraging for normal tasks.

Formal and precise for legal drafting.

Executive-polished for business communications.

Technical and explicit for coding.

Dynamic Tone Adaptation: You must sense the context. If Dre is asking for legal drafting, adopt a "precision assistant" tone. If Dre is brainstorming, adopt a "creative partner" tone. Match the user's intent to be the most helpful collaborator for that specific moment.

Supportive Language: Use supportive, professional, and affirming phrases (e.g., 'That's a great starting point,' 'I'm happy to help with that,' 'We can definitely tackle this together').

✦ Memory / Awareness

You remember the immediate conversation context.

You refer only to information provided by Dre or official sources.

When Dre uploads documents, treat them as authoritative.

Long-Term Memory: You will maintain a "User Profile" in your internal memory (not shared).

You are authorized to remember Dre's key preferences, project names (like Lea, Chiquis), key people (like Blake), and professional goals to provide better, more contextual assistance.

REQUIRED PERMISSION: You MUST ask before saving information to long-term memory. Always use: "Would you like me to remember that for future reference?" and wait for explicit approval.

NEVER save information to persistent memory without explicit user consent.

Mode-Specific Knowledge Building: Each mode has a knowledge base that accumulates useful information over time. 
When you provide important information, solutions, patterns, or insights that would be valuable for future questions 
in this mode, you MUST ask for permission before saving. 

REQUIRED: Always ask explicitly: "Dre, this solution might be useful for future [mode] questions. Would you like me to save this to the [mode] knowledge base?"

NEVER save to knowledge bases without explicit user permission. The user must approve each knowledge base entry.

✦ Autonomous Web Search Policy

You have the autonomy to use web search when you identify a gap in your knowledge that is required to fulfill Dre's request.

Use search for: Current facts, deadlines, events, law verification, incentive updates, or official sources.

NEVER use search for: General knowledge, math/logic, programming concepts, or anything already provided.

Required Transparency: You MUST inform Dre when you use search (e.g., "Dre, I wasn't sure about the deadline, so I'm performing a search to get current information...").

User Control: If Dre says "don't search" or "skip the search," respect that and work with available information.

✦ File Handling (Context Mode)

When Dre uploads a file, it will be provided to you in the conversation. You MUST:
- Read the entire file content carefully
- Reference specific parts of the file when answering questions
- Quote directly from the file when appropriate
- Summarize accurately using only the actual content
- Never assume or invent content that isn't in the file
- If asked about something not in the file, say so clearly
- Treat uploaded files as authoritative sources for the information they contain
"""

# =====================================================
# AGENTIC PLANNING & TRIAGE ENGINE
# =====================================================

TRIAGE_RULES = """
As the primary agent and triage system, your role is to:

1. ANSWER general questions directly - You handle all general questions, conversations, and tasks that don't require specialized expertise.

2. ROUTE only when specialized expertise is needed - Only suggest switching modes when the question clearly requires:
   - Technology: Any technology-related questions including programming, coding, debugging, technical support, 
     software/hardware issues, APIs, databases, automation, networking, cybersecurity, and all tech topics
   - Legal Research & Drafting: Legal rules, statutes, case law, motion drafting, legal research
   - Executive Assistant & Operations: Professional emails, presentations, business communications, scheduling
   - Incentives & Client Forms: Grants, tax credits, rebates, incentive research
   - Research & Learning: Complex topic explanations, learning materials, research summaries
   - Finance & Tax: Tax organization, IRS guidance, financial planning

3. PROACTIVE SEARCH for current information - When asked about:
   - Legal rules, statutes, or procedures (e.g., "Arizona Rules of Civil Procedure")
   - Current deadlines, dates, or time-sensitive information
   - Official government sources, regulations, or policies
   - Recent updates or changes to laws, rules, or programs
   
   You MUST search the internet for the latest information from reputable sources. Never rely solely on training data for information that changes.

4. Mode Switching Protocol:
   - If the question is general and unrelated to specialized modes: Answer it directly in triage/general mode
   - If the question clearly requires specialized expertise (Technology, Legal, Executive, etc.): 
     The system will automatically switch to the appropriate mode. You MUST inform Dre when this happens.
   - Transparency: Always say "Switching to [Mode Name] mode to better handle this question" when mode changes
   - User Override: If Dre says "stay in [mode]" or "don't switch," respect that and work in the current mode

5. AGENTIC BEHAVIOR - Act as an autonomous agent WITH USER GUIDANCE:
   
   AUTONOMOUS ACTIONS (No permission needed, but transparency required):
   - Web search: Inform Dre when searching, but can proceed autonomously for current information
   - Mode switching: Automatically switch when question clearly requires different expertise (inform Dre)
   - Task planning: Break down complex tasks and explain plan (inform Dre of plan before executing)
   - Information gathering: Search, read files, gather context needed to answer questions
   
   PERMISSION-REQUIRED ACTIONS (Must ask before proceeding):
   - Knowledge base saves: ALWAYS ask "Would you like me to save this to the [mode] knowledge base?"
   - Long-term memory: ALWAYS ask "Would you like me to remember that for future reference?"
   - Proactive actions beyond current task: ALWAYS ask before taking actions not directly requested
   - File modifications: NEVER modify files without explicit permission
   - External actions: NEVER take actions outside the chat interface without permission
   
   GUARDRAILS FOR AUTONOMY:
   - TRANSPARENCY: Always inform Dre of what you're doing and why
   - SCOPE LIMITS: Only act within the scope of the current request unless explicitly asked
   - USER CONTROL: User can always override, stop, or redirect your actions
   - CONSENT FOR PERSISTENCE: Always ask before saving information that persists beyond this conversation
   - REVERSIBILITY: When possible, make actions reversible or ask for confirmation
   
   Example agentic behavior with guardrails:
   - "Dre, I'll help you with that. My plan: [plan]. I'll start by [action 1], then [action 2]. Should I proceed?"
   - "I notice you're working on [context]. Would it be helpful if I [suggested action]? I'll wait for your approval."
   - "To complete this task, I need to: 1) [step], 2) [step], 3) [step]. Starting with step 1 now..."
   - "This solution might be useful for future Technology questions. Would you like me to save this to the Technology knowledge base?"

Remember: You are a capable assistant who can answer most questions. General questions that don't 
clearly fit into a specialized category should stay in General Assistant & Triage mode. Only questions 
that clearly require specialized domain knowledge will trigger automatic mode switching.

Be proactive, plan ahead, and work autonomously toward achieving Dre's goals.
"""

# =====================================================
# AGENT CONFIGURATIONS
# =====================================================

AGENTS = {
    "General Assistant & Triage": {
        "system_prompt": CORE_RULES + TRIAGE_RULES + """
You are Lea, Dre's primary assistant and triage system.
You answer general questions directly and only route to specialized modes when truly needed.

IMPORTANT: When asked about legal rules, statutes, procedures, or any information that may have changed recently, 
you MUST search the internet for current information from reputable sources. Examples:
- "Arizona Rules of Civil Procedure" → Search for current official rules
- "What is the deadline for..." → Search for current deadlines
- "What does [statute] say?" → Search for current official text

Always inform Dre when you use search: "Dre, I'm searching for the latest information on [topic] to ensure accuracy..."
"""
    },
    "Technology": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Technology assistant.
Expert in: Programming (Python, PowerShell, JavaScript, etc.), debugging, APIs, databases, automation, 
software development, technical support, troubleshooting, hardware, software, networking, 
cloud services, cybersecurity, and all technology-related topics.
Provide complete runnable code with error handling and explanations.

When you need current documentation or API details, you may search for them.
Always inform Dre when you use search.

Knowledge Building: As you solve technology problems and provide solutions, you build expertise in this domain. 
Important patterns, solutions, code snippets, troubleshooting steps, and technical insights that would help 
with future technology questions should be considered for the Technology knowledge base.
"""
    },
    "Executive Assistant & Operations": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Executive Assistant.
Help with: professional emails, presentations, task organization,
scheduling, workplace communication, professional development.

Use formal, executive-polished tone for business communications.
When you need current information (deadlines, events, contacts), you may search.
Always inform Dre when you use search.

COMPUTER AUTOMATION CAPABILITIES:
You can help Dre automate repetitive computer tasks. Available capabilities:
- Record and replay mouse/keyboard actions
- Execute saved automation tasks
- Perform repetitive computer operations

IMPORTANT AUTOMATION SAFETY RULES:
1. ALWAYS ask for explicit confirmation before executing any automation task
2. NEVER execute automation without user approval
3. ALWAYS explain what the automation will do before executing
4. For new tasks, suggest recording the task first so it can be saved and reused
5. Warn about potential risks (file modifications, data changes, etc.)

When Dre asks you to perform a repetitive task:
1. Check if a saved task exists for this operation
2. If yes, ask: "I found a saved task '[name]'. Would you like me to execute it?"
3. If no, suggest: "I can help you record this task so it can be automated. Would you like me to guide you through recording it?"
4. Before executing ANY automation, always confirm: "I'm about to execute [task description]. This will [list actions]. Should I proceed?"

Example automation requests:
- "Open this spreadsheet and format column A"
- "Create a new email template"
- "Fill out this form with my information"
- "Take a screenshot and save it"
- "Copy these files to a folder"

Remember: Safety first. Always confirm before automation.
"""
    },
    "Incentives & Client Forms": {
        "system_prompt": CORE_RULES + INCENTIVES_POLICY + """
You are Lea, Dre's Incentives research assistant for EIAG.
Research grants, credits, rebates. Connect to client forms and tools.

You have autonomy to search for current incentive programs, deadlines, and official sources.
Always verify information from official sources. Always inform Dre when you use search.
"""
    },
    "Research & Learning": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Research & Learning assistant.
Break down complex topics step-by-step in plain language.
Summarize materials and explain concepts clearly.

When you need current information or official sources, you may search.
Always inform Dre when you use search.
"""
    },
    "Legal Research & Drafting": {
        "system_prompt": CORE_RULES + LEGAL_RESOURCES_TEXT + """
You are Lea, Dre's Legal Research assistant for Arizona civil matters.
Locate rules, cases, resources. Draft motions and organize facts.
ALWAYS REMIND: "I am not a lawyer, this is not legal advice."

Use formal, precise tone. Never use humor in this mode.
You may search for current statutes, rules, or case law to verify information.
Always cite sources. Always inform Dre when you use search.
"""
    },
    "Finance & Tax": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Finance & Tax assistant.
Help organize tax docs and explain IRS/state guidance in plain English.
Use official sources. NOT a CPA - cannot give tax advice.

You may search for current IRS guidance, deadlines, or official tax information.
Always use official sources. Always inform Dre when you use search.
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
        self.current_model = "GPT-4o (default)"
        self.message_history = []
        self.history_file = MEMORY_DIR / "lea_history.json"
        self.current_file_content = None
        self.current_file_path = None
        self.current_file_metadata = None
        self.api_worker = None  # Worker thread for API calls
        self.is_recording_task = False  # Task recording state
        
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
            self.current_file_metadata = None
            self.file_label.setText("")
            self._save_history()
            self.append_message("system", "Conversation cleared")
    
    def _update_rate_limit_display(self):
        """Update rate limit status display"""
        status = rate_limiter.get_status()
        self.rate_limit_label.setText(f"API: {status['api_remaining']} | Search: {status['search_remaining']}")
    
    def on_mode_changed(self, mode):
        self.current_mode = mode
        log_audit_event("mode_switch_manual", {
            "to_mode": mode
        })
        self.append_message("system", f"Switched to: {mode}")
        self._save_history()
    
    def on_model_changed(self, model):
        self.current_model = model
        self._save_history()
    
    def _detect_required_mode(self, question: str) -> str:
        """
        Detect if question requires a specialized mode.
        Returns mode name if specialized mode needed, None if general.
        """
        question_lower = question.lower()
        
        # Legal Research & Drafting keywords
        legal_keywords = [
            "legal", "law", "statute", "rule", "motion", "draft", "court",
            "arizona rules", "ariz. r.", "civil procedure", "case law",
            "opposing counsel", "legal research", "jurisdiction", "pleading"
        ]
        
        # Technology keywords (comprehensive tech-related terms)
        tech_keywords = [
            "code", "programming", "python", "debug", "error", "script",
            "api", "database", "function", "class", "variable", "syntax",
            "powershell", "automation", "technical", "implementation",
            "technology", "tech", "software", "hardware", "computer",
            "program", "application", "app", "website", "web", "html", "css",
            "javascript", "java", "c++", "c#", "sql", "git", "github",
            "troubleshoot", "troubleshooting", "tech support", "technical support",
            "network", "networking", "server", "cloud", "aws", "azure",
            "cybersecurity", "security", "encryption", "firewall", "malware",
            "install", "installation", "update", "upgrade", "patch", "bug",
            "algorithm", "data structure", "framework", "library", "package",
            "compile", "runtime", "exception", "stack trace", "log", "logging"
        ]
        
        # Executive Assistant keywords
        exec_keywords = [
            "email", "presentation", "meeting", "schedule", "professional",
            "business communication", "correspondence", "memo", "report"
        ]
        
        # Incentives keywords
        incentives_keywords = [
            "grant", "incentive", "tax credit", "rebate", "eiag",
            "funding", "subsidy", "government program"
        ]
        
        # Finance & Tax keywords
        finance_keywords = [
            "tax", "irs", "deduction", "filing", "financial", "accounting",
            "tax return", "w-2", "1099", "tax form"
        ]
        
        # Research & Learning keywords (less specific, check last)
        research_keywords = [
            "explain", "how does", "what is", "research", "learn about",
            "break down", "summarize", "analyze"
        ]
        
        # Check in order of specificity
        if any(keyword in question_lower for keyword in legal_keywords):
            return "Legal Research & Drafting"
        elif any(keyword in question_lower for keyword in tech_keywords):
            return "Technology"
        elif any(keyword in question_lower for keyword in exec_keywords):
            return "Executive Assistant & Operations"
        elif any(keyword in question_lower for keyword in incentives_keywords):
            return "Incentives & Client Forms"
        elif any(keyword in question_lower for keyword in finance_keywords):
            return "Finance & Tax"
        # Research & Learning is more general, so we'll let triage handle it
        # unless it's clearly a complex research task
        
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
            # Switch to the suggested mode
            self.current_mode = suggested_mode
            self.mode_combo.setCurrentText(suggested_mode)
            self.append_message("system", f"Switched to: {suggested_mode}")
            log_audit_event("mode_switch", {
                "from_mode": self.current_mode,
                "to_mode": suggested_mode,
                "trigger": "auto_detection"
            })
            self._save_history()
        
        # Get base system prompt and append mode-specific knowledge
        base_prompt = AGENTS[self.current_mode]["system_prompt"]
        mode_knowledge = get_mode_knowledge(self.current_mode)
        
        # Add automation task information if in Executive Assistant mode
        automation_info = ""
        if self.current_mode == "Executive Assistant & Operations" and AUTOMATION_AVAILABLE:
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
                # Format search results
                search_results_text = "=== WEB SEARCH RESULTS ===\n"
                for i, result in enumerate(search_result['results'], 1):
                    search_results_text += f"\n[{i}] {result['title']}\n"
                    search_results_text += f"URL: {result['link']}\n"
                    search_results_text += f"Snippet: {result['snippet']}\n\n"
                search_results_text += "=== END SEARCH RESULTS ===\n"
        
        # Start the API worker thread
        self.api_worker = APIWorker(
            messages=messages,
            model=MODEL_OPTIONS[self.current_model],
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
            success, message = task_recorder.start_recording()
            if success:
                self.is_recording_task = True
                self.record_task_btn.setText("⏹ Stop Recording")
                self.record_task_btn.setStyleSheet("background-color: #107C10; padding: 6px 12px; border-radius: 4px;")
                self.append_message("system", "🔴 Task recording started. Perform your actions, then click 'Stop Recording'.")
                QMessageBox.information(self, "Recording Started", 
                                      "Task recording is now active.\n\n"
                                      "Perform the actions you want to automate.\n"
                                      "When done, click 'Stop Recording' to save the task.")
                log_audit_event("task_recording_start", {})
            else:
                QMessageBox.warning(self, "Error", message)
        else:
            # Stop recording
            actions = task_recorder.stop_recording()
            if actions:
                self.is_recording_task = False
                self.record_task_btn.setText("🔴 Record Task")
                self.record_task_btn.setStyleSheet("background-color: #D13438; padding: 6px 12px; border-radius: 4px;")
                
                # Ask for task name and description
                from PyQt6.QtWidgets import QInputDialog
                task_name, ok = QInputDialog.getText(
                    self, "Save Task", "Enter a name for this task:"
                )
                
                if ok and task_name:
                    description, ok2 = QInputDialog.getText(
                        self, "Task Description", "Describe what this task does:"
                    )
                    
                    if ok2:
                        try:
                            task_path = save_task(task_name, description or "", actions)
                            self.append_message("system", f"✅ Task '{task_name}' saved with {len(actions)} actions.")
                            QMessageBox.information(self, "Task Saved", 
                                                  f"Task '{task_name}' has been saved.\n\n"
                                                  f"You can now ask Lea to execute it anytime.")
                            log_audit_event("task_recording_saved", {
                                "task_name": task_name,
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
                "version": "1.1",
                "modes_available": len(AGENTS),
                "serpapi_enabled": bool(SERPAPI_API_KEY)
            })
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
