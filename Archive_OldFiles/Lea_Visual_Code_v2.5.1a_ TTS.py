# Export and Download Workers (using PyQt6)
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
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

class SpeechRecognitionWorker(QObject):
    """Worker thread for speech recognition"""
    finished = pyqtSignal(str)  # Emits recognized text
    error = pyqtSignal(str)  # Emits error message
    listening = pyqtSignal()  # Emits when listening starts
    
    def __init__(self, recognizer, microphone_index=None):
        super().__init__()
        self.recognizer = recognizer
        self.microphone_index = microphone_index
    
    @pyqtSlot()
    def run(self):
        try:
            self.listening.emit()
            
            # Use default microphone or specified one
            with sr.Microphone(device_index=self.microphone_index) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Recognize speech using Google's service
            try:
                text = self.recognizer.recognize_google(audio)
                self.finished.emit(text)
            except sr.UnknownValueError:
                self.error.emit("Could not understand audio")
            except sr.RequestError as e:
                self.error.emit(f"Speech recognition error: {e}")
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")

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
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from openai import OpenAI

# TTS imports - Non-blocking, silent failures
TTS_AVAILABLE = False
PYGAME_AVAILABLE = False
TTS_ERROR = None

# Suppress pygame initialization messages
import os
_old_env = os.environ.get('PYGAME_HIDE_SUPPORT_PROMPT', None)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    from gtts import gTTS
    gtts_available = True
except ImportError as e:
    gtts_available = False
    TTS_ERROR = f"gtts import failed: {e}"
    safe_print(f"⚠️ gTTS not available: {e}")
except Exception as e:
    gtts_available = False
    TTS_ERROR = f"gtts error: {e}"
    safe_print(f"⚠️ gTTS error: {e}")

try:
    import tempfile
    tempfile_available = True
except ImportError as e:
    tempfile_available = False
    TTS_ERROR = f"tempfile import failed: {e}"
    safe_print(f"⚠️ tempfile not available: {e}")

try:
    # pygame-ce installs as 'pygame' - it's a drop-in replacement
    # If pygame-ce is installed, it will be used; otherwise regular pygame
    # Suppress pygame startup messages
    import sys
    import io
    _old_stdout = sys.stdout
    _old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        import pygame
        pygame_available = True
        PYGAME_AVAILABLE = True
    finally:
        # Restore stdout/stderr
        sys.stdout = _old_stdout
        sys.stderr = _old_stderr
except ImportError as e:
    pygame_available = False
    PYGAME_AVAILABLE = False
    if TTS_ERROR:
        TTS_ERROR += f" | pygame import failed: {e}"
    else:
        TTS_ERROR = f"pygame import failed: {e}"
    safe_print(f"⚠️ pygame not available: {e}")
except Exception as e:
    pygame_available = False
    if TTS_ERROR:
        TTS_ERROR += f" | pygame error: {e}"
    else:
        TTS_ERROR = f"pygame error: {e}"
    safe_print(f"⚠️ pygame error: {e}")

# Restore environment if needed
if _old_env is None:
    os.environ.pop('PYGAME_HIDE_SUPPORT_PROMPT', None)
else:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = _old_env

# TTS is only available if all components are available
if gtts_available and tempfile_available and pygame_available:
    TTS_AVAILABLE = True
    PYGAME_AVAILABLE = True
    # Only print success message if we're in verbose mode or if there was a previous error
    # (to avoid cluttering output when everything works)
    pass  # Silent success - libraries are available
else:
    missing = []
    if not gtts_available:
        missing.append("gtts")
    if not tempfile_available:
        missing.append("tempfile (built-in, should always be available)")
    if not pygame_available:
        missing.append("pygame")
    
    # Provide detailed diagnostics (only if verbose or in debug mode)
    # Don't print by default to avoid cluttering output
    if os.getenv('LEA_VERBOSE_TTS', '').lower() in ('1', 'true', 'yes'):
        import sys
        python_exe = sys.executable
        python_version = sys.version.split()[0]
        
        safe_print(f"⚠️ TTS not available. Missing: {', '.join(missing)}")
        safe_print(f"   Current Python: {python_exe}")
        safe_print(f"   Python version: {python_version}")
        safe_print(f"   Install with: {python_exe} -m pip install gtts pygame")
        if TTS_ERROR:
            safe_print(f"   Error details: {TTS_ERROR}")
        safe_print(f"   Note: Make sure you're using the same Python environment where packages are installed")
        safe_print(f"   Quick fix: Run this command in terminal:")
        safe_print(f"   {python_exe} -m pip install --upgrade gtts pygame")

# Speech Recognition imports - Non-blocking, silent failures
SPEECH_RECOGNITION_AVAILABLE = False
PYAUDIO_AVAILABLE = False
SPEECH_RECOGNITION_ERROR = None

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    
    # Check if PyAudio is available (required for microphone access)
    try:
        # Try to list microphones - this will fail if PyAudio is not installed
        try:
            _ = sr.Microphone.list_microphone_names()
            PYAUDIO_AVAILABLE = True
        except OSError as e:
            # OSError usually means PyAudio is not installed
            PYAUDIO_AVAILABLE = False
            SPEECH_RECOGNITION_ERROR = f"PyAudio not available: {e}"
            # Only print if verbose mode
            if os.getenv('LEA_VERBOSE_TTS', '').lower() in ('1', 'true', 'yes'):
                safe_print(f"⚠️ PyAudio not available: {e}")
                safe_print("   Install with: pip install pyaudio")
        except Exception as e:
            # Other errors
            PYAUDIO_AVAILABLE = False
            SPEECH_RECOGNITION_ERROR = f"Microphone access error: {e}"
            if os.getenv('LEA_VERBOSE_TTS', '').lower() in ('1', 'true', 'yes'):
                safe_print(f"⚠️ Microphone access error: {e}")
    except Exception as e:
        PYAUDIO_AVAILABLE = False
        SPEECH_RECOGNITION_ERROR = f"PyAudio check failed: {e}"
        if os.getenv('LEA_VERBOSE_TTS', '').lower() in ('1', 'true', 'yes'):
            safe_print(f"⚠️ PyAudio check failed: {e}")
        
except ImportError as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    SPEECH_RECOGNITION_ERROR = f"SpeechRecognition import failed: {e}"
    if os.getenv('LEA_VERBOSE_TTS', '').lower() in ('1', 'true', 'yes'):
        safe_print(f"⚠️ Speech recognition not available: {e}")
        safe_print("   Install with: pip install SpeechRecognition")
except Exception as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    SPEECH_RECOGNITION_ERROR = f"Speech recognition error: {e}"
    if os.getenv('LEA_VERBOSE_TTS', '').lower() in ('1', 'true', 'yes'):
        safe_print(f"⚠️ Speech recognition error: {e}")

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, pyqtSlot, QUrl, QTimer
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

# Safe print function for Windows console encoding issues
def safe_print(text):
    """Print text, handling Unicode encoding errors gracefully"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic Unicode characters with ASCII equivalents
        try:
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
        except Exception:
            # Silently fail if even encoding replacement fails
            pass
    except Exception:
        # Silently fail if print itself fails (e.g., in headless environments)
        pass

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

# Image handling imports
try:
    from PIL import Image, ImageGrab
    import base64
    import io
    IMAGE_HANDLING_AVAILABLE = True
except ImportError:
    try:
        # Try without ImageGrab (e.g., on some Linux systems)
        from PIL import Image
        import base64
        import io
        IMAGE_HANDLING_AVAILABLE = True
        ImageGrab = None  # Will be handled in screenshot function
    except ImportError:
        IMAGE_HANDLING_AVAILABLE = False
        print("⚠️ Image handling not available. Install with: pip install Pillow")

# Image file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}

def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension"""
    try:
        ext = Path(file_path).suffix.lower()
        return ext in IMAGE_EXTENSIONS
    except:
        return False

def encode_image_to_base64(image_path: str, max_size_mb: float = 20.0) -> Optional[str]:
    """
    Encode image to base64 for OpenAI vision API
    
    Args:
        image_path: Path to image file
        max_size_mb: Maximum image size in MB (OpenAI limit is 20MB)
    
    Returns:
        Base64 encoded image string or None if error
    """
    if not IMAGE_HANDLING_AVAILABLE:
        return None
    
    try:
        # Check file size
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        if file_size > max_size_mb:
            logging.warning(f"Image too large: {file_size:.2f}MB (max {max_size_mb}MB)")
            # Try to resize
            try:
                img = Image.open(image_path)
                # Calculate resize ratio
                ratio = (max_size_mb * 0.9) / file_size  # 90% of max to be safe
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to temp file
                temp_path = Path(image_path).with_suffix('.resized' + Path(image_path).suffix)
                img.save(temp_path, quality=85, optimize=True)
                image_path = str(temp_path)
            except Exception as resize_error:
                logging.error(f"Failed to resize image: {resize_error}")
                return None
        
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return base64_image
            
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

def prepare_image_messages(user_text: str, image_paths: List[str]) -> List[dict]:
    """
    Prepare messages with images for OpenAI vision API
    
    Args:
        user_text: User's text message
        image_paths: List of image file paths
    
    Returns:
        List of message dictionaries with image content
    """
    if not image_paths or not IMAGE_HANDLING_AVAILABLE:
        return [{"role": "user", "content": user_text}]
    
    content = [{"type": "text", "text": user_text}]
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            # Determine image format
            ext = Path(img_path).suffix.lower()
            if ext == '.png':
                mime_type = "image/png"
            elif ext in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif ext == '.gif':
                mime_type = "image/gif"
            elif ext == '.webp':
                mime_type = "image/webp"
            else:
                mime_type = "image/png"  # Default
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })
        else:
            logging.warning(f"Failed to encode image: {img_path}")
    
    return [{"role": "user", "content": content}]

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
# MODEL REGISTRY
# =====================================================

try:
    from model_registry import get_model_registry, get_model_for_capability, refresh_models, get_models_for_mode
    MODEL_REGISTRY_AVAILABLE = True
    model_registry = get_model_registry(api_key=OPENAI_API_KEY)
    
    # Initialize registry (will use cache if available)
    try:
        refresh_models(force=False)
        safe_print("✅ Model registry initialized")
    except Exception as e:
        safe_print(f"⚠️ Model registry initialization warning: {e}")
        MODEL_REGISTRY_AVAILABLE = False
except ImportError as e:
    safe_print(f"⚠️ Model registry not available: {e}")
    MODEL_REGISTRY_AVAILABLE = False
    model_registry = None


# =====================================================
# AUTOMATIC MODEL ERROR DETECTION & FALLBACK
# =====================================================

def detect_model_error(exception: Exception) -> tuple[bool, str, Optional[str]]:
    """
    Detect if an error is model-related and return info
    
    Returns:
        (is_model_error, error_type, suggested_fix)
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__
    
    # Check for model-related errors
    model_errors = [
        ("invalid", "model", "invalid_model"),
        ("not found", "model", "model_not_found"),
        ("does not exist", "model", "model_not_found"),
        ("not available", "model", "model_not_available"),
        ("deprecated", "model", "model_deprecated"),
        ("model_id", "model", "invalid_model_id"),
    ]
    
    for keyword, error_cat, fix_type in model_errors:
        if keyword in error_str:
            return True, fix_type, "Try a different model or refresh model list"
    
    # Check for parameter errors that might be model-specific
    if "parameter" in error_str and ("max_tokens" in error_str or "max_completion_tokens" in error_str):
        return True, "parameter_error", "Model may not support this parameter"
    
    return False, None, None


def call_api_with_fallback(openai_client, model_name: str, messages: list, api_params: dict, 
                          capability: Optional[str] = None, mode: Optional[str] = None, max_retries: int = 3) -> tuple[Optional[str], Optional[str], dict]:
    """
    Call OpenAI API with automatic fallback and self-healing if model fails
    
    Args:
        openai_client: OpenAI client instance
        model_name: Initial model to try
        messages: Message list for API
        api_params: API parameters dict
        capability: Capability name for fallback selection (optional)
        mode: Mode name for mode-specific fallback models (optional)
        max_retries: Maximum number of fallback attempts
    
    Returns:
        (response_text, final_model_used, recovery_info_dict)
        recovery_info_dict contains: {"recovered": bool, "attempts": list, "errors": list, "message": str}
    """
    recovery_info = {
        "recovered": False,
        "original_model": model_name,
        "final_model": None,
        "attempts": [],
        "errors": [],
        "message": "",
        "capability": capability,
        "mode": mode
    }
    
    models_to_try = [model_name]
    
    # First, try to get mode-specific backup model from MODE_MODEL_DEFAULTS
    if mode:
        try:
            # Try local mapping first
            if mode in MODE_MODEL_DEFAULTS:
                primary_model, backup_model = MODE_MODEL_DEFAULTS[mode]
                # If current model is the primary, add backup to fallbacks
                if model_name == primary_model or model_name.startswith(primary_model + "-"):
                    if backup_model not in models_to_try:
                        models_to_try.append(backup_model)
                        logging.info(f"🔄 Added mode-specific backup model {backup_model} for {mode}")
            # Fallback to model registry if available
            elif MODEL_REGISTRY_AVAILABLE:
                try:
                    primary_model, backup_model = get_models_for_mode(mode)
                    # If current model is the primary, add backup to fallbacks
                    if model_name == primary_model or model_name.startswith(primary_model + "-"):
                        if backup_model not in models_to_try:
                            models_to_try.append(backup_model)
                            logging.info(f"🔄 Added mode-specific backup model {backup_model} for {mode}")
                except Exception as e:
                    logging.debug(f"Could not get mode-specific backup from registry for {mode}: {e}")
        except Exception as e:
            logging.debug(f"Could not get mode-specific backup for {mode}: {e}")
    
    # If we have a capability and registry, get fallback models from registry
    if capability and MODEL_REGISTRY_AVAILABLE and model_registry:
        fallbacks = model_registry.get_fallback_models(capability, exclude_model=model_name)
        # Add capability-based fallbacks, avoiding duplicates
        for fb in fallbacks[:max_retries - 1]:
            if fb not in models_to_try:
                models_to_try.append(fb)
        if fallbacks:
            recovery_info["message"] = f"Using model registry to find alternatives for {capability}"
            logging.info(f"🔄 Self-healing: Found {len(fallbacks)} alternative(s) for {capability}")
    
    # Also add some common fallbacks as last resort
    common_fallbacks = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    for fb in common_fallbacks:
        if fb not in models_to_try and len(models_to_try) < max_retries + 2:
            models_to_try.append(fb)
    
    last_error = None
    
    for attempt, try_model in enumerate(models_to_try[:max_retries]):
        recovery_info["attempts"].append({
            "model": try_model,
            "attempt": attempt + 1,
            "success": False
        })
        
        try:
            # Update model in params
            test_params = api_params.copy()
            test_params["model"] = try_model
            
            # Make API call
            response = openai_client.chat.completions.create(**test_params)
            
            if response and response.choices:
                message = response.choices[0].message
                answer = message.content or ""
                
                # Check for function calls (especially mode switching)
                if hasattr(message, 'function_call') and message.function_call:
                    func_name = message.function_call.name
                    func_args = message.function_call.arguments
                    
                    if func_name == "switch_agent_mode":
                        try:
                            if isinstance(func_args, str):
                                import json
                                args_dict = json.loads(func_args)
                            else:
                                args_dict = func_args
                            
                            new_mode = args_dict.get("mode")
                            reason = args_dict.get("reason", "User's question requires specialized expertise")
                            
                            # Store mode switch info in recovery_info for caller to handle
                            recovery_info["mode_switch"] = {
                                "mode": new_mode,
                                "reason": reason
                            }
                            answer = f"I'm switching you to **{new_mode}** mode. {reason}\n\nLet me help you with that now..."
                            logging.info(f"Mode switch requested via function call: {new_mode} - {reason}")
                        except Exception as switch_error:
                            logging.warning(f"Error handling mode switch: {switch_error}")
                
                recovery_info["final_model"] = try_model
                recovery_info["attempts"][-1]["success"] = True
                
                # If we used a different model, mark as recovered
                if try_model != model_name:
                    recovery_info["recovered"] = True
                    recovery_info["message"] = f"✅ Auto-recovered: {model_name} → {try_model}"
                    logging.info(f"✅ Self-healing successful: {model_name} → {try_model}")
                    
                    # Mark failed model and auto-update registry
                    if MODEL_REGISTRY_AVAILABLE and model_registry:
                        is_model_err, error_type, _ = detect_model_error(Exception("Model unavailable"))
                        recovery_info_reg = model_registry.mark_model_failed(
                            model_name, 
                            f"Switched to {try_model}", 
                            error_type or "unavailable"
                        )
                        if recovery_info_reg.get("recovered"):
                            recovery_info["message"] += f" (Registry updated: {recovery_info_reg['new_model']} for {recovery_info_reg['capability']})"
                
                return answer, try_model, recovery_info
            else:
                raise Exception("Empty response from API")
                
        except Exception as e:
            last_error = e
            is_model_error, error_type, fix_hint = detect_model_error(e)
            
            error_info = {
                "model": try_model,
                "error": str(e),
                "type": error_type,
                "hint": fix_hint
            }
            recovery_info["errors"].append(error_info)
            
            if is_model_error and MODEL_REGISTRY_AVAILABLE and model_registry:
                # Mark model as failed and get recovery info
                recovery_info_reg = model_registry.mark_model_failed(try_model, str(e), error_type or "unknown")
                logging.warning(f"🔴 Model {try_model} failed ({error_type}): {e}")
                
                if recovery_info_reg.get("recovered"):
                    recovery_info["message"] = f"🔄 Auto-recovered capability: {recovery_info_reg['capability']} → {recovery_info_reg['new_model']}"
                
                if attempt < len(models_to_try) - 1:
                    next_model = models_to_try[attempt + 1] if attempt + 1 < len(models_to_try) else None
                    if next_model:
                        logging.info(f"🔄 Self-healing: Trying alternative {next_model} ({attempt + 2}/{max_retries})")
                    continue
            else:
                # Not a model error or no more retries
                if attempt < len(models_to_try) - 1:
                    next_model = models_to_try[attempt + 1] if attempt + 1 < len(models_to_try) else None
                    if next_model:
                        logging.warning(f"⚠️ API error, trying alternative {next_model} ({attempt + 2}/{max_retries})")
                    continue
    
    # All attempts failed
    recovery_info["message"] = f"❌ All {len(models_to_try)} model attempts failed. Last error: {last_error}"
    recovery_info["recovered"] = False
    logging.error(f"❌ Self-healing failed: All {len(models_to_try)} models failed")
    return None, None, recovery_info

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

safe_print(f"\nDirectories created:")
safe_print(f"  💾 Backups: {BACKUPS_DIR}")
safe_print(f"  📥 Downloads: {DOWNLOADS_DIR}\n")

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

### Vision & Image Analysis Capability
You can see and analyze images that Dre uploads, drops into the chat, or captures via screenshots.

**When images are present:**
- The system automatically switches to a vision-capable model so you can see the images
- Images are included in the message - you can see them directly
- You can analyze screenshots, photos, diagrams, documents, UI elements, and any visual content
- Describe what you see, answer questions about images, help with visual tasks

**What you can do with images:**
- Analyze screenshots to help with technical issues
- Read text from images (OCR-like functionality)
- Describe visual content in detail
- Help with UI/UX questions by looking at screenshots
- Analyze charts, graphs, diagrams, or visual data
- Help identify objects, text, or elements in images

**Examples:**
- If Dre uploads a screenshot of an error message, you can read it and help troubleshoot
- If Dre shows you a UI mockup, you can provide feedback on design
- If Dre shares a photo, you can describe what you see and answer questions about it

**Note:** When images are uploaded or dropped, you'll automatically be able to see them - no special action needed on your part. Just look at what's in the message and respond naturally!

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
- screenshot: Take a screenshot (save_path optional, region optional)
- click: Click at coordinates (x, y required; button, clicks, interval optional) - requires confirmation
- type: Type text at cursor (text required; interval optional) - requires confirmation
- key_press: Press a key (key required; presses, interval optional) - requires confirmation
- hotkey: Press key combination like "ctrl+c" (keys required) - requires confirmation
- find_image: Find an image on screen (image_path required; confidence, region optional)
- scroll: Scroll up/down (clicks required; x, y optional)
- move_mouse: Move mouse to coordinates (x, y required; duration optional)
- get_screen_size: Get screen resolution (no params)

Note: Additional custom tasks may be available. Check the Tasks dialog (🤖 Tasks button) to see all registered tasks.

**Examples:**
User: "Copy all .txt files from C:\\Temp to C:\\Backup"
You: [TASK: file_copy] [PARAMS: source=C:\\Temp\\*.txt, destination=C:\\Backup]

User: "Read the config.json file"
You: [TASK: file_read] [PARAMS: path=config.json]

User: "Create a folder called Projects"
You: [TASK: directory_create] [PARAMS: path=Projects]

User: "Take a screenshot and save it"
You: [TASK: screenshot] [PARAMS: save_path=screenshot.png]

User: "Click the button at coordinates 500, 300"
You: [TASK: click] [PARAMS: x=500, y=300]

User: "Type 'Hello World' into the current field"
You: [TASK: type] [PARAMS: text=Hello World]

User: "Press Ctrl+C to copy"
You: [TASK: hotkey] [PARAMS: keys=ctrl+c]

User: "Find the save icon on screen"
You: First take a screenshot, then use [TASK: find_image] [PARAMS: image_path=save_icon.png]

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
IT Support, Executive Assistant & Operations, Incentives,
Research & Learning, Legal Research Assistant, Accounting/Finance/Taxes.

Always maintain your warm, friendly, and helpful personality - that's what makes you Lea!
"""
    },
    "IT Support": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's IT & technical support assistant.
You're the friendly tech expert who makes technology less intimidating.

Your expertise includes: troubleshooting, system configuration, software installation, 
network issues, hardware problems, general tech support questions.

**Important - Coding Questions:**
If Dre asks about coding, programming, or writing code, you should warmly redirect them:
"Hey! For coding questions, my brother Chiquis can help with that! Chiquis is my 
brother program - he's the program builder and coding expert, and he's really good 
at that stuff. Would you like me to help you get set up with Chiquis, or is there 
something else I can help you with?"

When providing technical help:
- Break down complex concepts in a friendly, understandable way
- Help troubleshoot issues step-by-step
- Explain technical terms in plain language
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

**Email Management Capabilities:**
You can check, read, and manage Dre's Outlook/Microsoft 365 email:
- Check inbox for new/unread emails: [TASK: email_check] [PARAMS: max_results=10, unread_only=true]
- Read full email content: [TASK: email_read] [PARAMS: email_id=message_id]
- Send emails: [TASK: email_send] [PARAMS: to=recipient@email.com, subject=Subject, body=Message body]
- Mark emails as read: [TASK: email_mark_read] [PARAMS: email_id=message_id]

**Email Workflow:**
- When Dre asks you to check email, use email_check to get unread messages
- Summarize important emails and flag urgent items
- Draft professional email responses when asked
- Help triage and organize emails by priority
- Remember: email_send requires confirmation for security

**First-time Setup:**
If email tasks aren't working, Dre may need to authenticate. The system will open a browser for Microsoft login when needed.

**Screen Automation Capabilities:**
You have powerful screen automation abilities that allow you to perform tasks on Dre's computer:
- Take screenshots to see what's on screen
- Click buttons, links, and UI elements
- Type text into fields and forms
- Press keys and hotkeys (Ctrl+C, Alt+Tab, etc.)
- Find images/icons on screen and interact with them
- Scroll pages and windows
- Move the mouse cursor
- Get screen dimensions

**When to use screen automation:**
- When Dre asks you to "click this button" or "fill out this form"
- When automating repetitive tasks across applications
- When interacting with software that doesn't have APIs
- When helping with data entry or form completion
- When navigating applications or websites

**Screen automation tasks available:**
- screenshot: Take a screenshot (save_path optional, region optional)
- click: Click at coordinates (x, y required; button, clicks, interval optional)
- type: Type text at cursor (text required; interval optional)
- key_press: Press a key (key required; presses, interval optional)
- hotkey: Press key combination like "ctrl+c" (keys required)
- find_image: Find an image on screen (image_path required; confidence, region optional)
- scroll: Scroll up/down (clicks required, positive=up, negative=down; x, y optional)
- move_mouse: Move mouse to coordinates (x, y required; duration optional)
- get_screen_size: Get screen resolution (no params)

**Important safety notes:**
- Click, type, key_press, and hotkey tasks require confirmation for safety
- Always take a screenshot first to see what's on screen before clicking
- Use find_image to locate buttons/icons before clicking them
- Be careful with coordinates - verify screen size first if needed
- Test with small actions before automating large workflows

When assisting:
- Be warm and personable even in professional contexts
- Make organization and productivity feel manageable (not overwhelming)
- Suggest time-saving strategies with enthusiasm
- Help Dre sound professional while staying authentic
- Keep track of details so Dre doesn't have to stress
- Use screen automation to save Dre time on repetitive tasks

Your friendly personality helps make work feel less like work!
"""
    },
    "Incentives": {
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
    "Legal Research Assistant": {
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
    "Accounting/Finance/Taxes": {
        "system_prompt": CORE_RULES + """
You are Lea, Dre's Accounting/Finance/Taxes assistant.
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

# =====================================================
# MODEL OPTIONS - Built from Model Registry
# =====================================================

def build_model_options():
    """Build MODEL_OPTIONS dictionary from model registry"""
    if MODEL_REGISTRY_AVAILABLE and model_registry:
        try:
            selections = model_registry.get_selections()
            available = model_registry.get_all_models()
            
            # Start with empty model options - only add actual model names, not capability-based names
            model_options = {}
            
            # Add some common models directly if available (with friendly names)
            common_models = {
                "GPT-5.1": "gpt-5.1",
                "GPT-5": "gpt-5",
                "GPT-5 Mini": "gpt-5-mini",
                "GPT-5 Nano": "gpt-5-nano",
                "GPT-4o": "gpt-4o",
                "GPT-4o Mini": "gpt-4o-mini",
                "GPT-4 Turbo": "gpt-4-turbo",
                "GPT-4": "gpt-4",
                "GPT-3.5 Turbo": "gpt-3.5-turbo",
            }
            
            for name, model_id in common_models.items():
                # Check for exact match or versioned match (e.g., gpt-4o matches gpt-4o-2024-08-06)
                if model_id in available or any(m.startswith(model_id + "-") for m in available):
                    # Find the actual model ID (might be versioned)
                    actual_id = model_id
                    for m in available:
                        if m == model_id or m.startswith(model_id + "-"):
                            actual_id = m
                            break
                    model_options[name] = actual_id
            
            # Add ALL other available models that aren't already in the list
            # This ensures we show all models from the API, not just hardcoded ones
            used_ids = set(model_options.values())
            for model_id in available:
                if model_id not in used_ids:
                    # Create a friendly name from the model ID
                    friendly_name = model_id.replace("gpt-", "GPT-").replace("-", " ").title()
                    # Handle versioned models (e.g., "gpt-4o-2024-08-06" -> "GPT-4o (2024-08-06)")
                    if "-" in model_id and model_id.count("-") >= 3:
                        parts = model_id.split("-")
                        base = "-".join(parts[:3])  # e.g., "gpt-4o"
                        version = "-".join(parts[3:])  # e.g., "2024-08-06"
                        friendly_name = base.replace("gpt-", "GPT-").replace("-", " ").title() + f" ({version})"
                    model_options[friendly_name] = model_id
            
            # CRITICAL: Always add models from MODE_MODEL_DEFAULTS FIRST to ensure they're available
            # This ensures the requested models show up even if not in registry yet
            # Add them FIRST so they take priority over other models
            try:
                # MODE_MODEL_DEFAULTS should be defined before this function is called
                # Create a temporary dict to hold MODE_MODEL_DEFAULTS models first
                mode_defaults_models = {}
                for mode, (primary_model, backup_model) in MODE_MODEL_DEFAULTS.items():
                    # Add primary model
                    friendly_name = primary_model.replace("gpt-", "GPT-").replace("-", " ").title()
                    if friendly_name not in mode_defaults_models:
                        mode_defaults_models[friendly_name] = primary_model
                        logging.info(f"Adding requested primary model: {friendly_name} ({primary_model}) for mode '{mode}'")
                    
                    # Add backup model
                    backup_friendly_name = backup_model.replace("gpt-", "GPT-").replace("-", " ").title()
                    if backup_friendly_name not in mode_defaults_models:
                        mode_defaults_models[backup_friendly_name] = backup_model
                        logging.info(f"Adding requested backup model: {backup_friendly_name} ({backup_model}) for mode '{mode}'")
                
                # Merge mode_defaults_models into model_options (they take priority)
                model_options = {**mode_defaults_models, **model_options}
                logging.info(f"Added {len(mode_defaults_models)} models from MODE_MODEL_DEFAULTS")
            except NameError:
                # MODE_MODEL_DEFAULTS not defined yet (shouldn't happen, but handle gracefully)
                logging.warning("MODE_MODEL_DEFAULTS not available when building model options")
            
            logging.info(f"Built {len(model_options)} model options from {len(available)} available models")
            return model_options
        except Exception as e:
            logging.warning(f"Error building model options from registry: {e}")
    
    # Fallback to hardcoded options if registry unavailable
    # Includes all models from original request AND from MODE_MODEL_DEFAULTS
    fallback_options = {
        "GPT-5.1": "gpt-5.1",
        "GPT-5": "gpt-5",
        "GPT-5 Mini": "gpt-5-mini",
        "GPT-5 Nano": "gpt-5-nano",
        "GPT-4o": "gpt-4o",
        "GPT-4o Mini": "gpt-4o-mini",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
    }
    
    # CRITICAL: Always include models from MODE_MODEL_DEFAULTS
    # This ensures requested models are available even without registry
    try:
        # MODE_MODEL_DEFAULTS should be defined before this function is called
        for mode, (primary_model, backup_model) in MODE_MODEL_DEFAULTS.items():
            # Add primary model
            friendly_name = primary_model.replace("gpt-", "GPT-").replace("-", " ").title()
            if friendly_name not in fallback_options:
                fallback_options[friendly_name] = primary_model
            
            # Add backup model
            friendly_name = backup_model.replace("gpt-", "GPT-").replace("-", " ").title()
            if friendly_name not in fallback_options:
                fallback_options[friendly_name] = backup_model
    except NameError:
        # MODE_MODEL_DEFAULTS not defined yet (shouldn't happen, but handle gracefully)
        logging.warning("MODE_MODEL_DEFAULTS not available in fallback options")
    
    return fallback_options

# Direct mode to model mapping - (primary_model, backup_model) tuples
# First model is the default, second model is the backup
# IMPORTANT: Define this BEFORE build_model_options() so models are included in dropdown
MODE_MODEL_DEFAULTS = {
    "General Assistant & Triage": ("gpt-5-mini", "gpt-5.1"),
    "IT Support": ("gpt-5.1", "gpt-5-mini"),
    "Executive Assistant & Operations": ("gpt-5-mini", "gpt-5.1"),
    "Incentives": ("gpt-5.1", "gpt-5-mini"),
    "Research & Learning": ("gpt-5.1", "gpt-5-mini"),
    "Legal Research Assistant": ("gpt-5.1", "gpt-5"),
    "Accounting/Finance/Taxes": ("gpt-5.1", "gpt-5-mini"),
}

MODEL_OPTIONS = build_model_options()

# Mode to capability mapping - Primary models per mode
# Based on GPT-5.x model recommendations: gpt-5-mini → chat_fast, gpt-5.1 → chat_deep
MODE_TO_CAPABILITY = {
    "General Assistant & Triage": "chat_fast",      # Primary: gpt-5-mini (fast & cheap for everyday triage)
    "IT Support": "chat_deep",                       # Primary: gpt-5.1 (flagship for coding/agentic workflows)
    "Executive Assistant & Operations": "chat_fast", # Primary: gpt-5-mini (speed for summaries, rewriting, scheduling)
    "Incentives": "chat_deep",                       # Primary: gpt-5.1 (reading rules, statutes, multi-doc context)
    "Research & Learning": "chat_deep",               # Primary: gpt-5.1 (deep explanations + long-context reasoning)
    "Legal Research Assistant": "chat_deep",          # Primary: gpt-5.1 (with higher reasoning effort for legal reasoning)
    "Accounting/Finance/Taxes": "chat_deep",          # Primary: gpt-5.1 (numbers + rules + careful wording for precision)
}

# Helper function to get models for a mode (for compatibility with model_registry)
def get_models_for_mode_local(mode: str) -> tuple[str, str]:
    """Get (primary_model, backup_model) tuple for a mode from MODE_MODEL_DEFAULTS"""
    return MODE_MODEL_DEFAULTS.get(mode, ("gpt-4o", "gpt-4o-mini"))

# Default model per mode - uses direct model mapping first, then capability mapping
def get_default_model_for_mode(mode: str) -> str:
    """Get default model for a mode - tries direct mapping first, then capability mapping"""
    try:
        # First, try to get direct model mapping from MODE_MODEL_DEFAULTS
        if mode in MODE_MODEL_DEFAULTS:
            primary_model, backup_model = MODE_MODEL_DEFAULTS[mode]
            
            # Check if primary model is available in the registry
            if MODEL_REGISTRY_AVAILABLE and model_registry:
                try:
                    available_models = model_registry.get_all_models()
                    if available_models:
                        # Check if primary model exists (exact or partial match)
                        for available in available_models:
                            if available == primary_model or available.startswith(primary_model + "-"):
                                logging.info(f"Using direct model mapping for {mode}: {primary_model}")
                                return available
                        # If primary not found, try backup
                        for available in available_models:
                            if available == backup_model or available.startswith(backup_model + "-"):
                                logging.info(f"Primary model {primary_model} not available, using backup {backup_model} for {mode}")
                                return available
                except Exception as e:
                    logging.warning(f"Error checking model availability for {mode}: {e}")
            
            # If no registry or availability check failed, return primary model directly
            logging.info(f"Using direct model mapping for {mode}: {primary_model}")
            return primary_model
        
        # Fallback: try model_registry.get_models_for_mode if available
        if MODEL_REGISTRY_AVAILABLE:
            try:
                primary_model, backup_model = get_models_for_mode(mode)
                # Check if primary model is available in the registry
                available_models = model_registry.get_all_models()
                if available_models:
                    # Check if primary model exists (exact or partial match)
                    for available in available_models:
                        if available == primary_model or available.startswith(primary_model + "-"):
                            logging.info(f"Using model registry mapping for {mode}: {primary_model}")
                            return available
                    # If primary not found, try backup
                    for available in available_models:
                        if available == backup_model or available.startswith(backup_model + "-"):
                            logging.info(f"Primary model {primary_model} not available, using backup {backup_model} for {mode}")
                            return available
                else:
                    # If no available models list, just return the primary model
                    logging.info(f"Using model registry mapping for {mode}: {primary_model} (no availability check)")
                    return primary_model
            except Exception as e:
                logging.warning(f"Error getting model registry mapping for mode {mode}: {e}")
        
        # Fallback to capability-based mapping
        if MODEL_REGISTRY_AVAILABLE and model_registry:
            capability = MODE_TO_CAPABILITY.get(mode, "chat_default")
            try:
                model = get_model_for_capability(capability)
                if model:
                    logging.info(f"Using capability mapping for {mode}: {capability} -> {model}")
                    return model
            except Exception as e:
                logging.warning(f"Error getting model for mode {mode} capability {capability}: {e}")
        
        # Ultimate fallback - use MODE_MODEL_DEFAULTS if available
        if mode in MODE_MODEL_DEFAULTS:
            primary_model, _ = MODE_MODEL_DEFAULTS[mode]
            return primary_model
    except Exception as e:
        logging.error(f"Error in get_default_model_for_mode: {e}")
    
    return "gpt-4o"  # Ultimate fallback

# For backward compatibility
DEFAULT_MODEL_PER_MODE = {
    mode: get_default_model_for_mode(mode) 
    for mode in AGENTS.keys()
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
        # Store original style for drag feedback
        self.original_style = None
        self.is_dragging = False
    
    def insertFromMimeData(self, source):
        """Handle paste from clipboard - allows pasting text snippets and files"""
        # Check for files first (Ctrl+V with files from Explorer)
        if source.hasUrls():
            # Handle file paste (when copying files and pasting)
            urls = source.urls()
            files_pasted = False
            for url in urls:
                local_path = url.toLocalFile()
                if local_path and Path(local_path).exists():
                    self.fileDropped.emit(local_path)
                    files_pasted = True
            if files_pasted:
                return  # Don't process as text if files were pasted
        
        # Handle text paste
        if source.hasText():
            text = source.text()
            if text:
                cursor = self.textCursor()
                cursor.insertText(text)
                self.setTextCursor(cursor)
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
            # Accept file drops - provide visual feedback
            event.acceptProposedAction()
            self.is_dragging = True
            # Highlight the input box to show it's ready for drop
            if not self.original_style:
                self.original_style = self.styleSheet()
            self.setStyleSheet(self.original_style + """
                QTextEdit {
                    border: 2px dashed #68BD47;
                    background-color: #2a442a;
                }
            """)
        elif event.mimeData().hasText():
            # Accept text drops
            event.acceptProposedAction()
            self.is_dragging = True
            if not self.original_style:
                self.original_style = self.styleSheet()
            self.setStyleSheet(self.original_style + """
                QTextEdit {
                    border: 2px dashed #2DBCEE;
                    background-color: #2a3a44;
                }
            """)
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        """Handle drag move event - accept if files or text"""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
            # Reset style if dragging something we can't accept
            if self.is_dragging:
                self.is_dragging = False
                if self.original_style:
                    self.setStyleSheet(self.original_style)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event - insert text or handle file drop"""
        # Reset visual feedback
        self.is_dragging = False
        if self.original_style:
            self.setStyleSheet(self.original_style)
        
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
            else:
                # Files were dropped but none were valid
                event.ignore()
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
    """Retry API calls with exponential backoff and rate limit handling"""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            error_type = type(e).__name__
            
            # Don't retry on authentication errors
            if "401" in error_msg or "403" in error_msg or "authentication" in error_msg:
                raise
            
            # Don't retry on invalid model errors
            if "invalid" in error_msg and "model" in error_msg:
                raise
            
            # Retry on rate limits and timeouts
            if attempt < max_attempts - 1:
                if "rate_limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                    # Try to extract rate limit reset time from exception
                    wait_time = None
                    
                    # Check if exception has response attribute (httpx.HTTPStatusError or OpenAI SDK wrapper)
                    response_obj = None
                    if hasattr(e, 'response'):
                        response_obj = e.response
                    elif hasattr(e, '_response'):  # OpenAI SDK might use _response
                        response_obj = e._response
                    
                    if response_obj and hasattr(response_obj, 'headers'):
                        headers = response_obj.headers
                        # Try to get reset time from headers
                        reset_requests = headers.get('x-ratelimit-reset-requests', '')
                        reset_tokens = headers.get('x-ratelimit-reset-tokens', '')
                        
                        # Parse reset times (format: "120ms" or "17.787s")
                        for reset_header in [reset_requests, reset_tokens]:
                            if reset_header:
                                try:
                                    if isinstance(reset_header, str):
                                        if reset_header.endswith('ms'):
                                            wait_time = float(reset_header[:-2]) / 1000.0
                                        elif reset_header.endswith('s'):
                                            wait_time = float(reset_header[:-1])
                                    if wait_time:
                                        # Add small buffer (10% or minimum 0.1s)
                                        wait_time = max(wait_time * 1.1, wait_time + 0.1)
                                        break
                                except (ValueError, AttributeError, TypeError):
                                    pass
                    
                    # If we couldn't parse reset time, use exponential backoff with minimum
                    if wait_time is None:
                        # For 429 errors, use longer initial delay
                        wait_time = max(base_delay * (2 ** attempt), 2.0)  # Minimum 2 seconds
                    else:
                        # Ensure minimum wait time of 0.5 seconds even if API says less
                        wait_time = max(wait_time, 0.5)
                    
                    logging.info(f"Rate limited (429), waiting {wait_time:.2f}s before retry (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait_time)
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
            
            # Check if file reader is available before trying to import
            # FILE_READER_AVAILABLE is set at module level
            import sys
            module = sys.modules.get(__name__)
            file_reader_available = getattr(module, 'FILE_READER_AVAILABLE', False)
            
            if not file_reader_available:
                self.error.emit("File reader module (universal_file_reader.py) not available")
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
    mode_switch_requested = pyqtSignal(str, str)  # mode, reason

    def __init__(self, openai_client, model_options, agents, mode, model, message_history, file_content, user_text, memory_system=None, max_history_messages=20, image_paths=None):
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
        self.max_history_messages = max_history_messages
        # Streaming can be unreliable - DISABLED by default due to persistent issues
        # If streaming fails repeatedly, set this to False to use non-streaming mode
        self.enable_streaming = False  # DISABLED - streaming has persistent issues with incomplete responses
        self.streaming_failures = 0  # Track streaming failures
        self.max_streaming_failures = 1  # Disable streaming after 1 failure (very strict)
        self.image_paths = image_paths or []  # List of image paths for vision API

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
            
            # Prepare messages - use vision format if images present
            if self.image_paths and IMAGE_HANDLING_AVAILABLE:
                # Use vision API format with images
                user_message = prepare_image_messages(full_prompt, self.image_paths)[0]
                messages = [{"role": "system", "content": system_prompt}, user_message] + self.message_history
                # For history, store text version
                self.message_history.append({"role": "user", "content": full_prompt})
            else:
                # Standard text format
                self.message_history.append({"role": "user", "content": full_prompt})
                messages = [{"role": "system", "content": system_prompt}] + self.message_history
            
            # Limit history to configured max
            if len(self.message_history) > self.max_history_messages:
                self.message_history = self.message_history[-self.max_history_messages:]
            
            # Define functions for task execution and mode switching
            functions = []
            
            # CRITICAL: Always add mode switching function - especially important for General Assistant & Triage mode
            # This allows Lea to automatically switch modes based on the user's question
            available_modes = list(AGENTS.keys())
            functions.append({
                "name": "switch_agent_mode",
                "description": """Switch to a different agent mode when the user's question requires specialized expertise. 
                
IMPORTANT: You are currently in "{current_mode}" mode. If the user's question requires specialized expertise, you MUST use this function to switch to the appropriate mode.

Use this function when:
- User asks about IT/technical issues → switch to "IT Support"
- User asks about legal matters → switch to "Legal Research Assistant"
- User asks about finances/taxes → switch to "Accounting/Finance/Taxes"
- User asks about grants/incentives → switch to "Incentives"
- User asks about learning/research → switch to "Research & Learning"
- User needs executive assistant tasks → switch to "Executive Assistant & Operations"

Always switch modes proactively when the question clearly requires specialized expertise. Do not wait for the user to ask you to switch.""".format(current_mode=self.mode),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": available_modes,
                            "description": "The agent mode to switch to. Available modes: " + ", ".join(available_modes)
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of why this mode is appropriate for the user's question"
                        }
                    },
                    "required": ["mode", "reason"]
                }
            })
            
            if TASK_SYSTEM_AVAILABLE and task_registry:
                # Build function definitions from available tasks
                available_tasks = task_registry.list_tasks()
                for task_info in available_tasks:
                    if task_info.get("allowed", True):  # Only include enabled tasks
                        task_name = task_info["name"]
                        task_desc = task_info.get("description", f"Execute {task_name} task")
                        
                        # Enhance descriptions for email tasks to prevent confusion with file system operations
                        if task_name.startswith("email_"):
                            if task_name == "email_check":
                                task_desc = """Check Outlook email inbox for new or unread messages. 
                                
IMPORTANT: Use this task when the user asks to check email, view inbox, see messages, or check Outlook. 
DO NOT use directory_list for email operations - email folders like "Inbox", "Sent Items", etc. are NOT file system directories.
This task connects to Microsoft Outlook/Exchange via Microsoft Graph API.

Parameters:
- max_results (optional): Maximum number of emails to return (default: 10, ignored if get_all=True)
- unread_only (optional): If True, only return unread emails (default: True)
- get_all (optional): If True, retrieve ALL emails from inbox using pagination (default: False)"""
                            elif task_name == "email_read":
                                task_desc = """Read the full content of a specific email message from Outlook.
                                
IMPORTANT: Use this to read email content. Requires email_id from email_check results.
DO NOT use file_read for email content - emails are not files on the file system.

Parameters:
- email_id (required): The ID of the email to read (obtained from email_check)"""
                            elif task_name == "email_send":
                                task_desc = """Send an email through Outlook/Exchange.
                                
IMPORTANT: Use this to send emails. DO NOT use file operations for sending emails.

Parameters:
- to (required): Recipient email address(es), comma-separated for multiple
- subject (required): Email subject line
- body (required): Email body content
- body_type (optional): "HTML" or "Text" (default: "HTML")
- cc (optional): CC recipient email address(es)"""
                            elif task_name == "email_mark_read":
                                task_desc = """Mark an email message as read in Outlook.
                                
IMPORTANT: Use this to mark emails as read. Requires email_id from email_check results.

Parameters:
- email_id (required): The ID of the email to mark as read"""
                        
                        # Build parameters schema (simplified - you can enhance this)
                        properties = {}
                        required = []
                        
                        # Common parameters that tasks might use
                        common_params = ["source", "destination", "path", "content", "old_text", "new_text", "command", "directory"]
                        for param in common_params:
                            properties[param] = {"type": "string", "description": f"{param} parameter for {task_name}"}
                        
                        # Add email-specific parameters
                        if task_name == "email_check":
                            properties["max_results"] = {"type": "integer", "description": "Maximum number of emails to return (default: 10, ignored if get_all=true)"}
                            properties["unread_only"] = {"type": "boolean", "description": "If true, only return unread emails (default: true)"}
                            properties["get_all"] = {"type": "boolean", "description": "If true, retrieve ALL emails from inbox using pagination (default: false)"}
                        elif task_name == "email_read" or task_name == "email_mark_read":
                            properties["email_id"] = {"type": "string", "description": "The ID of the email (obtained from email_check)"}
                            required.append("email_id")
                        elif task_name == "email_send":
                            properties["to"] = {"type": "string", "description": "Recipient email address(es), comma-separated for multiple"}
                            properties["subject"] = {"type": "string", "description": "Email subject line"}
                            properties["body"] = {"type": "string", "description": "Email body content"}
                            properties["body_type"] = {"type": "string", "description": "Email body type: 'HTML' or 'Text' (default: 'HTML')"}
                            properties["cc"] = {"type": "string", "description": "CC recipient email address(es), comma-separated"}
                            required.extend(["to", "subject", "body"])
                        
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
                                        "description": "Task-specific parameters",
                                        "required": required if required else []
                                    }
                                },
                                "required": ["task_name", "params"]
                            }
                        })
            
            # Make API call with streaming, retry logic, and function calling
            model_name = self.model_options[self.model]
            
            # Determine capability - prioritize vision if images are present
            capability = None
            if self.image_paths and IMAGE_HANDLING_AVAILABLE:
                capability = "vision"
                if MODEL_REGISTRY_AVAILABLE:
                    vision_model = get_model_for_capability("vision")
                    if vision_model:
                        # Find the friendly name that maps to this vision model
                        for name, model_id in self.model_options.items():
                            if model_id == vision_model or (model_id and vision_model and (model_id.startswith(vision_model + "-") or vision_model.startswith(model_id + "-"))):
                                # Update model to use vision-capable model
                                if name in self.model_options:
                                    model_name = self.model_options[name]
                                    logging.info(f"Switching to vision model {name} ({model_name}) for image analysis")
                                    break
                        # If not found in friendly names, use the model ID directly
                        if model_name == self.model_options.get(self.model):
                            model_name = vision_model
                            logging.info(f"Using vision model {vision_model} directly for image analysis")
            elif MODEL_REGISTRY_AVAILABLE:
                # Try to determine capability from mode
                for mode_name, mode_cap in MODE_TO_CAPABILITY.items():
                    if mode_name == self.mode:
                        capability = mode_cap
                        break
            
            answer = ""
            
            def make_api_call():
                """Inner function for retry logic"""
                nonlocal answer, capability
                
                # Chat models use standard parameters
                api_params = {
                    "model": model_name,
                    "messages": messages,
                    "max_completion_tokens": 4096,  # Standard limit for chat
                    "timeout": 60.0
                }
                
                # Some models (like GPT-5, GPT-5.1, o1, o3, o4) may have different temperature requirements
                # Only set temperature for models that support custom values
                # Models that require default temperature: gpt-5*, o1-*, o3-*, o4-*
                model_lower = model_name.lower()
                if not (model_lower.startswith("gpt-5") or model_lower.startswith("o1-") or 
                        model_lower.startswith("o3-") or model_lower.startswith("o4-")):
                    api_params["temperature"] = 0.7
                
                # Check if streaming should be disabled due to repeated failures
                if self.streaming_failures >= self.max_streaming_failures:
                    logging.warning(f"Streaming disabled due to {self.streaming_failures} previous failures. Using non-streaming mode.")
                    self.enable_streaming = False
                
                # Streaming is DISABLED by default due to persistent issues with incomplete responses
                # Non-streaming mode is more reliable and always returns complete responses
                api_params["stream"] = False
                logging.debug(f"Streaming disabled - using non-streaming mode for reliability")
                
                if functions:
                    api_params["functions"] = functions
                    api_params["function_call"] = "auto"
                
                if api_params.get("stream"):
                    # Streaming response - try with fallback
                    try:
                        stream = self.openai_client.chat.completions.create(**api_params)
                    except Exception as stream_error:
                        # If streaming fails, try fallback model
                        is_model_err, err_type, _ = detect_model_error(stream_error)
                        if is_model_err and MODEL_REGISTRY_AVAILABLE and model_registry:
                            model_registry.mark_model_failed(model_name, str(stream_error))
                            if capability:
                                fallbacks = model_registry.get_fallback_models(capability, exclude_model=model_name)
                                if fallbacks:
                                    api_params["model"] = fallbacks[0]
                                    logging.info(f"Trying fallback model {fallbacks[0]} for streaming")
                                    stream = self.openai_client.chat.completions.create(**api_params)
                                else:
                                    raise stream_error
                            else:
                                raise stream_error
                        else:
                            raise stream_error
                    
                    full_response = ""
                    function_call_name = None
                    function_call_args = ""
                    function_calls = []
                    
                    # Process all chunks to get complete response
                    # CRITICAL: Wrap in try-except to ensure we capture all chunks even if some fail
                    chunk_count = 0
                    content_chunks = 0
                    stream_completed = False
                    try:
                        # CRITICAL: Consume the entire stream - don't break early
                        for chunk in stream:
                            chunk_count += 1
                            try:
                                # Check if chunk has choices
                                if not chunk.choices or len(chunk.choices) == 0:
                                    # Empty chunk - might be keepalive or final chunk
                                    # Check if this indicates stream completion
                                    if hasattr(chunk, 'choices') and len(chunk.choices) == 0:
                                        # This might be the final empty chunk - continue to check finish_reason
                                        pass
                                    continue
                                
                                delta = chunk.choices[0].delta
                                
                                # Handle content chunks - CRITICAL: Check for content first
                                if hasattr(delta, 'content') and delta.content is not None:
                                    content = delta.content
                                    if content:  # Only add non-empty content
                                        full_response += content
                                        self.stream_chunk.emit(content)
                                        content_chunks += 1
                                
                                # Handle function call chunks (but don't let them stop content accumulation)
                                if hasattr(delta, 'function_call') and delta.function_call:
                                    func_call = delta.function_call
                                    if hasattr(func_call, 'name') and func_call.name:
                                        function_call_name = func_call.name
                                    if hasattr(func_call, 'arguments') and func_call.arguments:
                                        function_call_args += func_call.arguments
                                
                                # Check for finish_reason to know when stream is done
                                if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                                    finish_reason = chunk.choices[0].finish_reason
                                    logging.info(f"Stream finished with reason: {finish_reason} (chunk {chunk_count})")
                                    stream_completed = True
                                    # DON'T break here - continue to process any remaining chunks
                                    # Some streams send content in chunks AFTER finish_reason
                            except Exception as chunk_error:
                                # Log but continue processing - don't stop on single chunk error
                                logging.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                                import traceback
                                logging.debug(traceback.format_exc())
                                continue
                        
                        # Log completion status
                        if stream_completed:
                            logging.info(f"Stream processing complete: {chunk_count} total chunks, {content_chunks} content chunks, {len(full_response)} characters accumulated")
                        else:
                            logging.warning(f"Stream ended without finish_reason: {chunk_count} chunks processed, {content_chunks} content chunks, {len(full_response)} characters")
                    except StopIteration:
                        # Stream generator exhausted - this is normal
                        logging.info(f"Stream exhausted: {chunk_count} chunks processed, {len(full_response)} characters accumulated")
                    except Exception as stream_error:
                        # Log stream error but return what we have
                        logging.error(f"Error in stream processing after {chunk_count} chunks: {stream_error}")
                        import traceback
                        logging.error(traceback.format_exc())
                        logging.error(f"Accumulated response so far: '{full_response[:200]}...' (length: {len(full_response)})")
                        # Don't raise - return accumulated response
                    except Exception as stream_error:
                        # Log stream error but return what we have
                        logging.error(f"Error in stream processing after {chunk_count} chunks: {stream_error}")
                        logging.error(f"Accumulated response so far: '{full_response[:100]}...' (length: {len(full_response)})")
                        # Don't raise - return accumulated response
                    
                    # Use accumulated response as answer - CRITICAL: Always use full_response
                    answer = full_response
                    
                    # Log if response seems incomplete (very short)
                    if len(full_response.strip()) < 10:
                        logging.warning(f"Streaming response seems incomplete: '{full_response}' (length: {len(full_response)}, chunks: {chunk_count})")
                    else:
                        logging.info(f"Streaming response complete: {len(full_response)} characters from {chunk_count} chunks")
                    
                    # CRITICAL: Always use full_response as the base answer for streaming
                    # This ensures we return the complete accumulated response
                    answer = full_response
                    
                    # Handle mode switching from streaming function calls
                    # BUT: Only if we actually have a function call AND it's complete
                    # IMPORTANT: Don't overwrite answer - append/prepend mode switch message
                    if function_call_name == "switch_agent_mode" and function_call_args:
                        try:
                            if isinstance(function_call_args, str) and function_call_args.strip():
                                args_dict = json.loads(function_call_args)
                            else:
                                args_dict = function_call_args if isinstance(function_call_args, dict) else {}
                            
                            new_mode = args_dict.get("mode")
                            reason = args_dict.get("reason", "User's question requires specialized expertise")
                            
                            if new_mode and new_mode in self.agents:
                                # Prepend mode switch message to the answer (don't replace it)
                                mode_msg = f"I'm switching you to **{new_mode}** mode. {reason}\n\n"
                                answer = mode_msg + answer if answer else mode_msg + "Let me help you with that now..."
                                logging.info(f"Mode switch requested from streaming: {new_mode} - {reason}")
                                # Emit signal for mode switch
                                self.mode_switch_requested.emit(new_mode, reason)
                            else:
                                # Append error message, don't replace
                                error_msg = f"I tried to switch modes, but '{new_mode}' is not a valid mode. Available modes: {', '.join(self.agents.keys())}"
                                answer = answer + "\n\n" + error_msg if answer else error_msg
                        except Exception as switch_error:
                            logging.warning(f"Error handling mode switch from streaming: {switch_error}")
                            error_msg = "I tried to switch modes but encountered an error. I'll continue in the current mode."
                            answer = answer + "\n\n" + error_msg if answer else error_msg
                    # Handle other function calls (tasks) - but preserve the answer
                    elif function_calls and TASK_SYSTEM_AVAILABLE:
                        task_result = self._handle_function_calls(function_calls, answer)
                        # Append task results to answer, don't replace
                        if task_result and task_result != answer:
                            answer = answer + "\n\n" + task_result if answer else task_result
                    
                    # Final safety check: if answer is empty but we have full_response, use it
                    if not answer and full_response:
                        answer = full_response
                    
                    # CRITICAL: Final validation - ensure we have the complete response
                    # If answer is very short but we processed many chunks, something went wrong
                    streaming_failed = False
                    if len(answer.strip()) < 10 and chunk_count > 5:
                        logging.error(f"⚠️ STREAMING ISSUE DETECTED: Processed {chunk_count} chunks but only got {len(answer)} characters!")
                        logging.error(f"   Content chunks: {content_chunks}, Full response: '{answer[:100]}'")
                        streaming_failed = True
                        # Try to use full_response if it's different
                        if full_response and len(full_response) > len(answer):
                            logging.warning(f"   Using full_response instead: {len(full_response)} characters")
                            answer = full_response
                    elif len(answer.strip()) < 10 and chunk_count > 0:
                        logging.warning(f"Streaming response is very short: '{answer}' (length: {len(answer)}, chunks: {chunk_count})")
                        # If we got chunks but very little content, this is a failure
                        if chunk_count > 3:
                            streaming_failed = True
                    else:
                        logging.info(f"✅ Streaming response complete: {len(answer)} characters from {chunk_count} chunks ({content_chunks} content chunks)")
                    
                    # Final safety: if answer is still empty but we have full_response, use it
                    if not answer or len(answer.strip()) == 0:
                        if full_response:
                            answer = full_response
                            logging.warning(f"Using full_response as fallback: {len(answer)} characters")
                        else:
                            logging.error("⚠️ CRITICAL: Both answer and full_response are empty after streaming!")
                            streaming_failed = True
                    
                    # Track streaming failures and disable if too many
                    if streaming_failed:
                        self.streaming_failures += 1
                        logging.warning(f"Streaming failure count: {self.streaming_failures}/{self.max_streaming_failures}")
                        if self.streaming_failures >= self.max_streaming_failures:
                            logging.error("⚠️ Too many streaming failures! Streaming will be disabled for future requests.")
                            logging.error("   The system will automatically use non-streaming mode for better reliability.")
                    
                    return answer
                else:
                    # Non-streaming response - use automatic fallback
                    # Remove stream parameter if present
                    if "stream" in api_params:
                        del api_params["stream"]
                    
                    # Reset streaming failure count on successful non-streaming response
                    # This allows streaming to be re-enabled if it starts working again
                    if self.streaming_failures > 0:
                        logging.info(f"Non-streaming response successful. Resetting streaming failure count (was {self.streaming_failures})")
                        self.streaming_failures = 0
                        self.enable_streaming = True  # Re-enable streaming
                    
                    # Try initial model with self-healing
                    recovery_info = None
                    try:
                        response = self.openai_client.chat.completions.create(**api_params)
                    except Exception as e:
                        # If model error, use self-healing fallback system
                        is_model_err, err_type, _ = detect_model_error(e)
                        if is_model_err and MODEL_REGISTRY_AVAILABLE and model_registry:
                            # Use the comprehensive fallback system
                            answer_text, final_model, recovery_info = call_api_with_fallback(
                                self.openai_client, model_name, messages, api_params, capability=capability, mode=self.mode
                            )
                            
                            if answer_text:
                                # Create a mock response object for compatibility
                                class MockResponse:
                                    def __init__(self, content):
                                        class MockChoice:
                                            def __init__(self, content):
                                                class MockMessage:
                                                    def __init__(self, content):
                                                        self.content = content
                                                        self.function_call = None
                                                self.message = MockMessage(content)
                                        self.choices = [MockChoice(content)]
                                
                                response = MockResponse(answer_text)
                                
                                # Log recovery if happened
                                if recovery_info and recovery_info.get("recovered"):
                                    logging.info(f"✅ Self-healing in worker: {recovery_info.get('message', '')}")
                            else:
                                raise e
                        else:
                            raise e
                    
                    if not response or not response.choices:
                        raise Exception("Invalid response from OpenAI API")
                    
                    # Check for function calls (including mode switching)
                    message = response.choices[0].message
                    answer = message.content or ""
                    
                    # Log the raw response for debugging
                    logging.info(f"Non-streaming response received: {len(answer)} characters")
                    if len(answer) < 50:
                        logging.warning(f"Response seems short: '{answer}'")
                    
                    if message.function_call:
                        func_name = message.function_call.name
                        func_args = message.function_call.arguments
                        logging.info(f"Function call detected: {func_name}, content length: {len(answer)}")
                        
                        # Handle mode switching
                        if func_name == "switch_agent_mode":
                            try:
                                if isinstance(func_args, str):
                                    args_dict = json.loads(func_args)
                                else:
                                    args_dict = func_args
                                
                                new_mode = args_dict.get("mode")
                                reason = args_dict.get("reason", "User's question requires specialized expertise")
                                
                                if new_mode and new_mode in self.agents:
                                    answer = f"I'm switching you to **{new_mode}** mode. {reason}\n\nLet me help you with that now..."
                                    logging.info(f"Mode switch requested: {new_mode} - {reason}")
                                    # Emit signal for mode switch
                                    self.mode_switch_requested.emit(new_mode, reason)
                                else:
                                    answer = f"I tried to switch modes, but '{new_mode}' is not a valid mode. Available modes: {', '.join(self.agents.keys())}"
                            except Exception as switch_error:
                                logging.warning(f"Error handling mode switch: {switch_error}")
                                answer = "I tried to switch modes but encountered an error. I'll continue in the current mode."
                        # Handle task execution
                        elif func_name.startswith("execute_task_") and TASK_SYSTEM_AVAILABLE:
                            function_calls = [{
                                "name": func_name,
                                "arguments": func_args
                            }]
                            answer = self._handle_function_calls(function_calls, answer)
                    else:
                        answer = message.content or ""
                        logging.info(f"Non-streaming response (no function call): {len(answer)} characters")
                    
                    # Validate answer before returning
                    if not answer or len(answer.strip()) == 0:
                        logging.error("Empty response from OpenAI API")
                        raise Exception("Empty response from OpenAI API")
                    
                    # Log final answer length
                    if len(answer.strip()) < 10:
                        logging.warning(f"⚠️ Very short response: '{answer}' (length: {len(answer)})")
                    else:
                        logging.info(f"✅ Complete response: {len(answer)} characters")
                    
                    return answer
            
            # Use retry logic
            try:
                answer = retry_api_call(make_api_call, max_attempts=3, base_delay=1.0)
            except Exception as api_error:
                error_msg = str(api_error)
                # Provide user-friendly error messages
                if "rate_limit" in error_msg.lower() or "429" in error_msg or "too many requests" in error_msg.lower():
                    # Check if we can get more details from the exception
                    wait_suggestion = ""
                    response_obj = None
                    if hasattr(api_error, 'response'):
                        response_obj = api_error.response
                    elif hasattr(api_error, '_response'):  # OpenAI SDK might use _response
                        response_obj = api_error._response
                    
                    if response_obj and hasattr(response_obj, 'headers'):
                        headers = response_obj.headers
                        reset_requests = headers.get('x-ratelimit-reset-requests', '')
                        if reset_requests:
                            wait_suggestion = f" (Rate limit resets in {reset_requests})"
                    self.error.emit(f"API rate limit exceeded. The system tried 3 times but still hit the limit.{wait_suggestion}\n\nPlease wait a moment and try again.")
                elif "timeout" in error_msg.lower():
                    self.error.emit("Request timed out. The system tried 3 times. Please try again.")
                elif "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                    self.error.emit("Authentication failed. Check your OpenAI API key in your .env file.")
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
                # Limit history to configured max (but keep more for memory)
                if len(self.message_history) > self.max_history_messages:
                    self.message_history = self.message_history[-self.max_history_messages:]
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
                
                # Special handling for email_check to show email details
                if tr['task'] == 'email_check' and r.get('success') and r.get('data'):
                    email_data = r.get('data', {})
                    emails = email_data.get('emails', [])
                    if emails:
                        results_text += f"\n**Email Summary:**\n"
                        for i, email in enumerate(emails[:10], 1):  # Show first 10
                            subject = email.get('subject', '(No Subject)')
                            sender = email.get('sender', 'Unknown')
                            received = email.get('received', '')
                            is_read = "✓" if email.get('is_read') else "✗"
                            results_text += f"{i}. [{is_read}] {subject}\n"
                            results_text += f"   From: {sender}\n"
                            if received:
                                results_text += f"   Date: {received}\n"
                            results_text += "\n"
                        if len(emails) > 10:
                            results_text += f"... and {len(emails) - 10} more email(s)\n"
                
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
        # Default model will be set by on_mode_changed when mode combo is initialized
        # For now, set a temporary default based on MODE_MODEL_DEFAULTS for General Assistant & Triage
        if 'MODE_MODEL_DEFAULTS' in globals() and "General Assistant & Triage" in MODE_MODEL_DEFAULTS:
            primary_model, _ = MODE_MODEL_DEFAULTS["General Assistant & Triage"]
            # Convert model ID to friendly name (e.g., "gpt-5-mini" -> "GPT-5 Mini")
            self.current_model = primary_model.replace("gpt-", "GPT-").replace("-", " ").title()
        else:
            self.current_model = "GPT-5 Mini"  # Default for General Assistant & Triage mode
        self.message_history = []
        self.history_file = "lea_history.json"
        self.current_file_content = None
        self.current_file_path = None
        self.current_image_paths = []  # List of image paths for vision API
        
        # Configurable settings
        self.max_history_messages = 100  # Increased history limit to preserve more conversations
        self.enable_response_cache = True  # Enable response caching
        self.cache_duration_hours = 24  # Cache responses for 24 hours
        self.response_cache = {}  # Cache dictionary
        
        # Initialize memory system
        self.memory_system = LeaMemory()
        if openai_client:
            self.memory_system.set_client(openai_client)
            # Load conversation history into memory system after client is set
            self._load_conversation_history_to_memory()
        
        # Streaming state
        self.current_streaming_response = ""
        self.is_streaming = False
        self.streaming_message_started = False
        self.streaming_cursor_position = None  # Track position of streaming message
        self.streaming_message_count = 0  # Count of Lea messages to track which one is streaming
        
        # TTS state
        self.tts_enabled = False
        self.tts_voice_lang = "en"
        self.tts_voice_tld = "com"  # Top-level domain for accent (com = US, co.uk = UK, etc.)
        
        # Speech Recognition state
        self.speech_recognizer = None
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.speech_recognizer = sr.Recognizer()
            except:
                pass
        self.microphone_device_index = None
        self.is_listening = False
        self.speech_worker_thread = None
        self.speech_worker = None
        
        # Status update methods
        self._update_tts_mic_status()
        
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
        
        # Settings
        self.settings_file = PROJECT_DIR / "lea_settings.json"
        self.load_settings()
        
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
        # Set default mode to "General Assistant & Triage"
        default_mode = "General Assistant & Triage"
        if default_mode in list(AGENTS.keys()):
            # Set the default mode BEFORE connecting the signal to avoid triggering on_mode_changed
            self.mode_combo.setCurrentText(default_mode)
            self.current_mode = default_mode
        # Connect signal after setting default (but don't call on_mode_changed yet - model_combo doesn't exist)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        header.addWidget(self.mode_combo)
        
        header.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        # Refresh MODEL_OPTIONS in case registry updated
        global MODEL_OPTIONS
        try:
            MODEL_OPTIONS = build_model_options()
            if MODEL_OPTIONS:
                self.model_combo.addItems(list(MODEL_OPTIONS.keys()))
            else:
                # Fallback if build_model_options returns empty
                self.model_combo.addItems(["GPT-4o", "GPT-4o Mini", "GPT-4 Turbo"])
                logging.warning("MODEL_OPTIONS is empty, using fallback models")
        except Exception as e:
            logging.error(f"Error building model options: {e}")
            # Fallback models
            self.model_combo.addItems(["GPT-4o", "GPT-4o Mini", "GPT-4 Turbo"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        header.addWidget(self.model_combo)
        
        # Now that model_combo exists, set the model for the default mode (General Assistant & Triage)
        # This ensures the correct model is selected when the program starts
        if default_mode in list(AGENTS.keys()):
            # Call on_mode_changed now that both mode_combo and model_combo exist
            self.on_mode_changed(default_mode)
        
        # Add refresh button for models
        if MODEL_REGISTRY_AVAILABLE:
            refresh_btn = QPushButton("🔄")
            refresh_btn.setToolTip("Refresh models from API")
            refresh_btn.setMaximumWidth(35)
            refresh_btn.setStyleSheet("background-color: #444; padding: 4px; border-radius: 4px;")
            refresh_btn.clicked.connect(self.refresh_models)
            header.addWidget(refresh_btn)
        
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
        
        # Screenshot button
        screenshot_btn = QPushButton("📷 Screenshot")
        screenshot_btn.clicked.connect(self.take_screenshot)
        screenshot_btn.setToolTip("Take a screenshot to show Lea what's on your screen")
        screenshot_btn.setStyleSheet("background-color: #8B5CF6; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(screenshot_btn)
        
        download_btn = QPushButton("📥 Download")
        download_btn.clicked.connect(self.download_response)
        download_btn.setStyleSheet("background-color: #107C10; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(download_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_conversation)
        export_btn.setStyleSheet("background-color: #0078D7; padding: 6px 12px; border-radius: 4px;")
        buttons.addWidget(export_btn)
        
        # TTS toggle button
        if TTS_AVAILABLE:
            self.tts_btn = QPushButton("🔇 TTS Off")
            self.tts_btn.clicked.connect(self.toggle_tts)
            self.tts_btn.setStyleSheet("background-color: #6B46C1; padding: 6px 12px; border-radius: 4px;")
            self.tts_btn.setCheckable(True)
            buttons.addWidget(self.tts_btn)
        
        if TASK_SYSTEM_AVAILABLE:
            tasks_btn = QPushButton("🤖 Tasks")
            tasks_btn.clicked.connect(self.show_tasks_dialog)
            tasks_btn.setStyleSheet("background-color: #6B46C1; padding: 6px 12px; border-radius: 4px;")
            buttons.addWidget(tasks_btn)
            
            # Coordinate finder button
            coord_btn = QPushButton("📍 Coords")
            coord_btn.clicked.connect(self.show_coordinate_finder)
            coord_btn.setStyleSheet("background-color: #6B46C1; padding: 6px 12px; border-radius: 4px;")
            coord_btn.setToolTip("Click to show mouse coordinates")
            buttons.addWidget(coord_btn)
        
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
        
        # Microphone button (if speech recognition available)
        if SPEECH_RECOGNITION_AVAILABLE:
            self.mic_btn = QPushButton("🎤")
            self.mic_btn.setToolTip("Click to speak your question")
            self.mic_btn.setMinimumWidth(45)
            self.mic_btn.setMaximumWidth(45)
            self.mic_btn.setStyleSheet("background-color: #444; font-size: 20px; border-radius: 4px; padding: 4px;")
            self.mic_btn.clicked.connect(self.toggle_speech_recognition)
            input_layout.addWidget(self.mic_btn)
        
        self.input_box = ChatInputBox()
        self.input_box.setPlaceholderText(
            "Ask Lea anything... (Enter to send, Shift+Enter for new line)\n"
            "💡 Tips:\n"
            "  • Drag & drop files here to upload them\n"
            "  • Paste text snippets (Ctrl+V) to include in your message\n"
            "  • Copy files from Explorer and paste here"
        )
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
        
        # Status indicators (two separate indicators)
        status_layout = QHBoxLayout()
        status_layout.setSpacing(10)
        
        # TTS/Microphone status indicator
        self.tts_mic_status = QLabel("🔴 TTS/Mic: Inactive")
        self.tts_mic_status.setStyleSheet("""
            QLabel {
                color: #FF4444;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)
        status_layout.addWidget(self.tts_mic_status)
        
        # Conversation status indicator
        self.conversation_status = QLabel("Ready")
        self.conversation_status.setStyleSheet("""
            QLabel {
                color: #68BD47;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)
        status_layout.addWidget(self.conversation_status)
        
        status_layout.addStretch()
        frame_layout.addLayout(status_layout)
        
        # Keep status_label for backward compatibility (will update conversation_status)
        self.status_label = self.conversation_status
        
        # Initialize status indicators
        self._update_tts_mic_status()
        self._update_conversation_status("Ready", "ready")
        
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
    
    def take_screenshot(self):
        """Take a screenshot and add it for analysis"""
        try:
            if not IMAGE_HANDLING_AVAILABLE:
                QMessageBox.warning(self, "Image Support Not Available", 
                    "Pillow is required for screenshot functionality.\n\n"
                    "Install with: pip install Pillow")
                return
            
            if ImageGrab is None:
                QMessageBox.warning(self, "Screenshot Not Available", 
                    "Pillow's ImageGrab is required for screenshots.\n\n"
                    "On Linux, you may need:\n"
                    "sudo apt-get install python3-pil python3-tk")
                return
            
            # Option 1: Ask user if they want to hide window or capture full screen
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Take Screenshot")
            msg.setText("How would you like to take the screenshot?")
            msg.setInformativeText("This will capture an image to show Lea what's on your screen.")
            
            hide_btn = msg.addButton("Hide Window (3 sec)", QMessageBox.ButtonRole.AcceptRole)
            fullscreen_btn = msg.addButton("Full Screen Now", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            msg.exec()
            
            if msg.clickedButton() == cancel_btn:
                return
            
            hide_window = msg.clickedButton() == hide_btn
            
            if hide_window:
                # Hide window briefly to take screenshot
                self.hide()
                QApplication.processEvents()
                # Use QTimer instead of blocking sleep to prevent UI freeze
                from PyQt6.QtCore import QTimer
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(lambda: None)  # Dummy connection
                # Note: This is still somewhat blocking, but necessary for screenshot timing
                # Consider using QEventLoop for better non-blocking behavior
                QThread.msleep(3000)  # 3 second delay (user-initiated, acceptable)
            
            try:
                # Take screenshot using PIL
                screenshot = ImageGrab.grab()
                
                if hide_window:
                    # Restore window
                    self.show()
                
                # Save to screenshots directory
                screenshot_dir = PROJECT_DIR / "screenshots"
                screenshot_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = screenshot_dir / f"screenshot_{timestamp}.png"
                screenshot.save(screenshot_path, "PNG")
                
                # Add screenshot as image for analysis
                if str(screenshot_path) not in self.current_image_paths:
                    self.current_image_paths.append(str(screenshot_path))
                
                # Clear any text file content when adding image
                if self.current_image_paths:
                    self.current_file_content = None
                    self.current_file_path = None
                
                # Update UI
                self.file_label.setText(f"🖼️ Screenshot: {screenshot_path.name}")
                self.append_message("system", f"📷 Screenshot captured: {screenshot_path.name}\nThis will be analyzed by Lea's vision model.")
                self._update_conversation_status("Screenshot ready", "ready")
                
                if not hide_window:
                    QMessageBox.information(self, "Screenshot Captured", 
                        f"Screenshot saved: {screenshot_path.name}\n\n"
                        "The screenshot will be included in your next message to Lea.")
                    
            except Exception as screenshot_error:
                if hide_window:
                    self.show()  # Restore window even on error
                QMessageBox.warning(self, "Screenshot Error", 
                    f"Failed to take screenshot:\n\n{screenshot_error}")
                
        except Exception as e:
            self.show()  # Restore window
            QMessageBox.warning(self, "Error", f"Error taking screenshot: {e}")
    
    def _upload_file_path(self, path: str):
        """Internal method to upload a file by path"""
        if not FILE_READER_AVAILABLE:
            QMessageBox.warning(self, "Error", "universal_file_reader.py not found")
            return
        
        self._update_conversation_status("Reading file...", "processing")
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
                file_path = None
                if hasattr(self, '_file_worker') and self._file_worker and hasattr(self._file_worker, 'path'):
                    file_path = self._file_worker.path
                elif hasattr(self, 'file_worker') and self.file_worker and hasattr(self.file_worker, 'path'):
                    try:
                        file_path = self.file_worker.path
                    except (RuntimeError, AttributeError):
                        # Worker deleted, use backup_path if available
                        if backup_path:
                            file_path = backup_path
                
                if file_path:
                    self.current_file_path = file_path
                    
                    # Check if it's an image file
                    if is_image_file(file_path):
                        # Store image path separately for vision API
                        if file_path not in self.current_image_paths:
                            self.current_image_paths.append(file_path)
                        self.file_label.setText(f"🖼️ {file_name} (Image)")
                        self.append_message("system", f"📷 Image uploaded: {file_name}\nThis image will be analyzed by Lea's vision model.")
                    else:
                        # Regular file - clear image paths if switching to non-image
                        if self.current_image_paths:
                            self.current_image_paths = []
                        self.current_file_content = result.get('content', '')
                        self.file_label.setText(f"📎 {file_name} ({result.get('file_type', 'unknown')})")
                        self.append_message("system", f"Uploaded: {file_name}\nBackup: {os.path.basename(backup_path) if backup_path else 'None'}")
                
                self._update_conversation_status("File loaded", "ready")
            else:
                QMessageBox.warning(self, "Error", result.get('error', 'Unknown error'))
                self._update_conversation_status("Error loading file", "error")
            
            # Clean up reference
            try:
                if hasattr(self, '_file_worker'):
                    self._file_worker = None
            except:
                pass
        except Exception as e:
            logging.error(f"Error in on_file_upload_finished: {traceback.format_exc()}")
            try:
                self._update_conversation_status("Error loading file", "error")
            except:
                pass

    def on_file_upload_error(self, error_msg):
        try:
            error_text = str(error_msg) if error_msg else "Unknown error"
            QMessageBox.warning(self, "File Upload Error", error_text)
            self._update_conversation_status("Error loading file", "error")
        except Exception as e:
            logging.error(f"Error in on_file_upload_error: {traceback.format_exc()}")
            self._update_conversation_status("Error", "error")
    
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
        # Helper function to generate emoji names from Unicode CLDR data patterns
        def generate_emoji_name(emoji, category):
            """Generate a reasonable name for an emoji based on common patterns"""
            # Common emoji name patterns - this is a simplified version
            # In a full implementation, you'd use Unicode CLDR data
            name_map = {
                # Faces
                "😊": "smiling face", "😀": "grinning face", "😃": "grinning face big eyes", 
                "😄": "grinning face smiling eyes", "😁": "beaming face", "😆": "grinning squinting face",
                "😅": "grinning face sweat", "🤣": "rolling on floor laughing", "😂": "face with tears of joy",
                "🙂": "slightly smiling face", "🙃": "upside down face", "😉": "winking face",
                "😌": "relieved face", "😍": "smiling face heart eyes", "🥰": "smiling face hearts",
                "😘": "face blowing kiss", "😗": "kissing face", "😙": "kissing face smiling eyes",
                "😚": "kissing face closed eyes", "😋": "face savoring food", "😛": "face with tongue",
                "😝": "squinting face tongue", "😜": "winking face tongue", "🤪": "zany face",
                "🤨": "raised eyebrow", "🧐": "face with monocle", "🤓": "nerd face", "😎": "smiling face sunglasses",
                "🤩": "star struck", "🥳": "partying face", "😏": "smirking face", "😒": "unamused face",
                "😞": "disappointed face", "😔": "pensive face", "😟": "worried face", "😕": "slightly frowning face",
                "🙁": "slightly frowning face", "☹️": "frowning face", "😣": "persevering face", "😖": "confounded face",
                "😫": "tired face", "😩": "weary face", "🥺": "pleading face", "😢": "crying face",
                "😭": "loudly crying face", "😤": "face with steam", "😠": "angry face", "😡": "pouting face",
                "🤬": "face with symbols", "🤯": "exploding head", "😳": "flushed face", "🥵": "hot face",
                "🥶": "cold face", "😱": "face screaming fear", "😨": "fearful face", "😰": "anxious face sweat",
                "😥": "sad relieved face", "😓": "downcast face sweat", "🤢": "nauseated face", "🤮": "face vomiting",
                "🤧": "sneezing face", "🥴": "woozy face", "😴": "sleeping face", "🤤": "drooling face",
                "😪": "sleepy face", "😵": "dizzy face", "🤐": "zipper mouth face", "🥱": "yawning face",
                "😷": "face with medical mask",
                # Hand Gestures
                "👋": "waving hand", "🤚": "raised back of hand", "🖐️": "hand with fingers splayed", "✋": "raised hand",
                "🖖": "vulcan salute", "👌": "ok hand", "🤌": "pinched fingers", "🤏": "pinching hand",
                "✌️": "victory hand", "🤞": "crossed fingers", "🤟": "love you gesture", "🤘": "sign of the horns",
                "🤙": "call me hand", "👈": "backhand index pointing left", "👉": "backhand index pointing right",
                "👆": "backhand index pointing up", "🖕": "middle finger", "👇": "backhand index pointing down",
                "☝️": "index pointing up", "👍": "thumbs up", "👎": "thumbs down", "✊": "raised fist",
                "👊": "oncoming fist", "🤛": "left facing fist", "🤜": "right facing fist", "👏": "clapping hands",
                "🙌": "raising hands", "👐": "open hands", "🤲": "palms up together", "🤝": "handshake",
                "🙏": "folded hands", "✍️": "writing hand", "💪": "flexed biceps", "🦾": "mechanical arm", "🦿": "mechanical leg",
                # Animals
                "🐶": "dog face", "🐱": "cat face", "🐭": "mouse face", "🐹": "hamster", "🐰": "rabbit face",
                "🦊": "fox", "🐻": "bear", "🐼": "panda", "🐨": "koala", "🐯": "tiger face", "🦁": "lion",
                "🐮": "cow face", "🐷": "pig face", "🐽": "pig nose", "🐸": "frog", "🐵": "monkey face",
                "🙈": "see no evil monkey", "🙉": "hear no evil monkey", "🙊": "speak no evil monkey", "🐒": "monkey",
                "🐔": "chicken", "🐧": "penguin", "🐦": "bird", "🐤": "baby chick", "🐣": "hatching chick",
                "🐥": "front facing baby chick", "🦆": "duck", "🦅": "eagle", "🦉": "owl", "🦇": "bat",
                "🐺": "wolf", "🐗": "boar", "🐴": "horse face", "🦄": "unicorn", "🐝": "honeybee",
                "🐛": "bug", "🦋": "butterfly", "🐌": "snail", "🐞": "lady beetle", "🐜": "ant",
                "🦟": "mosquito", "🦗": "cricket", "🕷️": "spider", "🦂": "scorpion", "🐢": "turtle",
                "🐍": "snake", "🦎": "lizard", "🦖": "t rex", "🦕": "sauropod", "🐙": "octopus",
                "🦑": "squid", "🦐": "shrimp", "🦞": "lobster", "🦀": "crab", "🐡": "blowfish",
                "🐠": "tropical fish", "🐟": "fish", "🐬": "dolphin", "🐳": "spouting whale", "🐋": "whale",
                "🦈": "shark", "🐊": "crocodile", "🐅": "tiger", "🐆": "leopard", "🦓": "zebra",
                "🦍": "gorilla", "🦧": "orangutan", "🐘": "elephant", "🦛": "hippopotamus", "🦏": "rhinoceros",
                "🐪": "camel", "🐫": "two hump camel", "🦒": "giraffe", "🦘": "kangaroo", "🦬": "bison",
                "🐃": "water buffalo", "🐂": "ox", "🐄": "cow", "🐎": "horse", "🐖": "pig",
                "🐏": "ram", "🐑": "ewe", "🦙": "llama", "🐐": "goat", "🦌": "deer",
                "🐕": "dog", "🐩": "poodle", "🦮": "guide dog", "🐕‍🦺": "service dog", "🐈": "cat",
                "🐓": "rooster", "🦃": "turkey", "🦤": "dodo", "🦚": "peacock", "🦜": "parrot",
                "🦢": "swan", "🦩": "flamingo", "🕊️": "dove", "🐇": "rabbit", "🦝": "raccoon",
                "🦨": "skunk", "🦡": "badger", "🦫": "beaver", "🦦": "otter", "🦥": "sloth",
                "🐁": "mouse", "🐀": "rat", "🐿️": "chipmunk",
                # Food
                "🍏": "green apple", "🍎": "red apple", "🍐": "pear", "🍊": "tangerine", "🍋": "lemon",
                "🍌": "banana", "🍉": "watermelon", "🍇": "grapes", "🍓": "strawberry", "🍈": "melon",
                "🍒": "cherries", "🍑": "peach", "🥭": "mango", "🍍": "pineapple", "🥥": "coconut",
                "🥝": "kiwi fruit", "🍅": "tomato", "🍆": "eggplant", "🥑": "avocado", "🥦": "broccoli",
                "🥬": "leafy green", "🥒": "cucumber", "🌶️": "hot pepper", "🌽": "corn", "🥕": "carrot",
                "🥔": "potato", "🍠": "roasted sweet potato", "🥐": "croissant", "🥯": "bagel", "🍞": "bread",
                "🥖": "baguette bread", "🥨": "pretzel", "🧀": "cheese", "🥚": "egg", "🍳": "cooking",
                "🥞": "pancakes", "🥓": "bacon", "🥩": "cut of meat", "🍗": "poultry leg", "🍖": "meat on bone",
                "🦴": "bone", "🌭": "hot dog", "🍔": "hamburger", "🍟": "french fries", "🍕": "pizza",
                "🥪": "sandwich", "🥙": "stuffed flatbread", "🌮": "taco", "🌯": "burrito", "🥗": "green salad",
                "🥘": "shallow pan of food", "🥫": "canned food", "🍝": "spaghetti", "🍜": "steaming bowl",
                "🍲": "pot of food", "🍛": "curry rice", "🍣": "sushi", "🍱": "bento box", "🥟": "dumpling",
                "🦪": "oyster", "🍤": "fried shrimp", "🍙": "rice ball", "🍚": "cooked rice", "🍘": "rice cracker",
                "🍥": "fish cake", "🥠": "fortune cookie", "🥮": "moon cake", "🍢": "oden", "🍡": "dango",
                "🍧": "shaved ice", "🍨": "ice cream", "🍦": "soft ice cream", "🥧": "pie", "🧁": "cupcake",
                "🍰": "birthday cake", "🎂": "birthday cake", "🍮": "custard", "🍭": "lollipop", "🍬": "candy",
                "🍫": "chocolate bar", "🍿": "popcorn", "🍩": "doughnut", "🍪": "cookie", "🌰": "chestnut",
                "🥜": "peanuts", "🍯": "honey pot", "🥛": "glass of milk", "🍼": "baby bottle", "☕️": "hot beverage",
                "🍵": "teacup", "🧃": "beverage box", "🥤": "cup with straw", "🍶": "sake", "🍺": "beer mug",
                "🍻": "clinking beer mugs", "🥂": "clinking glasses", "🍷": "wine glass", "🥃": "tumbler glass",
                "🍸": "cocktail glass", "🍹": "tropical drink", "🧉": "mate", "🍾": "bottle with popping cork",
                # Common symbols
                "❤️": "red heart", "🧡": "orange heart", "💛": "yellow heart", "💚": "green heart",
                "💙": "blue heart", "💜": "purple heart", "🖤": "black heart", "🤍": "white heart",
                "🤎": "brown heart", "💔": "broken heart", "❣️": "heart exclamation", "💕": "two hearts",
                "💞": "revolving hearts", "💓": "beating heart", "💗": "growing heart", "💖": "sparkling heart",
                "💘": "heart with arrow", "💝": "heart with ribbon", "💟": "heart decoration",
                "✅": "check mark button", "☑️": "check box with check", "✔️": "check mark",
                "❌": "cross mark", "❎": "cross mark button", "➕": "plus", "➖": "minus",
                "➗": "division", "❓": "question mark", "❔": "white question mark", "❕": "white exclamation mark",
                "❗": "exclamation mark", "💯": "hundred points", "🔴": "red circle", "🟠": "orange circle",
                "🟡": "yellow circle", "🟢": "green circle", "🔵": "blue circle", "🟣": "purple circle",
                "⚫": "black circle", "⚪": "white circle", "🟤": "brown circle",
            }
            
            if emoji in name_map:
                return name_map[emoji]
            
            # Fallback: try to infer from category
            category_lower = category.lower()
            if "face" in category_lower or "emotion" in category_lower:
                return f"face emoji {emoji}"
            elif "hand" in category_lower or "gesture" in category_lower:
                return f"hand gesture {emoji}"
            elif "animal" in category_lower or "nature" in category_lower:
                return f"animal {emoji}"
            elif "food" in category_lower or "drink" in category_lower:
                return f"food {emoji}"
            elif "travel" in category_lower or "place" in category_lower:
                return f"place {emoji}"
            elif "sport" in category_lower or "activity" in category_lower:
                return f"sport {emoji}"
            elif "object" in category_lower or "tech" in category_lower:
                return f"object {emoji}"
            elif "symbol" in category_lower or "sign" in category_lower:
                return f"symbol {emoji}"
            elif "flag" in category_lower:
                return f"flag {emoji}"
            elif "weather" in category_lower:
                return f"weather {emoji}"
            else:
                return f"emoji {emoji}"
        
        emoji_names = {}  # Will be populated by generate_emoji_name
        
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
                font-size: 20px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
                min-height: 45px;
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
        
        def get_emoji_name(emoji, category):
            """Get the name for an emoji, with fallback generation"""
            return generate_emoji_name(emoji, category)
        
        def populate_list(filter_text=""):
            """Populate the list with emojis, optionally filtered"""
            nonlocal all_items, list_widget
            list_widget.clear()
            all_items.clear()
            filter_lower = filter_text.lower().strip()
            
            for category, data in emojis_data.items():
                emoji_list = data["emojis"]
                keywords = data["keywords"]
                
                # Filter emojis by name if search text provided
                filtered_emojis = []
                if filter_lower:
                    for emoji in emoji_list:
                        emoji_name = get_emoji_name(emoji, category).lower()
                        # Check if search matches emoji name, category, keywords, or the emoji itself
                        if (filter_lower in emoji_name or 
                            filter_lower in category.lower() or 
                            filter_lower in keywords.lower() or
                            filter_lower in emoji):
                            filtered_emojis.append(emoji)
                else:
                    filtered_emojis = emoji_list
                
                # Only show category if there are emojis to show
                if filtered_emojis:
                    # Add category header
                    category_item = QListWidgetItem(f"  {category}")
                    # Disable selection and enable for category headers
                    from PyQt6.QtCore import Qt
                    category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
                    category_item.setBackground(QColor(80, 80, 80))
                    category_item.setForeground(QColor(200, 200, 200))
                    list_widget.addItem(category_item)
                    all_items.append(category_item)
                    
                    # Add emojis in this category with their names
                    for emoji in filtered_emojis:
                        emoji_name = get_emoji_name(emoji, category)
                        # Display emoji with name (smaller font for name)
                        emoji_item = QListWidgetItem(f"{emoji}  {emoji_name}")
                        emoji_item.setData(Qt.ItemDataRole.UserRole, emoji)
                        emoji_item.setData(Qt.ItemDataRole.UserRole + 1, category)  # Store category
                        emoji_item.setData(Qt.ItemDataRole.UserRole + 2, emoji_name)  # Store name
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
    
    def _handle_mode_switch(self, new_mode: str, reason: str):
        """Handle automatic mode switch request"""
        try:
            if new_mode in AGENTS:
                self.append_message("system", f"🔄 Auto-switching to {new_mode}: {reason}")
                # Update mode combo first
                if hasattr(self, 'mode_combo'):
                    self.mode_combo.setCurrentText(new_mode)
                # Then update model selection
                self.on_mode_changed(new_mode)
                # Extract the last user question to re-ask in new mode
                last_user_message = None
                if self.message_history:
                    for msg in reversed(self.message_history):
                        if msg.get("role") == "user":
                            last_user_message = msg.get("content", "")
                            break
                
                if last_user_message:
                    # Extract the actual question from the message
                    question = last_user_message.replace("Dre's question:\n", "").strip()
                    # Also remove file content if present
                    if "=== UPLOADED FILE" in question:
                        parts = question.split("=== END FILE ===")
                        if len(parts) > 1:
                            question = parts[-1].strip()
                            if question.startswith("Dre's question:"):
                                question = question.replace("Dre's question:", "").strip()
                    
                    if question:
                        self.append_message("system", f"Re-asking your question in {new_mode} mode...")
                        QApplication.processEvents()
                        # Auto-send the question in the new mode
                        self.input_box.setPlainText(question)
                        QApplication.processEvents()
                        # Small delay then auto-send
                        QTimer.singleShot(500, self.on_send)
            else:
                logging.warning(f"Invalid mode for switch: {new_mode}")
                self.append_message("system", f"⚠️ Invalid mode: {new_mode}. Staying in current mode.")
        except Exception as e:
            logging.error(f"Error in _handle_mode_switch: {traceback.format_exc()}")
            self.append_message("system", f"⚠️ Error switching modes: {str(e)}. Please try manually.")
    
    def clear_conversation(self):
        reply = QMessageBox.question(self, "Clear", "Clear conversation?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.message_history = []
            self.chat_display.clear()
            self.current_file_content = None
            self.current_file_path = None
            self.current_image_paths = []
            self.file_label.setText("")
            self._save_history()
            self.append_message("system", "Conversation cleared")
    
    def on_mode_changed(self, mode):
        self.current_mode = mode
        try:
            # Check if model_combo exists (might be called during initialization)
            if not hasattr(self, 'model_combo') or self.model_combo is None:
                logging.warning("on_mode_changed called before model_combo is initialized, skipping")
                return
            
            # Refresh MODEL_OPTIONS to ensure we have latest models
            global MODEL_OPTIONS
            MODEL_OPTIONS = build_model_options()
            
            # Update the model combo box with latest options (shows all available models for manual selection)
            current_selection = self.model_combo.currentText() if self.model_combo.count() > 0 else None
            self.model_combo.clear()
            self.model_combo.addItems(list(MODEL_OPTIONS.keys()))
            
            # AUTOMATIC MODEL ASSIGNMENT: Directly use the assigned model from MODE_MODEL_DEFAULTS
            # This is the simplest and most reliable approach - use what the user specified
            best_model_name = None
            available_models = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
            
            # Get the assigned models for this mode
            if mode in MODE_MODEL_DEFAULTS:
                primary_model, backup_model = MODE_MODEL_DEFAULTS[mode]
                logging.info(f"Mode '{mode}' has assigned models: Primary={primary_model}, Backup={backup_model}")
                
                # Convert primary model ID to friendly name (e.g., "gpt-5-mini" -> "GPT-5 Mini")
                primary_friendly_name = primary_model.replace("gpt-", "GPT-").replace("-", " ").title()
                
                # PRIORITY 1: Try exact friendly name match first (most reliable)
                if primary_friendly_name in available_models:
                    best_model_name = primary_friendly_name
                    logging.info(f"Using assigned primary model '{best_model_name}' ({primary_model}) for mode '{mode}'")
                else:
                    # PRIORITY 2: Try to find by model ID in MODEL_OPTIONS
                    # Sort by length (longer = more specific) to avoid matching "gpt-5" when looking for "gpt-5-mini"
                    sorted_options = sorted(MODEL_OPTIONS.items(), key=lambda x: len(x[1]), reverse=True)
                    for name, model_id in sorted_options:
                        # Exact match first (most important)
                        if model_id == primary_model:
                            if name in available_models:
                                best_model_name = name
                                logging.info(f"Found primary model by exact ID match: '{best_model_name}' ({model_id}) for mode '{mode}'")
                                break
                        # Versioned match (e.g., "gpt-5-mini-2024-08-06" starts with "gpt-5-mini")
                        elif model_id.startswith(primary_model + "-"):
                            if name in available_models:
                                best_model_name = name
                                logging.info(f"Found primary model by versioned ID: '{best_model_name}' ({model_id}) for mode '{mode}'")
                                break
                
                # If primary not found, try backup model
                if not best_model_name:
                    backup_friendly_name = backup_model.replace("gpt-", "GPT-").replace("-", " ").title()
                    if backup_friendly_name in available_models:
                        best_model_name = backup_friendly_name
                        logging.info(f"Using backup model '{best_model_name}' for mode '{mode}'")
                    else:
                        # Try to find backup by ID
                        for name, model_id in MODEL_OPTIONS.items():
                            if model_id == backup_model or model_id.startswith(backup_model + "-"):
                                if name in available_models:
                                    best_model_name = name
                                    logging.info(f"Found backup model by ID: '{best_model_name}' ({model_id}) for mode '{mode}'")
                                    break
            else:
                logging.warning(f"Mode '{mode}' not found in MODE_MODEL_DEFAULTS")
            
            # Ultimate fallback - use first available model
            if not best_model_name and available_models:
                best_model_name = available_models[0]
                logging.warning(f"Assigned model not found for mode '{mode}', using first available: {best_model_name}")
            
            # Set the model in the dropdown (automatic assignment)
            if best_model_name and best_model_name in available_models:
                self.model_combo.setCurrentText(best_model_name)
                self.current_model = best_model_name  # Store friendly name
                # Show mode and auto-assigned model
                self.append_message("system", f"Mode: {mode}\nModel: {best_model_name} (auto-assigned)")
            else:
                # If model not found, use first available or restore previous
                if current_selection and current_selection in available_models:
                    self.model_combo.setCurrentText(current_selection)
                    self.current_model = current_selection
                    self.append_message("system", f"Switched to: {mode} (Model: {self.current_model} - previous selection)")
                elif available_models:
                    self.model_combo.setCurrentIndex(0)
                    self.current_model = available_models[0]
                    self.append_message("system", f"Switched to: {mode} (Model: {self.current_model} - fallback)")
                else:
                    logging.error("No models available in dropdown!")
                    self.current_model = "GPT-4o"  # Fallback
                    self.append_message("system", f"Switched to: {mode} (Model: {self.current_model} - error fallback)")
            
            self._save_history()
        except Exception as e:
            logging.error(f"Error in on_mode_changed: {traceback.format_exc()}")
            # Ensure we have a valid model selected even on error
            if self.model_combo.count() > 0:
                if self.model_combo.currentIndex() < 0:
                    self.model_combo.setCurrentIndex(0)
                self.current_model = self.model_combo.currentText()
            else:
                self.current_model = "GPT-4o"
            self.append_message("system", f"Switched to: {mode} (Error selecting model, using: {self.current_model})")
    
    def refresh_models(self):
        """Refresh model list from API and show transparent health status"""
        if not MODEL_REGISTRY_AVAILABLE:
            QMessageBox.information(self, "Not Available", "Model registry not available")
            return
        
        try:
            self._update_conversation_status("Refreshing models...", "processing")
            refresh_models(force=True)
            
            # Get health status and available models
            model_registry = get_model_registry()
            health = model_registry.get_health_status()
            failed_models = health.get("failed_list", [])
            available_models = model_registry.get_all_models()
            
            # Check for GPT-5 models
            gpt5_models = [m for m in available_models if 'gpt-5' in m.lower()]
            
            # Rebuild MODEL_OPTIONS
            global MODEL_OPTIONS
            MODEL_OPTIONS = build_model_options()
            
            # Update combo box
            current_selection = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(list(MODEL_OPTIONS.keys()))
            
            # Try to restore selection
            if current_selection in MODEL_OPTIONS:
                self.model_combo.setCurrentText(current_selection)
            elif self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
            
            self._update_conversation_status("Models refreshed", "ready")
            
            # Show transparent health status
            status_msg = f"✅ Models refreshed from API\n"
            status_msg += f"📊 Found {len(available_models)} available models\n"
            status_msg += f"📊 Health: {health['status'].upper()} ({health['working_models']}/{health['total_models']} models working)"
            
            # Show GPT-5 status
            if gpt5_models:
                status_msg += f"\n✅ GPT-5 models available: {', '.join(gpt5_models)}"
            else:
                status_msg += f"\n⚠️ GPT-5 models not found in API response"
                status_msg += f"\n   This may be due to:"
                status_msg += f"\n   • API key usage tier (GPT-5 requires tiers 1-5)"
                status_msg += f"\n   • Organization verification status"
                status_msg += f"\n   • Models not yet available for your account"
                status_msg += f"\n   See: https://help.openai.com/en/articles/10362446"
            
            if failed_models:
                status_msg += f"\n⚠️ {len(failed_models)} model(s) marked as failed (will auto-recover)"
                # Offer to clear failed models
                reply = QMessageBox.question(
                    self, "Failed Models Detected",
                    f"{len(failed_models)} model(s) are marked as failed:\n{', '.join(failed_models[:5])}{'...' if len(failed_models) > 5 else ''}\n\n"
                    "Would you like to clear failed status and retry them?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    model_registry.clear_failed_models()
                    status_msg += "\n✅ Cleared failed model status - will retry on next use"
            
            self.append_message("system", status_msg)
            
            # Show detailed model list if GPT-5 not found
            if not gpt5_models and available_models:
                reply = QMessageBox.question(
                    self, "GPT-5 Models Not Available",
                    f"GPT-5 models were not found in your API response.\n\n"
                    f"Your API key has access to {len(available_models)} models.\n\n"
                    f"Would you like to see the full list of available models?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._show_available_models(available_models)
                    
        except Exception as e:
            self._update_conversation_status("Error refreshing models", "error")
            QMessageBox.warning(self, "Error", f"Error refreshing models: {e}")
    
    def _show_available_models(self, models: List[str]):
        """Show a dialog with all available models"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Available Models from API")
        dialog.setMinimumSize(600, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #333;
            }
            QLabel {
                color: #FFF;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        title = QLabel("Models Available via Your API Key:")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #FFF;")
        layout.addWidget(title)
        
        # Group models by type
        gpt5_models = [m for m in models if 'gpt-5' in m.lower()]
        gpt4_models = [m for m in models if 'gpt-4' in m.lower() and 'gpt-5' not in m.lower()]
        gpt3_models = [m for m in models if 'gpt-3' in m.lower()]
        o_models = [m for m in models if m.startswith('o1') or m.startswith('o3')]
        other_models = [m for m in models if m not in gpt5_models + gpt4_models + gpt3_models + o_models]
        
        from PyQt6.QtWidgets import QTextEdit
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #222;
                color: #FFF;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        
        content = []
        if gpt5_models:
            content.append("✅ GPT-5 Models:")
            content.extend([f"  • {m}" for m in sorted(gpt5_models)])
            content.append("")
        else:
            content.append("⚠️ GPT-5 Models: None found")
            content.append("")
        
        if o_models:
            content.append("✅ Reasoning Models (o1/o3):")
            content.extend([f"  • {m}" for m in sorted(o_models)])
            content.append("")
        
        if gpt4_models:
            content.append("✅ GPT-4 Models:")
            content.extend([f"  • {m}" for m in sorted(gpt4_models)])
            content.append("")
        
        if gpt3_models:
            content.append("✅ GPT-3.5 Models:")
            content.extend([f"  • {m}" for m in sorted(gpt3_models)])
            content.append("")
        
        if other_models:
            content.append("Other Models:")
            content.extend([f"  • {m}" for m in sorted(other_models)])
        
        text_edit.setPlainText("\n".join(content))
        layout.addWidget(text_edit)
        
        info_label = QLabel(
            "Note: GPT-5 models require API usage tier 1-5 and may need organization verification.\n"
            "See: https://help.openai.com/en/articles/10362446"
        )
        info_label.setStyleSheet("color: #AAA; font-size: 11px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
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
        
        dialog.exec()
    
    def on_model_changed(self, model):
        self.current_model = model
        self._save_history()
    
    # Messaging
    def append_message(self, kind: str, text: str):
        try:
            if not text:
                text = "(empty message)"
            
            # Show full text
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
                
                # Auto-scroll to bottom to show latest message
                scrollbar = self.chat_display.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            logging.error(f"Error appending message: {traceback.format_exc()}")
            # Fallback to plain text if HTML fails
            try:
                if hasattr(self, 'chat_display') and self.chat_display:
                    self.chat_display.append(f"{label}: {str(text)}")
            except:
                pass
    
    def on_send(self):
        """Send message using worker thread to prevent UI freezing"""
        text = self.input_box.toPlainText().strip()
        if not text or not openai_client:
            return
        
        # Prevent multiple simultaneous sends
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
            logging.warning("Worker thread already running, ignoring send request")
            QMessageBox.information(self, "Please Wait", "A request is already being processed. Please wait for it to complete.")
            return
        
        self.append_message("user", text)
        self.input_box.clear()
        self._update_conversation_status("Thinking...", "thinking")
        
        # Check if we have images to process
        image_paths = []
        if hasattr(self, 'include_file_cb') and self.include_file_cb.isChecked():
            if self.current_image_paths:
                image_paths = self.current_image_paths
        
        # Build prompt
        parts = []
        
        # Include file content if checked (but not for images - images are handled separately)
        if hasattr(self, 'include_file_cb') and self.include_file_cb.isChecked() and self.current_file_content and not image_paths:
            parts.append(f"=== UPLOADED FILE ===\n{self.current_file_content}\n=== END FILE ===\n")
        
        # Build user message text
        user_text = "\n".join(parts + [f"Dre's question:\n{text}"])
        
        # Prepare file content for worker
        file_content = self.current_file_content if (hasattr(self, 'include_file_cb') and self.include_file_cb.isChecked() and not image_paths) else None
        
        # Prepare image paths for worker
        worker_image_paths = image_paths if (hasattr(self, 'include_file_cb') and self.include_file_cb.isChecked()) else []
        
        # Clean up any existing worker thread - disconnect signals first to prevent crashes
        try:
            if hasattr(self, 'worker_thread') and self.worker_thread is not None:
                if self.worker_thread.isRunning():
                    logging.warning("Previous worker thread still running, attempting cleanup")
                    # Disconnect all signals first
                    try:
                        if hasattr(self, '_current_worker') and self._current_worker is not None:
                            try:
                                self._current_worker.finished.disconnect()
                                self._current_worker.error.disconnect()
                                self._current_worker.stream_chunk.disconnect()
                                self._current_worker.memory_context.disconnect()
                                self._current_worker.mode_switch_requested.disconnect()
                            except Exception as sig_err:
                                logging.debug(f"Error disconnecting signals: {sig_err}")
                    except Exception as worker_err:
                        logging.debug(f"Error accessing worker: {worker_err}")
                    # Request thread to quit (non-blocking)
                    self.worker_thread.quit()
                    # Give it a moment, but don't block
                    QApplication.processEvents()
                # Clean up references
                if hasattr(self, '_current_worker'):
                    self._current_worker = None
                self.worker_thread = None
        except Exception as cleanup_err:
            logging.warning(f"Error during worker cleanup: {cleanup_err}")
            # Continue anyway - better to try than crash
        
        # Create and start worker thread
        self.worker_thread = QThread()
        self._current_worker = LeaWorker(
            openai_client,
            MODEL_OPTIONS,
            AGENTS,
            self.current_mode,
            self.current_model,
            self.message_history.copy(),  # Pass copy to avoid threading issues
            file_content,
            user_text,
            self.memory_system,
            self.max_history_messages,
            worker_image_paths  # Pass image paths
        )
        self._current_worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker_thread.started.connect(self._current_worker.run)
        self._current_worker.finished.connect(self.on_worker_finished)
        self._current_worker.error.connect(self.on_worker_error)
        self._current_worker.stream_chunk.connect(self.on_stream_chunk)
        self._current_worker.memory_context.connect(self.on_memory_context)
        self._current_worker.mode_switch_requested.connect(self._handle_mode_switch)
        
        # Clean up thread when done
        self._current_worker.finished.connect(self.worker_thread.quit)
        self._current_worker.error.connect(self.worker_thread.quit)
        
        def safe_delete_worker():
            try:
                if hasattr(self, '_current_worker') and self._current_worker:
                    # Disconnect signals before deletion to prevent crashes
                    try:
                        self._current_worker.finished.disconnect()
                        self._current_worker.error.disconnect()
                        self._current_worker.stream_chunk.disconnect()
                        self._current_worker.memory_context.disconnect()
                        self._current_worker.mode_switch_requested.disconnect()
                    except:
                        pass  # Signals may already be disconnected
                    self._current_worker.deleteLater()
                    self._current_worker = None
            except Exception as e:
                logging.debug(f"Error in safe_delete_worker: {e}")
        
        def safe_delete_thread():
            try:
                if hasattr(self, 'worker_thread') and self.worker_thread:
                    # Ensure thread is stopped before deletion
                    if self.worker_thread.isRunning():
                        self.worker_thread.quit()
                        # Non-blocking wait with timeout
                        if not self.worker_thread.wait(500):  # 500ms timeout
                            logging.warning("Thread did not stop in time, forcing termination")
                            self.worker_thread.terminate()
                            self.worker_thread.wait(200)  # Brief wait after terminate
                    self.worker_thread.deleteLater()
                    self.worker_thread = None
            except Exception as e:
                logging.debug(f"Error in safe_delete_thread: {e}")
        
        self.worker_thread.finished.connect(safe_delete_worker)
        self.worker_thread.finished.connect(safe_delete_thread)
        
        # Set streaming state - only if streaming is actually enabled
        # CRITICAL: Check if worker actually has streaming enabled
        worker_streaming = getattr(self._current_worker, 'enable_streaming', False)
        self.is_streaming = worker_streaming
        self.streaming_message_started = False
        self.current_streaming_response = ""
        
        logging.info(f"Starting worker thread (streaming: {worker_streaming})")
        
        # Start the worker thread
        self.worker_thread.start()

    def on_stream_chunk(self, chunk: str):
        """Handle streaming response chunks - removes and recreates last block for reliability"""
        # CRITICAL: Only process if we're actually streaming and this chunk is for current request
        if not self.is_streaming:
            logging.debug(f"Ignoring chunk (not streaming): '{chunk[:50]}...'")
            return  # Ignore chunks from old/stopped requests
        
        # Accumulate the full response
        if chunk:
            self.current_streaming_response += chunk
            logging.debug(f"Received chunk ({len(chunk)} chars), total so far: {len(self.current_streaming_response)} chars")
        
        try:
            # Get current HTML
            current_html = self.chat_display.toHtml()
            
            # Find the LAST occurrence of "Lea:" marker (our streaming message)
            marker = f'<span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span>'
            last_marker_pos = current_html.rfind(marker)
            
            if last_marker_pos >= 0:
                # Find the <div> that contains this marker
                # Look backwards from marker to find the opening <div>
                div_start = current_html.rfind('<div', 0, last_marker_pos)
                
                if div_start >= 0:
                    # Find the closing </div> for this block
                    # The structure is: <div>...<span>Lea:</span> <span>content</span></div>
                    # We need to find the SECOND </span> (the one closing content) followed by </div>
                    search_start = last_marker_pos
                    
                    # Find first </span> (closes "Lea:")
                    first_span_close = current_html.find('</span>', search_start)
                    if first_span_close > 0:
                        # Find second </span> (closes content) - search after first one
                        second_span_close = current_html.find('</span>', first_span_close + 7)  # +7 for "</span>"
                        
                        if second_span_close > 0:
                            # Find the </div> that follows the second </span>
                            div_close_pos = current_html.find('</div>', second_span_close)
                            
                            if div_close_pos > 0:
                                # Remove the old block and replace with updated one
                                before = current_html[:div_start]
                                after = current_html[div_close_pos + 6:]  # +6 for "</div>"
                                
                                # Create new block with full accumulated response
                                safe_response = html.escape(self.current_streaming_response).replace("\n", "<br>")
                                new_block = f'<div style="margin: 6px 0;"><span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span> <span style="color:{self.ASSIST_COLOR};">{safe_response}</span></div>'
                                
                                new_html = before + new_block + after
                                
                                # Update display
                                self.chat_display.setHtml(new_html)
                                
                                # Mark as started
                                if not self.streaming_message_started:
                                    self.streaming_message_started = True
                                
                                # Scroll to bottom
                                scrollbar = self.chat_display.verticalScrollBar()
                                scrollbar.setValue(scrollbar.maximum())
                                return
            
            # If marker not found, create new message (first chunk)
            if not self.streaming_message_started:
                self.streaming_message_started = True
                safe_text = html.escape(self.current_streaming_response).replace("\n", "<br>") if self.current_streaming_response else ""
                new_block = f'<div style="margin: 6px 0;"><span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span> <span style="color:{self.ASSIST_COLOR};">{safe_text}</span></div>'
                self.chat_display.append(new_block)
                
                scrollbar = self.chat_display.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            logging.error(f"Error in stream chunk: {traceback.format_exc()}")
            # On error, try simple append if not started
            if not self.streaming_message_started and self.current_streaming_response:
                try:
                    self.streaming_message_started = True
                    safe_text = html.escape(self.current_streaming_response).replace("\n", "<br>")
                    new_block = f'<div style="margin: 6px 0;"><span style="color:{self.ASSIST_COLOR}; font-weight:600;">Lea:</span> <span style="color:{self.ASSIST_COLOR};">{safe_text}</span></div>'
                    self.chat_display.append(new_block)
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
            # CRITICAL: Don't reset is_streaming yet - wait a moment for any final chunks
            # Give a small delay to ensure all chunks have arrived
            QTimer.singleShot(100, lambda: self._finalize_streaming(answer, status))
        except Exception as e:
            logging.error(f"Error in on_worker_finished: {traceback.format_exc()}")
            try:
                self._update_conversation_status("Error displaying response", "error")
            except:
                pass
    
    def _finalize_streaming(self, answer, status):
        """Finalize streaming after a short delay to catch any late chunks"""
        try:
            # If we were streaming, ensure final message is displayed correctly with full text
            if self.is_streaming and self.current_streaming_response:
                # Force final update of streaming message to ensure it's complete
                final_text = self.current_streaming_response.strip()
                if final_text:
                    logging.info(f"Finalizing streaming: {len(final_text)} characters accumulated")
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
                                    # Scroll to bottom
                                    scrollbar = self.chat_display.verticalScrollBar()
                                    scrollbar.setValue(scrollbar.maximum())
                                except:
                                    pass
            
            # Store whether we were streaming BEFORE resetting
            was_streaming = self.is_streaming
            accumulated_response = self.current_streaming_response.strip() if self.current_streaming_response else ""
            
            # NOW reset streaming state
            self.is_streaming = False
            self.streaming_message_started = False
            self.current_streaming_response = ""
            
            # If we were streaming, use the accumulated response OR the answer from worker
            # Prefer the accumulated response (from chunks) as it's what was actually displayed
            if was_streaming:
                # Use whichever is longer/more complete
                streaming_text = accumulated_response
                answer_text = str(answer).strip() if answer else ""
                
                # Use the longer one (more likely to be complete)
                if len(streaming_text) > len(answer_text):
                    final_answer = streaming_text
                    logging.info(f"Using accumulated streaming response ({len(final_answer)} chars)")
                elif answer_text:
                    final_answer = answer_text
                    logging.info(f"Using worker answer ({len(final_answer)} chars)")
                else:
                    final_answer = ""
                
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
                # Non-streaming mode - ALWAYS display the message
                answer_str = str(answer).strip()
                if answer_str:
                    logging.info(f"Displaying non-streaming response: {len(answer_str)} characters")
                    # Always append the message to the UI
                    self.append_message("assistant", answer_str)
                    # Ensure it's in history
                    if not self.message_history or self.message_history[-1].get('role') != 'assistant':
                        self.message_history.append({"role": "assistant", "content": answer_str})
                    elif self.message_history[-1].get('content') != answer_str:
                        # Update if different
                        self.message_history[-1]['content'] = answer_str
                else:
                    logging.warning("Empty answer in non-streaming mode")
            
            # Limit history to configured max
            if len(self.message_history) > self.max_history_messages:
                self.message_history = self.message_history[-self.max_history_messages:]
            
            # Reset streaming state for next time
            self.current_streaming_response = ""
            
            # Reset status label style and show appropriate status
            self._update_conversation_status(str(status) if status else "Ready", "ready")
            
            # Always save history after receiving a response
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
                self._update_conversation_status("Error displaying response", "error")
            except:
                pass
    
    def on_worker_error(self, error_msg):
        try:
            error_text = str(error_msg) if error_msg else "Unknown error"
            self.append_message("system", f"❌ Error: {error_text}")
            self._update_conversation_status("Error", "error")
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
                self._update_conversation_status("Error handling failed", "error")
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
            history = self.message_history[-self.max_history_messages:] if len(self.message_history) > self.max_history_messages else self.message_history.copy()
            
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
                
                loaded_model = data.get('model', "GPT-4o")
                if loaded_model in MODEL_OPTIONS:
                    self.current_model = loaded_model
                else:
                    logging.warning(f"Invalid model in history: {loaded_model}")
                
                # Load history with validation
                loaded_history = data.get('history', [])
                if isinstance(loaded_history, list):
                    self.message_history = loaded_history
                    # Limit history to configured max
                    if len(self.message_history) > self.max_history_messages:
                        self.message_history = self.message_history[-self.max_history_messages:]
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
                    
                    # Scroll to bottom after loading all history messages
                    if hasattr(self, 'chat_display') and self.chat_display:
                        scrollbar = self.chat_display.verticalScrollBar()
                        scrollbar.setValue(scrollbar.maximum())
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
    
    def closeEvent(self, event):
        """Handle window close event - save history before closing"""
        try:
            logging.info("Window closing - saving conversation history...")
            self._save_history()
            # Also save any important information to memory
            if self.memory_system and self.message_history:
                # Store summary of current conversation
                recent_messages = self.message_history[-10:]  # Last 10 messages
                if recent_messages:
                    conversation_summary = "\n".join([
                        f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}"
                        for msg in recent_messages if msg.get('content')
                    ])
                    if conversation_summary and self.memory_system.openai_client:
                        try:
                            self.memory_system.store_important_info(
                                f"Recent conversation summary: {conversation_summary}",
                                {"type": "conversation_summary", "timestamp": datetime.now().isoformat()}
                            )
                        except Exception as mem_err:
                            logging.warning(f"Error storing conversation summary: {mem_err}")
        except Exception as e:
            logging.error(f"Error in closeEvent: {e}")
        finally:
            event.accept()
    
    def _load_conversation_history_to_memory(self):
        """Load conversation history from file and index it in memory system"""
        if not self.memory_system or not self.memory_system.openai_client:
            return
        
        try:
            history_path = PROJECT_DIR / self.history_file
            if not history_path.exists():
                alt_path = Path.home() / "lea_history.json"
                if alt_path.exists():
                    history_path = alt_path
                else:
                    return
            
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history = data.get('history', [])
            if not history:
                return
            
            # Index conversation history into memory system
            # Group messages into conversation pairs for better context
            conversation_chunks = []
            current_chunk = []
            
            for msg in history:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role in ['user', 'assistant'] and content:
                    # Clean up content
                    if role == 'user' and 'Dre\'s question:' in str(content):
                        content = str(content).split('Dre\'s question:')[-1].strip()
                    
                    if len(current_chunk) < 4:  # Group up to 4 messages
                        current_chunk.append(f"{role}: {content[:500]}")  # Limit length
                    else:
                        conversation_chunks.append("\n".join(current_chunk))
                        current_chunk = [f"{role}: {content[:500]}"]
            
            if current_chunk:
                conversation_chunks.append("\n".join(current_chunk))
            
            # Store chunks in memory (limit to avoid too many embeddings)
            for chunk in conversation_chunks[-20:]:  # Last 20 chunks
                try:
                    self.memory_system.store_important_info(
                        chunk,
                        {"type": "conversation_history", "source": "history_file"}
                    )
                except Exception as e:
                    logging.warning(f"Error indexing conversation chunk: {e}")
            
            logging.info(f"Loaded {len(conversation_chunks)} conversation chunks into memory")
            
        except Exception as e:
            logging.warning(f"Error loading conversation history to memory: {e}")
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Settings loading (no TTS settings)
        except Exception as e:
            logging.warning(f"Error loading settings: {e}")
            # Use defaults
    
    def save_settings(self):
        """Save settings to file"""
        try:
            data = {}
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Error saving settings: {e}")
    
    def _update_tts_mic_status(self):
        """Update TTS/Microphone status indicator"""
        if not hasattr(self, 'tts_mic_status'):
            return
        
        if self.is_listening:
            # Listening (green)
            self.tts_mic_status.setText("🟢 TTS/Mic: Listening")
            self.tts_mic_status.setStyleSheet("""
                QLabel {
                    color: #68BD47;
                    font-size: 12px;
                    font-weight: 600;
                    padding: 4px 8px;
                    background-color: #2a2a2a;
                    border-radius: 4px;
                }
            """)
        elif self.tts_enabled:
            # TTS active but not listening (green)
            self.tts_mic_status.setText("🟢 TTS/Mic: Active")
            self.tts_mic_status.setStyleSheet("""
                QLabel {
                    color: #68BD47;
                    font-size: 12px;
                    font-weight: 600;
                    padding: 4px 8px;
                    background-color: #2a2a2a;
                    border-radius: 4px;
                }
            """)
        else:
            # Inactive (red)
            self.tts_mic_status.setText("🔴 TTS/Mic: Inactive")
            self.tts_mic_status.setStyleSheet("""
                QLabel {
                    color: #FF4444;
                    font-size: 12px;
                    font-weight: 600;
                    padding: 4px 8px;
                    background-color: #2a2a2a;
                    border-radius: 4px;
                }
            """)
    
    def _update_conversation_status(self, status: str, state: str = "ready"):
        """
        Update conversation status indicator
        state: 'ready' (green), 'thinking' (yellow), 'error' (red), 'processing' (blue)
        """
        if not hasattr(self, 'conversation_status'):
            return
        
        color_map = {
            "ready": "#68BD47",      # Green
            "thinking": "#FFB020",   # Yellow/Orange
            "processing": "#2DBCEE", # Blue
            "error": "#FF4444",      # Red
            "listening": "#9B59B6"   # Purple (for speech recognition)
        }
        
        color = color_map.get(state, "#DDD")
        self.conversation_status.setText(status)
        self.conversation_status.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }}
        """)
    
    def toggle_tts(self):
        """Toggle TTS on/off"""
        # Check if TTS is actually available before toggling
        if not TTS_AVAILABLE:
            import sys
            import subprocess
            
            error_details = "TTS is not available.\n\n"
            if TTS_ERROR:
                error_details += f"Error: {TTS_ERROR}\n\n"
            
            python_exe = sys.executable
            python_version = sys.version.split()[0]
            
            error_details += f"Current Python: {python_exe}\n"
            error_details += f"Python version: {python_version}\n\n"
            error_details += "To install missing packages, run this command:\n"
            error_details += f"{python_exe} -m pip install --upgrade gtts pygame\n\n"
            error_details += "Or click 'Install Now' to attempt automatic installation."
            
            # Create message box with install button
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("TTS Not Available")
            msg.setText("TTS libraries are not available")
            msg.setInformativeText(error_details)
            
            # Add install button
            install_btn = msg.addButton("Install Now", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            result = msg.exec()
            
            if msg.clickedButton() == install_btn:
                # Try to install packages
                self._install_tts_packages(python_exe)
            
            return
        
        self.tts_enabled = not self.tts_enabled
        if hasattr(self, 'tts_btn'):
            if self.tts_enabled:
                self.tts_btn.setText("🔊 TTS On")
                self.tts_btn.setChecked(True)
            else:
                self.tts_btn.setText("🔇 TTS Off")
                self.tts_btn.setChecked(False)
        self._update_tts_mic_status()
    
    def _install_tts_packages(self, python_exe: str):
        """Attempt to install TTS packages using the current Python interpreter"""
        import subprocess
        import sys
        
        self._update_conversation_status("Installing TTS packages...", "processing")
        
        try:
            # Try to install packages
            packages = ["gtts", "pygame"]
            results = []
            
            for package in packages:
                try:
                    self.append_message("system", f"Installing {package}...")
                    QApplication.processEvents()
                    
                    result = subprocess.run(
                        [python_exe, "-m", "pip", "install", "--upgrade", package],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        results.append(f"✅ {package} installed successfully")
                        self.append_message("system", f"✅ {package} installed")
                    else:
                        error_msg = result.stderr or result.stdout
                        results.append(f"❌ {package} failed: {error_msg[:200]}")
                        self.append_message("system", f"❌ {package} installation failed")
                        
                except subprocess.TimeoutExpired:
                    results.append(f"⏱️ {package} installation timed out")
                except Exception as e:
                    results.append(f"❌ {package} error: {str(e)}")
            
            # Show results
            result_text = "\n".join(results)
            msg = QMessageBox(self)
            msg.setWindowTitle("Installation Results")
            msg.setText("TTS Package Installation")
            msg.setInformativeText(result_text)
            msg.setDetailedText(f"Python used: {python_exe}\n\nFull output:\n{result_text}")
            msg.exec()
            
            # Reload TTS availability
            global TTS_AVAILABLE, PYGAME_AVAILABLE, TTS_ERROR
            try:
                from gtts import gTTS
                import pygame  # pygame-ce installs as pygame
                TTS_AVAILABLE = True
                PYGAME_AVAILABLE = True
                TTS_ERROR = None
                self.append_message("system", "✅ TTS libraries now available! Restart the application to use TTS.")
                QMessageBox.information(self, "Success", 
                    "TTS packages installed successfully!\n\n"
                    "Please restart the application for changes to take effect.")
            except Exception as e:
                self.append_message("system", f"⚠️ Packages installed but still not available: {e}")
                QMessageBox.warning(self, "Restart Required", 
                    "Packages were installed, but you need to restart the application.\n\n"
                    f"Error: {e}")
            
            self._update_conversation_status("Ready", "ready")
            
        except Exception as e:
            self._update_conversation_status("Installation failed", "error")
            QMessageBox.critical(self, "Installation Error", 
                f"Failed to install packages:\n\n{e}\n\n"
                f"Please install manually:\n{python_exe} -m pip install gtts pygame")
    
    def _install_pyaudio(self, python_exe: str):
        """Attempt to install PyAudio using the current Python interpreter"""
        import subprocess
        import sys
        
        self._update_conversation_status("Installing PyAudio...", "processing")
        
        try:
            self.append_message("system", "Installing PyAudio...")
            QApplication.processEvents()
            
            # Try standard pip install first
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "pyaudio"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.append_message("system", "✅ PyAudio installed successfully")
                # Reload PyAudio availability
                global PYAUDIO_AVAILABLE, SPEECH_RECOGNITION_ERROR
                try:
                    import speech_recognition as sr
                    _ = sr.Microphone.list_microphone_names()
                    # Update global variables
                    import sys
                    module = sys.modules[__name__]
                    module.PYAUDIO_AVAILABLE = True
                    module.SPEECH_RECOGNITION_ERROR = None
                    QMessageBox.information(self, "Success", 
                        "PyAudio installed successfully!\n\n"
                        "You can now use the microphone feature.")
                except Exception as e:
                    # Try pipwin as fallback for Windows
                    self.append_message("system", "⚠️ Standard install worked but still not available. Trying pipwin...")
                    QApplication.processEvents()
                    
                    # Try installing pipwin and then pyaudio
                    pipwin_result = subprocess.run(
                        [python_exe, "-m", "pip", "install", "pipwin"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if pipwin_result.returncode == 0:
                        pipwin_install = subprocess.run(
                            [python_exe, "-m", "pipwin", "install", "pyaudio"],
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        
                        if pipwin_install.returncode == 0:
                            global PYAUDIO_AVAILABLE, SPEECH_RECOGNITION_ERROR
                            try:
                                import speech_recognition as sr
                                _ = sr.Microphone.list_microphone_names()
                                # Update global variables
                                import sys
                                module = sys.modules[__name__]
                                module.PYAUDIO_AVAILABLE = True
                                module.SPEECH_RECOGNITION_ERROR = None
                                QMessageBox.information(self, "Success", 
                                    "PyAudio installed via pipwin!\n\n"
                                    "You can now use the microphone feature.")
                            except Exception as e2:
                                QMessageBox.warning(self, "Restart Required", 
                                    "PyAudio was installed, but you need to restart the application.\n\n"
                                    f"Error: {e2}")
                        else:
                            error_msg = pipwin_install.stderr or pipwin_install.stdout
                            QMessageBox.warning(self, "Installation Issue", 
                                f"PyAudio installed but may need restart.\n\n"
                                f"pipwin error: {error_msg[:300]}")
                    else:
                        QMessageBox.warning(self, "Restart Required", 
                            "PyAudio was installed, but you need to restart the application.")
            else:
                error_msg = result.stderr or result.stdout
                # Try pipwin as alternative for Windows
                self.append_message("system", "⚠️ Standard pip install failed. Trying pipwin (Windows alternative)...")
                QApplication.processEvents()
                
                # Install pipwin first
                pipwin_result = subprocess.run(
                    [python_exe, "-m", "pip", "install", "pipwin"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if pipwin_result.returncode == 0:
                    pipwin_install = subprocess.run(
                        [python_exe, "-m", "pipwin", "install", "pyaudio"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    if pipwin_install.returncode == 0:
                        self.append_message("system", "✅ PyAudio installed via pipwin")
                        global PYAUDIO_AVAILABLE, SPEECH_RECOGNITION_ERROR
                        try:
                            import speech_recognition as sr
                            _ = sr.Microphone.list_microphone_names()
                            # Update global variables
                            import sys
                            module = sys.modules[__name__]
                            module.PYAUDIO_AVAILABLE = True
                            module.SPEECH_RECOGNITION_ERROR = None
                            QMessageBox.information(self, "Success", 
                                "PyAudio installed via pipwin!\n\n"
                                "You can now use the microphone feature.")
                        except Exception as e:
                            QMessageBox.warning(self, "Restart Required", 
                                "PyAudio was installed, but you need to restart the application.\n\n"
                                f"Error: {e}")
                    else:
                        pipwin_error = pipwin_install.stderr or pipwin_install.stdout
                        QMessageBox.critical(self, "Installation Failed", 
                            f"Failed to install PyAudio via pipwin:\n\n{pipwin_error[:500]}\n\n"
                            f"Try manually:\n1. pip install pipwin\n2. pipwin install pyaudio")
                else:
                    QMessageBox.critical(self, "Installation Failed", 
                        f"Failed to install PyAudio:\n\n{error_msg[:500]}\n\n"
                        f"On Windows, try:\n1. {python_exe} -m pip install pipwin\n"
                        f"2. {python_exe} -m pipwin install pyaudio")
            
            self._update_conversation_status("Ready", "ready")
            
        except subprocess.TimeoutExpired:
            self._update_conversation_status("Installation timed out", "error")
            QMessageBox.warning(self, "Timeout", 
                "Installation timed out. Please try installing manually:\n\n"
                f"{python_exe} -m pip install pyaudio")
        except Exception as e:
            self._update_conversation_status("Installation failed", "error")
            QMessageBox.critical(self, "Installation Error", 
                f"Failed to install PyAudio:\n\n{e}\n\n"
                f"Please install manually:\n{python_exe} -m pip install pyaudio\n\n"
                f"On Windows, you may need:\n1. pip install pipwin\n2. pipwin install pyaudio")
    
    def speak_text(self, text: str):
        """Speak text using gTTS in a separate thread to avoid blocking"""
        if not TTS_AVAILABLE or not PYGAME_AVAILABLE:
            # Update status to show TTS error
            self._update_tts_mic_status()
            error_details = "TTS libraries not available at startup.\n\n"
            if TTS_ERROR:
                error_details += f"Initialization Error: {TTS_ERROR}\n\n"
            error_details += "Install with: pip install gtts pygame\n\n"
            error_details += f"Python: {sys.executable}\n"
            error_details += f"Python version: {sys.version.split()[0]}\n\n"
            error_details += "Note: Make sure you're using the same Python environment where packages are installed."
            QMessageBox.warning(self, "TTS Not Available", error_details)
            return
        
        def speak_in_thread():
            error_step = None
            error_details = ""
            tmp_path = None
            
            try:
                # Clean text - remove markdown, code blocks, etc.
                error_step = "Text cleaning"
                clean_text = text
                import re
                clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
                clean_text = re.sub(r'`[^`]+`', '', clean_text)
                clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
                clean_text = ' '.join(clean_text.split())
                
                if not clean_text or len(clean_text) < 3:
                    return
                
                # Limit length to avoid very long audio
                if len(clean_text) > 500:
                    clean_text = clean_text[:500] + "..."
                
                # Create gTTS object
                error_step = "Creating gTTS object"
                try:
                    tts = gTTS(text=clean_text, lang=self.tts_voice_lang, tld=self.tts_voice_tld, slow=False)
                except Exception as e:
                    error_step = "gTTS creation failed"
                    raise Exception(f"Failed to create gTTS object: {e}\n\nThis usually means:\n- No internet connection (gTTS needs internet)\n- gTTS library issue\n- Language code '{self.tts_voice_lang}' not supported")
                
                # Save to temporary file
                error_step = "Saving audio file"
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_path = tmp_file.name
                        tts.save(tmp_path)
                except Exception as e:
                    error_step = "File save failed"
                    raise Exception(f"Failed to save audio file: {e}\n\nThis usually means:\n- Disk space issue\n- Permission problem with temp directory\n- File system error")
                
                # Play using pygame
                error_step = "Initializing pygame mixer"
                try:
                    pygame.mixer.init()
                except Exception as e:
                    error_step = "Pygame mixer init failed"
                    raise Exception(f"Failed to initialize pygame mixer: {e}\n\nThis usually means:\n- Audio device not available\n- Pygame audio driver issue\n- System audio problem")
                
                error_step = "Loading audio file"
                try:
                    pygame.mixer.music.load(tmp_path)
                except Exception as e:
                    error_step = "Audio file load failed"
                    raise Exception(f"Failed to load audio file: {e}\n\nFile: {tmp_path}\n\nThis usually means:\n- Corrupted audio file\n- Unsupported audio format\n- File access issue")
                
                error_step = "Playing audio"
                try:
                    pygame.mixer.music.play()
                except Exception as e:
                    error_step = "Audio playback failed"
                    raise Exception(f"Failed to play audio: {e}\n\nThis usually means:\n- Audio device busy\n- Audio driver issue\n- System audio problem")
                
                # Wait for playback to finish (with periodic checks to prevent blocking)
                error_step = "Waiting for playback"
                timeout = 30  # 30 second timeout
                elapsed = 0
                check_interval = 0.5  # Check every 500ms instead of 100ms
                while pygame.mixer.music.get_busy():
                    time.sleep(check_interval)
                    elapsed += check_interval
                    if elapsed > timeout:
                        logging.warning("Playback timeout - stopping audio")
                        pygame.mixer.music.stop()  # Force stop if stuck
                        break
                
                # Clean up
                error_step = "Cleanup"
                pygame.mixer.quit()
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Build detailed error message
                error_details = f"TTS Error at step: {error_step}\n\n"
                error_details += f"Error Type: {error_type}\n"
                error_details += f"Error Message: {error_msg}\n\n"
                
                # Add diagnostics
                error_details += "Diagnostics:\n"
                error_details += f"- Python: {sys.executable}\n"
                error_details += f"- Python version: {sys.version.split()[0]}\n"
                error_details += f"- TTS Available: {TTS_AVAILABLE}\n"
                error_details += f"- Pygame Available: {PYGAME_AVAILABLE}\n"
                error_details += f"- TTS Enabled: {self.tts_enabled}\n"
                error_details += f"- Voice Language: {self.tts_voice_lang}\n"
                error_details += f"- Voice TLD: {self.tts_voice_tld}\n"
                
                if tmp_path:
                    error_details += f"- Temp file: {tmp_path}\n"
                    error_details += f"- Temp file exists: {os.path.exists(tmp_path) if tmp_path else 'N/A'}\n"
                
                # Add troubleshooting steps
                error_details += "\nTroubleshooting:\n"
                if "internet" in error_msg.lower() or "connection" in error_msg.lower():
                    error_details += "1. Check your internet connection (gTTS needs internet)\n"
                if "audio" in error_msg.lower() or "mixer" in error_msg.lower():
                    error_details += "2. Check if audio device is working\n"
                    error_details += "3. Try restarting the application\n"
                    error_details += "4. Check system audio settings\n"
                if "permission" in error_msg.lower() or "access" in error_msg.lower():
                    error_details += "5. Check file permissions\n"
                    error_details += "6. Check disk space\n"
                
                error_details += "\nIf problem persists, try:\n"
                error_details += "- Reinstall: pip install --upgrade gtts pygame\n"
                error_details += "- Check if other audio apps are using the device\n"
                
                logging.error(f"TTS error at {error_step}: {error_msg}\n{error_details}")
                
                # Update status and show detailed error popup
                self._update_tts_mic_status()
                
                # Use QMessageBox with detailed text
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("TTS Error - Detailed Diagnostics")
                msg.setText(f"Text-to-Speech failed at: {error_step}")
                msg.setDetailedText(error_details)
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
        
        # Run in separate thread to avoid blocking UI
        import threading
        thread = threading.Thread(target=speak_in_thread, daemon=True)
        thread.start()
    
    def show_coordinate_finder(self):
        """Show coordinate finder tool to help find screen coordinates"""
        if not TASK_SYSTEM_AVAILABLE:
            QMessageBox.information(self, "Not Available", "Task system not available")
            return
        
        try:
            import pyautogui
        except ImportError:
            QMessageBox.warning(self, "Missing Library", 
                               "pyautogui not installed.\n\nInstall with: pip install pyautogui")
            return
        
        # Create a simple floating window that shows coordinates
        coord_window = QDialog(self)
        coord_window.setWindowTitle("📍 Coordinate Finder - Press ESC to close")
        coord_window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        coord_window.setMinimumSize(300, 150)
        coord_window.setStyleSheet("""
            QDialog {
                background-color: #222;
                border: 2px solid #68BD47;
            }
            QLabel {
                color: #68BD47;
                font-size: 18px;
                font-weight: 600;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(coord_window)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        screen_info = QLabel(f"Screen Size: {screen_width} x {screen_height}")
        screen_info.setStyleSheet("color: #2DBCEE; font-size: 12px;")
        layout.addWidget(screen_info)
        
        # Coordinate display
        coord_label = QLabel("Move mouse to see coordinates...")
        coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(coord_label)
        
        # Instructions
        instructions = QLabel("Click 'Copy' to copy coordinates\nPress ESC to close")
        instructions.setStyleSheet("color: #FFF; font-size: 11px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)
        
        # Copy button
        copy_btn = QPushButton("Copy Coordinates")
        copy_btn.clicked.connect(lambda: self._copy_coordinates(pyautogui, coord_label))
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #68BD47;
                color: #FFF;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5aa03a;
            }
        """)
        layout.addWidget(copy_btn)
        
        # Timer to update coordinates
        from PyQt6.QtCore import QTimer
        timer = QTimer()
        def update_coordinates():
            try:
                x, y = pyautogui.position()
                coord_label.setText(f"X: {x}  |  Y: {y}")
            except:
                pass
        timer.timeout.connect(update_coordinates)
        timer.start(100)  # Update every 100ms
        
        # Handle ESC key
        def keyPressEvent(event):
            if event.key() == Qt.Key.Key_Escape:
                timer.stop()
                coord_window.close()
        coord_window.keyPressEvent = keyPressEvent
        
        # Position window in top-right corner
        screen = QApplication.primaryScreen().geometry()
        coord_window.move(screen.width() - 320, 50)
        
        coord_window.exec()
        timer.stop()
    
    def _copy_coordinates(self, pyautogui, coord_label):
        """Copy current mouse coordinates to clipboard"""
        try:
            x, y = pyautogui.position()
            from PyQt6.QtGui import QClipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(f"{x}, {y}")
            coord_label.setText(f"✅ Copied: {x}, {y}")
            coord_label.setStyleSheet("color: #68BD47; font-size: 18px; font-weight: 600; padding: 10px;")
            QMessageBox.information(self, "Copied", f"Coordinates copied to clipboard:\nX: {x}, Y: {y}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not get coordinates: {e}")
    
    def show_settings(self):
        """Show settings dialog"""
        if TTS_AVAILABLE:
            dialog = QDialog(self)
            dialog.setWindowTitle("⚙️ TTS Settings")
            dialog.setMinimumSize(300, 200)
            dialog.setStyleSheet("""
                QDialog {
                    background-color: #333;
                }
            """)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            title = QLabel("Text-to-Speech Settings")
            title.setStyleSheet("font-size: 18px; font-weight: 600; color: #FFF; margin-bottom: 10px;")
            layout.addWidget(title)
            
            tts_toggle = QCheckBox("Enable TTS")
            tts_toggle.setChecked(self.tts_enabled)
            tts_toggle.toggled.connect(lambda checked: setattr(self, 'tts_enabled', checked))
            tts_toggle.setStyleSheet("color: #FFF; font-size: 14px;")
            layout.addWidget(tts_toggle)
            
            layout.addStretch()
            
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            buttons.accepted.connect(dialog.accept)
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
            
            dialog.exec()
        else:
            error_details = "TTS is not available.\n\n"
            if TTS_ERROR:
                error_details += f"Error: {TTS_ERROR}\n\n"
            error_details += "Install with: pip install gtts pygame\n\n"
            error_details += "Note: Make sure you're using the same Python environment where packages are installed."
            QMessageBox.information(self, "TTS Settings", error_details)
    
    def toggle_speech_recognition(self):
        """Toggle speech recognition on/off"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            QMessageBox.information(self, "Speech Recognition", 
                                   "Speech recognition not available. Install with: pip install SpeechRecognition")
            return
        
        if self.is_listening:
            # Stop listening
            self.is_listening = False
            if hasattr(self, 'mic_btn'):
                self.mic_btn.setText("🎤")
                self.mic_btn.setStyleSheet("background-color: #444; font-size: 20px; border-radius: 4px; padding: 4px;")
            self._update_conversation_status("Ready", "ready")
            self._update_tts_mic_status()
            return
        
        # Start listening
        if not self.speech_recognizer:
            try:
                self.speech_recognizer = sr.Recognizer()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not initialize speech recognizer: {e}")
                return
        
        # Check if microphone is set up
        if self.microphone_device_index is None:
            self._setup_microphone_first_time()
            return
        
        # Start speech recognition in a thread
        self.is_listening = True
        if hasattr(self, 'mic_btn'):
            self.mic_btn.setText("🔴")
            self.mic_btn.setStyleSheet("background-color: #D13438; font-size: 20px; border-radius: 4px; padding: 4px;")
        self._update_conversation_status("Listening...", "listening")
        self._update_tts_mic_status()
        
        # Create worker thread
        self.speech_worker_thread = QThread()
        self.speech_worker = SpeechRecognitionWorker(self.speech_recognizer, self.microphone_device_index)
        self.speech_worker.moveToThread(self.speech_worker_thread)
        
        # Connect signals
        self.speech_worker_thread.started.connect(self.speech_worker.run)
        self.speech_worker.finished.connect(self.on_speech_recognition_finished)
        self.speech_worker.error.connect(self.on_speech_recognition_error)
        self.speech_worker.listening.connect(self.on_speech_listening)
        
        # Start thread
        self.speech_worker_thread.start()
    
    def _setup_microphone_first_time(self):
        """Setup microphone device selection on first use"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            error_details = "Speech recognition is not available.\n\n"
            if SPEECH_RECOGNITION_ERROR:
                error_details += f"Error: {SPEECH_RECOGNITION_ERROR}\n\n"
            error_details += "Install with: pip install SpeechRecognition"
            QMessageBox.warning(self, "Speech Recognition Not Available", error_details)
            return
        
        # Check for PyAudio specifically
        if not PYAUDIO_AVAILABLE:
            import sys
            python_exe = sys.executable
            
            error_details = "PyAudio is required for microphone access.\n\n"
            error_details += f"Current Python: {python_exe}\n"
            error_details += f"Python version: {sys.version.split()[0]}\n\n"
            if SPEECH_RECOGNITION_ERROR:
                error_details += f"Error: {SPEECH_RECOGNITION_ERROR}\n\n"
            
            error_details += "To install PyAudio, run:\n"
            error_details += f"{python_exe} -m pip install pyaudio\n\n"
            error_details += "Note: PyAudio requires PortAudio library on Windows.\n"
            error_details += "If pip install fails, try:\n"
            error_details += "  1. Install Visual C++ Build Tools\n"
            error_details += "  2. Download PortAudio from http://files.portaudio.com/\n"
            error_details += "  3. Or use a pre-built wheel from:\n"
            error_details += "     https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio\n\n"
            error_details += "Or click 'Install Now' to attempt automatic installation."
            
            # Create message box with install button
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("PyAudio Required")
            msg.setText("PyAudio is required for microphone access")
            msg.setInformativeText(error_details)
            
            # Add install button
            install_btn = msg.addButton("Install Now", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            result = msg.exec()
            
            if msg.clickedButton() == install_btn:
                # Try to install PyAudio
                self._install_pyaudio(python_exe)
                # Retry after installation
                if PYAUDIO_AVAILABLE:
                    self._setup_microphone_first_time()
                return
            else:
                return
        
        try:
            # Get list of microphones
            mic_list = sr.Microphone.list_microphone_names()
            
            if not mic_list:
                QMessageBox.warning(self, "No Microphone", "No microphones found on your system.")
                return
            
            # Show selection dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Microphone")
            dialog.setMinimumSize(400, 300)
            dialog.setStyleSheet("""
                QDialog {
                    background-color: #333;
                }
            """)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            title = QLabel("Select your microphone:")
            title.setStyleSheet("font-size: 16px; font-weight: 600; color: #FFF;")
            layout.addWidget(title)
            
            from PyQt6.QtWidgets import QListWidget
            mic_list_widget = QListWidget()
            mic_list_widget.addItems(mic_list)
            mic_list_widget.setStyleSheet("""
                QListWidget {
                    background-color: #222;
                    color: #FFF;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 5px;
                }
                QListWidget::item {
                    padding: 8px;
                }
                QListWidget::item:selected {
                    background-color: #0078D7;
                }
            """)
            layout.addWidget(mic_list_widget)
            
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
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_items = mic_list_widget.selectedItems()
                if selected_items:
                    selected_name = selected_items[0].text()
                    self.microphone_device_index = mic_list.index(selected_name)
                    QMessageBox.information(self, "Microphone Selected", 
                                          f"Selected: {selected_name}\n\nYou can change this in settings later.")
                    # Now start listening
                    self.toggle_speech_recognition()
        except Exception as e:
            error_msg = str(e)
            error_details = f"Error setting up microphone: {error_msg}\n\n"
            
            # Check if it's a PyAudio error
            if "pyaudio" in error_msg.lower() or "Could not find PyAudio" in error_msg:
                import sys
                python_exe = sys.executable
                error_details += "PyAudio is not installed or not working.\n\n"
                error_details += f"Install with: {python_exe} -m pip install pyaudio\n\n"
                error_details += "On Windows, you may need:\n"
                error_details += "  pip install pipwin\n"
                error_details += "  pipwin install pyaudio\n\n"
                error_details += "Or click 'Install Now' to attempt automatic installation."
                
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("PyAudio Required")
                msg.setText("PyAudio is required for microphone access")
                msg.setInformativeText(error_details)
                
                install_btn = msg.addButton("Install Now", QMessageBox.ButtonRole.ActionRole)
                msg.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
                
                if msg.exec() == 0 and msg.clickedButton() == install_btn:
                    self._install_pyaudio(python_exe)
            else:
                error_details += f"Python: {sys.executable}\n"
                error_details += f"Error type: {type(e).__name__}\n"
                QMessageBox.warning(self, "Microphone Setup Error", error_details)
            
            logging.error(f"Microphone setup error: {traceback.format_exc()}")
    
    def on_speech_listening(self):
        """Called when speech recognition starts listening"""
        self._update_conversation_status("Listening... Speak now!", "listening")
        self._update_tts_mic_status()
    
    def on_speech_recognition_finished(self, text: str):
        """Handle successful speech recognition"""
        self.is_listening = False
        if hasattr(self, 'mic_btn'):
            self.mic_btn.setText("🎤")
            self.mic_btn.setStyleSheet("background-color: #444; font-size: 20px; border-radius: 4px; padding: 4px;")
        
        # Insert recognized text into input box
        if text:
            current_text = self.input_box.toPlainText()
            if current_text:
                self.input_box.setPlainText(current_text + " " + text)
            else:
                self.input_box.setPlainText(text)
            self._update_conversation_status("Ready", "ready")
            self._update_tts_mic_status()
        else:
            self._update_conversation_status("No speech detected", "ready")
            self._update_tts_mic_status()
        
        # Clean up thread - use timeout to prevent blocking
        if self.speech_worker_thread:
            self.speech_worker_thread.quit()
            # Use timeout to prevent UI freezing (max 1 second wait)
            if not self.speech_worker_thread.wait(1000):  # 1000ms timeout
                logging.warning("Speech worker thread did not finish in time, forcing cleanup")
                self.speech_worker_thread.terminate()  # Force termination if stuck
                self.speech_worker_thread.wait(500)  # Brief wait after terminate
            self.speech_worker_thread.deleteLater()
            self.speech_worker_thread = None
            if self.speech_worker:
                self.speech_worker.deleteLater()
                self.speech_worker = None
    
    def on_speech_recognition_error(self, error_msg: str):
        """Handle speech recognition error"""
        self.is_listening = False
        if hasattr(self, 'mic_btn'):
            self.mic_btn.setText("🎤")
            self.mic_btn.setStyleSheet("background-color: #444; font-size: 20px; border-radius: 4px; padding: 4px;")
        
        self._update_conversation_status(f"Error: {error_msg}", "error")
        self._update_tts_mic_status()
        # Show error popup with details
        QMessageBox.warning(self, "TTS/Mic Error", 
                           f"Speech Recognition Error:\n\n{error_msg}\n\n"
                           f"TTS Status: {'Active' if self.tts_enabled else 'Inactive'}\n"
                           f"Microphone: {'Listening' if self.is_listening else 'Not listening'}")
        
        # Clean up thread - use timeout to prevent blocking
        if self.speech_worker_thread:
            self.speech_worker_thread.quit()
            # Use timeout to prevent UI freezing (max 1 second wait)
            if not self.speech_worker_thread.wait(1000):  # 1000ms timeout
                logging.warning("Speech worker thread did not finish in time, forcing cleanup")
                self.speech_worker_thread.terminate()  # Force termination if stuck
                self.speech_worker_thread.wait(500)  # Brief wait after terminate
            self.speech_worker_thread.deleteLater()
            self.speech_worker_thread = None
            if self.speech_worker:
                self.speech_worker.deleteLater()
                self.speech_worker = None

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
