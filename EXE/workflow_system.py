"""
Workflow Automation System for Lea Assistant

This module provides workflow recording, storage, and playback capabilities.
Workflows can be taught to Lea by demonstration, then replayed automatically.

Workflows are stored as JSON files and can include:
- Mouse clicks and movements
- Keyboard input
- Screen verification points
- Parameter substitution
- Loops and conditions
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import threading

# Try to import desktop automation tools
DESKTOP_TOOLS_AVAILABLE = False
desktop_tools = None
try:
    import desktop_tools
    DESKTOP_TOOLS_AVAILABLE = True
except ImportError:
    DESKTOP_TOOLS_AVAILABLE = False
    logging.warning("desktop_tools not available - workflow automation limited")
except Exception as e:
    DESKTOP_TOOLS_AVAILABLE = False
    logging.warning(f"Error importing desktop_tools: {e} - workflow automation limited")

# Try to import action capture libraries
PYNPUT_AVAILABLE = False
mouse = None
keyboard = None
try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logging.warning("pynput not available - install with: pip install pynput")
except Exception as e:
    PYNPUT_AVAILABLE = False
    logging.warning(f"Error importing pynput: {e}")

PYAUTOGUI_AVAILABLE = False
pyautogui = None
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("pyautogui not available - install with: pip install pyautogui")
except Exception as e:
    PYAUTOGUI_AVAILABLE = False
    logging.warning(f"Error importing pyautogui: {e}")


@dataclass
class WorkflowAction:
    """Represents a single action in a workflow"""
    action_type: str  # "click", "type", "key", "hotkey", "wait", "verify", "screenshot"
    timestamp: float
    parameters: Dict[str, Any]  # Action-specific parameters
    description: str = ""  # Human-readable description


@dataclass
class Workflow:
    """Represents a complete workflow"""
    name: str
    description: str
    created: str  # ISO timestamp
    modified: str  # ISO timestamp
    actions: List[WorkflowAction]
    parameters: Dict[str, str]  # Parameter names and descriptions
    category: str = "general"  # Category for organization


class WorkflowRecorder:
    """Records user actions to create workflows with actual mouse/keyboard capture"""
    
    def __init__(self):
        self.is_recording = False
        self.workflow_actions: List[WorkflowAction] = []
        self.start_time: Optional[float] = None
        self.mouse_listener = None
        self.keyboard_listener = None
        self.last_mouse_pos = None
        self.last_action_time = None
        self.recording_lock = threading.Lock()
    
    def start_recording(self, workflow_name: str) -> Tuple[bool, str]:
        """Start recording a new workflow with mouse and keyboard listeners"""
        if self.is_recording:
            return False, "Already recording a workflow"
        
        if not PYNPUT_AVAILABLE:
            return False, "pynput not available - install with: pip install pynput"
        
        self.workflow_actions = []
        self.start_time = time.time()
        self.last_action_time = self.start_time
        self.last_mouse_pos = None
        self.is_recording = True
        
        try:
            # Start mouse listener (if pynput available)
            if mouse:
                try:
                    self.mouse_listener = mouse.Listener(
                        on_click=self._on_mouse_click,
                        on_move=self._on_mouse_move,
                        on_scroll=self._on_mouse_scroll
                    )
                    self.mouse_listener.start()
                    logging.info("Mouse listener started")
                except Exception as e:
                    logging.warning(f"Failed to start mouse listener: {e}")
                    self.mouse_listener = None
            
            # Start keyboard listener (if pynput available)
            if keyboard:
                try:
                    self.keyboard_listener = keyboard.Listener(
                        on_press=self._on_key_press,
                        on_release=self._on_key_release
                    )
                    self.keyboard_listener.start()
                    logging.info("Keyboard listener started")
                except Exception as e:
                    logging.warning(f"Failed to start keyboard listener: {e}")
                    self.keyboard_listener = None
            
            if not self.mouse_listener and not self.keyboard_listener:
                # No listeners started - pynput not available
                self.is_recording = False
                return False, "pynput not available - install with: pip install pynput"
            
            logging.info(f"Started recording workflow: {workflow_name} (listeners active)")
            return True, f"Recording started for workflow: {workflow_name}. Perform your actions now."
        except Exception as e:
            self.is_recording = False
            error_msg = f"Error starting recording: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
    def stop_recording(self) -> Tuple[bool, str, List[WorkflowAction]]:
        """Stop recording and return the recorded actions"""
        if not self.is_recording:
            return False, "Not currently recording", []
        
        self.is_recording = False
        
        # Stop listeners
        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
                self.mouse_listener = None
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
        except Exception as e:
            logging.warning(f"Error stopping listeners: {e}")
        
        actions = self.workflow_actions.copy()
        self.workflow_actions = []
        self.start_time = None
        self.last_action_time = None
        
        logging.info(f"Stopped recording. Captured {len(actions)} actions")
        return True, f"Recording stopped. Captured {len(actions)} actions", actions
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if not self.is_recording or not pressed:  # Only record press, not release
            return
        
        with self.recording_lock:
            current_time = time.time()
            relative_time = current_time - self.start_time if self.start_time else 0.0
            
            # Determine button name
            button_name = "left"
            if button == mouse.Button.right:
                button_name = "right"
            elif button == mouse.Button.middle:
                button_name = "middle"
            
            # Add click action
            action = WorkflowAction(
                action_type="click",
                timestamp=relative_time,
                parameters={
                    "x": int(x),
                    "y": int(y),
                    "button": button_name,
                    "clicks": 1
                },
                description=f"Click at ({int(x)}, {int(y)}) with {button_name} button"
            )
            self.workflow_actions.append(action)
            self.last_action_time = current_time
            self.last_mouse_pos = (x, y)
            logging.debug(f"Recorded click: {button_name} at ({int(x)}, {int(y)})")
    
    def _on_mouse_move(self, x, y):
        """Handle mouse move events (optional - can be filtered)"""
        # Only record significant moves (more than 10 pixels) to reduce noise
        if not self.is_recording:
            return
        
        if self.last_mouse_pos:
            dx = abs(x - self.last_mouse_pos[0])
            dy = abs(y - self.last_mouse_pos[1])
            if dx < 10 and dy < 10:  # Ignore small movements
                return
        
        # Don't record every move - too noisy
        # Only update position for reference
        self.last_mouse_pos = (x, y)
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        if not self.is_recording:
            return
        
        with self.recording_lock:
            current_time = time.time()
            relative_time = current_time - self.start_time if self.start_time else 0.0
            
            action = WorkflowAction(
                action_type="scroll",
                timestamp=relative_time,
                parameters={
                    "x": int(x),
                    "y": int(y),
                    "dx": int(dx),
                    "dy": int(dy)
                },
                description=f"Scroll at ({int(x)}, {int(y)}) - dx={int(dx)}, dy={int(dy)}"
            )
            self.workflow_actions.append(action)
            self.last_action_time = current_time
            logging.debug(f"Recorded scroll at ({int(x)}, {int(y)})")
    
    def _on_key_press(self, key):
        """Handle keyboard key press events"""
        if not self.is_recording:
            return
        
        try:
            # Get key name
            if hasattr(key, 'char') and key.char:
                key_name = key.char
            elif hasattr(key, 'name'):
                key_name = key.name
            else:
                key_name = str(key)
            
            # Skip special keys that are just modifiers
            if key_name in ['ctrl_l', 'ctrl_r', 'alt_l', 'alt_r', 'shift_l', 'shift_r', 'cmd_l', 'cmd_r']:
                return
            
            with self.recording_lock:
                current_time = time.time()
                relative_time = current_time - self.start_time if self.start_time else 0.0
                
                # Check if this is part of a hotkey combination
                # For now, record individual key presses
                # Hotkey detection can be enhanced later
                
                action = WorkflowAction(
                    action_type="key",
                    timestamp=relative_time,
                    parameters={
                        "key": key_name,
                        "presses": 1
                    },
                    description=f"Press key: {key_name}"
                )
                self.workflow_actions.append(action)
                self.last_action_time = current_time
                logging.debug(f"Recorded key press: {key_name}")
        
        except Exception as e:
            logging.warning(f"Error recording key press: {e}")
    
    def _on_key_release(self, key):
        """Handle keyboard key release events (usually not needed for recording)"""
        # We typically only care about key presses, not releases
        pass
    
    def add_action(self, action_type: str, parameters: Dict[str, Any], description: str = ""):
        """Manually add an action to the current recording (for programmatic actions)"""
        if not self.is_recording:
            return
        
        with self.recording_lock:
            current_time = time.time()
            relative_time = current_time - self.start_time if self.start_time else 0.0
            action = WorkflowAction(
                action_type=action_type,
                timestamp=relative_time,
                parameters=parameters,
                description=description
            )
            self.workflow_actions.append(action)
            self.last_action_time = current_time
            logging.debug(f"Manually recorded action: {action_type} - {description}")


class WorkflowPlayer:
    """Plays back recorded workflows with retry logic and error handling"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, action_timeout: float = 10.0):
        self.is_playing = False
        self.current_workflow: Optional[Workflow] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.action_timeout = action_timeout
        self.failed_actions = []  # Track failed actions for reporting
    
    def play_workflow(self, workflow: Workflow, parameters: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """Play back a workflow with optional parameter substitution, retry logic, and error handling"""
        if self.is_playing:
            return False, "A workflow is already playing"
        
        if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
            return False, "Desktop automation tools not available"
        
        try:
            if not desktop_tools.is_automation_enabled():
                return False, "Desktop automation is disabled"
        except Exception as e:
            return False, f"Error checking automation status: {str(e)}"
        
        self.is_playing = True
        self.current_workflow = workflow
        self.failed_actions = []
        
        try:
            logging.info(f"Playing workflow: {workflow.name} ({len(workflow.actions)} actions)")
            
            for i, action in enumerate(workflow.actions):
                if not self.is_playing:  # Check if playback was stopped
                    return False, "Workflow playback was stopped"
                
                # Wait for the action's timestamp (relative delay)
                if i > 0:
                    delay = action.timestamp - workflow.actions[i-1].timestamp
                    if delay > 0:
                        # Cap delay at 5 seconds to avoid long waits
                        time.sleep(min(delay, 5.0))
                
                # Execute the action with retry logic
                success, message = self._execute_action_with_retry(action, parameters, action_num=i+1)
                
                if not success:
                    self.failed_actions.append({
                        "action_num": i + 1,
                        "action_type": action.action_type,
                        "description": action.description,
                        "error": message
                    })
                    logging.warning(f"Action {i+1} failed after retries: {message}")
                    # Decide whether to continue or stop
                    # For critical actions (like clicks on specific buttons), we might want to stop
                    # For now, continue but track failures
            
            # Report results
            if self.failed_actions:
                failed_count = len(self.failed_actions)
                total_count = len(workflow.actions)
                success_count = total_count - failed_count
                return True, f"Workflow '{workflow.name}' completed with {success_count}/{total_count} actions successful. {failed_count} action(s) failed."
            else:
                return True, f"Workflow '{workflow.name}' completed successfully ({len(workflow.actions)} actions)"
        
        except Exception as e:
            error_msg = f"Error playing workflow: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
        
        finally:
            self.is_playing = False
            self.current_workflow = None
            self.failed_actions = []
    
    def _execute_action_with_retry(self, action: WorkflowAction, parameters: Optional[Dict[str, str]] = None, action_num: int = 0) -> Tuple[bool, str]:
        """Execute an action with retry logic and timeout handling"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Execute with timeout (using threading for timeout)
                import threading
                result_container = {"success": False, "message": "", "done": False}
                
                def execute():
                    try:
                        result_container["success"], result_container["message"] = self._execute_action(action, parameters)
                    except Exception as e:
                        result_container["success"] = False
                        result_container["message"] = str(e)
                    finally:
                        result_container["done"] = True
                
                thread = threading.Thread(target=execute, daemon=True)
                thread.start()
                thread.join(timeout=self.action_timeout)
                
                if not result_container["done"]:
                    # Timeout occurred
                    last_error = f"Action timed out after {self.action_timeout} seconds"
                    logging.warning(f"Action {action_num} attempt {attempt + 1} timed out")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
                
                if result_container["success"]:
                    if attempt > 0:
                        logging.info(f"Action {action_num} succeeded on attempt {attempt + 1}")
                    return True, result_container["message"]
                else:
                    last_error = result_container["message"]
                    logging.warning(f"Action {action_num} attempt {attempt + 1} failed: {last_error}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
            
            except Exception as e:
                last_error = f"Exception during action execution: {str(e)}"
                logging.error(f"Action {action_num} attempt {attempt + 1} exception: {last_error}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        return False, f"Failed after {self.max_retries} attempts: {last_error}"
    
    def _execute_action(self, action: WorkflowAction, parameters: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """Execute a single workflow action with parameter substitution and error handling"""
        params = action.parameters.copy()
        
        # Substitute parameters if provided
        if parameters:
            for key, value in parameters.items():
                # Replace parameter placeholders in string values
                for param_key, param_value in params.items():
                    if isinstance(param_value, str) and f"{{{key}}}" in param_value:
                        params[param_key] = param_value.replace(f"{{{key}}}", value)
        
        try:
            if action.action_type == "click":
                if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
                    return False, "Desktop automation tools not available"
                
                x = int(params.get("x", 0))
                y = int(params.get("y", 0))
                button = params.get("button", "left")
                clicks = int(params.get("clicks", 1))
                
                # Verify coordinates are valid
                if x < 0 or y < 0:
                    return False, f"Invalid click coordinates: ({x}, {y})"
                
                return desktop_tools.click_at_position(x, y, button, clicks)
            
            elif action.action_type == "type":
                if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
                    return False, "Desktop automation tools not available"
                
                text = str(params.get("text", ""))
                interval = float(params.get("interval", 0.05))
                
                if not text:
                    return False, "No text provided to type"
                
                return desktop_tools.type_text(text, interval)
            
            elif action.action_type == "key":
                if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
                    return False, "Desktop automation tools not available"
                
                key = str(params.get("key", ""))
                presses = int(params.get("presses", 1))
                interval = float(params.get("interval", 0.1))
                
                if not key:
                    return False, "No key specified"
                
                return desktop_tools.press_key(key, presses, interval)
            
            elif action.action_type == "hotkey":
                if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
                    return False, "Desktop automation tools not available"
                
                keys = params.get("keys", [])
                if not keys or not isinstance(keys, list):
                    return False, "No keys specified for hotkey"
                
                return desktop_tools.press_hotkey(*keys)
            
            elif action.action_type == "scroll":
                # Handle scroll action using desktop_tools or pyautogui directly
                try:
                    x = int(params.get("x", 0))
                    y = int(params.get("y", 0))
                    dy = int(params.get("dy", 0))
                    
                    # Try desktop_tools first, fallback to pyautogui
                    if DESKTOP_TOOLS_AVAILABLE and desktop_tools and hasattr(desktop_tools, 'scroll_at_position'):
                        success, message = desktop_tools.scroll_at_position(x, y, dy)
                        return success, message
                    elif PYAUTOGUI_AVAILABLE and pyautogui:
                        # Fallback to pyautogui directly
                        if x > 0 and y > 0:
                            pyautogui.moveTo(x, y, duration=0.1)
                        pyautogui.scroll(dy, x=x, y=y)
                        return True, f"Scrolled {dy} units at ({x}, {y})"
                    else:
                        return False, "Neither desktop_tools nor pyautogui available for scrolling"
                except Exception as e:
                    return False, f"Error scrolling: {str(e)}"
            
            elif action.action_type == "wait":
                duration = float(params.get("duration", 1.0))
                if duration < 0:
                    duration = 0
                if duration > 10:  # Cap wait time at 10 seconds
                    duration = 10
                time.sleep(duration)
                return True, f"Waited {duration} seconds"
            
            elif action.action_type == "verify":
                if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
                    return False, "Desktop automation tools not available for verification"
                
                # Verify screen state (e.g., check if text/image is present)
                verify_type = params.get("type", "text")  # "text" or "image"
                if verify_type == "text":
                    text = params.get("text", "")
                    region = params.get("region")
                    if not text:
                        return False, "No text specified for verification"
                    
                    success, message, extracted_text = desktop_tools.read_text_from_screen(region)
                    if not success:
                        return False, f"OCR failed: {message}"
                    
                    if text.lower() in extracted_text.lower():
                        return True, f"Verified text '{text}' is present"
                    else:
                        return False, f"Text '{text}' not found on screen. Found: {extracted_text[:100]}"
                
                elif verify_type == "image":
                    image_path = params.get("image_path", "")
                    confidence = float(params.get("confidence", 0.8))
                    
                    if not image_path:
                        return False, "No image path specified for verification"
                    
                    success, message, position = desktop_tools.find_image_on_screen(image_path, confidence)
                    return success, message
                
                else:
                    return False, f"Unknown verify type: {verify_type}"
            
            elif action.action_type == "screenshot":
                if not DESKTOP_TOOLS_AVAILABLE or not desktop_tools:
                    return False, "Desktop automation tools not available for screenshots"
                
                file_path = params.get("file_path")
                success, message, saved_path = desktop_tools.take_screenshot(file_path)
                return success, message
            
            else:
                return False, f"Unknown action type: {action.action_type}"
        
        except KeyError as e:
            return False, f"Missing required parameter: {str(e)}"
        except ValueError as e:
            return False, f"Invalid parameter value: {str(e)}"
        except Exception as e:
            error_msg = f"Error executing {action.action_type} action: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
    def stop_playback(self):
        """Stop current workflow playback"""
        if self.is_playing:
            logging.info("Stopping workflow playback")
            self.is_playing = False


class WorkflowManager:
    """Manages workflow storage and retrieval"""
    
    def __init__(self, workflows_dir: Optional[Path] = None):
        if workflows_dir is None:
            # Default to workflows directory in project folder
            workflows_dir = Path("F:/Dre_Programs/LeaAssistant/workflows")
        
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(exist_ok=True)
        self.recorder = WorkflowRecorder()
        self.player = WorkflowPlayer()
    
    def save_workflow(self, workflow: Workflow) -> Tuple[bool, str]:
        """Save a workflow to disk"""
        try:
            # Convert workflow to dict
            workflow_dict = {
                "name": workflow.name,
                "description": workflow.description,
                "created": workflow.created,
                "modified": datetime.now().isoformat(),
                "category": workflow.category,
                "parameters": workflow.parameters,
                "actions": [asdict(action) for action in workflow.actions]
            }
            
            # Save to JSON file
            filename = f"{workflow.name.replace(' ', '_').lower()}.json"
            filepath = self.workflows_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workflow_dict, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved workflow: {workflow.name} to {filepath}")
            return True, f"Workflow '{workflow.name}' saved successfully"
        
        except Exception as e:
            error_msg = f"Error saving workflow: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
    def load_workflow(self, workflow_name: str) -> Tuple[bool, str, Optional[Workflow]]:
        """Load a workflow from disk"""
        try:
            filename = f"{workflow_name.replace(' ', '_').lower()}.json"
            filepath = self.workflows_dir / filename
            
            if not filepath.exists():
                return False, f"Workflow '{workflow_name}' not found", None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                workflow_dict = json.load(f)
            
            # Reconstruct workflow
            actions = [
                WorkflowAction(**action_dict)
                for action_dict in workflow_dict.get("actions", [])
            ]
            
            workflow = Workflow(
                name=workflow_dict["name"],
                description=workflow_dict.get("description", ""),
                created=workflow_dict.get("created", datetime.now().isoformat()),
                modified=workflow_dict.get("modified", datetime.now().isoformat()),
                actions=actions,
                parameters=workflow_dict.get("parameters", {}),
                category=workflow_dict.get("category", "general")
            )
            
            return True, f"Workflow '{workflow_name}' loaded", workflow
        
        except Exception as e:
            error_msg = f"Error loading workflow: {str(e)}"
            logging.error(error_msg)
            return False, error_msg, None
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows"""
        workflows = []
        
        for filepath in self.workflows_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    workflow_dict = json.load(f)
                
                workflows.append({
                    "name": workflow_dict["name"],
                    "description": workflow_dict.get("description", ""),
                    "created": workflow_dict.get("created", ""),
                    "modified": workflow_dict.get("modified", ""),
                    "category": workflow_dict.get("category", "general"),
                    "action_count": len(workflow_dict.get("actions", [])),
                    "parameters": workflow_dict.get("parameters", {})
                })
            except Exception as e:
                logging.warning(f"Error reading workflow file {filepath}: {e}")
        
        return workflows
    
    def delete_workflow(self, workflow_name: str) -> Tuple[bool, str]:
        """Delete a workflow"""
        try:
            filename = f"{workflow_name.replace(' ', '_').lower()}.json"
            filepath = self.workflows_dir / filename
            
            if not filepath.exists():
                return False, f"Workflow '{workflow_name}' not found"
            
            filepath.unlink()
            logging.info(f"Deleted workflow: {workflow_name}")
            return True, f"Workflow '{workflow_name}' deleted successfully"
        
        except Exception as e:
            error_msg = f"Error deleting workflow: {str(e)}"
            logging.error(error_msg)
            return False, error_msg


# Global workflow manager instance
_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager(workflows_dir: Optional[Path] = None) -> WorkflowManager:
    """Get or create the global workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None:
        try:
            _workflow_manager = WorkflowManager(workflows_dir)
            logging.info("Workflow manager initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing workflow manager: {e}")
            raise
    return _workflow_manager


def check_workflow_system_availability() -> Tuple[bool, str]:
    """Check if workflow system is available and report status"""
    issues = []
    
    if not DESKTOP_TOOLS_AVAILABLE:
        issues.append("desktop_tools module not available")
    
    if not PYNPUT_AVAILABLE:
        issues.append("pynput not available (workflow recording will not work)")
    
    if not PYAUTOGUI_AVAILABLE:
        issues.append("pyautogui not available (some automation features limited)")
    
    try:
        manager = get_workflow_manager()
        if issues:
            return True, f"Workflow system available with limitations: {', '.join(issues)}"
        else:
            return True, "Workflow system fully available"
    except Exception as e:
        return False, f"Workflow system not available: {str(e)}"

