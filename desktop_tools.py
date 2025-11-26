"""
Desktop Automation Tools for Lea Assistant

This module provides safe, structured automation functions that Lea can call
to perform repetitive tasks on the user's computer.

Each function:
- Has a clear name and docstring
- Uses safe automation libraries (pyautogui, keyboard, etc.)
- Returns success/failure and optional results
- Logs all actions for transparency
- Never bypasses system security

Master switch: Set ENABLE_DESKTOP_AUTOMATION = False to disable all automation
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Master switch for desktop automation
ENABLE_DESKTOP_AUTOMATION = True

# Try to import automation libraries (optional dependencies)
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    # Set failsafe - move mouse to corner to abort
    pyautogui.FAILSAFE = True
    # Set pause between actions for safety
    pyautogui.PAUSE = 0.5
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("pyautogui not available - install with: pip install pyautogui")

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logging.warning("keyboard not available - install with: pip install keyboard")

PLAYWRIGHT_AVAILABLE = False
sync_playwright = None
Browser = None
Page = None
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("playwright not available - install with: pip install playwright")
except Exception as e:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning(f"Error importing playwright: {e}")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR not available - install with: pip install pytesseract pillow")


class DesktopAutomationError(Exception):
    """Custom exception for desktop automation errors"""
    pass


def _check_automation_enabled() -> bool:
    """Check if desktop automation is enabled"""
    if not ENABLE_DESKTOP_AUTOMATION:
        logging.info("Desktop automation is disabled")
        return False
    return True


def _log_action(action: str, details: str = ""):
    """Log an automation action for transparency"""
    if details:
        logging.info(f"Desktop Automation: {action} - {details}")
    else:
        logging.info(f"Desktop Automation: {action}")


# =====================================================
# BASIC AUTOMATION FUNCTIONS
# =====================================================

def click_at_position(x: int, y: int, button: str = "left", clicks: int = 1) -> Tuple[bool, str]:
    """
    Click at a specific screen position.
    
    Args:
        x: X coordinate
        y: Y coordinate
        button: "left", "right", or "middle"
        clicks: Number of clicks (default: 1)
    
    Returns:
        (success: bool, message: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled"
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui"
    
    try:
        _log_action(f"Clicking at ({x}, {y})", f"button={button}, clicks={clicks}")
        pyautogui.click(x, y, button=button, clicks=clicks)
        return True, f"Successfully clicked at ({x}, {y})"
    except Exception as e:
        error_msg = f"Error clicking at ({x}, {y}): {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def type_text(text: str, interval: float = 0.05) -> Tuple[bool, str]:
    """
    Type text using keyboard simulation.
    
    Args:
        text: Text to type
        interval: Delay between keystrokes in seconds
    
    Returns:
        (success: bool, message: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled"
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui"
    
    try:
        _log_action("Typing text", f"length={len(text)} characters")
        pyautogui.write(text, interval=interval)
        return True, f"Successfully typed {len(text)} characters"
    except Exception as e:
        error_msg = f"Error typing text: {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def press_key(key: str, presses: int = 1, interval: float = 0.1) -> Tuple[bool, str]:
    """
    Press a keyboard key.
    
    Args:
        key: Key to press (e.g., "enter", "tab", "ctrl", "alt", "f1")
        presses: Number of times to press (default: 1)
        interval: Delay between presses in seconds
    
    Returns:
        (success: bool, message: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled"
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui"
    
    try:
        _log_action(f"Pressing key", f"key={key}, presses={presses}")
        pyautogui.press(key, presses=presses, interval=interval)
        return True, f"Successfully pressed {key} {presses} time(s)"
    except Exception as e:
        error_msg = f"Error pressing key {key}: {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def press_hotkey(*keys: str) -> Tuple[bool, str]:
    """
    Press a keyboard hotkey combination (e.g., Ctrl+C, Alt+Tab).
    
    Args:
        *keys: Keys to press simultaneously (e.g., "ctrl", "c")
    
    Returns:
        (success: bool, message: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled"
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui"
    
    try:
        key_combo = "+".join(keys)
        _log_action("Pressing hotkey", f"keys={key_combo}")
        pyautogui.hotkey(*keys)
        return True, f"Successfully pressed hotkey: {key_combo}"
    except Exception as e:
        error_msg = f"Error pressing hotkey {key_combo}: {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def take_screenshot(file_path: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Take a screenshot of the entire screen.
    
    Args:
        file_path: Optional path to save screenshot (default: auto-generate)
    
    Returns:
        (success: bool, message: str, file_path: Optional[str])
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled", None
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui", None
    
    try:
        if not file_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"screenshot_{timestamp}.png"
        
        _log_action("Taking screenshot", f"file={file_path}")
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)
        return True, f"Screenshot saved to {file_path}", file_path
    except Exception as e:
        error_msg = f"Error taking screenshot: {str(e)}"
        logging.error(error_msg)
        return False, error_msg, None


def scroll_at_position(x: int, y: int, scroll_amount: int = 3) -> Tuple[bool, str]:
    """
    Scroll at a specific screen position.
    
    Args:
        x: X coordinate to scroll at
        y: Y coordinate to scroll at
        scroll_amount: Number of scroll units (positive = down, negative = up)
    
    Returns:
        (success: bool, message: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled"
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui"
    
    try:
        _log_action(f"Scrolling at ({x}, {y})", f"amount={scroll_amount}")
        # Move mouse to position first
        pyautogui.moveTo(x, y, duration=0.1)
        # Perform scroll
        pyautogui.scroll(scroll_amount, x=x, y=y)
        return True, f"Successfully scrolled {scroll_amount} units at ({x}, {y})"
    except Exception as e:
        error_msg = f"Error scrolling: {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def find_image_on_screen(image_path: str, confidence: float = 0.8) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    """
    Find an image on the screen using template matching.
    
    Args:
        image_path: Path to the image file to search for
        confidence: Confidence threshold (0.0 to 1.0, default: 0.8)
    
    Returns:
        (success: bool, message: str, position: Optional[Tuple[int, int]])
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled", None
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui", None
    
    try:
        _log_action("Searching for image", f"image={image_path}, confidence={confidence}")
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            center = pyautogui.center(location)
            return True, f"Found image at {center}", center
        else:
            return False, f"Image not found on screen (confidence threshold: {confidence})", None
    except Exception as e:
        error_msg = f"Error finding image: {str(e)}"
        logging.error(error_msg)
        return False, error_msg, None


# =====================================================
# WORKFLOW FUNCTIONS (Higher-level automation)
# =====================================================

def open_application(application_name: str, wait_time: float = 2.0) -> Tuple[bool, str]:
    """
    Open an application by name (Windows).
    
    Args:
        application_name: Name of the application to open
        wait_time: Time to wait after opening (seconds)
    
    Returns:
        (success: bool, message: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled"
    
    try:
        import subprocess
        _log_action("Opening application", f"app={application_name}")
        subprocess.Popen([application_name], shell=True)
        time.sleep(wait_time)
        return True, f"Opened {application_name}"
    except Exception as e:
        error_msg = f"Error opening application {application_name}: {str(e)}"
        logging.error(error_msg)
        return False, error_msg


def wait_for_image(image_path: str, timeout: float = 10.0, confidence: float = 0.8) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    """
    Wait for an image to appear on screen.
    
    Args:
        image_path: Path to the image file to wait for
        timeout: Maximum time to wait in seconds
        confidence: Confidence threshold for matching
    
    Returns:
        (success: bool, message: str, position: Optional[Tuple[int, int]])
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled", None
    
    start_time = time.time()
    _log_action("Waiting for image", f"image={image_path}, timeout={timeout}")
    
    while time.time() - start_time < timeout:
        success, message, position = find_image_on_screen(image_path, confidence)
        if success:
            return True, f"Image found after {time.time() - start_time:.1f} seconds", position
        time.sleep(0.5)  # Check every 0.5 seconds
    
    return False, f"Image not found within {timeout} seconds", None


# =====================================================
# BROWSER AUTOMATION (Playwright)
# =====================================================

def open_browser_and_navigate(url: str, browser_type: str = "chromium", headless: bool = False) -> Tuple[bool, str, Optional[Any], Optional[Any]]:
    """
    Open a browser and navigate to a URL using Playwright.
    
    Args:
        url: URL to navigate to
        browser_type: "chromium", "firefox", or "webkit"
        headless: Run browser in headless mode
    
    Returns:
        (success: bool, message: str, browser: Optional[Browser], page: Optional[Page])
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled", None, None
    
    if not PLAYWRIGHT_AVAILABLE:
        return False, "playwright not available - install with: pip install playwright", None, None
    
    try:
        _log_action("Opening browser", f"url={url}, browser={browser_type}")
        playwright = sync_playwright().start()
        
        if browser_type == "chromium":
            browser = playwright.chromium.launch(headless=headless)
        elif browser_type == "firefox":
            browser = playwright.firefox.launch(headless=headless)
        elif browser_type == "webkit":
            browser = playwright.webkit.launch(headless=headless)
        else:
            return False, f"Unknown browser type: {browser_type}", None, None
        
        page = browser.new_page()
        page.goto(url)
        return True, f"Opened browser and navigated to {url}", browser, page
    except Exception as e:
        error_msg = f"Error opening browser: {str(e)}"
        logging.error(error_msg)
        return False, error_msg, None, None


# =====================================================
# OCR FUNCTIONS (Text reading from screen)
# =====================================================

def read_text_from_screen(region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, str, str]:
    """
    Read text from screen using OCR.
    
    Args:
        region: Optional (x, y, width, height) region to read from (None = entire screen)
    
    Returns:
        (success: bool, message: str, extracted_text: str)
    """
    if not _check_automation_enabled():
        return False, "Desktop automation is disabled", ""
    
    if not OCR_AVAILABLE:
        return False, "OCR not available - install with: pip install pytesseract pillow", ""
    
    if not PYAUTOGUI_AVAILABLE:
        return False, "pyautogui not available - install with: pip install pyautogui", ""
    
    try:
        _log_action("Reading text from screen", f"region={region}")
        screenshot = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
        text = pytesseract.image_to_string(screenshot)
        return True, "Successfully read text from screen", text.strip()
    except Exception as e:
        error_msg = f"Error reading text from screen: {str(e)}"
        logging.error(error_msg)
        return False, error_msg, ""


# =====================================================
# TOOL REGISTRY (For Lea to discover available tools)
# =====================================================

DESKTOP_TOOLS = {
    "click_at_position": click_at_position,
    "type_text": type_text,
    "press_key": press_key,
    "press_hotkey": press_hotkey,
    "take_screenshot": take_screenshot,
    "scroll_at_position": scroll_at_position,
    "find_image_on_screen": find_image_on_screen,
    "open_application": open_application,
    "wait_for_image": wait_for_image,
    "open_browser_and_navigate": open_browser_and_navigate,
    "read_text_from_screen": read_text_from_screen,
}


def list_available_tools() -> Dict[str, Dict[str, Any]]:
    """
    List all available desktop automation tools with their metadata.
    
    Returns:
        Dictionary mapping tool names to their metadata (description, parameters, etc.)
    """
    tools_info = {}
    for tool_name, tool_func in DESKTOP_TOOLS.items():
        tools_info[tool_name] = {
            "name": tool_name,
            "description": tool_func.__doc__ or f"Desktop automation tool: {tool_name}",
            "function": tool_func,
        }
    return tools_info


def get_tool(tool_name: str):
    """Get a desktop automation tool by name"""
    return DESKTOP_TOOLS.get(tool_name)


def is_automation_enabled() -> bool:
    """Check if desktop automation is enabled"""
    return ENABLE_DESKTOP_AUTOMATION

