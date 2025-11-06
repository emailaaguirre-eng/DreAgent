"""Safety policies: path validation, command whitelist, logging."""
import json
from pathlib import Path
from datetime import datetime

BLOCKED_PATTERNS = {".ssh", ".aws", ".gnupg", "AppData", ".git/config", ".env"}
ALLOWED_COMMANDS = {"pytest -q", "ruff check .", "black --check .", "git status", "git diff"}


def load_config():
    """Load config.json from src/."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def validate_path(target: Path, workspace_root: Path) -> tuple[bool, str]:
    """Check path is in workspace and not blocked."""
    try:
        resolved = target.resolve()
        
        # Must be in workspace
        if not str(resolved).startswith(str(workspace_root)):
            return False, "Path outside workspace"
        
        # Check blocked patterns
        for blocked in BLOCKED_PATTERNS:
            if blocked in str(resolved):
                return False, f"Blocked pattern: {blocked}"
        
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_command(cmd: str) -> tuple[bool, str]:
    """Check command is whitelisted."""
    normalized = " ".join(cmd.split())
    for allowed in ALLOWED_COMMANDS:
        if normalized.startswith(allowed):
            return True, ""
    return False, f"Command not whitelisted: {cmd}"


def log_action(operation: str, metadata: dict):
    """Log to logs/*.jsonl (metadata only, no secrets/contents)."""
    log_dir = Path(__file__).parent.parent / "logs"
    log_file = log_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    entry = {"timestamp": datetime.now().isoformat(), "operation": operation, **metadata}
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
