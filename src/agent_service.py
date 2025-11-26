"""FastAPI service with /propose_edit, /apply_edit, /run_command endpoints."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import difflib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from policies import validate_path, validate_command, log_action, load_config
from model_client import call_model

app = FastAPI(title="AI Agent Service")
config = load_config()
WORKSPACE_ROOT = Path(config["WORKSPACE_ROOT"]).resolve()


class ProposeEditRequest(BaseModel):
    path: str
    instruction: str
    content: Optional[str] = None


class ApplyEditRequest(BaseModel):
    path: str
    new_text: str
    approved: bool = False
    dry_run: bool = True


class RunCommandRequest(BaseModel):
    cmd: str


@app.get("/")
def health():
    return {"status": "running", "workspace": str(WORKSPACE_ROOT)}


@app.post("/propose_edit")
def propose_edit(req: ProposeEditRequest):
    """Generate unified diff via model; MVP: TODO→DONE replacement."""
    file_path = WORKSPACE_ROOT / req.path
    
    # Validate path
    is_safe, reason = validate_path(file_path, WORKSPACE_ROOT)
    if not is_safe:
        raise HTTPException(403, detail=reason)
    
    # Get current content
    if file_path.exists():
        old_content = file_path.read_text()
    else:
        old_content = ""
    
    # Call model (or stub)
    new_content = call_model(req.instruction, old_content)
    
    # Generate diff
    diff = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{req.path}",
        tofile=f"b/{req.path}"
    ))
    
    log_action("propose_edit", {"path": req.path, "safe": True})
    
    return {
        "diff": "".join(diff),
        "new_text": new_content,
        "path": req.path
    }


@app.post("/apply_edit")
def apply_edit(req: ApplyEditRequest):
    """Apply edit with approval + policy checks."""
    file_path = WORKSPACE_ROOT / req.path
    
    # Validate path
    is_safe, reason = validate_path(file_path, WORKSPACE_ROOT)
    if not is_safe:
        raise HTTPException(403, detail=reason)
    
    # Check size
    if len(req.new_text.encode()) > 50 * 1024:
        raise HTTPException(403, detail="Exceeds 50KB limit")
    
    # Dry run mode
    if req.dry_run:
        log_action("apply_edit", {"path": req.path, "dry_run": True})
        return {"message": "Dry run: validation passed", "applied": False}
    
    # Require approval
    if not req.approved:
        raise HTTPException(403, detail="Approval required")
    
    # Write file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(req.new_text)
    
    log_action("apply_edit", {"path": req.path, "applied": True})
    return {"message": "Applied successfully", "applied": True}


@app.post("/run_command")
def run_command(req: RunCommandRequest):
    """Execute whitelisted command."""
    is_allowed, reason = validate_command(req.cmd)
    if not is_allowed:
        raise HTTPException(403, detail=reason)
    
    import subprocess
    result = subprocess.run(
        req.cmd.split(),
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        shell=False
    )
    
    log_action("run_command", {"cmd": req.cmd, "returncode": result.returncode})
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }
