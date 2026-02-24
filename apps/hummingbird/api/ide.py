"""
=============================================================================
HUMMINGBIRD-LEA - IDE API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
API endpoints for the Chiquis Agentic IDE.

Features:
- File management (read, write, list)
- AI code completion
- Inline code edits (Cmd+K)
- Code chat with context
- Project indexing for RAG
=============================================================================
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from core.utils.config import get_settings
from core.utils.auth import get_current_user
from core.providers.ollama import get_ollama_client, Message as OllamaMessage, ModelType, Message as OllamaMessage, ModelType
from core.services.rag import get_rag_engine, get_rag_context

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.scss',
    '.json', '.yaml', '.yml', '.md', '.txt', '.sql', '.sh', '.bash',
    '.env', '.gitignore', '.dockerfile', '.toml', '.ini', '.cfg',
    '.xml', '.csv', '.go', '.rs', '.java', '.c', '.cpp', '.h',
}

# Project root for IDE (configurable)
IDE_PROJECT_ROOT = Path(settings.knowledge_dir).parent / "projects"


# =============================================================================
# Request/Response Models
# =============================================================================

class FileContent(BaseModel):
    """File content response"""
    path: str
    content: str
    language: str
    size: int
    modified: str


class FileNode(BaseModel):
    """File tree node"""
    name: str
    path: str
    type: str  # "file" or "directory"
    children: Optional[List["FileNode"]] = None
    language: Optional[str] = None
    size: Optional[int] = None


class WriteFileRequest(BaseModel):
    """Write file request"""
    path: str
    content: str


class CompletionRequest(BaseModel):
    """Code completion request"""
    code: str = Field(..., description="Current code in editor")
    cursor_position: int = Field(..., description="Cursor position in code")
    file_path: Optional[str] = Field(None, description="Current file path")
    language: str = Field("python", description="Programming language")
    max_tokens: int = Field(150, description="Max tokens for completion")


class CompletionResponse(BaseModel):
    """Code completion response"""
    completion: str
    confidence: float
    insert_position: int


class InlineEditRequest(BaseModel):
    """Inline edit request (Cmd+K)"""
    code: str = Field(..., description="Selected code or current line")
    instruction: str = Field(..., description="User instruction for edit")
    context_before: str = Field("", description="Code before selection")
    context_after: str = Field("", description="Code after selection")
    file_path: Optional[str] = Field(None, description="Current file path")
    language: str = Field("python", description="Programming language")


class InlineEditResponse(BaseModel):
    """Inline edit response"""
    edited_code: str
    explanation: str
    confidence: float


class CodeChatRequest(BaseModel):
    """Code chat request"""
    message: str
    code_context: Optional[str] = Field(None, description="Selected code for context")
    file_path: Optional[str] = None
    language: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)
    use_rag: bool = Field(True, description="Use RAG for codebase awareness")


class CodeChatResponse(BaseModel):
    """Code chat response"""
    response: str
    code_suggestions: Optional[List[str]] = None
    referenced_files: Optional[List[str]] = None
    confidence: float


class IndexProjectRequest(BaseModel):
    """Index project for RAG"""
    project_path: str
    extensions: List[str] = Field(
        default_factory=lambda: [".py", ".js", ".ts", ".md"]
    )


# =============================================================================
# Helper Functions
# =============================================================================

def get_language_from_extension(file_path: str) -> str:
    """Get language identifier from file extension"""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescriptreact',
        '.jsx': 'javascriptreact',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
    }
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext, 'plaintext')


def is_safe_path(base_path: Path, requested_path: str) -> bool:
    """Check if path is safe (no directory traversal)"""
    try:
        full_path = (base_path / requested_path).resolve()
        return str(full_path).startswith(str(base_path.resolve()))
    except Exception:
        return False


def is_allowed_file(file_path: str) -> bool:
    """Check if file extension is allowed"""
    ext = Path(file_path).suffix.lower()
    return ext in ALLOWED_EXTENSIONS or ext == ''


def build_file_tree(directory: Path, base_path: Path) -> List[FileNode]:
    """Build file tree structure"""
    nodes = []

    try:
        items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

        for item in items:
            # Skip hidden files and common ignored directories
            if item.name.startswith('.') or item.name in {'__pycache__', 'node_modules', 'venv', '.git'}:
                continue

            relative_path = str(item.relative_to(base_path))

            if item.is_dir():
                children = build_file_tree(item, base_path)
                if children:  # Only include non-empty directories
                    nodes.append(FileNode(
                        name=item.name,
                        path=relative_path,
                        type="directory",
                        children=children,
                    ))
            elif item.is_file() and is_allowed_file(item.name):
                nodes.append(FileNode(
                    name=item.name,
                    path=relative_path,
                    type="file",
                    language=get_language_from_extension(item.name),
                    size=item.stat().st_size,
                ))
    except PermissionError:
        pass

    return nodes


# =============================================================================
# File Management Endpoints
# =============================================================================

@router.get("/files/tree")
async def get_file_tree(
    project: str = Query("default", description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get file tree for a project.
    """
    project_path = IDE_PROJECT_ROOT / project

    if not project_path.exists():
        # Create default project directory
        project_path.mkdir(parents=True, exist_ok=True)
        # Create a sample file
        sample_file = project_path / "main.py"
        sample_file.write_text('# Welcome to Chiquis IDE!\n# Start coding here...\n\nprint("Hello, World!")\n')

    tree = build_file_tree(project_path, project_path)

    return {
        "project": project,
        "root": str(project_path),
        "tree": [node.model_dump() for node in tree],
    }


@router.get("/files/read")
async def read_file(
    path: str = Query(..., description="File path relative to project"),
    project: str = Query("default", description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> FileContent:
    """
    Read a file's content.
    """
    project_path = IDE_PROJECT_ROOT / project

    if not is_safe_path(project_path, path):
        raise HTTPException(status_code=400, detail="Invalid file path")

    file_path = project_path / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    if not is_allowed_file(path):
        raise HTTPException(status_code=400, detail="File type not allowed")

    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Cannot read binary file")

    stat = file_path.stat()

    return FileContent(
        path=path,
        content=content,
        language=get_language_from_extension(path),
        size=stat.st_size,
        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
    )


@router.post("/files/write")
async def write_file(
    request: WriteFileRequest,
    project: str = Query("default", description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Write content to a file.
    """
    project_path = IDE_PROJECT_ROOT / project

    if not is_safe_path(project_path, request.path):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not is_allowed_file(request.path):
        raise HTTPException(status_code=400, detail="File type not allowed")

    file_path = project_path / request.path

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(request.content, encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")

    return {
        "success": True,
        "path": request.path,
        "size": len(request.content),
        "modified": datetime.now().isoformat(),
    }


@router.delete("/files/delete")
async def delete_file(
    path: str = Query(..., description="File path"),
    project: str = Query("default", description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Delete a file.
    """
    project_path = IDE_PROJECT_ROOT / project

    if not is_safe_path(project_path, path):
        raise HTTPException(status_code=400, detail="Invalid file path")

    file_path = project_path / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if file_path.is_dir():
        raise HTTPException(status_code=400, detail="Cannot delete directories via this endpoint")

    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    return {"success": True, "path": path}


@router.post("/files/create-folder")
async def create_folder(
    path: str = Query(..., description="Folder path"),
    project: str = Query("default", description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Create a new folder.
    """
    project_path = IDE_PROJECT_ROOT / project

    if not is_safe_path(project_path, path):
        raise HTTPException(status_code=400, detail="Invalid folder path")

    folder_path = project_path / path

    try:
        folder_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {str(e)}")

    return {"success": True, "path": path}


# =============================================================================
# AI Code Completion
# =============================================================================

@router.post("/completion", response_model=CompletionResponse)
async def get_code_completion(
    request: CompletionRequest,
    current_user: dict = Depends(get_current_user),
) -> CompletionResponse:
    """
    Get AI-powered code completion using Chiquis (deepseek-coder).
    """
    ollama = get_ollama_client()

    # Extract context around cursor
    code_before = request.code[:request.cursor_position]
    code_after = request.code[request.cursor_position:]

    # Get last few lines for context
    lines_before = code_before.split('\n')[-10:]
    context = '\n'.join(lines_before)

    # Build prompt for completion
    prompt = f"""Complete the following {request.language} code. Only provide the completion, no explanation.
The code continues from where it ends:

```{request.language}
{context}
```

Continue the code naturally:"""

    try:
        response = await ollama.generate(
            prompt=prompt,
            model=settings.ollama_model_code,
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": request.max_tokens,
                "stop": ["\n\n\n", "```"],
            }
        )

        completion = response.get("response", "").strip()

        # Clean up completion
        if completion.startswith("```"):
            completion = completion.split("```")[1] if "```" in completion else completion

        # Remove any markdown artifacts
        completion = re.sub(r'^[a-z]+\n', '', completion)  # Remove language identifier

        return CompletionResponse(
            completion=completion,
            confidence=0.85,
            insert_position=request.cursor_position,
        )

    except Exception as e:
        logger.error(f"Completion error: {e}")
        return CompletionResponse(
            completion="",
            confidence=0.0,
            insert_position=request.cursor_position,
        )


# =============================================================================
# Inline Edit (Cmd+K)
# =============================================================================

@router.post("/inline-edit", response_model=InlineEditResponse)
async def inline_edit(
    request: InlineEditRequest,
    current_user: dict = Depends(get_current_user),
) -> InlineEditResponse:
    """
    Perform inline code edit based on natural language instruction.
    Similar to Cursor's Cmd+K feature.
    """
    ollama = get_ollama_client()

    # Build context
    full_context = ""
    if request.context_before:
        full_context += f"// Code before:\n{request.context_before[-500:]}\n\n"
    full_context += f"// Code to edit:\n{request.code}\n"
    if request.context_after:
        full_context += f"\n// Code after:\n{request.context_after[:500]}"

    prompt = f"""You are a code editor assistant. Edit the following {request.language} code according to the instruction.
Only output the edited code, no explanation or markdown.

{full_context}

Instruction: {request.instruction}

Edited code:"""

    try:
        response = await ollama.generate(
            prompt=prompt,
            model=settings.ollama_model_code,
            options={
                "temperature": 0.3,
                "top_p": 0.95,
                "num_predict": 500,
            }
        )

        edited_code = response.get("response", "").strip()

        # Clean up response
        if edited_code.startswith("```"):
            lines = edited_code.split("\n")
            edited_code = "\n".join(lines[1:])  # Remove first line
            if "```" in edited_code:
                edited_code = edited_code.split("```")[0]

        # Generate brief explanation
        explanation_prompt = f"In one short sentence, explain what was changed in this edit: {request.instruction}"
        explanation_response = await ollama.generate(
            prompt=explanation_prompt,
            model=settings.ollama_model_code,
            options={"num_predict": 50},
        )
        explanation = explanation_response.get("response", "Code edited as requested.").strip()

        return InlineEditResponse(
            edited_code=edited_code.strip(),
            explanation=explanation,
            confidence=0.85,
        )

    except Exception as e:
        logger.error(f"Inline edit error: {e}")
        return InlineEditResponse(
            edited_code=request.code,
            explanation=f"Error: {str(e)}",
            confidence=0.0,
        )


# =============================================================================
# Code Chat with RAG
# =============================================================================

@router.post("/chat", response_model=CodeChatResponse)
async def code_chat(
    request: CodeChatRequest,
    project: str = Query("default", description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> CodeChatResponse:
    """
    Chat with Chiquis about code with codebase awareness via RAG.
    """
    ollama = get_ollama_client()

    # Build context from RAG if enabled
    rag_context = ""
    referenced_files = []

    if request.use_rag:
        try:
            rag_result = await get_rag_context(request.message, top_k=3)
            if rag_result.get("context"):
                rag_context = f"\n\nRelevant codebase context:\n{rag_result['context']}"
                referenced_files = [r.get("source", "") for r in rag_result.get("results", [])]
        except Exception as e:
            logger.warning(f"RAG context failed: {e}")

    # Build system message
    system_message = """You are Chiquis, a friendly and supportive coding assistant. You help with:
- Writing clean, efficient code
- Debugging and fixing issues
- Explaining code concepts
- Suggesting improvements
- Answering programming questions

Be concise but thorough. Use code examples when helpful.
If you include code, use markdown code blocks with the language specified."""

    # Build conversation
    messages = [{"role": "system", "content": system_message}]

    # Add history
    for msg in request.history[-10:]:
        messages.append(msg)

    # Build user message with context
    user_message = request.message

    if request.code_context:
        user_message = f"Regarding this code:\n```{request.language or 'code'}\n{request.code_context}\n```\n\n{request.message}"

    if rag_context:
        user_message += rag_context

    messages.append({"role": "user", "content": user_message})

    try:
        ollama_messages = [
            OllamaMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in messages
        ]
        response = await ollama.chat(
            messages=ollama_messages,
            model_type=ModelType.CODE,
            model=settings.ollama_model_code,
            temperature=0.7,
        )

        response_text = getattr(response, "content", "") or 

        # Extract code suggestions if any
        code_suggestions = []
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', response_text, re.DOTALL)
        if code_blocks:
            code_suggestions = [block.strip() for block in code_blocks[:3]]

        return CodeChatResponse(
            response=response_text,
            code_suggestions=code_suggestions if code_suggestions else None,
            referenced_files=referenced_files if referenced_files else None,
            confidence=0.85,
        )

    except Exception as e:
        logger.error(f"Code chat error: {e}")
        return CodeChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            confidence=0.0,
        )


# =============================================================================
# Project Indexing
# =============================================================================

@router.post("/index-project")
async def index_project(
    request: IndexProjectRequest,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Index a project directory for RAG-based codebase awareness.
    """
    project_path = IDE_PROJECT_ROOT / request.project_path

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    rag_engine = get_rag_engine()
    indexed_files = 0
    errors = []

    for ext in request.extensions:
        for file_path in project_path.rglob(f"*{ext}"):
            # Skip common ignored directories
            if any(part in file_path.parts for part in ['node_modules', '__pycache__', '.git', 'venv']):
                continue

            try:
                result = await rag_engine.index_document(str(file_path))
                if result.success:
                    indexed_files += 1
            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")

    return {
        "success": True,
        "indexed_files": indexed_files,
        "project": request.project_path,
        "errors": errors if errors else None,
    }


# =============================================================================
# Project Management
# =============================================================================

@router.get("/projects")
async def list_projects(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List available projects.
    """
    IDE_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    projects = []
    for item in IDE_PROJECT_ROOT.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            file_count = sum(1 for _ in item.rglob('*') if _.is_file())
            projects.append({
                "name": item.name,
                "path": str(item.relative_to(IDE_PROJECT_ROOT)),
                "file_count": file_count,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
            })

    return {"projects": projects}


@router.post("/projects/create")
async def create_project(
    name: str = Query(..., description="Project name"),
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Create a new project.
    """
    # Sanitize name
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid project name")

    project_path = IDE_PROJECT_ROOT / safe_name

    if project_path.exists():
        raise HTTPException(status_code=400, detail="Project already exists")

    project_path.mkdir(parents=True)

    # Create initial file
    readme = project_path / "README.md"
    readme.write_text(f"# {name}\n\nCreated with Chiquis IDE\n")

    main_file = project_path / "main.py"
    main_file.write_text(f'"""\n{name}\nCreated with Chiquis IDE\n"""\n\nprint("Hello from {name}!")\n')

    return {
        "success": True,
        "name": safe_name,
        "path": str(project_path.relative_to(IDE_PROJECT_ROOT)),
    }
