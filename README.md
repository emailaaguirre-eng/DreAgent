# 🤖 AI Agent Template

**Reusable scaffold for local AI coding agents with strict safety policies.**

## What This Is

A production-ready template for building AI coding agents that:
- Generate code diffs for human review (`/propose_edit`)
- Apply changes only after explicit approval (`/apply_edit`)
- Run whitelisted commands safely (`/run_command`)
- Enforce workspace sandboxing and security policies

## Quick Start (Local)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update workspace root in src/config.json
# Change "/workspace" to your project path

# 4. Launch FastAPI service (F5 in VS Code or):
uvicorn src.agent_service:app --reload

# 5. Test with REST Client (open agent.http, click "Send Request")
```

## Quick Start (Devcontainer)

```bash
# 1. Open in VS Code
# 2. Click "Reopen in Container" when prompted
# 3. Wait for container build + pip install
# 4. Press F5 to launch service
```

## VS Code Extension

```bash
cd extension
npm install
npm run compile

# Press F5 to launch extension dev host
# In dev host: Ctrl+Shift+P → "AI Agent: Propose Edit"
```

## Safety Guarantees

✅ **Workspace Sandbox**: All operations restricted to `WORKSPACE_ROOT`  
✅ **Blocked Directories**: `.ssh`, `.aws`, `.git/config`, `AppData`, `.env`  
✅ **File Size Limit**: 50KB max per write  
✅ **Command Whitelist**: Only `pytest -q`, `ruff check .`, `git status`, etc.  
✅ **No Shell Injection**: Commands executed with `shell=False` only  
✅ **Dry Run Default**: `dry_run=true` by default; requires `dry_run=false` + `approved=true`  
✅ **Audit Logs**: JSONL logs in `logs/` (metadata only, no secrets)

## How to Reuse This Template

### On GitHub

1. In this repo: **Settings** → **Template repository** → ✅ Enable
2. For new projects: Click **"Use this template"** → **"Create a new repository"**
3. Clone your new repo and customize

### Locally

```bash
git clone https://github.com/youruser/ai-agent-template.git my-new-agent
cd my-new-agent
rm -rf .git
git init
# Customize and commit
```

## Enable Real Model

```bash
# 1. Copy .env.example to .env
cp .env.example .env

# 2. Add your API key to .env
echo "ANTHROPIC_API_KEY=sk-..." >> .env

# 3. Update src/model_client.py to call actual API
# (Currently uses stub: TODO→DONE replacement)
```

## Architecture

```
src/
  agent_service.py   # FastAPI endpoints
  policies.py        # Safety validators
  model_client.py    # LLM integration (stub)
  config.json        # WORKSPACE_ROOT config

extension/
  src/extension.ts   # VS Code command

tests/
  test_policies.py   # Security tests
  test_agent.py      # Endpoint tests
```

## Troubleshooting

**Service won't start?**
```bash
# Check port 8000 is free
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

**Tests failing?**
```bash
# Update WORKSPACE_ROOT in src/config.json to current directory
export WORKSPACE_ROOT=$(pwd)
pytest -q
```

**Extension not compiling?**
```bash
cd extension
npm install
npm run compile
```

## License

MIT - Use freely in your projects
