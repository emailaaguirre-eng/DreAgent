# Hummingbird-LEA Project Context

## Project Overview
**Hummingbird-LEA** is a private, self-hosted multi-agent AI platform powered by the **CoDre-X** engine.

- **Owner:** B & D Servicing LLC
- **Creator:** Dre
- **Purpose:** Internal AI assistant platform for Dre and EIAG team
- **Cost Target:** $0/month recurring (100% self-hosted with Ollama)

## The Three Agents

### Lea - Executive Assistant (Primary)
- **Role:** Executive operations, email/calendar, document generation, task management
- **Personality:** Warm, friendly, proactive, detail-oriented, humorous but professional
- **Model:** llama3.1:8b (via Ollama)
- **File:** `core/agents/lea.py`

### Chiquis - Coding Partner
- **Role:** Code writing, debugging, GitHub operations, technical documentation
- **Personality:** Sweet, supportive, encouraging, patient - Lea's brother
- **Model:** deepseek-coder:6.7b (via Ollama)
- **File:** `core/agents/chiquis.py`

### Grant - Economic Incentives Expert
- **Role:** EIAG subject matter expert - site selection, tax incentives, proposals
- **Personality:** Professional, thorough, analytical, precise
- **Model:** llama3.1:8b (via Ollama)
- **File:** `core/agents/grant.py`

## Tech Stack
- **Framework:** FastAPI (chosen over Django for async AI calls)
- **AI Provider:** Ollama (local, self-hosted)
- **Database:** SQLite (async with aiosqlite)
- **Vector DB:** ChromaDB (for RAG - Phase 3)
- **Frontend:** Vanilla HTML/CSS/JS (no framework)
- **Auth:** JWT tokens with simple username/password
- **Deployment:** Docker, systemd, Nginx reverse proxy

## Critical Design Principles

### Anti-Hallucination Strategy (VERY IMPORTANT)
The agents MUST follow these rules:
1. **NEVER ASSUME** - When in doubt, ASK before answering
2. **NEVER INVENT** - No made-up facts, dates, numbers
3. **SAY "I DON'T KNOW"** - Admit uncertainty clearly
4. **CITE SOURCES** - Reference where information came from
5. **CONFIRM CRITICAL ACTIONS** - Verify before irreversible actions
6. **CONFIDENCE SCORING** - If <85% confident, express uncertainty

### Ambiguity Detection
Agents should ask clarifying questions when they detect:
- Vague references ("it", "that", "the file")
- Incomplete actions ("send", "create" without objects)
- Vague time references ("soon", "later")

## Project Structure
```
hummingbird-lea/
├── core/                    # CoDre-X Engine
│   ├── agents/              # Lea, Chiquis, Grant agents
│   │   ├── base.py          # Base agent class (Phase 2 integrated)
│   │   ├── lea.py           # Lea agent
│   │   ├── chiquis.py       # Chiquis agent
│   │   └── grant.py         # Grant agent
│   ├── providers/           # AI providers
│   │   ├── ollama.py        # Ollama client
│   │   └── router.py        # Smart model router
│   ├── reasoning/           # Phase 2: Agentic Reasoning
│   │   ├── react.py         # ReAct loop implementation
│   │   ├── ambiguity.py     # Advanced ambiguity detection
│   │   ├── confidence.py    # Multi-factor confidence scoring
│   │   ├── clarifying.py    # Clarifying questions engine
│   │   ├── transparency.py  # Reasoning transparency
│   │   └── __init__.py      # ReasoningEngine unified class
│   ├── services/            # Core Services
│   │   ├── rag/             # Phase 3: RAG System
│   │   ├── vision/          # Phase 4: Vision & OCR
│   │   ├── documents/       # Phase 5: Document Generation
│   │   └── __init__.py
│   └── utils/               # Config, Auth, Security
│       ├── config.py        # Settings management
│       ├── auth.py          # JWT authentication
│       ├── monitoring.py    # Health checks & metrics (Phase 6)
│       └── security/        # Security utilities (Phase 6)
│           ├── middleware.py # Rate limiting, input validation
│           ├── errors.py     # Error handling
│           ├── cache.py      # Caching system
│           └── __init__.py
├── apps/
│   └── hummingbird/         # Web Application
│       ├── api/             # REST API endpoints
│       │   ├── auth.py      # Authentication
│       │   ├── chat.py      # Chat endpoint
│       │   ├── health.py    # Health checks & metrics
│       │   ├── knowledge.py # RAG/Knowledge base
│       │   ├── vision.py    # Vision/OCR API
│       │   ├── documents.py # Document generation API
│       │   └── ide.py       # Chiquis IDE API (Phase 7)
│       ├── static/          # Frontend (HTML/CSS/JS)
│       │   ├── index.html   # Main chat interface
│       │   ├── ide.html     # Chiquis IDE (Phase 7)
│       │   ├── ide.js       # IDE JavaScript
│       │   └── ide.css      # IDE styles
│       └── main.py          # FastAPI app
├── data/                    # Runtime data
├── config/                  # Configuration
│   ├── .env.example         # Environment template
│   ├── .env.production      # Production config
│   └── nginx/               # Nginx configuration
├── scripts/                 # Deployment scripts
│   ├── deploy.sh            # Main deployment script
│   ├── setup-ollama.sh      # Ollama setup
│   ├── hummingbird.service  # Systemd service
│   └── ollama.service       # Ollama systemd service
├── tests/                   # Test suite
├── Dockerfile               # Docker build
├── docker-compose.yml       # Docker orchestration
├── docker-compose.prod.yml  # Production overrides
├── requirements.txt
├── run.py                   # Entry point
└── CLAUDE.md               # This file
```

## Development Status - ALL PHASES COMPLETE

### Phase 1: COMPLETE - Foundation
- Project structure
- Base agent class with anti-hallucination
- All three agents (Lea, Chiquis, Grant)
- Ollama integration
- Smart router
- FastAPI backend
- Web chat interface
- JWT authentication
- Health checks

### Phase 2: COMPLETE - Agentic Reasoning
- **ReAct reasoning loop** (`core/reasoning/react.py`)
  - Thought → Action → Observation → Reflection pattern
  - Structured reasoning steps with confidence tracking
- **Advanced ambiguity detector** (`core/reasoning/ambiguity.py`)
  - Pattern-based detection with severity scoring
- **Multi-factor confidence scoring** (`core/reasoning/confidence.py`)
  - Uncertainty language, hedging phrases, complexity assessment
- **Clarifying questions engine** (`core/reasoning/clarifying.py`)
  - Intent detection and priority-based questions
- **Reasoning transparency** (`core/reasoning/transparency.py`)
  - Multiple transparency levels for debugging

### Phase 3: COMPLETE - RAG System
- **Document Loaders** - PDF, DOCX, TXT, MD, CSV, JSON
- **Text Chunking** - Recursive, Sentence, Paragraph strategies
- **Embedding Pipeline** - nomic-embed-text via Ollama with caching
- **Vector Store** - ChromaDB with semantic search
- **RAG Engine** - Unified indexing and retrieval
- **API Endpoints** - Upload, search, context retrieval

### Phase 4: COMPLETE - Vision & OCR
- **Vision Service** (`core/services/vision/vision.py`)
  - llava-llama3:8b integration for image understanding
  - Scene description, object detection, text reading
- **OCR Service** (`core/services/vision/ocr.py`)
  - EasyOCR with multi-language support
  - Text extraction with confidence scoring
- **Image Analysis Engine** (`core/services/vision/engine.py`)
  - Unified interface combining vision + OCR
  - Automatic analysis type detection
- **Vision Mixin** (`core/services/vision/agent_mixin.py`)
  - Easy integration with agents
- **API Endpoints** (`apps/hummingbird/api/vision.py`)
  - `POST /api/vision/analyze` - Analyze image
  - `POST /api/vision/ocr` - Extract text
  - `POST /api/vision/describe` - Describe image
  - `GET /api/vision/capabilities` - List capabilities

### Phase 5: COMPLETE - Document Generation
- **PowerPoint Generator** (`core/services/documents/powerpoint.py`)
  - Professional slide creation with python-pptx
  - EIAG branded templates
- **Word Generator** (`core/services/documents/word.py`)
  - Document creation with python-docx
  - Tables, headers, styled sections
- **PDF Generator** (`core/services/documents/pdf.py`)
  - PDF creation with reportlab
  - Custom layouts and branding
- **Document Engine** (`core/services/documents/engine.py`)
  - Unified interface for all formats
  - EIAG-specific templates (proposals, site selection, incentive summaries)
- **Document Mixin** (`core/services/documents/agent_mixin.py`)
  - Agent integration with intent detection
- **API Endpoints** (`apps/hummingbird/api/documents.py`)
  - `POST /api/documents/generate` - Generate document
  - `POST /api/documents/proposal` - EIAG proposal
  - `POST /api/documents/site-selection` - Site selection report
  - `GET /api/documents/templates` - List templates
  - `GET /api/documents/formats` - Available formats

### Phase 6: COMPLETE - Polish & Deploy
- **Security Middleware** (`core/utils/security/middleware.py`)
  - Rate limiting with sliding window
  - Input validation (SQL injection, XSS, path traversal protection)
  - Security headers (CSP, HSTS, X-Frame-Options)
  - Request ID tracking
- **Error Handling** (`core/utils/security/errors.py`)
  - Custom exception hierarchy
  - Error categories and severity levels
  - Structured error responses
  - Error recovery utilities (retry, fallback)
- **Caching System** (`core/utils/security/cache.py`)
  - LRU cache with TTL support
  - Response caching for API endpoints
  - Embedding cache for RAG
  - Cache decorators
- **Production Configuration** (`config/.env.production`)
  - Environment-specific settings
  - Security hardening options
- **Docker Deployment**
  - `Dockerfile` - Multi-stage build
  - `docker-compose.yml` - Full stack orchestration
  - `docker-compose.prod.yml` - Production overrides
  - Ollama service integration
- **Deployment Scripts** (`scripts/`)
  - `deploy.sh` - Automated deployment (Docker or native)
  - `setup-ollama.sh` - Model installation
  - `hummingbird.service` - Systemd service with security hardening
  - `ollama.service` - Ollama systemd service
- **Nginx Configuration** (`config/nginx/`)
  - Reverse proxy setup
  - SSL/TLS configuration
  - Rate limiting at proxy level
  - Security headers
- **Health Monitoring** (`core/utils/monitoring.py`)
  - Component health checks
  - System metrics (CPU, memory, disk)
  - Request metrics and latency tracking
  - Structured JSON logging
  - Kubernetes-style probes (/ready, /live)

### Phase 7: COMPLETE - Chiquis Agentic IDE
- **Monaco Editor Integration** (`apps/hummingbird/static/ide.html`, `ide.js`, `ide.css`)
  - Full VS Code editor experience in browser
  - Syntax highlighting for 20+ languages
  - Multiple file tabs with modification tracking
  - Dark theme optimized for coding
- **AI Code Completion** (`apps/hummingbird/api/ide.py`)
  - Real-time code suggestions powered by deepseek-coder
  - Context-aware completions based on cursor position
  - Language-specific completion providers
- **Cmd+K Inline Edits**
  - Natural language code transformations
  - Select code and describe changes
  - Inline edit dialog with context awareness
  - Explanation of changes made
- **Chat Sidebar**
  - Integrated chat with Chiquis while coding
  - Code context awareness (selected code)
  - Conversation history maintained
  - Code suggestions extracted from responses
- **Codebase Awareness via RAG**
  - Project indexing for code search
  - Context retrieval during chat
  - Referenced files shown in responses
- **File Explorer & Project Management**
  - Tree view of project files
  - Create/delete files and folders
  - Multiple project support
  - Project creation wizard
- **API Endpoints** (`apps/hummingbird/api/ide.py`)
  - `GET /api/ide/files/tree` - Get file tree
  - `GET /api/ide/files/read` - Read file content
  - `POST /api/ide/files/write` - Write file
  - `DELETE /api/ide/files/delete` - Delete file
  - `POST /api/ide/completion` - AI code completion
  - `POST /api/ide/inline-edit` - Cmd+K inline edit
  - `POST /api/ide/chat` - Chat with codebase context
  - `POST /api/ide/index-project` - Index for RAG
  - `GET /api/ide/projects` - List projects
  - `POST /api/ide/projects/create` - Create project
- **Access**: Navigate to `/ide` from main chat or directly

## Ollama Models Required
```bash
ollama pull llama3.1:8b        # Chat/reasoning
ollama pull deepseek-coder:6.7b # Coding
ollama pull llava-llama3:8b     # Vision (Phase 4)
ollama pull nomic-embed-text    # Embeddings (Phase 3)
```

## Running the Project

### Development
```bash
# Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy config
cp config/.env.example .env
# Edit .env with your settings

# Run
python run.py
```

### Production with Docker
```bash
# Copy and configure environment
cp config/.env.production .env
# Edit .env with production values (IMPORTANT: change all secrets!)

# Start services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Initialize Ollama models (first time only)
docker-compose --profile init up ollama-init
```

### Production with systemd
```bash
# Run deployment script
sudo ./scripts/deploy.sh --native

# Or manually:
# 1. Install Ollama
./scripts/setup-ollama.sh

# 2. Setup systemd services
sudo cp scripts/hummingbird.service /etc/systemd/system/
sudo cp scripts/ollama.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hummingbird ollama
sudo systemctl start ollama
sudo systemctl start hummingbird
```

## API Endpoints Summary

| Endpoint | Description |
|----------|-------------|
| `POST /api/auth/login` | JWT authentication |
| `POST /api/chat/` | Chat with agents |
| `GET /api/health/` | Basic health check |
| `GET /api/health/detailed` | Component health report |
| `GET /api/health/metrics` | Application metrics |
| `POST /api/knowledge/upload` | Index documents |
| `POST /api/knowledge/search` | Search knowledge base |
| `POST /api/vision/analyze` | Analyze images |
| `POST /api/vision/ocr` | Extract text from images |
| `POST /api/documents/generate` | Generate documents |
| `POST /api/documents/proposal` | EIAG proposals |
| `GET /api/ide/files/tree` | IDE file explorer |
| `POST /api/ide/completion` | AI code completion |
| `POST /api/ide/inline-edit` | Cmd+K inline edit |
| `POST /api/ide/chat` | Chat with code context |

## User Context
- **Dre** is a self-described "new coder" who needs **baby steps**
- He works at **EIAG** (Economic Incentives Advisory Group)
- He values **accuracy over speed** - agents should ask questions rather than guess
- The project should be **well-documented** with clear explanations

## Code Style
- Use clear, descriptive variable names
- Add docstrings to all functions and classes
- Include inline comments for complex logic
- Keep functions focused and not too long
- Use type hints where helpful

## Important Files to Know

### Core
- `core/agents/base.py` - Base agent class (Phase 2 integrated)
- `core/providers/ollama.py` - Ollama client
- `core/reasoning/__init__.py` - ReasoningEngine (Phase 2)

### Services
- `core/services/rag/engine.py` - RAG engine (Phase 3)
- `core/services/vision/engine.py` - Vision/OCR engine (Phase 4)
- `core/services/documents/engine.py` - Document engine (Phase 5)

### Security & Monitoring
- `core/utils/security/middleware.py` - Rate limiting, validation
- `core/utils/security/errors.py` - Error handling
- `core/utils/security/cache.py` - Caching system
- `core/utils/monitoring.py` - Health checks & metrics

### API
- `apps/hummingbird/api/chat.py` - Main chat endpoint
- `apps/hummingbird/api/health.py` - Health & monitoring API
- `apps/hummingbird/api/ide.py` - Chiquis IDE API (Phase 7)
- `apps/hummingbird/main.py` - FastAPI application

### IDE (Phase 7)
- `apps/hummingbird/static/ide.html` - IDE page
- `apps/hummingbird/static/ide.js` - Monaco Editor integration
- `apps/hummingbird/static/ide.css` - IDE styling

### Deployment
- `Dockerfile` - Container build
- `docker-compose.yml` - Service orchestration
- `scripts/deploy.sh` - Deployment automation
- `config/nginx/hummingbird.conf` - Nginx configuration
