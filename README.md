# 🐦 Hummingbird-LEA

**Your AI Team - Powered by CoDre-X**

A private, self-hosted AI assistant platform featuring three specialized agents:
- **Lea** 🐦 - Executive Assistant & Operations Lead
- **Chiquis** 💻 - Coding Partner
- **Grant** 🏛️ - Economic Incentives & Site Selection Expert

---

## Features

- ✅ **100% Self-Hosted** - Runs entirely on your server
- ✅ **$0 AI Costs** - Uses Ollama for local AI inference
- ✅ **Private & Secure** - No data leaves your network
- ✅ **Anti-Hallucination** - Agents ask questions when uncertain
- ✅ **Three Specialized Agents** - Each with unique expertise
- ✅ **Web-Based Interface** - Access from any device

---

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running
3. **Required Ollama models:**
   ```bash
   ollama pull llama3.1:8b
   ollama pull deepseek-coder:6.7b
   ollama pull llava-llama3:8b
   ollama pull nomic-embed-text
   ```

### Installation

1. **Clone or download this project**

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   # Copy the example config
   cp config/.env.example .env
   
   # Edit .env with your settings (especially change the passwords!)
   ```

5. **Run the application:**
   ```bash
   python run.py
   ```

6. **Open your browser:**
   ```
   http://localhost:8000
   ```

---

## Configuration

Edit the `.env` file to customize:

```env
# Change these for security!
ADMIN_USERNAME=your_username
ADMIN_PASSWORD=your_secure_password
SECRET_KEY=generate-a-random-string
JWT_SECRET_KEY=generate-another-random-string

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_CHAT=llama3.1:8b
OLLAMA_MODEL_CODE=deepseek-coder:6.7b

# Server settings
HOST=0.0.0.0
PORT=8000
```

---

## Project Structure

```
hummingbird-lea/
├── core/                    # CoDre-X Engine
│   ├── agents/              # Lea, Chiquis, Grant
│   ├── providers/           # Ollama integration
│   ├── reasoning/           # Agentic AI (Phase 2)
│   ├── services/            # RAG, Vision, etc. (Future)
│   └── utils/               # Config, Auth
│
├── apps/
│   └── hummingbird/         # Web Application
│       ├── api/             # REST API endpoints
│       ├── static/          # Frontend (HTML/CSS/JS)
│       └── main.py          # FastAPI app
│
├── data/                    # Data storage
│   ├── knowledge/           # RAG documents
│   ├── memory/              # Conversation history
│   └── uploads/             # User uploads
│
├── config/                  # Configuration
│   ├── .env.example         # Environment template
│   └── robots.txt           # Block search engines
│
├── requirements.txt         # Python dependencies
├── run.py                   # Start script
└── README.md               # This file
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Authenticate and get token |
| `/api/chat/` | POST | Send message to agent |
| `/api/chat/agents` | GET | List available agents |
| `/api/chat/greeting/{agent}` | GET | Get agent greeting |
| `/api/health/` | GET | Health check |
| `/api/health/ollama` | GET | Ollama status |

---

## The Agents

### 🐦 Lea - Executive Assistant

Lea is your primary assistant, handling:
- Email drafting and management
- Calendar coordination
- Document creation (PowerPoint, Word, PDF)
- Task management
- General assistance

**Personality:** Warm, friendly, proactive, detail-oriented

### 💻 Chiquis - Coding Partner

Chiquis is your coding companion, helping with:
- Writing and reviewing code
- Debugging errors
- Explaining concepts
- GitHub operations
- Project scaffolding

**Personality:** Sweet, supportive, patient, encouraging

### 🏛️ Grant - Incentives Expert

Grant is your EIAG domain expert, specializing in:
- State/local economic incentives
- Site selection analysis
- Tax credit programs
- Client proposals

**Personality:** Professional, thorough, analytical, precise

---

## Security

This is a **private application**. Security measures include:

1. **Authentication** - Username/password required
2. **JWT Tokens** - Secure session management
3. **No Indexing** - `robots.txt` and meta tags block crawlers
4. **Security Headers** - X-Robots-Tag, X-Frame-Options, etc.

**Important:** Always change the default credentials in production!

---

## Roadmap

- [x] **Phase 1:** Core agents, chat API, web interface
- [ ] **Phase 2:** Advanced reasoning (ReAct loop)
- [ ] **Phase 3:** RAG system for document knowledge
- [ ] **Phase 4:** Vision & OCR capabilities
- [ ] **Phase 5:** Document generation (PowerPoint, Word)
- [ ] **Phase 6:** Full deployment & polish

---

## Troubleshooting

### Ollama not connecting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Models not found
```bash
# Pull required models
ollama pull llama3.1:8b
ollama pull deepseek-coder:6.7b
```

### Permission errors
```bash
# Ensure data directories exist
mkdir -p data/knowledge data/memory data/uploads
```

---

## License

Proprietary - B & D Servicing LLC

---

## Credits

**Created by:** Dre  
**Platform:** CoDre-X  
**Company:** B & D Servicing LLC

---

*Powered by CoDre-X* ⚡
