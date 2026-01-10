# DreAgent Cloud

**AI Assistant powered by CoDre-X™**

Copyright © 2026 B&D Servicing LLC - All Rights Reserved

---

## Overview

DreAgent Cloud is a high-performance AI assistant built with Next.js and the Vercel AI SDK. It features:

- 🚀 **Streaming Responses** - True streaming via Vercel AI SDK (no timeouts)
- 🧠 **RAG Integration** - Semantic search with Supabase pgvector
- 📧 **Outlook Integration** - Email and calendar via Microsoft Graph
- 🎤 **Voice I/O** - Speech-to-text input support
- 🎨 **Beautiful UI** - Modern, responsive design
- ⚡ **Edge Performance** - ~50ms cold starts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js App Router                       │
│  ┌─────────────────┐  ┌────────────────────────────────┐    │
│  │   React Frontend │  │  API Routes (Node.js Runtime)  │    │
│  │   Chat Interface │  │  • /api/chat (streaming)       │    │
│  │   Voice Input    │  │  • /api/knowledge (RAG)        │    │
│  │   Mode Selector  │  │  • /api/outlook/* (Graph API)  │    │
│  └─────────────────┘  └────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Supabase                              │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │  Conversations  │  │  Knowledge Documents (pgvector)  │   │
│  │  Outlook Tokens │  │  Semantic Search via embeddings  │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone and Install

```bash
git clone <your-repo>
cd dreagent-cloud
npm install
```

### 2. Set Up Environment

```bash
cp .env.example .env.local
# Edit .env.local with your keys
```

### 3. Set Up Supabase

1. Create a [Supabase](https://supabase.com) project
2. Run `src/lib/db/schema.sql` in the SQL Editor
3. Copy your project URL and service role key

### 4. Set Up Azure AD (for Outlook)

1. Go to [Azure Portal](https://portal.azure.com) → Azure AD → App registrations
2. Create new registration
3. Add redirect URI: `http://localhost:3000/api/outlook/auth`
4. Add API permissions:
   - `Mail.Read`, `Mail.Send`
   - `Calendars.ReadWrite`
   - `User.Read`, `offline_access`
5. Create a client secret
6. Copy Client ID, Tenant ID, and Secret

### 5. Run Locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Add environment variables
vercel env add OPENAI_API_KEY
vercel env add NEXT_PUBLIC_SUPABASE_URL
vercel env add SUPABASE_SERVICE_ROLE_KEY
vercel env add OUTLOOK_CLIENT_ID
vercel env add OUTLOOK_CLIENT_SECRET
vercel env add OUTLOOK_TENANT_ID

# Deploy to production
vercel --prod
```

## API Endpoints

### Chat (Streaming)

```bash
POST /api/chat
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "Hello!"}],
  "mode": "general",
  "enableRag": true
}
```

**Modes:** `general`, `it-support`, `executive`, `legal`, `finance`, `research`, `incentives`

### Knowledge (RAG)

```bash
# Add document
POST /api/knowledge
{
  "userId": "user-123",
  "title": "Company Handbook",
  "content": "..."
}

# Search
POST /api/knowledge/search
{
  "query": "vacation policy",
  "userId": "user-123"
}
```

### Outlook

```bash
# Get auth URL
GET /api/outlook/auth

# Get emails
GET /api/outlook/emails
Authorization: Bearer <access_token>

# Get calendar
GET /api/outlook/calendar
Authorization: Bearer <access_token>
```

## Agent Modes

| Mode | Model | Use Case |
|------|-------|----------|
| General | gpt-4o-mini | Quick tasks, triage |
| IT Support | gpt-4o | Coding, debugging |
| Executive | gpt-4o-mini | Emails, scheduling |
| Legal | gpt-4-turbo | Legal research |
| Finance | gpt-4-turbo | Tax, accounting |
| Research | gpt-4o | Deep explanations |
| Incentives | gpt-4-turbo | Forms, compliance |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `OUTLOOK_CLIENT_ID` | No | Azure AD app client ID |
| `OUTLOOK_CLIENT_SECRET` | No | Azure AD app secret |
| `OUTLOOK_TENANT_ID` | No | Azure AD tenant ID |
| `SERPAPI_API_KEY` | No | For web search |

## License

Copyright © 2026 B&D Servicing LLC - All Rights Reserved

Powered by CoDre-X™
