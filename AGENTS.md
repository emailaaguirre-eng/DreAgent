# AGENTS.md

## Cursor Cloud specific instructions

DreAgent Cloud ("Lea") is a single **Next.js 14 (App Router) + TypeScript** app — frontend React UI and backend API routes live in the same deployable unit. There are no local databases or service containers to run; all backing services (OpenAI, Supabase, Microsoft Graph, SerpAPI) are remote SaaS reached via API keys. The package manager is **npm** (`package-lock.json`). Dependencies are installed automatically by the startup update script, so you normally don't need to run `npm install` yourself.

Standard commands (see `package.json`):
- Dev server: `npm run dev` (serves http://localhost:3000). Run it in a background/tmux session.
- Lint: `npm run lint`. Build: `npm run build`. Prod start: `npm start`.
- Offline test suite: `npm run test:lea-executive` — runs `scripts/lea-executive-checks.ts` via `tsx`. This exercises real core logic (executive intent routing, draft parsing, owner-session HMAC, provider abstraction) and needs **no** external services or env vars, so it always works in this environment.

Non-obvious caveats:
- **Clients are lazily initialized and features are gated**, so the dev server boots and the UI renders with no env vars set. Failures only surface at call time. In particular, `POST /api/chat` returns `{"error":"All model attempts failed: OpenAI API key is missing..."}` when `OPENAI_API_KEY` is absent — the app is healthy; the key just isn't configured.
- Put secrets in a gitignored `.env.local` (Next.js auto-loads it). `.env`/`.env.local` are gitignored — never commit them. See `.env.example` for the full variable list.
- **Owner auth is the gate for personal features.** Set `LEA_OWNER_SESSION_SECRET` (and optional `LEA_OWNER_ID`, `LEA_OWNER_COOKIE_SECURE=false` for local http). Mint a trusted session with `POST /api/auth/owner-session` body `{"secret":"<LEA_OWNER_SESSION_SECRET>"}`, which sets an HMAC-signed HTTP-only `lea_owner_session` cookie. A client-supplied `userId` is explicitly **not** trusted: without a valid owner session, RAG fails closed and executive mail/calendar flows report no connected provider.
- Minimum config to exercise the primary chat product end-to-end: `OPENAI_API_KEY` (chat + embeddings) plus `NEXT_PUBLIC_SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (conversations/RAG; apply `src/lib/db/schema.sql` in Supabase first). Outlook/Azure AD and SerpAPI are optional feature add-ons.
- `npm run build` prints a benign `Email history CSV export error: Dynamic server usage` log while prerendering `/api/outlook/email-history`; the build still succeeds (that route is correctly dynamic).
