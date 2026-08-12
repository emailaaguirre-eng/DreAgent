# LEA Gmail Read (M3 foundation)

## Gmail M3 design

Gmail plugs into the existing **read-only** `MailCalendarProvider` port from M2. This branch adds a **registered skeleton adapter** so LEA is provider-aware of Gmail without inventing live mailbox access.

```text
Chat / executive
      │
      ▼
resolveConnectedProvider()
      │
      ├─ Outlook adapter  → live Graph read when owner session + tokens exist
      └─ Gmail adapter    → always disconnected until OAuth + API read exist
```

Rules:

- **No send/write** on the Gmail (or LEA) provider port (`mailSend` / `calendarWrite` = false).
- **Do not invent** inbox/calendar contents.
- **Do not commit secrets.** Operators set env later; this branch does not create or edit `.env`.
- **M1 owner session** remains required for Outlook DB tokens; Gmail has no token store yet.

## What is implemented

| Piece | Status |
| --- | --- |
| `gmailProvider` (`src/lib/providers/gmail-adapter.ts`) | Registered |
| Registry entry beside Outlook | Yes |
| `getConnection` | Always `connected: false` with explicit reason |
| Capability flags | `mailRead`/`calendarRead`/`mailExport` = **false**; writes = **false** |
| `getGmailConfigStatus()` | Reports whether `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET` env names are present |
| Live Gmail API list/read | **Not implemented** |
| Gmail OAuth routes | **Not implemented** |
| Gmail token table | **Not implemented** |

## What is intentionally deferred

- Google OAuth authorize / callback routes (mirror of `/api/outlook/auth`, owner-session gated)
- Durable token storage (e.g. `gmail_tokens` analogous to `outlook_tokens`)
- Gmail API client for messages/calendar list
- Flipping `mailRead: true` (only when a real safe read path exists)
- Gmail CSV export route
- Calendar read for Google Calendar
- Any send / draft-send / create behavior (M4+)
- Multi-provider UI picker (Outlook still preferred when connected)

## How Gmail plugs into provider abstraction

1. Implement `MailCalendarProvider` as `gmailProvider`.
2. Register in `src/lib/providers/registry.ts`.
3. Chat continues to call `resolveConnectedProvider` only — no Gmail-specific Graph/API imports in chat.
4. Until `getConnection().connected === true`, chat fails soft to the existing “no connected provider” path (Outlook still works when connected).

## No send/write expansion

- Provider port has no send/create methods.
- `GMAIL_FOUNDATION_CAPABILITIES.mailSend` and `calendarWrite` are false.
- Chat must not call `sendEmail` / `createCalendarEvent` (unchanged Smart LEA rule).

## Required future env / OAuth setup (operators only)

Set on the server (never commit values):

| Env | Purpose |
| --- | --- |
| `GMAIL_CLIENT_ID` | Google OAuth client id |
| `GMAIL_CLIENT_SECRET` | Google OAuth client secret |
| `GMAIL_REDIRECT_URI` or `NEXT_PUBLIC_APP_URL` | Callback URL construction (optional until routes exist) |

Also required later (code, not this branch):

- Owner-session–gated OAuth connect flow
- Token persistence under trusted owner id
- Gmail API scopes limited to **read** for M3 completion (`gmail.readonly` / calendar read as decided)
- Staging validation before any deploy (M9); no live Hummingbird cutover here

## Why no live deployment

Hummingbird remains live at `lea.codre-x.com`. This is foundation-only (`Deployment allowed: no`). Enabling live Gmail read requires OAuth + tokens + API client + staging sign-off, not this skeleton alone.
