# LEA Auth & Owner Session (M1)

## Current M1 design

LEA v2 is a **single-owner** product on DreAgent. This branch adds a **server-trusted owner session**, not multi-user SaaS auth.

| Piece | Role |
| --- | --- |
| `LEA_OWNER_SESSION_SECRET` | Server-only secret used to sign session tokens and bootstrap a session |
| `LEA_OWNER_ID` (optional) | Durable owner key for DB rows; default `lea-owner` |
| Cookie `lea_owner_session` | HTTP-only, signed, time-limited owner session |
| `POST /api/auth/owner-session` | Bootstrap gate: body/header secret must match `LEA_OWNER_SESSION_SECRET` |
| `src/lib/auth/owner-session.ts` | Resolve/reject identity; never treats client claims as authority |

### Identity resolution

1. Verify the signed owner session cookie (HMAC-SHA256).
2. Owner id must match configured `LEA_OWNER_ID` / default.
3. If the client also sends `userId` / `x-user-id` that **differs** from the session owner → **fail closed**.
4. Client-only identity with **no** cookie → **fail closed** for sensitive data paths.

Chat **LLM drafting** can still run without an owner session. Personal knowledge, conversation store, and Outlook **DB-token** paths require the session (or a live Graph Bearer for mail/calendar Graph calls only).

## What is trusted

- Signed `lea_owner_session` cookie after bootstrap
- Server-configured owner id (`LEA_OWNER_ID` or `lea-owner`)
- `Authorization: Bearer <graph_access_token>` as a **live Graph credential** (not as multi-user app identity)

## What is not trusted

- `localStorage` / `dreagent_user_id` (UI preference only after M1)
- Body `userId`, query `userId`, header `x-user-id` as sole identity
- Forged or cross-user client ids pretending to be the owner

## Sensitive routes (fail closed without owner session for data bound to owner)

- `/api/knowledge`, `/api/knowledge/search`
- `/api/conversations`
- Outlook token **storage/lookup** via DB (`resolveOutlookAccessToken` without Bearer)
- Outlook OAuth token persist on `/api/outlook/auth` callback

## Remaining auth work (not this branch)

- Real login UI (password/OAuth for the human operator) beyond bootstrap secret
- Multi-user accounts and RLS policies
- Hardware/phishing-resistant MFA
- Migrating legacy multi-`userId` Outlook token rows into `lea-owner`
- Full cutover of any external clients still sending `userId`
- M4 draft approval gates (send/create remain out of chat; Graph write APIs still exist but no longer accept client-only identity for DB tokens)

## Risks partially mitigated

- **R1** (weak client userId): partial — server no longer treats client `userId` as authority; session required for owner-scoped data

## Why no live deployment is included

- Hummingbird remains live at `lea.codre-x.com`
- This is foundation hardening only (`Deployment allowed: no`)
- Staging/live must not pick this up until M9/M10 and explicit deploy permission
- Owner session secret must be set in the environment by operators; this branch does **not** edit `.env` or deploy

## Operator bootstrap (authorized environments only)

```http
POST /api/auth/owner-session
Content-Type: application/json

{ "secret": "<same value as LEA_OWNER_SESSION_SECRET>" }
```

Response sets HTTP-only cookie. `GET /api/auth/owner-session` returns status. `DELETE` clears the cookie.

Do not commit secrets. Do not put the bootstrap secret in the client app bundle.
