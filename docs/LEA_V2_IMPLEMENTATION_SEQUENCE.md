# LEA v2 Implementation Sequence

Ordering for coding branches after foundation docs. Product milestones: [LEA_V2_FOUNDATION_PLAN.md](./LEA_V2_FOUNDATION_PLAN.md). Source of truth: [LEA_V2_SOURCE_OF_TRUTH.md](./LEA_V2_SOURCE_OF_TRUTH.md). Auth design: [LEA_AUTH_SESSION.md](./LEA_AUTH_SESSION.md).

## Branches

| Branch | Scope | Status |
| --- | --- | --- |
| `feature/lea-v2-foundation-hardening` | Planning docs only | Merged to main as foundation path docs |
| `feature/lea-auth-and-owner-session` | **M1** single-owner session gate; stop trusting client `userId` | **Active coding branch** |

## M1 coding branch (current)

### `feature/lea-auth-and-owner-session`

- Server-authoritative owner session (`lea_owner_session` HTTP-only cookie)
- Bootstrap via `POST /api/auth/owner-session` (shared secret; not multi-user login UI)
- Knowledge, conversations, and Outlook DB-token paths fail closed without trusted owner
- Client `localStorage` / body / `x-user-id` are not authoritative
- Chat drafting still works without session; personal RAG/mail storage requires session
- Offline checks for owner-session helper and untrusted client userId
- **No deploy** in this branch

## Then

### 2. `feature/lea-provider-abstraction` (M2)

After identity can be trusted:

- Define email/calendar provider interfaces
- Move Outlook (and any existing) integrations behind ports
- Keep behavior parity for current Outlook users
- No requirement to enable Gmail in the same PR/branch

### 3. `feature/lea-gmail-read` (M3)

After abstraction exists:

- Gmail **read-only** support through the provider layer
- No send/write expansion in this branch unless re-scoped with M4 approval gates
- Secrets remain out of repo; config via env on authorized environments only

## Later branches (indicative)

| Branch (suggested name) | Milestone |
| --- | --- |
| `feature/lea-draft-approval-gates` | M4 |
| `feature/lea-secure-conversation-memory` | M5 |
| `feature/lea-document-library` | M6 |
| `feature/lea-executive-daily-briefing` | M7 |
| `feature/lea-health-weight-module` | M8 |
| Ops-only, explicit permissions | M9 staging |
| Decision record + ops, explicit permissions | M10 cutover |

**Ship order for platform risk remains M1 → M2 → M3.**

## Guardrails for every coding branch

1. Target DreAgent only unless the prompt overrides with another **named** repo/path.
2. Never deploy unless the prompt sets **Deployment allowed: yes** and names environment.
3. Do not touch live Hummingbird, PM2, cPanel, or domain routing as a side effect of a feature branch.
4. Prefer small vertical slices that map to one milestone each.
