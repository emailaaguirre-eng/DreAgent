# LEA v2 Implementation Sequence

Ordering for coding branches after foundation docs. Product milestones: [LEA_V2_FOUNDATION_PLAN.md](./LEA_V2_FOUNDATION_PLAN.md). Source of truth: [LEA_V2_SOURCE_OF_TRUTH.md](./LEA_V2_SOURCE_OF_TRUTH.md). Auth design: [LEA_AUTH_SESSION.md](./LEA_AUTH_SESSION.md). Provider port: [LEA_PROVIDER_ABSTRACTION.md](./LEA_PROVIDER_ABSTRACTION.md).

## Branches

| Branch | Scope | Status |
| --- | --- | --- |
| `feature/lea-v2-foundation-hardening` | Planning docs only | Merged |
| `feature/lea-auth-and-owner-session` | **M1** single-owner session gate | Merged (`#4`) |
| `feature/lea-provider-abstraction` | **M2** read-only mail/calendar provider port + Outlook adapter | **Active coding branch** |

## M2 coding branch (current)

### `feature/lea-provider-abstraction`

- Provider-neutral types and `MailCalendarProvider` (read/list/status/capabilities)
- Outlook adapter wraps existing Graph read helpers
- Chat executive intents resolve a connected provider instead of importing Graph types
- Gmail **not** implemented; id reserved only
- Send/create remain off the LEA port and unused by chat
- M1 owner-session token gates unchanged
- **No deploy** in this branch

## Then

### 3. `feature/lea-gmail-read` (M3)

After abstraction exists:

- Gmail **read-only** adapter registered beside Outlook
- No send/write expansion in that branch unless re-scoped with M4 approval gates
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

**Ship order for remaining platform risk is M2 → M3**, then M4+ as listed.

## Guardrails for every coding branch

1. Target DreAgent only unless the prompt overrides with another **named** repo/path.
2. Never deploy unless the prompt sets **Deployment allowed: yes** and names environment.
3. Do not touch live Hummingbird, PM2, cPanel, or domain routing as a side effect of a feature branch.
4. Prefer small vertical slices that map to one milestone each.
