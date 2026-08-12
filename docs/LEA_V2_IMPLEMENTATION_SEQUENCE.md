# LEA v2 Implementation Sequence

Ordering for coding branches after foundation docs. Product milestones: [LEA_V2_FOUNDATION_PLAN.md](./LEA_V2_FOUNDATION_PLAN.md). Source of truth: [LEA_V2_SOURCE_OF_TRUTH.md](./LEA_V2_SOURCE_OF_TRUTH.md). Auth design: [LEA_AUTH_SESSION.md](./LEA_AUTH_SESSION.md). Provider port: [LEA_PROVIDER_ABSTRACTION.md](./LEA_PROVIDER_ABSTRACTION.md). Gmail: [LEA_GMAIL_READ.md](./LEA_GMAIL_READ.md).

## Branches

| Branch | Scope | Status |
| --- | --- | --- |
| `feature/lea-v2-foundation-hardening` | Planning docs only | Merged |
| `feature/lea-auth-and-owner-session` | **M1** single-owner session gate | Merged (`#4`) |
| `feature/lea-provider-abstraction` | **M2** read-only mail/calendar provider port + Outlook adapter | Merged (`#5`) |
| `feature/lea-gmail-read` | **M3** Gmail read provider foundation (skeleton; no live Gmail yet) | **Active coding branch** |

## M3 coding branch (current)

### `feature/lea-gmail-read`

- Register `gmailProvider` beside Outlook on the LEA port
- Capabilities: no send/write; **no** live `mailRead` until OAuth + API exist
- `getConnection` always disconnected with explicit deferred reason
- Config status helper lists required future env names (does not edit `.env`)
- Preserve Outlook live read + M1 owner-session gates
- **No deploy** in this branch

## Then

### Next product milestones

| Branch (suggested name) | Milestone |
| --- | --- |
| Follow-on Gmail OAuth + API read (still no send) | Complete M3 live read |
| `feature/lea-draft-approval-gates` | M4 |
| `feature/lea-secure-conversation-memory` | M5 |
| `feature/lea-document-library` | M6 |
| `feature/lea-executive-daily-briefing` | M7 |
| `feature/lea-health-weight-module` | M8 |
| Ops-only, explicit permissions | M9 staging |
| Decision record + ops, explicit permissions | M10 cutover |

## Guardrails for every coding branch

1. Target DreAgent only unless the prompt overrides with another **named** repo/path.
2. Never deploy unless the prompt sets **Deployment allowed: yes** and names environment.
3. Do not touch live Hummingbird, PM2, cPanel, or domain routing as a side effect of a feature branch.
4. Prefer small vertical slices that map to one milestone each.
5. Do not invent Gmail mailbox data or commit secrets.
