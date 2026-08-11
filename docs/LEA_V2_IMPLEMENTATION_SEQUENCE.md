# LEA v2 Implementation Sequence

Ordering for **coding branches after** the docs/planning foundation branch. Product milestones: [LEA_V2_FOUNDATION_PLAN.md](./LEA_V2_FOUNDATION_PLAN.md). Source of truth: [LEA_V2_SOURCE_OF_TRUTH.md](./LEA_V2_SOURCE_OF_TRUTH.md).

## Current branch (docs only)

| Branch | Scope |
| --- | --- |
| `feature/lea-v2-foundation-hardening` | Planning docs only — no application code, no deploy, no env edits |

## Recommended first real coding branch

### 1. `feature/lea-auth-and-owner-session` (M1 first)

**Do this first.** Without owner session, every other feature inherits R1 (weak client identity).

Suggested focus:

- Server-authoritative session / auth hardening
- Owner binding for LEA executive and memory paths
- Remove or demote sole reliance on client `userId` / localStorage identity
- Tests proving unauthenticated and cross-user access fail closed

Prompt must still declare target repo/path/branch and **Deployment allowed: no** unless staging is intentionally in scope.

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

## Later branches (indicative, not started here)

| Branch (suggested name) | Milestone |
| --- | --- |
| `feature/lea-draft-approval-gates` | M4 |
| `feature/lea-secure-conversation-memory` | M5 |
| `feature/lea-document-library` | M6 |
| `feature/lea-executive-daily-briefing` | M7 |
| `feature/lea-health-weight-module` | M8 |
| Ops-only, explicit permissions | M9 staging |
| Decision record + ops, explicit permissions | M10 cutover |

Parallelism is possible after M1 for docs-heavy modules (e.g. health plan), but **ship order for platform risk is M1 → M2 → M3** on the branches above.

## Guardrails for every coding branch

1. Target DreAgent only unless the prompt overrides with another **named** repo/path.
2. Never deploy unless the prompt sets **Deployment allowed: yes** and names environment.
3. Do not touch live Hummingbird, PM2, cPanel, or domain routing as a side effect of a feature branch.
4. Do not install deps or change app code in docs-only work.
5. Prefer small vertical slices that map to one milestone each.

## Done definition for this sequence doc

- Engineering agrees M1 branch name and that it is the next code branch after docs merge.
- No application commits mixed into the foundation-hardening docs branch.
