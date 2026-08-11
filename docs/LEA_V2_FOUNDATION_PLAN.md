# LEA v2 Foundation Plan

DreAgent is the selected LEA v2 foundation. This plan sequences foundation hardening before any live cutover. Live Hummingbird at `lea.codre-x.com` stays production until M10 decides otherwise.

See also:

- [LEA_V2_SOURCE_OF_TRUTH.md](./LEA_V2_SOURCE_OF_TRUTH.md)
- [LEA_V2_RISK_REGISTER.md](./LEA_V2_RISK_REGISTER.md)
- [LEA_V2_IMPLEMENTATION_SEQUENCE.md](./LEA_V2_IMPLEMENTATION_SEQUENCE.md)

## Goals

- Make identity and session ownership trustworthy.
- Decouple mail/calendar from a single provider (Outlook-only today).
- Add Gmail **read** support under explicit policy.
- Enforce draft-only write paths until approval gates exist.
- Secure conversation memory; grow file library and executive modules with clear milestones.
- Prove on **staging** (`/home/codrex/lea-main` candidate) before any domain decision.
- Keep live Hummingbird untouched until cutover is explicitly approved.

## Non-goals (this foundation program)

- Live cutover or DNS/cPanel changes without a dedicated decision.
- Secret creation or `.env` commits.
- Coding product milestones until the docs branch lands and a coding branch is opened.

## Milestones

### M1 — Identity / auth hardening

**Outcome:** Server-trusted owner session; no reliance on client-only `userId` / localStorage as the sole identity.

**Includes:** Auth model, session cookies or equivalent server session, owner binding for LEA actions, regression expectations for unauthenticated access.

**Exit criteria:** Authenticated routes reject forged client identity; owner session documented for subsequent features.

### M2 — Provider abstraction for email / calendar

**Outcome:** Interface(s) for mail and calendar so features do not hardcode Outlook-only clients.

**Includes:** Provider ports, capability matrix (read/draft/send/calendar), configuration without hardcoding credentials in code.

**Exit criteria:** Existing Outlook paths call through abstraction; new providers can be registered without rewriting core LEA flows.

### M3 — Gmail read support

**Outcome:** Read-only Gmail via the abstraction (scopes and tokens under proper env config, not in repo).

**Includes:** OAuth/read path, pagination bounds, error surfaces, tests that do not require live secrets in CI where possible.

**Exit criteria:** Staged or mocked verification of list/read; no send path enabled by this milestone alone.

### M4 — Draft-only approval gates

**Outcome:** Outbound write operations (compose/send/calendar mutate where applicable) are draft-first and require explicit approval gates.

**Includes:** Draft storage, approval/reject workflow, audit trail of who approved what.

**Exit criteria:** No silent auto-send from executive agents by default; gated path is the only production write path until policy expands.

### M5 — Secure conversation memory

**Outcome:** Conversation/history storage respects auth boundaries and retention expectations.

**Includes:** Per-owner isolation, server-side access control, clear separation of personal vs shared memory if both exist.

**Exit criteria:** Cannot read another user’s memory with a forged client id; retention/export policy noted in docs or config surface.

### M6 — File / document library

**Outcome:** Owned document library for executive workflows (upload, index, retrieve) with auth and size policies.

**Includes:** Storage backend choice, metadata model, access control, basic search/list.

**Exit criteria:** Authenticated owner can manage their library; unauthenticated access denied.

### M7 — Executive daily briefing

**Outcome:** Daily briefing generation that pulls only from allowed sources under auth and draft/send policy.

**Includes:** Scheduling or on-demand generation, summary content model, delivery channel policy (in-app first preferred).

**Exit criteria:** Briefing works under M1 + provider rules; no live send without M4 gates if email delivery is used.

### M8 — Health / weight module

**Outcome:** Health/weight journey module aligned with existing product docs, under hardened auth and memory rules.

**Includes:** Data model, privacy sensitivity, alignment with `lea-health-weight-journey.md` where still valid.

**Exit criteria:** Module usable by authenticated owner; PHI-adjacent data not exposed cross-user.

### M9 — Staging deployment

**Outcome:** DreAgent candidate running at staging path `/home/codrex/lea-main` with required env present and smoke checks documented.

**Includes:** Deploy permission explicit, health checks, auth smoke, provider config verification.

**Exit criteria:** Staging smoke suite passes; live domain still Hummingbird.

### M10 — Live cutover decision

**Outcome:** Explicit go/no-go on pointing `lea.codre-x.com` (or successor) at DreAgent LEA v2.

**Includes:** Staging sign-off, rollback plan, Hummingbird freeze/export if needed, DNS/cPanel owner approval.

**Exit criteria:** Written go/no-go. **No cutover by default.** Foundation work must not imply M10 is approved.

## Milestone dependency sketch

```text
M1 → M2 → M3
M1 → M5
M2 → M4
M1 + M2 + M4 + M5 → M6 / M7 / M8 (can parallelize carefully)
M1–M8 sufficient quality → M9 staging
M9 green + deliberate decision → M10 only
```

## Current phase

**Docs + planning only** (`feature/lea-v2-foundation-hardening`). No application code changes in this phase.
