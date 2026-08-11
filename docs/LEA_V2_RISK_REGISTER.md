# LEA v2 Risk Register

Risks for building LEA v2 on DreAgent while Hummingbird remains live. Update status as work proceeds; do not treat this as an ops runbook that authorizes deploy.

| ID | Risk | Severity | Likelihood | Why it matters | Mitigation |
| --- | --- | --- | --- | --- | --- |
| R1 | Weak client `userId` / localStorage identity | **Critical** | High | Forged or shared client ids can impersonate an “owner” and access LEA memory, mail, or tools | M1 auth/session hardening; never trust client id alone; server session + owner binding |
| R2 | Outlook-only coupling | High | High | Product and architecture stuck on one mail/calendar stack; Gmail and multi-provider blocked | M2 provider abstraction before/alongside Gmail read (M3) |
| R3 | Missing server `.env` on `lea-main` | High | Medium–High | Staging candidate cannot run real integrations; risky “fix” attempts or wrong clone deploys | Require env inventory checklist; never commit secrets; staging only with deploy permission and verified env |
| R4 | Live domain still points to Hummingbird | High (ops) | Certain today | Confusion: “LEA live” is not DreAgent; accidental code paths or docs can imply cutover | Source of truth doc; prompts require deploy permission; M10 is separate decision |
| R5 | Stale local clones causing deployment confusion | High | Medium | Wrong tree/path ship or wrong branch deployed; live/legacy mixed | Single named local path per prompt; mark iCloud / lea-vnext / other clones do-not-deploy |
| R6 | Disk space pressure (server/local large trees) | Medium | Medium | Failed builds/deploys, partial updates, inability to stage cleanly | Capacity check before M9; prune stale artifacts carefully without touching live Hummingbird |
| R7 | No live cutover until staging passes | Policy / High if violated | Controllable | Premature DNS/cPanel/PM2 changes break live LEA | M9 full smoke; M10 written go/no-go; default Deployment allowed: no |

## Related operational constraints

- Do not edit live Hummingbird at `/home/codrex/lea` as part of DreAgent foundation coding.
- Do not change cPanel routing or PM2 for live without explicit ops permission in the prompt.
- Do not write or rewrite `.env` files from agent prompts that forbid it.
- `lea.codre-x.com` = Hummingbird until M10 says otherwise.

## Open questions (track, do not invent production answers)

- Exact session technology (cookie vs token header) for M1 — decide on coding branch with existing stack.
- Staging URL vs path-only access for `lea-main`.
- Whether Gmail read (M3) uses personal owner OAuth only vs workspace.

## Change log

| Date | Note |
| --- | --- |
| 2026-08-11 | Initial risk register for LEA v2 foundation path on DreAgent |
