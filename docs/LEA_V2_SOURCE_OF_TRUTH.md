# LEA v2 Source of Truth

## Selected future path

**DreAgent `main` is the selected LEA v2 product foundation.**

All future LEA v2 product work should land in the DreAgent repository unless an explicit exception is approved in writing (and still must not imply live cutover without a separate deploy decision).

## Live legacy vs future foundation

| Surface | Role | Status |
| --- | --- | --- |
| Hummingbird at `/home/codrex/lea` | Live production LEA | **Live legacy** — remain online; do not cut over from this doc work |
| `lea.codre-x.com` | Public domain | Currently serves **Hummingbird**, **not** DreAgent |
| `/home/codrex/lea-main` | Server staging candidate | Staging path for DreAgent; not live |
| DreAgent `main` | LEA v2 future path | Selected foundation for ongoing product work |

## Stale clones (do-not-deploy)

The following are **not** deploy targets and must not be treated as the source of truth for releases:

- Stale local clones that are not the approved target path
- iCloud DreAgent copies
- `dreagent-cloud` (unless a prompt explicitly nominates that repo **and** grants deploy permission)
- `C:\Users\email\Projects\lea-vnext`
- Any tree that is not the agreed DreAgent working path for the branch in question

Deploying from a stale clone has caused confusion. Prefer the explicit target path named in each prompt.

## Deployment policy (default)

- **No live cutover** from this foundation phase.
- Live Hummingbird remains live until a deliberate, separate cutover decision (see milestone M10).
- Staging may only proceed when a prompt explicitly allows deployment **and** names the correct staging path.
- Missing server `.env` / secrets on staging are a go/no-go gate, not something to invent from docs alone.

## Required prompt header (every future LEA prompt)

Every future LEA-related prompt **must** include all of the following:

1. **Target repo** (e.g. DreAgent)
2. **Target local path** (absolute path to the working tree)
3. **Target branch** (name to create or use)
4. **Deployment allowed** (`yes` / `no`, and if yes: which environment only)

Example:

```text
Target repo: DreAgent
Target local path: C:\Users\email\Projects\DreAgent-lea-main
Target branch: feature/lea-auth-and-owner-session
Deployment allowed: no
```

If any of these four fields is missing, stop and ask before code or ops work.

## Explicit do-not-touch (unless a prompt overrides with deploy permission)

- `/home/codrex/lea` (live Hummingbird)
- Live `lea.codre-x.com` routing/content
- `/home/codrex/lea-main` (server) without explicit staging deploy permission
- `C:\Users\email\Projects\lea-vnext`
- `dreagent-cloud`
- iCloud DreAgent
- Any `.env` / secret files
- PM2 process management without explicit ops permission
- cPanel routing without explicit ops permission

## How to use this doc

1. Confirm the prompt’s target matches this source of truth.
2. Prefer docs + planning branches before application changes when the foundation story is unclear.
3. Treat Hummingbird as continuity; treat DreAgent as the build surface.
4. Never assume domain cutover from “foundation” or “staging green.”
