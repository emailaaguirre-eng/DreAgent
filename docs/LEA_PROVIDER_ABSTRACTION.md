# LEA Provider Abstraction

## Current provider model

LEA executive mail/calendar access goes through a **read-only provider port**, not directly through Microsoft Graph types in chat.

```text
Chat / executive intents
        │
        ▼
src/lib/providers  (MailCalendarProvider)
        │
        ├─► Outlook adapter  ──►  src/lib/outlook/*  (Graph client + token store)
        └─► Gmail adapter    ──►  foundation only (no live read yet; see LEA_GMAIL_READ.md)
```

- **Registry** (`src/lib/providers/registry.ts`) lists registered adapters (Outlook + Gmail).
- **Resolve** (`resolveConnectedProvider`) returns the first **connected** provider or `null` (fail closed).
- **M1 owner session** still gates Outlook **DB token** lookup. The adapters do not reintroduce client `userId` authority.
- **Writes are not on the port.** Send/create remain Outlook-specific HTTP routes and are unused by chat.

## Outlook as the first live adapter

`outlookProvider` (`src/lib/providers/outlook-adapter.ts`):

| Capability (LEA port) | Value | Notes |
| --- | --- | --- |
| `mailRead` | true | Wraps `getEmails` |
| `calendarRead` | true | Wraps `getCalendarEvents` |
| `mailExport` | true | Count/preview via list; CSV still `/api/outlook/email-history` |
| `mailSend` | **false** | Graph send exists only on Outlook API POST, not this port |
| `calendarWrite` | **false** | Graph create exists only on Outlook API POST, not this port |

## Gmail (M3 foundation)

See [LEA_GMAIL_READ.md](./LEA_GMAIL_READ.md).

- `gmailProvider` is **registered**.
- `getConnection` always returns `connected: false` until OAuth + token store + API read exist.
- Capabilities: all read/export **false**; send/write **false** (`mailRead` stays false until a real safe path exists).
- Do not imply Gmail inbox access is live.

## Provider-neutral vs provider-specific

### Neutral (use these in executive/chat)

- `MailMessage`, `CalendarItem`, `MailQuery`, `CalendarQuery`
- `ProviderCapabilities`, `ProviderConnection`, `ProviderId`
- `MailCalendarProvider.listMail` / `listCalendar` / `getConnection`
- Registry helpers

### Specific (stay in provider modules/routes)

- Microsoft Graph client, `GraphApiError`, OAuth authorize/token URLs
- `outlook_tokens` table and refresh
- `/api/outlook/*` including CSV export and Graph write POSTs
- Chat UI download URLs that still hit `/api/outlook/email-history`
- Optional body `outlookAccessToken` (legacy Graph Bearer override)
- Future Gmail OAuth/API modules under `src/lib/gmail/*` (config status only in M3 foundation)

## What remains coupled

- CSV export HTTP path is Outlook-only.
- Chat footer download buttons still call the Outlook export route.
- Outlook OAuth connect UX (`/api/outlook/auth`).
- Graph write helpers (`sendEmail`, `createCalendarEvent`) still exist for the Outlook routes; chat must not call them (Smart LEA draft-only).
- Multi-provider selection UI does not exist; first connected wins (Outlook today).

## Why no live deployment

Hummingbird remains live at `lea.codre-x.com`. Provider work is foundation only unless a prompt sets **Deployment allowed: yes**.
