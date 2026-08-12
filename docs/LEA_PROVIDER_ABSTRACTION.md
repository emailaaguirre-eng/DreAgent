# LEA Provider Abstraction (M2)

## Current provider model

LEA executive mail/calendar access goes through a **read-only provider port**, not directly through Microsoft Graph types in chat.

```text
Chat / executive intents
        │
        ▼
src/lib/providers  (MailCalendarProvider)
        │
        ▼
Outlook adapter  ──►  src/lib/outlook/*  (Graph client + token store)
```

- **Registry** (`src/lib/providers/registry.ts`) lists registered adapters.
- **Resolve** (`resolveConnectedProvider`) returns the first connected provider or `null` (fail closed).
- **M1 owner session** still gates Outlook **DB token** lookup. The adapter does not reintroduce client `userId` authority.
- **Writes are not on the port.** Send/create remain Outlook-specific HTTP routes and are unused by chat.

## Outlook as the first adapter

`outlookProvider` (`src/lib/providers/outlook-adapter.ts`):

| Capability (LEA port) | Value | Notes |
| --- | --- | --- |
| `mailRead` | true | Wraps `getEmails` |
| `calendarRead` | true | Wraps `getCalendarEvents` |
| `mailExport` | true | Count/preview via list; CSV still `/api/outlook/email-history` |
| `mailSend` | **false** | Graph send exists only on Outlook API POST, not this port |
| `calendarWrite` | **false** | Graph create exists only on Outlook API POST, not this port |

Token storage (`outlook_tokens`, `resolveOutlookAccessToken`) is unchanged except that chat no longer calls it directly.

## Gmail deferred to M3

- `ProviderId` includes `'gmail'` as a reserved id.
- **No Gmail adapter is registered.**
- `isProviderRegistered('gmail')` is false.
- Do not imply Gmail is connected.

## Provider-neutral vs provider-specific

### Neutral (use these in executive/chat)

- `MailMessage`, `CalendarItem`, `MailQuery`, `CalendarQuery`
- `ProviderCapabilities`, `ProviderConnection`, `ProviderId`
- `MailCalendarProvider.listMail` / `listCalendar` / `getConnection`
- Registry helpers

### Specific (stay in Outlook modules/routes)

- Microsoft Graph client, `GraphApiError`, OAuth authorize/token URLs
- `outlook_tokens` table and refresh
- `/api/outlook/*` including CSV export and Graph write POSTs
- Chat UI download URLs that still hit `/api/outlook/email-history`
- Optional body `outlookAccessToken` (legacy Graph Bearer override)

## What remains coupled

- CSV export HTTP path is Outlook-only.
- Chat footer download buttons still call the Outlook export route.
- Outlook OAuth connect UX (`/api/outlook/auth`).
- Graph write helpers (`sendEmail`, `createCalendarEvent`) still exist for the Outlook routes; chat must not call them (Smart LEA draft-only).
- Single registered provider: Outlook. Multi-provider selection UI does not exist.

## Why no live deployment

Hummingbird remains live at `lea.codre-x.com`. This branch is foundation only (`Deployment allowed: no`). M3 can add a Gmail **read** adapter behind the same port without rewriting chat intents.
