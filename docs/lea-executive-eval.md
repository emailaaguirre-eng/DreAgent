# Lea Executive evaluation checklist (Smart LEA v1)

Offline automated checks:

```bash
npm run test:lea-executive
```

v1 scope:
- Smarter executive planning/drafts, RAG/web, mode UX
- Live inbox optional via **connected email provider** (Outlook supported if configured; Gmail planned, not implemented)
- **No** Graph email send or calendar create from chat
- **No** conversation save/load UI (API route left as pre-existing)
- Provider-agnostic product language (Outlook is not required for LEA to be useful)

Manual product-path checks:

1. Default mode is Lea Executive (or last-used mode); capability chips do **not** imply Outlook-only.
2. Grant empty-state suggestions never claim live inbox access.
3. Without a connected provider, inbox check fails soft; Lea still drafts/plans and mentions Gmail is planned if relevant.
4. With Outlook configured (legacy): “Check my inbox” uses last-7-day default + previews.
5. Draft email / “send it” must **not** send — explain sending not enabled.
6. Draft calendar / “create it” must **not** create — explain create not enabled.
7. Empty RAG/web status is disclosed; Lea still drafts from user text.
8. Trust filter still rejects random commercial domains (script covers domain helpers).
9. Chat does not call `/api/conversations` for save/load in the UI.

Deferred roadmap (docs only, not in product yet):
- Multi-provider email (Gmail OAuth)—see architecture notes in conversation history
- [`docs/lea-wellness-support.md`](lea-wellness-support.md) — LEA Wellness / Health Journey Support
