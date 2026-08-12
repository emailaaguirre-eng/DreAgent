// DreAgent Cloud - Offline evaluation checklist for Lea Executive (Smart LEA v1)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Run: npm run test:lea-executive

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  detectExecutiveIntent,
  hasCalendarDraftArtifact,
  hasEmailDraftArtifact,
  parseCalendarDraftFromText,
  parseEmailDraftFromText,
  parseExecutiveParams,
  shouldUseWebSearch,
  truncatePreview,
} from '../src/lib/ai/executive';
import { MODES } from '../src/lib/ai/prompts';
import { isReliableDomainForTests } from '../src/lib/tools/web-search';
import {
  createOwnerSessionToken,
  getConfiguredOwnerId,
  isOwnerSessionConfigured,
  isValidOwnerBootstrapSecret,
  OWNER_SESSION_COOKIE,
  resolveTrustedOwnerId,
  verifyOwnerSessionToken,
} from '../src/lib/auth/owner-session';
import {
  getProvider,
  isProviderRegistered,
  listRegisteredProviders,
  outlookProvider,
  READ_ONLY_LEA_CAPABILITIES,
} from '../src/lib/providers';

function section(name: string) {
  console.log(`\n== ${name} ==`);
}

section('Mode matrix');
assert.equal(MODES.general.name, 'Grant');
assert.equal(MODES.executive.name, 'Lea');
assert.equal(MODES.executive.model, 'gpt-4o');
assert.equal(MODES['it-support'].name, 'Chiquis');
// Specialty modes remain configured server-side but are not separate product agents in UI.
assert.ok(MODES.legal && MODES.finance && MODES.research && MODES.incentives);
assert.ok(
  MODES.general.systemPrompt.includes('do not have a connected email provider') ||
    MODES.general.systemPrompt.toLowerCase().includes('connected email'),
  'Grant should not claim live mailbox provider access'
);
assert.ok(
  MODES.executive.systemPrompt.includes('Gmail support is planned'),
  'Lea should note Gmail is planned, not live'
);
assert.ok(
  MODES.executive.systemPrompt.includes('Outlook is currently supported'),
  'Lea should note Outlook as optional/current provider'
);
assert.ok(
  MODES.executive.systemPrompt.includes('Email draft'),
  'Lea should document draft format'
);
assert.ok(
  MODES.executive.systemPrompt.includes('sending is not enabled') ||
    MODES.executive.systemPrompt.includes('not enabled in Smart LEA v1'),
  'Lea should state send/create not enabled'
);
assert.ok(
  MODES.executive.systemPrompt.includes('one assistant') ||
    MODES.executive.systemPrompt.includes('Introduce yourself simply as Lea'),
  'Lea should present as a single assistant'
);

section('Visible product agents');
const ui = readFileSync(join(__dirname, '../src/components/chat/mode-selector.tsx'), 'utf8');
assert.ok(ui.includes('VISIBLE_AGENT_MODES'), 'Mode selector uses visible agent list');
assert.ok(!ui.includes('Object.values(MODES)'), 'Mode selector no longer lists all MODES');
const chatUi = readFileSync(join(__dirname, '../src/components/chat/chat-interface.tsx'), 'utf8');
assert.ok(chatUi.includes('normalizeToVisibleMode'), 'Chat normalizes specialty modes to Lea');
assert.ok(
  chatUi.includes("I'm Lea, your executive assistant") ||
    chatUi.includes('I can help with planning, drafting, research'),
  'Lea empty-state identity copy present'
);
assert.ok(
  chatUi.includes('Now / Next / Later') ||
    chatUi.includes("I'm overwhelmed"),
  'Lea empty state offers overwhelm / triage entry'
);
assert.ok(
  MODES.executive.systemPrompt.includes('project-aware') ||
    MODES.executive.systemPrompt.includes('Now / Next / Later'),
  'Lea prompt includes project-aware / triage framing'
);
assert.ok(
  MODES.executive.systemPrompt.includes('overwhelmed') ||
    MODES.executive.systemPrompt.includes('When the user is overwhelmed'),
  'Lea prompt includes overwhelm next-step behavior'
);
assert.ok(
  MODES.executive.systemPrompt.includes('Never invent access to files') ||
    MODES.executive.systemPrompt.includes('health records'),
  'Lea prompt forbids inventing unconnected systems'
);

section('Executive intent routing');
assert.equal(detectExecutiveIntent('check my inbox'), 'email_summary');
assert.equal(detectExecutiveIntent("what's on my calendar this week"), 'calendar_summary');
assert.equal(
  detectExecutiveIntent('export email history as csv'),
  'email_history_export'
);
assert.equal(
  detectExecutiveIntent('draft a follow-up email to the client'),
  'draft_email'
);
assert.equal(
  detectExecutiveIntent('schedule a meeting with the partner next Tuesday'),
  'draft_calendar_event'
);

const draftEmail = `## Email draft
To: someone@example.com
Subject: Follow up
Body:
Hello — checking in.`;
assert.equal(
  detectExecutiveIntent('send it', draftEmail),
  'confirm_send_email',
  'detect confirm intent so UI/model can soft-block writes'
);

const draftCal = `## Calendar event draft
Subject: Planning sync
Start: 2026-08-10T15:00:00
End: 2026-08-10T15:30:00`;
assert.equal(
  detectExecutiveIntent('create it', draftCal),
  'confirm_create_calendar'
);

section('Chat route must not Graph-send or Graph-create in v1');
const chatRoute = readFileSync(
  join(__dirname, '../src/app/api/chat/route.ts'),
  'utf8'
);
assert.equal(
  /sendEmail|createCalendarEvent/.test(chatRoute),
  false,
  'chat route must not import or call Graph send/create in Smart LEA v1'
);
assert.ok(
  chatRoute.includes('write_disabled') || chatRoute.includes('draft_only'),
  'chat route should surface draft-only / write_disabled messaging'
);

section('Chat UI must not wire conversation persistence in v1');
assert.equal(
  /\/api\/conversations/.test(chatUi),
  false,
  'chat UI must not call /api/conversations until ownership IDOR is fixed'
);

section('Default email window assumptions');
const params = parseExecutiveParams('check my inbox');
assert.equal(params.usedDefaultEmailWindow, true);
assert.ok(params.startDate);
assert.ok(params.endDate);

const paramsDays = parseExecutiveParams('check my email for the last 3 days');
assert.equal(paramsDays.usedDefaultEmailWindow, false);
assert.equal(paramsDays.daysBehind, 3);

section('Draft parsers and artifacts (helpers only; no Graph write)');
assert.equal(hasEmailDraftArtifact(draftEmail), true);
assert.equal(hasCalendarDraftArtifact(draftCal), true);
const parsedMail = parseEmailDraftFromText(draftEmail);
assert.ok(parsedMail);
assert.deepEqual(parsedMail?.to, ['someone@example.com']);
assert.equal(parsedMail?.subject, 'Follow up');
const parsedCal = parseCalendarDraftFromText(draftCal);
assert.ok(parsedCal);
assert.equal(parsedCal?.subject, 'Planning sync');

section('Web search triggers');
assert.equal(shouldUseWebSearch('hello there'), false);
assert.equal(shouldUseWebSearch('latest news on trade'), true);
assert.equal(shouldUseWebSearch('who is the governor', 'executive'), true);

section('Preview truncation / grounding helpers');
assert.equal(truncatePreview('short'), 'short');
assert.ok(truncatePreview('x'.repeat(300)).endsWith('…'));

section('Trust filter still blocks random commercial hosts');
assert.equal(isReliableDomainForTests('example.com'), false);
assert.equal(isReliableDomainForTests('reuters.com'), true);
assert.equal(isReliableDomainForTests('cdc.gov'), true);

section('Owner session foundation (M1)');
// Save/restore env so offline checks do not depend on the host machine secrets.
const prevSecret = process.env.LEA_OWNER_SESSION_SECRET;
const prevOwner = process.env.LEA_OWNER_ID;
process.env.LEA_OWNER_SESSION_SECRET = 'offline-test-owner-session-secret';
process.env.LEA_OWNER_ID = 'lea-owner';

assert.equal(getConfiguredOwnerId(), 'lea-owner');
assert.equal(isOwnerSessionConfigured(), true);

const token = createOwnerSessionToken('lea-owner');
const verified = verifyOwnerSessionToken(token);
assert.equal(verified.ok, true);
if (verified.ok) {
  assert.equal(verified.payload.ownerId, 'lea-owner');
}

const badSig = token.slice(0, -4) + 'xxxx';
assert.equal(verifyOwnerSessionToken(badSig).ok, false);

const forgedOnly = resolveTrustedOwnerId(
  new Request('http://localhost/api/knowledge', {
    headers: { 'x-user-id': 'forged-user-123' },
  }),
  { untrustedClientUserId: 'forged-user-123' }
);
assert.equal(forgedOnly.ok, false, 'forged client userId alone must fail closed');
assert.ok(
  forgedOnly.source === 'untrusted_client_rejected' ||
    forgedOnly.source === 'none' ||
    forgedOnly.source === 'invalid_session',
  'forged client identity rejected'
);

const withCookie = resolveTrustedOwnerId(
  new Request('http://localhost/api/knowledge', {
    headers: {
      cookie: `${OWNER_SESSION_COOKIE}=${token}`,
    },
  })
);
assert.equal(withCookie.ok, true);
if (withCookie.ok) assert.equal(withCookie.ownerId, 'lea-owner');

const mismatch = resolveTrustedOwnerId(
  new Request('http://localhost/api/knowledge', {
    headers: {
      cookie: `${OWNER_SESSION_COOKIE}=${token}`,
      'x-user-id': 'other-user',
    },
  }),
  { untrustedClientUserId: 'other-user' }
);
assert.equal(mismatch.ok, false, 'mismatched client userId must fail closed');
assert.equal(mismatch.source, 'owner_mismatch');

assert.equal(
  isValidOwnerBootstrapSecret('offline-test-owner-session-secret'),
  true
);
assert.equal(isValidOwnerBootstrapSecret('wrong-secret'), false);

// Restore env
if (prevSecret === undefined) delete process.env.LEA_OWNER_SESSION_SECRET;
else process.env.LEA_OWNER_SESSION_SECRET = prevSecret;
if (prevOwner === undefined) delete process.env.LEA_OWNER_ID;
else process.env.LEA_OWNER_ID = prevOwner;

section('Sensitive routes must not trust client userId alone');
const knowledgeRoute = readFileSync(
  join(__dirname, '../src/app/api/knowledge/route.ts'),
  'utf8'
);
const conversationsRoute = readFileSync(
  join(__dirname, '../src/app/api/conversations/route.ts'),
  'utf8'
);
const tokensLib = readFileSync(
  join(__dirname, '../src/lib/outlook/tokens.ts'),
  'utf8'
);
assert.ok(
  knowledgeRoute.includes('requireTrustedOwner'),
  'knowledge route must require trusted owner'
);
assert.ok(
  conversationsRoute.includes('requireTrustedOwner'),
  'conversations route must require trusted owner'
);
assert.ok(
  tokensLib.includes('resolveTrustedOwnerId'),
  'Outlook token resolver must use trusted owner for DB path'
);
assert.ok(
  !tokensLib.includes('if (explicitUserId?.trim()) return explicitUserId.trim()'),
  'resolveUserId must not treat explicit client userId as authoritative owner'
);
assert.ok(
  chatRoute.includes('resolveTrustedOwnerId') ||
    chatRoute.includes('owner_session_required'),
  'chat route must not treat client userId as sole RAG/identity authority'
);
assert.ok(
  chatUi.includes("credentials: 'same-origin'") ||
    !chatUi.includes("'x-user-id'"),
  'chat UI must not send x-user-id as trusted identity header'
);
assert.ok(
  !chatUi.includes('body: { mode, enableRag: true, userId }'),
  'chat UI must not send userId as authoritative body identity'
);

section('Provider abstraction foundation (M2)');
const registered = listRegisteredProviders();
assert.equal(registered.length, 1, 'only Outlook is registered in M2');
assert.equal(registered[0].id, 'outlook');
assert.equal(isProviderRegistered('outlook'), true);
assert.equal(isProviderRegistered('gmail'), false, 'Gmail adapter must not be registered yet');
assert.equal(getProvider('gmail'), null);

assert.equal(outlookProvider.id, 'outlook');
assert.equal(outlookProvider.capabilities.mailRead, true);
assert.equal(outlookProvider.capabilities.calendarRead, true);
assert.equal(outlookProvider.capabilities.mailExport, true);
assert.equal(
  outlookProvider.capabilities.mailSend,
  false,
  'LEA provider port must not enable send'
);
assert.equal(
  outlookProvider.capabilities.calendarWrite,
  false,
  'LEA provider port must not enable calendar write'
);
assert.equal(READ_ONLY_LEA_CAPABILITIES.mailSend, false);
assert.equal(READ_ONLY_LEA_CAPABILITIES.calendarWrite, false);

assert.ok(
  typeof outlookProvider.listMail === 'function' &&
    typeof outlookProvider.listCalendar === 'function' &&
    typeof outlookProvider.getConnection === 'function',
  'Outlook adapter exposes read/status methods'
);
assert.equal(
  'sendMail' in outlookProvider || 'createEvent' in outlookProvider,
  false,
  'Outlook adapter must not expose write methods on the LEA port'
);

assert.ok(
  chatRoute.includes('resolveConnectedProvider'),
  'chat must resolve a provider-neutral connected adapter'
);
assert.ok(
  !chatRoute.includes("from '@/lib/outlook/client'"),
  'chat must not import Outlook Graph client directly'
);
assert.ok(
  !chatRoute.includes('resolveOutlookAccessToken'),
  'chat must not call Outlook token helper directly (adapter owns that)'
);
assert.equal(
  /sendEmail|createCalendarEvent/.test(chatRoute),
  false,
  'provider work must not introduce chat mail/calendar writes'
);

const providerTypes = readFileSync(
  join(__dirname, '../src/lib/providers/types.ts'),
  'utf8'
);
assert.ok(providerTypes.includes('mailRead'), 'capabilities include mailRead');
assert.ok(providerTypes.includes('calendarRead'), 'capabilities include calendarRead');
assert.ok(
  !providerTypes.includes('sendEmail') && !providerTypes.includes('createCalendarEvent'),
  'provider types must not include Graph write APIs'
);

console.log('\nAll Smart LEA v1 offline checks passed.');
