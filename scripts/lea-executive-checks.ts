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

console.log('\nAll Smart LEA v1 offline checks passed.');
