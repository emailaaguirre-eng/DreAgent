// DreAgent Cloud - Streaming Chat API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { NextRequest } from 'next/server';
import { getSystemPrompt, getModeConfig, type AgentMode } from '@/lib/ai/prompts';
import {
  detectExecutiveIntent,
  findLastAssistantContent,
  parseExecutiveParams,
  shouldUseWebSearch,
  truncatePreview,
  type ExecutiveIntent,
} from '@/lib/ai/executive';
import {
  formatRagStatusLine,
  getRelevantContext,
} from '@/lib/rag/query';
import {
  getCalendarEvents,
  getEmails,
  type Email,
} from '@/lib/outlook/client';
import { resolveOutlookAccessToken } from '@/lib/outlook/tokens';
import {
  webSearch,
  filterReliableSearchResults,
  formatSearchResults,
} from '@/lib/tools/web-search';

export const runtime = 'nodejs';
export const maxDuration = 60;

type SupportedModel = 'gpt-4o-mini' | 'gpt-4o' | 'gpt-4-turbo';

const EMAIL_SAMPLE_LIMIT = 20;
const CALENDAR_SAMPLE_LIMIT = 20;

function getLastUserMessage(
  messages: { role: 'user' | 'assistant'; content: string }[]
): string {
  return messages.filter((m) => m.role === 'user').pop()?.content || '';
}

function getAmbiguityPrompt(
  mode: AgentMode,
  userMessage: string,
  intent: ExecutiveIntent
): string | null {
  const text = userMessage.trim().toLowerCase();
  if (!text) {
    return 'I need a bit more detail before I can help. What would you like me to do?';
  }

  const veryShort = text.split(/\s+/).length <= 2;
  const pronounOnlyPattern =
    /(this|that|it|them|those|these)\??$/.test(text) ||
    /^(help|fix|review|summarize)\s+(this|that|it)\??$/.test(text);

  if (veryShort || pronounOnlyPattern) {
    return 'I want to make sure I give one accurate answer. Can you clarify exactly what item, timeframe, and output format you want?';
  }

  if (mode === 'executive') {
    if (intent === 'email_history_export') {
      const hasFormatHint = /csv|excel|xlsx|powerpoint|pptx/.test(text);
      if (!hasFormatHint) {
        return 'I can export this accurately, but I need one format choice: CSV (Excel-compatible) or PowerPoint (.pptx). Which one do you want?';
      }
    }

    if (intent === 'confirm_send_email') {
      // handled in action path if draft missing
    }
  }

  return null;
}

function getModelFallbackOrder(primary: SupportedModel): SupportedModel[] {
  const ordered: SupportedModel[] = [primary];
  const fallbacks: SupportedModel[] = ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'];
  for (const model of fallbacks) {
    if (!ordered.includes(model)) {
      ordered.push(model);
    }
  }
  return ordered;
}

function filterEmails(
  emails: Email[],
  params: ReturnType<typeof parseExecutiveParams>
): Email[] {
  let list = emails;
  if (params.senderHint) {
    const hint = params.senderHint.toLowerCase();
    list = list.filter(
      (e) =>
        e.from.toLowerCase().includes(hint) ||
        e.fromName.toLowerCase().includes(hint)
    );
  }
  if (params.subjectHint) {
    const hint = params.subjectHint.toLowerCase();
    list = list.filter((e) => e.subject.toLowerCase().includes(hint));
  }
  if (params.searchKeyword) {
    const hint = params.searchKeyword.toLowerCase();
    list = list.filter(
      (e) =>
        e.subject.toLowerCase().includes(hint) ||
        e.preview.toLowerCase().includes(hint) ||
        e.from.toLowerCase().includes(hint) ||
        e.fromName.toLowerCase().includes(hint)
    );
  }
  return list;
}

function formatEmailSamples(emails: Email[]): string {
  const sample = emails.slice(0, EMAIL_SAMPLE_LIMIT);
  if (sample.length === 0) return 'No emails matched the requested parameters.';
  return sample
    .map((email, index) => {
      const preview = truncatePreview(email.preview || '');
      return `${index + 1}. [${email.isRead ? 'read' : 'UNREAD'}] ${email.subject}\n   From: ${email.fromName || email.from} <${email.from}>\n   Received: ${email.received}\n   Preview: ${preview || '(no preview)'}`;
    })
    .join('\n\n');
}

async function buildExecutiveActionContext(
  req: NextRequest,
  intent: ExecutiveIntent,
  lastUserMessage: string,
  userId?: string,
  outlookAccessToken?: string
): Promise<string> {
  // v1 Smart LEA: draft-only for outbound actions — never send mail or create calendar events.
  if (intent === 'draft_email') {
    return `Lea should draft an email only. Sending is not enabled in this version.
Action status: draft_only
Required draft format: ## Email draft with To, Subject, and Body fields.
User-facing: make clear you can draft this, but sending is not enabled yet.
Do not claim the email was sent. Never imply a mailbox send completed.`;
  }
  if (intent === 'draft_calendar_event') {
    return `Lea should draft a calendar event only. Creating events on a connected calendar provider is not enabled in this version.
Action status: draft_only
Required draft format: ## Calendar event draft with Subject, Start, End fields.
User-facing: make clear you can draft this, but creating calendar events is not enabled yet.
Do not claim the event was created. Never imply a calendar write completed.`;
  }
  if (intent === 'confirm_send_email') {
    return `User asked to send an email. Send is disabled for Smart LEA v1.
Action status: write_disabled
User-facing must say: "I can draft this, but sending/creating is not enabled yet."
Do not claim an email was sent. Offer to refine the email draft instead.`;
  }
  if (intent === 'confirm_create_calendar') {
    return `User asked to create a calendar event. Calendar create is disabled for Smart LEA v1.
Action status: write_disabled
User-facing must say: "I can draft this, but sending/creating is not enabled yet."
Do not claim a calendar event was created. Offer to refine the event draft instead.`;
  }
  if (intent === 'none') {
    return '';
  }

  const resolved = await resolveOutlookAccessToken({ req, userId });
  const accessToken = outlookAccessToken || resolved.accessToken;

  if (!accessToken) {
    return `Lea could not complete a live mail/calendar check because no connected email/calendar provider is available for this user.
Action status: failed
Provider note: Outlook is currently supported if configured; Gmail support is planned and not implemented yet.
User-facing: do not invent inbox/calendar contents. Stay useful with planning and drafts from user text or pasted messages. Explain that live access needs a connected email provider (and calendar provider when relevant).
Legacy connect path if Outlook is already set up for this deployment: /api/outlook/auth?userId=<your-user-id>`;
  }

  const params = parseExecutiveParams(lastUserMessage);

  if (intent === 'email_summary') {
    const emails = await getEmails(accessToken, {
      folder: 'inbox',
      limit: params.limit,
      unreadOnly: params.unreadOnly,
      startDate: params.startDate,
      endDate: params.endDate,
    });
    const filtered = filterEmails(emails, params);
    const assumptions: string[] = [];
    if (params.usedDefaultEmailWindow) {
      assumptions.push(
        'Assumed email window: last 7 days ending now (user did not specify dates).'
      );
    }
    if (params.unreadOnly) {
      assumptions.push('Filtered to unread messages only.');
    }

    return `Lea completed a connected email-provider inbox check (current implementation: Outlook/Microsoft Graph when configured).
Action status: success
Provider: outlook (legacy convenience if configured)
Filters: ${JSON.stringify({
  folder: 'inbox',
  limit: params.limit,
  unreadOnly: params.unreadOnly,
  startDate: params.startDate || null,
  endDate: params.endDate,
  senderHint: params.senderHint || null,
  subjectHint: params.subjectHint || null,
  searchKeyword: params.searchKeyword || null,
})}
Email count returned from provider: ${emails.length}
Email count after local filters: ${filtered.length}
Sample size shown: ${Math.min(filtered.length, EMAIL_SAMPLE_LIMIT)} of ${filtered.length} (truncated if larger)
${assumptions.length ? `Assumptions:\n- ${assumptions.join('\n- ')}\n` : ''}
Sample results:
${formatEmailSamples(filtered)}`;
  }

  if (intent === 'calendar_summary') {
    const events = await getCalendarEvents(accessToken, {
      daysAhead: params.daysAhead,
      daysBehind: params.daysBehind,
    });
    const assumptions: string[] = [];
    if (params.usedDefaultCalendarWindow) {
      assumptions.push(
        `Assumed calendar window: ${params.daysBehind} day(s) behind and ${params.daysAhead} day(s) ahead.`
      );
    }
    const sample = events.slice(0, CALENDAR_SAMPLE_LIMIT);
    const sampleText =
      sample.length === 0
        ? 'No events matched the requested parameters.'
        : sample
            .map(
              (event, index) =>
                `${index + 1}. ${event.subject}\n   When: ${event.start} -> ${event.end}\n   Location: ${event.location || 'No location'}\n   Organizer: ${event.organizer || 'n/a'}`
            )
            .join('\n\n');

    return `Lea completed a connected calendar-provider check (current implementation: Outlook/Microsoft Graph when configured).
Action status: success
Provider: outlook (legacy convenience if configured)
Filters: ${JSON.stringify({
  daysAhead: params.daysAhead,
  daysBehind: params.daysBehind,
})}
Event count returned: ${events.length}
Sample size shown: ${Math.min(events.length, CALENDAR_SAMPLE_LIMIT)} of ${events.length}
${assumptions.length ? `Assumptions:\n- ${assumptions.join('\n- ')}\n` : ''}
Sample results:
${sampleText}`;
  }

  if (intent === 'email_history_export') {
    const emails = await getEmails(accessToken, {
      folder: 'inbox',
      limit: params.limit,
      unreadOnly: params.unreadOnly,
      startDate: params.startDate,
      endDate: params.endDate,
    });

    let calendarCount = 0;
    if (params.includeCalendar) {
      const events = await getCalendarEvents(accessToken, {
        daysAhead: params.daysAhead,
        daysBehind: params.daysBehind,
      });
      calendarCount = events.length;
    }

    const exportParams = new URLSearchParams({
      folder: 'inbox',
      limit: String(params.limit),
      unread_only: String(params.unreadOnly),
      include_calendar: String(params.includeCalendar),
      days_behind: String(params.daysBehind),
      days_ahead: String(params.daysAhead),
    });
    if (userId) exportParams.set('userId', userId);
    if (params.startDate) exportParams.set('start_date', params.startDate);
    if (params.endDate) exportParams.set('end_date', params.endDate);

    return `Lea prepared an email-history CSV export request via the currently connected email provider path (Outlook endpoint when configured).
Action status: ready
Provider: outlook (legacy convenience if configured)
Records to export now: ${emails.length} emails${params.includeCalendar ? `, ${calendarCount} calendar events` : ''}
Export endpoint: /api/outlook/email-history?${exportParams.toString()}
UI: Lea Executive footer buttons "Download Email CSV" / "Download Email + Calendar CSV" use folder=inbox, limit=200, days_behind=30 (email+calendar path days_ahead=30).
Note: Use the same authenticated session or provide Authorization: Bearer <access_token> when downloading. Gmail export is not implemented yet.`;
  }

  return '';
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      messages,
      mode = 'executive',
      userId,
      enableRag = true,
      enableWebSearch = true,
      outlookAccessToken,
    } = body as {
      messages: { role: 'user' | 'assistant'; content: string }[];
      mode?: AgentMode;
      userId?: string;
      enableRag?: boolean;
      enableWebSearch?: boolean;
      outlookAccessToken?: string;
    };

    if (!messages || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: 'No messages provided' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const lastUserMessage = getLastUserMessage(messages);
    const recentAssistant = findLastAssistantContent(messages);
    const executiveIntent =
      mode === 'executive'
        ? detectExecutiveIntent(lastUserMessage, recentAssistant)
        : 'none';

    const ambiguityPrompt = getAmbiguityPrompt(
      mode,
      lastUserMessage,
      executiveIntent
    );

    const modeConfig = getModeConfig(mode);

    let ragContext = '';
    let ragStatusBlock = '';
    if (enableRag && lastUserMessage) {
      const ragOpts =
        mode === 'executive'
          ? { threshold: 0.65, limit: 5 }
          : { threshold: 0.72, limit: 3 };
      const rag = await getRelevantContext(lastUserMessage, userId, ragOpts);
      ragContext = rag.context;
      ragStatusBlock = formatRagStatusLine(rag);
    }

    let webContext = '';
    if (
      enableWebSearch &&
      lastUserMessage &&
      shouldUseWebSearch(lastUserMessage, mode)
    ) {
      const rawSearch = await webSearch(lastUserMessage);
      const reliableSearch = filterReliableSearchResults(rawSearch, 5);
      webContext = formatSearchResults(reliableSearch);
      if (!webContext) {
        webContext =
          '## Web Search Results\nNo reliable web evidence passed domain filters for this query.\nGuidance: For external factual claims, say they are unverified. You may still help with process, drafts, planning, and any verified mail/calendar provider knowledge/user content without inventing sources.';
      }
    }

    let executiveActionContext = '';
    if (mode === 'executive' && lastUserMessage && !ambiguityPrompt) {
      executiveActionContext = await buildExecutiveActionContext(
        req,
        executiveIntent,
        lastUserMessage,
        userId,
        outlookAccessToken
      );
    } else if (mode === 'executive' && lastUserMessage && executiveIntent !== 'none' && ambiguityPrompt) {
      // still allow draft / write-disabled guidance without other high-stakes actions
      if (
        executiveIntent === 'draft_email' ||
        executiveIntent === 'draft_calendar_event' ||
        executiveIntent === 'confirm_send_email' ||
        executiveIntent === 'confirm_create_calendar'
      ) {
        executiveActionContext = await buildExecutiveActionContext(
          req,
          executiveIntent,
          lastUserMessage,
          userId,
          outlookAccessToken
        );
      }
    }

    const systemPrompt = getSystemPrompt(mode, ragContext);
    let systemPromptWithContext = systemPrompt;

    if (ragStatusBlock) {
      systemPromptWithContext += `\n\n## Knowledge retrieval status\n${ragStatusBlock}`;
    }

    if (ambiguityPrompt) {
      systemPromptWithContext += `\n\n## Clarification Required\n\nThe current user request is ambiguous in a high-stakes way. Ask this clarification question first and do not provide a final high-stakes answer yet:\n"${ambiguityPrompt}"`;
    }
    if (webContext) {
      systemPromptWithContext += `\n\n## Verified Web Context\n\n${webContext}\n\nUse only these web sources for web-backed claims in this response.`;
    }
    if (executiveActionContext) {
      systemPromptWithContext += `\n\n## Verified Mail/Calendar Action Context\n\n${executiveActionContext}\n\nUse this verified action context for your response. Do not claim any mail or calendar action succeeded unless it appears in this context. Live mailbox access requires a connected email provider; Outlook is currently supported if configured and Gmail support is planned (not implemented yet).`;
    }

    const candidateModels = getModelFallbackOrder(modeConfig.model);
    let lastError: unknown = null;
    let result: Awaited<ReturnType<typeof streamText>> | null = null;

    for (const modelName of candidateModels) {
      try {
        result = await streamText({
          model: openai(modelName),
          system: systemPromptWithContext,
          messages,
          temperature: modeConfig.temperature,
          maxTokens: 4096,
        });
        break;
      } catch (error) {
        lastError = error;
        console.warn(`Model ${modelName} failed, trying fallback model.`);
      }
    }

    if (!result) {
      throw new Error(
        lastError instanceof Error
          ? `All model attempts failed: ${lastError.message}`
          : 'All model attempts failed'
      );
    }

    return result.toDataStreamResponse();
  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : 'Internal server error',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
