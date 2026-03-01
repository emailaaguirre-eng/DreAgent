// DreAgent Cloud - Streaming Chat API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { NextRequest } from 'next/server';
import { getSystemPrompt, getModeConfig, type AgentMode } from '@/lib/ai/prompts';
import { getRelevantContext } from '@/lib/rag/query';
import { getCalendarEvents, getEmails } from '@/lib/outlook/client';
import { resolveOutlookAccessToken } from '@/lib/outlook/tokens';
import {
  webSearch,
  filterReliableSearchResults,
  formatSearchResults,
} from '@/lib/tools/web-search';

export const runtime = 'nodejs'; // Use Node.js for full OpenAI SDK support
export const maxDuration = 60;   // Vercel Pro limit

type ExecutiveIntent = 'email_summary' | 'calendar_summary' | 'email_history_export' | 'none';

function getLastUserMessage(
  messages: { role: 'user' | 'assistant'; content: string }[]
): string {
  return messages.filter((m) => m.role === 'user').pop()?.content || '';
}

function detectExecutiveIntent(input: string): ExecutiveIntent {
  const text = input.toLowerCase();

  if (
    (text.includes('export') || text.includes('download')) &&
    text.includes('csv') &&
    (text.includes('email') || text.includes('history'))
  ) {
    return 'email_history_export';
  }

  if (text.includes('calendar') || text.includes('meeting') || text.includes('schedule')) {
    return 'calendar_summary';
  }

  if (text.includes('email') || text.includes('inbox')) {
    return 'email_summary';
  }

  return 'none';
}

function parseExecutiveParams(input: string): {
  limit: number;
  unreadOnly: boolean;
  includeCalendar: boolean;
  daysAhead: number;
  daysBehind: number;
  startDate?: string;
  endDate?: string;
} {
  const text = input.toLowerCase();
  const limitMatch = text.match(/(?:last|top|limit)\s+(\d{1,4})/);
  const daysMatch = text.match(/last\s+(\d{1,4})\s+days?/);

  const limit = limitMatch ? Math.min(parseInt(limitMatch[1], 10), 500) : 50;
  const unreadOnly = text.includes('unread');
  const includeCalendar = text.includes('calendar') || text.includes('meeting');
  const daysBehind = daysMatch ? Math.min(parseInt(daysMatch[1], 10), 365) : 30;
  const daysAhead = text.includes('next') ? 30 : 7;

  let startDate: string | undefined;
  const endDate = new Date().toISOString();
  if (daysMatch) {
    const start = new Date();
    start.setDate(start.getDate() - daysBehind);
    startDate = start.toISOString();
  }

  return {
    limit,
    unreadOnly,
    includeCalendar,
    daysAhead,
    daysBehind,
    startDate,
    endDate,
  };
}

function shouldUseWebSearch(input: string): boolean {
  const text = input.toLowerCase();
  return (
    text.includes('latest') ||
    text.includes('current') ||
    text.includes('today') ||
    text.includes('recent') ||
    text.includes('news') ||
    text.includes('update') ||
    text.includes('according to')
  );
}

function getAmbiguityPrompt(
  mode: AgentMode,
  userMessage: string
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
    const intent = detectExecutiveIntent(text);
    if (intent === 'email_summary') {
      const hasTimeHint = /last\s+\d+\s+days?|today|yesterday|this week|this month/.test(text);
      if (!hasTimeHint) {
        return 'For an accurate email check, what timeframe should I use (for example: today, last 7 days, or a specific date range)?';
      }
    }

    if (intent === 'email_history_export') {
      const hasFormatHint = /csv|excel|xlsx|powerpoint|pptx/.test(text);
      if (!hasFormatHint) {
        return 'I can export this accurately, but I need one format choice: CSV (Excel-compatible) or PowerPoint (.pptx). Which one do you want?';
      }
    }
  }

  return null;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      messages,
      mode = 'general',
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

    const ambiguityPrompt = getAmbiguityPrompt(mode, lastUserMessage);

    // Get mode configuration
    const modeConfig = getModeConfig(mode);

    // Get RAG context if enabled
    let ragContext = '';
    if (enableRag && lastUserMessage) {
      ragContext = await getRelevantContext(lastUserMessage, userId);
    }

    // Get web context when query likely needs current info
    let webContext = '';
    if (enableWebSearch && lastUserMessage && shouldUseWebSearch(lastUserMessage)) {
      const rawSearch = await webSearch(lastUserMessage);
      const reliableSearch = filterReliableSearchResults(rawSearch, 5);
      webContext = formatSearchResults(reliableSearch);
      if (!webContext) {
        webContext =
          '## Web Search Results\nNo reliable web sources were found for this query. Ask clarifying questions or explain what verified source type is needed.';
      }
    }

    let executiveActionContext = '';
    if (mode === 'executive' && lastUserMessage) {
      const intent = detectExecutiveIntent(lastUserMessage);
      if (intent !== 'none') {
        const resolved = await resolveOutlookAccessToken({ req, userId });
        const accessToken = outlookAccessToken || resolved.accessToken;

        if (!accessToken) {
          executiveActionContext = `Lea could not complete the requested Outlook action because no token is connected for this user.
Action status: failed
Required fix: connect Outlook with /api/outlook/auth?userId=<your-user-id> and retry.`;
        } else {
          const params = parseExecutiveParams(lastUserMessage);

          if (intent === 'email_summary') {
            const emails = await getEmails(accessToken, {
              folder: 'inbox',
              limit: params.limit,
              unreadOnly: params.unreadOnly,
              startDate: params.startDate,
              endDate: params.endDate,
            });

            const sample = emails.slice(0, 5).map((email, index) => (
              `${index + 1}. ${email.subject} | ${email.from} | ${email.received}`
            )).join('\n');

            executiveActionContext = `Lea completed an Outlook email check.
Action status: success
Filters: ${JSON.stringify({
  folder: 'inbox',
  limit: params.limit,
  unreadOnly: params.unreadOnly,
  startDate: params.startDate || null,
  endDate: params.endDate,
})}
Email count returned: ${emails.length}
Sample results:
${sample || 'No emails matched the requested parameters.'}`;
          }

          if (intent === 'calendar_summary') {
            const events = await getCalendarEvents(accessToken, {
              daysAhead: params.daysAhead,
              daysBehind: params.daysBehind,
            });

            const sample = events.slice(0, 5).map((event, index) => (
              `${index + 1}. ${event.subject} | ${event.start} -> ${event.end} | ${event.location || 'No location'}`
            )).join('\n');

            executiveActionContext = `Lea completed an Outlook calendar check.
Action status: success
Filters: ${JSON.stringify({
  daysAhead: params.daysAhead,
  daysBehind: params.daysBehind,
})}
Event count returned: ${events.length}
Sample results:
${sample || 'No events matched the requested parameters.'}`;
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

            executiveActionContext = `Lea prepared an Outlook history CSV export request.
Action status: ready
Records to export now: ${emails.length} emails${params.includeCalendar ? `, ${calendarCount} calendar events` : ''}
Export endpoint: /api/outlook/email-history?${exportParams.toString()}
Note: Use the same authenticated session or provide Authorization: Bearer <access_token> when downloading.`;
          }
        }
      }
    }

    // Build system prompt with RAG context
    const systemPrompt = getSystemPrompt(mode, ragContext);
    let systemPromptWithContext = systemPrompt;
    if (ambiguityPrompt) {
      systemPromptWithContext += `\n\n## Clarification Required\n\nThe current user request is ambiguous. Ask this clarification question first and do not provide a final answer yet:\n"${ambiguityPrompt}"`;
    }
    if (webContext) {
      systemPromptWithContext += `\n\n## Verified Web Context\n\n${webContext}\n\nUse only these web sources for web-backed claims in this response.`;
    }
    if (executiveActionContext) {
      systemPromptWithContext += `\n\n## Verified Outlook Action Context\n\n${executiveActionContext}\n\nUse this verified action context for your response. Do not claim any Outlook action succeeded unless it appears in this context.`;
    }

    // Stream the response using Vercel AI SDK
    const result = await streamText({
      model: openai(modeConfig.model),
      system: systemPromptWithContext,
      messages,
      temperature: modeConfig.temperature,
      maxTokens: 4096,
    });

    // Return streaming response
    return result.toDataStreamResponse();

  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : 'Internal server error' 
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
