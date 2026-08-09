// DreAgent Cloud - Executive Lea helpers
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

export type ExecutiveIntent =
  | 'email_summary'
  | 'calendar_summary'
  | 'email_history_export'
  | 'draft_email'
  | 'confirm_send_email'
  | 'draft_calendar_event'
  | 'confirm_create_calendar'
  | 'none';

export interface ExecutiveParams {
  limit: number;
  unreadOnly: boolean;
  includeCalendar: boolean;
  daysAhead: number;
  daysBehind: number;
  startDate?: string;
  endDate?: string;
  usedDefaultEmailWindow: boolean;
  usedDefaultCalendarWindow: boolean;
  senderHint?: string;
  subjectHint?: string;
  searchKeyword?: string;
}

const CONFIRM_SEND =
  /\b(send\s+it|send\s+the\s+email|send\s+that\s+email|yes[,\s]+send|confirm\s+send|go\s+ahead\s+and\s+send)\b/i;
const CONFIRM_CREATE =
  /\b(create\s+it|create\s+the\s+event|add\s+it\s+to\s+(my\s+)?calendar|yes[,\s]+create|confirm\s+(create|add)|schedule\s+it)\b/i;
const DRAFT_EMAIL =
  /\b(draft|compose|write)\b[\s\S]{0,40}\b(email|message|reply|follow[\s-]?up)\b|\b(email|message)\s+(draft|to\s+\S+)/i;
const DRAFT_CALENDAR =
  /\b(draft|propose|prepare)\b[\s\S]{0,40}\b(meeting|event|calendar)\b|\bschedule\s+(a\s+)?(meeting|event|call)\b/i;
const EXPORT =
  /\b(export|download)\b.*\b(csv|excel|xlsx|pptx|powerpoint)?\b.*\b(email|history|inbox)\b|\bemail\s+history\b.*\b(export|download|csv)\b/i;
const EMAIL_CHECK =
  /\b(inbox|emails?|mail|unread|messages?)\b|\bwhat'?s\s+waiting\b|\bwho\s+has\s+not\s+replied\b|\bcheck\s+my\s+(inbox|email)\b/i;
const CALENDAR_CHECK =
  /\b(calendar|meetings?|agenda|schedule)\b|\bwhat('?s| is)\s+on\s+my\s+calendar\b|\btoday'?s\s+(meetings?|agenda)\b/i;

export function detectExecutiveIntent(
  input: string,
  recentAssistantContent = ''
): ExecutiveIntent {
  const text = input.trim();
  if (!text) return 'none';

  if (CONFIRM_SEND.test(text) && hasEmailDraftArtifact(recentAssistantContent)) {
    return 'confirm_send_email';
  }
  if (
    CONFIRM_CREATE.test(text) &&
    hasCalendarDraftArtifact(recentAssistantContent)
  ) {
    return 'confirm_create_calendar';
  }

  if (EXPORT.test(text) || (
    (text.toLowerCase().includes('export') || text.toLowerCase().includes('download')) &&
    text.toLowerCase().includes('csv') &&
    (text.toLowerCase().includes('email') || text.toLowerCase().includes('history'))
  )) {
    return 'email_history_export';
  }

  // Prefer explicit draft intent over inbox "email" keyword false positives
  if (DRAFT_EMAIL.test(text) && !/\b(inbox|unread)\b/i.test(text)) {
    return 'draft_email';
  }
  if (DRAFT_CALENDAR.test(text) && !/\b(what'?s on|list|show|check)\b/i.test(text)) {
    return 'draft_calendar_event';
  }

  // Calendar before email when both could match "schedule"
  if (CALENDAR_CHECK.test(text) && !EMAIL_CHECK.test(text)) {
    return 'calendar_summary';
  }
  if (EMAIL_CHECK.test(text) && !DRAFT_EMAIL.test(text)) {
    return 'email_summary';
  }
  if (CALENDAR_CHECK.test(text)) {
    return 'calendar_summary';
  }

  return 'none';
}

export function parseExecutiveParams(input: string): ExecutiveParams {
  const text = input.toLowerCase();
  const limitMatch = text.match(/(?:last|top|limit)\s+(\d{1,4})/);
  const daysMatch = text.match(/last\s+(\d{1,4})\s+days?/);
  const today = /\btoday\b/.test(text);
  const thisWeek = /\bthis\s+week\b/.test(text);
  const thisMonth = /\bthis\s+month\b/.test(text);
  const yesterday = /\byesterday\b/.test(text);

  const limit = limitMatch
    ? Math.min(parseInt(limitMatch[1], 10), 200)
    : 50;
  const unreadOnly =
    text.includes('unread') || text.includes('waiting on me');
  const includeCalendar =
    text.includes('calendar') || text.includes('meeting');

  let daysBehind = 7;
  let usedDefaultEmailWindow = true;
  if (daysMatch) {
    daysBehind = Math.min(parseInt(daysMatch[1], 10), 365);
    usedDefaultEmailWindow = false;
  } else if (today || yesterday) {
    daysBehind = yesterday ? 1 : 0;
    usedDefaultEmailWindow = false;
  } else if (thisWeek) {
    daysBehind = 7;
    usedDefaultEmailWindow = false;
  } else if (thisMonth) {
    daysBehind = 30;
    usedDefaultEmailWindow = false;
  }

  let daysAhead = 7;
  let usedDefaultCalendarWindow = true;
  if (text.includes('next week') || thisWeek) {
    daysAhead = 7;
    usedDefaultCalendarWindow = false;
  } else if (text.includes('next month') || thisMonth) {
    daysAhead = 30;
    usedDefaultCalendarWindow = false;
  } else if (today) {
    daysAhead = 1;
    usedDefaultCalendarWindow = false;
  } else if (/\bnext\s+(\d{1,3})\s+days?\b/.test(text)) {
    const m = text.match(/\bnext\s+(\d{1,3})\s+days?\b/);
    daysAhead = m ? Math.min(parseInt(m[1], 10), 90) : 7;
    usedDefaultCalendarWindow = false;
  }

  const endDate = new Date().toISOString();
  const start = new Date();
  if (today) {
    start.setHours(0, 0, 0, 0);
  } else {
    start.setDate(start.getDate() - Math.max(daysBehind, 0));
  }
  // For "today" with daysBehind 0, still set start of day
  const startDate = start.toISOString();

  const fromMatch = input.match(
    /\bfrom\s+([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|[A-Za-z][A-Za-z0-9 .'-]{1,40})/i
  );
  const subjectMatch = input.match(/\bsubject\s*[:=]\s*["']?([^"'\n]+)["']?/i);
  const aboutMatch = input.match(/\babout\s+["']?([^"'\n.?!]+)["']?/i);

  return {
    limit,
    unreadOnly,
    includeCalendar,
    daysAhead,
    daysBehind: Math.max(daysBehind, 0),
    startDate,
    endDate,
    usedDefaultEmailWindow,
    usedDefaultCalendarWindow,
    senderHint: fromMatch?.[1]?.trim(),
    subjectHint: subjectMatch?.[1]?.trim(),
    searchKeyword: aboutMatch?.[1]?.trim(),
  };
}

export function hasEmailDraftArtifact(content: string): boolean {
  return (
    /##\s*Email draft/i.test(content) ||
    (/\bTo:\s*\S+@\S+/i.test(content) &&
      /\bSubject:\s*.+/i.test(content) &&
      /\bBody:\s*/i.test(content))
  );
}

export function hasCalendarDraftArtifact(content: string): boolean {
  return (
    /##\s*Calendar event draft/i.test(content) ||
    (/\bSubject:\s*.+/i.test(content) &&
      /\bStart:\s*.+/i.test(content) &&
      /\bEnd:\s*.+/i.test(content))
  );
}

export function parseEmailDraftFromText(content: string): {
  to: string[];
  subject: string;
  body: string;
  cc?: string[];
} | null {
  if (!content) return null;

  const toMatch = content.match(/\bTo:\s*(.+)/i);
  const ccMatch = content.match(/\bCc:\s*(.+)/i);
  const subjectMatch = content.match(/\bSubject:\s*(.+)/i);
  const bodyMatch = content.match(
    /\bBody:\s*([\s\S]*?)(?:\n##\s|\n---|\n\*\*Note:|\nAssumption:|$)/i
  );

  if (!toMatch || !subjectMatch || !bodyMatch) return null;

  const to = toMatch[1]
    .split(/[,;]/)
    .map((s) => s.trim())
    .filter((s) => s.includes('@'));
  const subject = subjectMatch[1].trim();
  const body = bodyMatch[1].trim();
  if (to.length === 0 || !subject || !body) return null;

  const cc = ccMatch
    ? ccMatch[1]
        .split(/[,;]/)
        .map((s) => s.trim())
        .filter((s) => s.includes('@'))
    : undefined;

  return { to, subject, body, cc };
}

export function parseCalendarDraftFromText(content: string): {
  subject: string;
  start: string;
  end: string;
  location?: string;
  body?: string;
  attendees?: string[];
} | null {
  if (!content) return null;

  const subjectMatch = content.match(/\bSubject:\s*(.+)/i);
  const startMatch = content.match(/\bStart:\s*(.+)/i);
  const endMatch = content.match(/\bEnd:\s*(.+)/i);
  const locationMatch = content.match(/\bLocation:\s*(.+)/i);
  const attendeesMatch = content.match(/\bAttendees:\s*(.+)/i);
  const bodyMatch = content.match(
    /\bBody:\s*([\s\S]*?)(?:\n##\s|\n---|\n\*\*Note:|$)/i
  );

  if (!subjectMatch || !startMatch || !endMatch) return null;

  const startRaw = startMatch[1].trim();
  const endRaw = endMatch[1].trim();
  const start = tryIsoDateTime(startRaw);
  const end = tryIsoDateTime(endRaw);
  if (!start || !end) return null;

  const attendees = attendeesMatch
    ? attendeesMatch[1]
        .split(/[,;]/)
        .map((s) => s.trim())
        .filter((s) => s.includes('@'))
    : undefined;

  return {
    subject: subjectMatch[1].trim(),
    start,
    end,
    location: locationMatch?.[1]?.trim(),
    body: bodyMatch?.[1]?.trim(),
    attendees,
  };
}

function tryIsoDateTime(value: string): string | null {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().replace(/\.\d{3}Z$/, '');
}

export function findLastAssistantContent(
  messages: { role: string; content: string }[]
): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'assistant' && messages[i].content) {
      return messages[i].content;
    }
  }
  return '';
}

export function truncatePreview(text: string, max = 180): string {
  const clean = (text || '').replace(/\s+/g, ' ').trim();
  if (clean.length <= max) return clean;
  return `${clean.slice(0, max - 1)}…`;
}

export function shouldUseWebSearch(input: string, mode?: string): boolean {
  const text = input.toLowerCase();
  const base =
    text.includes('latest') ||
    text.includes('current') ||
    text.includes('today') ||
    text.includes('recent') ||
    text.includes('news') ||
    text.includes('update') ||
    text.includes('according to') ||
    text.includes('who is') ||
    text.includes("who's") ||
    text.includes('when is') ||
    text.includes('when was') ||
    text.includes('deadline') ||
    text.includes('public record');

  if (mode === 'executive') {
    return (
      base ||
      text.includes('company') ||
      text.includes('regulation') ||
      text.includes('statute') ||
      /\bwhat is the (status|law|rule)\b/.test(text)
    );
  }

  return base;
}
