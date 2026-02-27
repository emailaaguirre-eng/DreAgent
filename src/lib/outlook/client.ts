// DreAgent Cloud - Outlook Client
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

const GRAPH_API = 'https://graph.microsoft.com/v1.0';

export class GraphApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'GraphApiError';
    this.status = status;
    this.code = code;
  }
}

interface GraphRequest {
  accessToken: string;
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  endpoint: string;
  body?: unknown;
  params?: Record<string, string>;
}

async function graphRequest<T>({
  accessToken,
  method = 'GET',
  endpoint,
  body,
  params,
}: GraphRequest): Promise<T> {
  const url = new URL(`${GRAPH_API}${endpoint}`);
  
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
  }

  const response = await fetch(url.toString(), {
    method,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new GraphApiError(
      error.error?.message || `Graph API error: ${response.status}`,
      response.status,
      error.error?.code
    );
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

// ============================================================================
// Email Operations
// ============================================================================

export interface Email {
  id: string;
  subject: string;
  from: string;
  fromName: string;
  received: string;
  preview: string;
  isRead: boolean;
  hasAttachments: boolean;
}

export async function getEmails(
  accessToken: string,
  options: {
    folder?: string;
    limit?: number;
    unreadOnly?: boolean;
    startDate?: string;
    endDate?: string;
  } = {}
): Promise<Email[]> {
  const {
    folder = 'inbox',
    limit = 50,
    unreadOnly = false,
    startDate,
    endDate,
  } = options;

  const params: Record<string, string> = {
    $top: String(limit),
    $orderby: 'receivedDateTime desc',
    $select: 'id,subject,from,receivedDateTime,bodyPreview,isRead,hasAttachments',
  };

  const filters: string[] = [];

  if (unreadOnly) {
    filters.push('isRead eq false');
  }
  if (startDate) {
    filters.push(`receivedDateTime ge ${startDate}`);
  }
  if (endDate) {
    filters.push(`receivedDateTime le ${endDate}`);
  }
  if (filters.length > 0) {
    params.$filter = filters.join(' and ');
  }

  const response = await graphRequest<{ value: Array<{
    id: string;
    subject: string;
    from: { emailAddress: { address: string; name: string } };
    receivedDateTime: string;
    bodyPreview: string;
    isRead: boolean;
    hasAttachments: boolean;
  }> }>({
    accessToken,
    endpoint: `/me/mailFolders/${folder}/messages`,
    params,
  });

  return response.value.map((email) => ({
    id: email.id,
    subject: email.subject || '(No Subject)',
    from: email.from?.emailAddress?.address || 'Unknown',
    fromName: email.from?.emailAddress?.name || '',
    received: email.receivedDateTime,
    preview: email.bodyPreview || '',
    isRead: email.isRead,
    hasAttachments: email.hasAttachments,
  }));
}

export async function sendEmail(
  accessToken: string,
  to: string[],
  subject: string,
  body: string,
  options: {
    cc?: string[];
    isHtml?: boolean;
  } = {}
): Promise<void> {
  const { cc, isHtml = true } = options;

  await graphRequest({
    accessToken,
    method: 'POST',
    endpoint: '/me/sendMail',
    body: {
      message: {
        subject,
        body: {
          contentType: isHtml ? 'HTML' : 'Text',
          content: body,
        },
        toRecipients: to.map((addr) => ({ emailAddress: { address: addr } })),
        ccRecipients: cc?.map((addr) => ({ emailAddress: { address: addr } })),
      },
    },
  });
}

// ============================================================================
// Calendar Operations
// ============================================================================

export interface CalendarEvent {
  id: string;
  subject: string;
  start: string;
  end: string;
  location: string;
  isAllDay: boolean;
  organizer: string;
}

export async function getCalendarEvents(
  accessToken: string,
  options: {
    daysAhead?: number;
    daysBehind?: number;
  } = {}
): Promise<CalendarEvent[]> {
  const { daysAhead = 30, daysBehind = 0 } = options;

  const now = new Date();
  const start = new Date(now);
  start.setDate(start.getDate() - daysBehind);
  const end = new Date(now);
  end.setDate(end.getDate() + daysAhead);

  const response = await graphRequest<{ value: Array<{
    id: string;
    subject: string;
    start: { dateTime: string };
    end: { dateTime: string };
    location: { displayName: string };
    isAllDay: boolean;
    organizer: { emailAddress: { address: string } };
  }> }>({
    accessToken,
    endpoint: '/me/calendar/calendarView',
    params: {
      startDateTime: start.toISOString(),
      endDateTime: end.toISOString(),
      $select: 'id,subject,start,end,location,isAllDay,organizer',
      $orderby: 'start/dateTime',
    },
  });

  return response.value.map((event) => ({
    id: event.id,
    subject: event.subject || '(No Subject)',
    start: event.start.dateTime,
    end: event.end.dateTime,
    location: event.location?.displayName || '',
    isAllDay: event.isAllDay,
    organizer: event.organizer?.emailAddress?.address || '',
  }));
}

export async function createCalendarEvent(
  accessToken: string,
  subject: string,
  start: string,
  end: string,
  options: {
    location?: string;
    body?: string;
    attendees?: string[];
  } = {}
): Promise<{ id: string }> {
  const { location, body, attendees } = options;

  const event: Record<string, unknown> = {
    subject,
    start: { dateTime: start, timeZone: 'UTC' },
    end: { dateTime: end, timeZone: 'UTC' },
  };

  if (location) event.location = { displayName: location };
  if (body) event.body = { contentType: 'HTML', content: body };
  if (attendees) {
    event.attendees = attendees.map((email) => ({
      emailAddress: { address: email },
      type: 'required',
    }));
  }

  return graphRequest({
    accessToken,
    method: 'POST',
    endpoint: '/me/calendar/events',
    body: event,
  });
}

// ============================================================================
// OAuth Helpers
// ============================================================================

export function getAuthUrl(redirectUri: string): string {
  const clientId = process.env.OUTLOOK_CLIENT_ID;
  const tenantId = process.env.OUTLOOK_TENANT_ID || 'common';

  if (!clientId) throw new Error('OUTLOOK_CLIENT_ID is not configured');

  const scopes = [
    'https://graph.microsoft.com/Mail.Read',
    'https://graph.microsoft.com/Mail.Send',
    'https://graph.microsoft.com/Calendars.ReadWrite',
    'https://graph.microsoft.com/User.Read',
    'offline_access',
  ].join(' ');

  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: redirectUri,
    scope: scopes,
    response_mode: 'query',
  });

  return `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/authorize?${params.toString()}`;
}

export async function exchangeCodeForTokens(
  code: string,
  redirectUri: string
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
}> {
  const clientId = process.env.OUTLOOK_CLIENT_ID;
  const clientSecret = process.env.OUTLOOK_CLIENT_SECRET;
  const tenantId = process.env.OUTLOOK_TENANT_ID || 'common';

  if (!clientId) throw new Error('OUTLOOK_CLIENT_ID is not configured');
  if (!clientSecret) throw new Error('OUTLOOK_CLIENT_SECRET is not configured');

  const scopes = [
    'https://graph.microsoft.com/Mail.Read',
    'https://graph.microsoft.com/Mail.Send',
    'https://graph.microsoft.com/Calendars.ReadWrite',
    'https://graph.microsoft.com/User.Read',
    'offline_access',
  ].join(' ');

  const response = await fetch(
    `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: clientId,
        client_secret: clientSecret,
        grant_type: 'authorization_code',
        code,
        redirect_uri: redirectUri,
        scope: scopes,
      }).toString(),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const desc = error.error_description || error.error || 'Token exchange failed';
    console.error('Token exchange failed:', JSON.stringify(error));
    throw new Error(desc);
  }

  return response.json();
}

export async function refreshAccessToken(
  refreshToken: string
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
}> {
  const clientId = process.env.OUTLOOK_CLIENT_ID;
  const clientSecret = process.env.OUTLOOK_CLIENT_SECRET;
  const tenantId = process.env.OUTLOOK_TENANT_ID || 'common';

  if (!clientId) throw new Error('OUTLOOK_CLIENT_ID is not configured');
  if (!clientSecret) throw new Error('OUTLOOK_CLIENT_SECRET is not configured');

  const scopes = [
    'https://graph.microsoft.com/Mail.Read',
    'https://graph.microsoft.com/Mail.Send',
    'https://graph.microsoft.com/Calendars.ReadWrite',
    'https://graph.microsoft.com/User.Read',
    'offline_access',
  ].join(' ');

  const response = await fetch(
    `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: clientId,
        client_secret: clientSecret,
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        scope: scopes,
      }).toString(),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const desc = error.error_description || error.error || 'Token refresh failed';
    console.error('Token refresh failed:', JSON.stringify(error));
    throw new Error(desc);
  }

  return response.json();
}
