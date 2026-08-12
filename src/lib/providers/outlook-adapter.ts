// DreAgent - Outlook mail/calendar adapter (M2)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Wraps existing Graph client + owner-session token resolution.
// Does not expose send/create on the LEA provider port.

import {
  getCalendarEvents,
  getEmails,
  type CalendarEvent,
  type Email,
} from '@/lib/outlook/client';
import { resolveOutlookAccessToken } from '@/lib/outlook/tokens';
import {
  READ_ONLY_LEA_CAPABILITIES,
  type CalendarItem,
  type CalendarQuery,
  type MailCalendarProvider,
  type MailMessage,
  type MailQuery,
  type ProviderConnection,
  type ProviderRequestContext,
} from '@/lib/providers/types';

function toMailMessage(email: Email): MailMessage {
  return {
    id: email.id,
    subject: email.subject,
    from: email.from,
    fromName: email.fromName,
    received: email.received,
    preview: email.preview,
    isRead: email.isRead,
    hasAttachments: email.hasAttachments,
  };
}

function toCalendarItem(event: CalendarEvent): CalendarItem {
  return {
    id: event.id,
    subject: event.subject,
    start: event.start,
    end: event.end,
    location: event.location,
    isAllDay: event.isAllDay,
    organizer: event.organizer,
  };
}

async function resolveOutlookCredential(ctx: ProviderRequestContext): Promise<{
  accessToken: string | null;
  source: ProviderConnection['source'];
  reason?: string;
}> {
  if (ctx.accessTokenOverride?.trim()) {
    return { accessToken: ctx.accessTokenOverride.trim(), source: 'bearer' };
  }

  const resolved = await resolveOutlookAccessToken({ req: ctx.req });
  if (!resolved.accessToken) {
    return {
      accessToken: null,
      source: 'none',
      reason:
        resolved.identityReason ||
        'Outlook not connected for the trusted owner session',
    };
  }

  return {
    accessToken: resolved.accessToken,
    source: resolved.source === 'header' ? 'bearer' : 'session',
  };
}

export const outlookProvider: MailCalendarProvider = {
  id: 'outlook',
  displayName: 'Outlook',
  capabilities: { ...READ_ONLY_LEA_CAPABILITIES },

  async getConnection(ctx): Promise<ProviderConnection> {
    const cred = await resolveOutlookCredential(ctx);
    return {
      connected: Boolean(cred.accessToken),
      providerId: 'outlook',
      displayName: 'Outlook',
      source: cred.source,
      reason: cred.reason,
    };
  },

  async listMail(ctx, query: MailQuery = {}): Promise<MailMessage[]> {
    const cred = await resolveOutlookCredential(ctx);
    if (!cred.accessToken) {
      throw new Error(cred.reason || 'Outlook mail read requires a connected provider');
    }
    const emails = await getEmails(cred.accessToken, {
      folder: query.folder,
      limit: query.limit,
      unreadOnly: query.unreadOnly,
      startDate: query.startDate,
      endDate: query.endDate,
    });
    return emails.map(toMailMessage);
  },

  async listCalendar(
    ctx,
    query: CalendarQuery = {}
  ): Promise<CalendarItem[]> {
    const cred = await resolveOutlookCredential(ctx);
    if (!cred.accessToken) {
      throw new Error(
        cred.reason || 'Outlook calendar read requires a connected provider'
      );
    }
    const events = await getCalendarEvents(cred.accessToken, {
      daysAhead: query.daysAhead,
      daysBehind: query.daysBehind,
    });
    return events.map(toCalendarItem);
  },
};
