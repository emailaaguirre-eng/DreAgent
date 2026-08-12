// DreAgent - Provider-neutral mail/calendar types (M2)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Read-focused LEA port. Send/create stay on provider-specific API routes until M4.
// Gmail is reserved as an id only — no adapter in this milestone.

import type { NextRequest } from 'next/server';

export type ProviderId = 'outlook' | 'gmail';

/**
 * What LEA executive may use through the shared provider port.
 * mailSend / calendarWrite are policy flags for the port (false in M2),
 * not a claim that Graph write helpers do not exist on Outlook-specific routes.
 */
export type ProviderCapabilities = {
  mailRead: boolean;
  calendarRead: boolean;
  mailSend: boolean;
  calendarWrite: boolean;
  mailExport: boolean;
};

export type MailMessage = {
  id: string;
  subject: string;
  from: string;
  fromName: string;
  received: string;
  preview: string;
  isRead: boolean;
  hasAttachments: boolean;
};

export type CalendarItem = {
  id: string;
  subject: string;
  start: string;
  end: string;
  location: string;
  isAllDay: boolean;
  organizer: string;
};

export type MailQuery = {
  folder?: string;
  limit?: number;
  unreadOnly?: boolean;
  startDate?: string;
  endDate?: string;
};

export type CalendarQuery = {
  daysAhead?: number;
  daysBehind?: number;
};

export type ProviderCredentialSource = 'session' | 'bearer' | 'none';

export type ProviderConnection = {
  connected: boolean;
  providerId: ProviderId;
  displayName: string;
  source: ProviderCredentialSource;
  reason?: string;
};

export type ProviderRequestContext = {
  req: NextRequest;
  /** Live provider credential override (e.g. Graph Bearer). Not a user identity. */
  accessTokenOverride?: string;
};

export type MailCalendarProvider = {
  readonly id: ProviderId;
  readonly displayName: string;
  readonly capabilities: ProviderCapabilities;
  getConnection(ctx: ProviderRequestContext): Promise<ProviderConnection>;
  listMail(ctx: ProviderRequestContext, query?: MailQuery): Promise<MailMessage[]>;
  listCalendar(
    ctx: ProviderRequestContext,
    query?: CalendarQuery
  ): Promise<CalendarItem[]>;
};

export const READ_ONLY_LEA_CAPABILITIES: ProviderCapabilities = {
  mailRead: true,
  calendarRead: true,
  mailSend: false,
  calendarWrite: false,
  mailExport: true,
};
