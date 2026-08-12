// DreAgent - Gmail read adapter foundation (M3)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Registered skeleton only. No OAuth, no token store, no Gmail API calls.
// Never reports connected / never invents inbox or calendar data.
// mailSend / calendarWrite stay false. mailRead stays false until a real path exists.

import { getGmailConfigStatus } from '@/lib/gmail/config';
import type {
  CalendarItem,
  MailCalendarProvider,
  MailMessage,
  ProviderCapabilities,
  ProviderConnection,
} from '@/lib/providers/types';

/**
 * Gmail LEA port capabilities for M3 foundation.
 * mailRead is false because there is no safe live Gmail API path yet.
 */
export const GMAIL_FOUNDATION_CAPABILITIES: ProviderCapabilities = {
  mailRead: false,
  calendarRead: false,
  mailSend: false,
  calendarWrite: false,
  mailExport: false,
};

function notReadyError(action: string): Error {
  const status = getGmailConfigStatus();
  return new Error(
    `Gmail ${action} is not available in the M3 foundation. ${status.reason}`
  );
}

export const gmailProvider: MailCalendarProvider = {
  id: 'gmail',
  displayName: 'Gmail',
  capabilities: { ...GMAIL_FOUNDATION_CAPABILITIES },

  async getConnection(): Promise<ProviderConnection> {
    const status = getGmailConfigStatus();
    return {
      connected: false,
      providerId: 'gmail',
      displayName: 'Gmail',
      source: 'none',
      reason: status.reason,
    };
  },

  async listMail(): Promise<MailMessage[]> {
    throw notReadyError('mail read');
  },

  async listCalendar(): Promise<CalendarItem[]> {
    throw notReadyError('calendar read');
  },
};
