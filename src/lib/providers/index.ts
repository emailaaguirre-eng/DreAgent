// DreAgent - Provider abstraction public surface
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

export {
  getProvider,
  isProviderRegistered,
  listRegisteredProviders,
  resolveConnectedProvider,
  type ResolvedProvider,
} from '@/lib/providers/registry';
export { gmailProvider, GMAIL_FOUNDATION_CAPABILITIES } from '@/lib/providers/gmail-adapter';
export { outlookProvider } from '@/lib/providers/outlook-adapter';
export {
  READ_ONLY_LEA_CAPABILITIES,
  type CalendarItem,
  type CalendarQuery,
  type MailCalendarProvider,
  type MailMessage,
  type MailQuery,
  type ProviderCapabilities,
  type ProviderConnection,
  type ProviderId,
  type ProviderRequestContext,
} from '@/lib/providers/types';
export {
  getGmailConfigStatus,
  GMAIL_OAUTH_ENV_NAMES,
  isGmailLiveReadAvailable,
} from '@/lib/gmail/config';
