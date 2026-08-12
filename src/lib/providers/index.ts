// DreAgent - Provider abstraction public surface (M2)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

export {
  getProvider,
  isProviderRegistered,
  listRegisteredProviders,
  resolveConnectedProvider,
  type ResolvedProvider,
} from '@/lib/providers/registry';
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
