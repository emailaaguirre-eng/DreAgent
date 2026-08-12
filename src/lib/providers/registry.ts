// DreAgent - Mail/calendar provider registry
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Outlook (live read when connected) + Gmail (M3 foundation skeleton, never connected yet).

import { gmailProvider } from '@/lib/providers/gmail-adapter';
import { outlookProvider } from '@/lib/providers/outlook-adapter';
import type {
  MailCalendarProvider,
  ProviderConnection,
  ProviderId,
  ProviderRequestContext,
} from '@/lib/providers/types';

const REGISTERED_PROVIDERS: readonly MailCalendarProvider[] = [
  outlookProvider,
  gmailProvider,
];

export function listRegisteredProviders(): readonly MailCalendarProvider[] {
  return REGISTERED_PROVIDERS;
}

export function getProvider(id: ProviderId): MailCalendarProvider | null {
  return REGISTERED_PROVIDERS.find((provider) => provider.id === id) ?? null;
}

export type ResolvedProvider = {
  provider: MailCalendarProvider;
  connection: ProviderConnection;
};

/**
 * First connected registered provider.
 * Outlook can connect today; Gmail foundation never reports connected until OAuth/read exist.
 * Fail closed (null) when none are connected.
 */
export async function resolveConnectedProvider(
  ctx: ProviderRequestContext
): Promise<ResolvedProvider | null> {
  for (const provider of REGISTERED_PROVIDERS) {
    const connection = await provider.getConnection(ctx);
    if (connection.connected) {
      return { provider, connection };
    }
  }
  return null;
}

export function isProviderRegistered(id: ProviderId): boolean {
  return REGISTERED_PROVIDERS.some((provider) => provider.id === id);
}
