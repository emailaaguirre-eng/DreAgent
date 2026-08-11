// DreAgent Cloud - Outlook Token Resolution and Refresh
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import type { NextRequest } from 'next/server';
import {
  extractUntrustedClientUserId,
  resolveTrustedOwnerId,
} from '@/lib/auth/owner-session';
import { getOutlookTokens, saveOutlookTokens } from '@/lib/db/supabase';
import { refreshAccessToken } from '@/lib/outlook/client';

interface ResolveTokenOptions {
  req: NextRequest;
  /**
   * @deprecated Untrusted client claim only. Never used as sole authority for DB token lookup.
   */
  userId?: string;
}

interface ResolvedTokenResult {
  accessToken: string | null;
  userId: string | null;
  source: 'header' | 'database' | 'database_refreshed' | 'none';
  identitySource?: string;
  identityReason?: string;
}

/**
 * @deprecated Client-supplied identity is not trusted.
 * Prefer extractUntrustedClientUserId + resolveTrustedOwnerId.
 * Kept as a thin untrusted claim extractor for migration / call-site clarity.
 */
export function resolveUserId(
  req: NextRequest,
  explicitUserId?: string
): string | null {
  return extractUntrustedClientUserId(req, explicitUserId);
}

export async function resolveOutlookAccessToken(
  options: ResolveTokenOptions
): Promise<ResolvedTokenResult> {
  const { req, userId: explicitUserId } = options;
  const untrustedClientUserId = extractUntrustedClientUserId(req, explicitUserId);

  const authHeader = req.headers.get('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    // Bearer is a live Graph credential, not a user identity claim.
    // Owner binding for token *storage* still uses trusted session elsewhere.
    return {
      accessToken: authHeader.slice(7),
      userId: null,
      source: 'header',
      identitySource: 'bearer_graph_token',
    };
  }

  const trusted = resolveTrustedOwnerId(req, {
    untrustedClientUserId,
  });

  if (!trusted.ok) {
    return {
      accessToken: null,
      userId: null,
      source: 'none',
      identitySource: trusted.source,
      identityReason: trusted.reason,
    };
  }

  const userId = trusted.ownerId;
  const tokenRecord = await getOutlookTokens(userId);
  if (!tokenRecord) {
    return {
      accessToken: null,
      userId,
      source: 'none',
      identitySource: trusted.source,
    };
  }

  const expiresAtMs = new Date(tokenRecord.expires_at).getTime();
  const nowMs = Date.now();
  const refreshBufferMs = 2 * 60 * 1000; // Refresh slightly before expiration

  if (expiresAtMs > nowMs + refreshBufferMs) {
    return {
      accessToken: tokenRecord.access_token,
      userId,
      source: 'database',
      identitySource: trusted.source,
    };
  }

  const refreshed = await refreshAccessToken(tokenRecord.refresh_token);
  const refreshedExpiresAt = new Date(Date.now() + refreshed.expires_in * 1000).toISOString();

  await saveOutlookTokens(
    userId,
    refreshed.access_token,
    refreshed.refresh_token || tokenRecord.refresh_token,
    refreshedExpiresAt
  );

  return {
    accessToken: refreshed.access_token,
    userId,
    source: 'database_refreshed',
    identitySource: trusted.source,
  };
}
