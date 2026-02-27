// DreAgent Cloud - Outlook Token Resolution and Refresh
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import type { NextRequest } from 'next/server';
import { getOutlookTokens, saveOutlookTokens } from '@/lib/db/supabase';
import { refreshAccessToken } from '@/lib/outlook/client';

interface ResolveTokenOptions {
  req: NextRequest;
  userId?: string;
}

interface ResolvedTokenResult {
  accessToken: string | null;
  userId: string | null;
  source: 'header' | 'database' | 'database_refreshed' | 'none';
}

export function resolveUserId(req: NextRequest, explicitUserId?: string): string | null {
  if (explicitUserId?.trim()) return explicitUserId.trim();

  const headerUserId = req.headers.get('x-user-id');
  if (headerUserId?.trim()) return headerUserId.trim();

  const { searchParams } = new URL(req.url);
  const queryUserId = searchParams.get('userId');
  if (queryUserId?.trim()) return queryUserId.trim();

  return null;
}

export async function resolveOutlookAccessToken(
  options: ResolveTokenOptions
): Promise<ResolvedTokenResult> {
  const { req, userId: explicitUserId } = options;

  const authHeader = req.headers.get('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    return {
      accessToken: authHeader.slice(7),
      userId: resolveUserId(req, explicitUserId),
      source: 'header',
    };
  }

  const userId = resolveUserId(req, explicitUserId);
  if (!userId) {
    return { accessToken: null, userId: null, source: 'none' };
  }

  const tokenRecord = await getOutlookTokens(userId);
  if (!tokenRecord) {
    return { accessToken: null, userId, source: 'none' };
  }

  const expiresAtMs = new Date(tokenRecord.expires_at).getTime();
  const nowMs = Date.now();
  const refreshBufferMs = 2 * 60 * 1000; // Refresh slightly before expiration

  if (expiresAtMs > nowMs + refreshBufferMs) {
    return {
      accessToken: tokenRecord.access_token,
      userId,
      source: 'database',
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
  };
}
