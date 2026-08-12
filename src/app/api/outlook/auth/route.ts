// DreAgent Cloud - Outlook Auth API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import {
  getConfiguredOwnerId,
  ownerAuthErrorResponse,
  requireTrustedOwner,
} from '@/lib/auth/owner-session';
import { getAuthUrl, exchangeCodeForTokens, refreshAccessToken } from '@/lib/outlook/client';
import { saveOutlookTokens } from '@/lib/db/supabase';

export const runtime = 'nodejs';

function getRedirectUri(): string {
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://lea.codrex.com';
  return `${appUrl.replace(/\/$/, '')}/api/outlook/auth`;
}

// GET - Get auth URL or handle OAuth callback from Microsoft
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const code = searchParams.get('code');
    const error = searchParams.get('error');
    const errorDescription = searchParams.get('error_description');
    const redirectUri = getRedirectUri();

    // Microsoft returned an error (user denied consent, etc.)
    if (error) {
      console.error(`OAuth error: ${error} - ${errorDescription}`);
      const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://lea.codrex.com';
      const params = new URLSearchParams({
        outlook_error: error,
        outlook_error_description: errorDescription || 'Authentication failed',
      });
      return NextResponse.redirect(`${appUrl}?${params.toString()}`);
    }

    // If code present, exchange for tokens — store only for trusted owner session.
    if (code) {
      const trusted = requireTrustedOwner(req);
      if (!trusted.ok) {
        return ownerAuthErrorResponse(trusted);
      }

      const tokens = await exchangeCodeForTokens(code, redirectUri);
      const expiresAt = new Date(Date.now() + tokens.expires_in * 1000).toISOString();
      await saveOutlookTokens(
        trusted.ownerId,
        tokens.access_token,
        tokens.refresh_token,
        expiresAt
      );

      return NextResponse.json({
        user_id: trusted.ownerId,
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_in: tokens.expires_in,
        status: 'connected',
        identitySource: trusted.source,
      });
    }

    // No code — return the auth URL. Prefer trusted owner session; do not bind to client userId.
    const trusted = requireTrustedOwner(req);
    const authUrl = getAuthUrl(redirectUri);
    if (!trusted.ok) {
      return NextResponse.json({
        auth_url: authUrl,
        configuredOwnerId: getConfiguredOwnerId(),
        note:
          'Owner session required before Outlook tokens can be stored. Establish session via POST /api/auth/owner-session, then retry. Client userId query/header is not trusted.',
        identity: trusted,
        redirect_uri: redirectUri,
      });
    }

    const withState = `${authUrl}&state=${encodeURIComponent(`owner:${trusted.ownerId}`)}`;

    return NextResponse.json({
      auth_url: withState,
      ownerId: trusted.ownerId,
      note: 'Complete Microsoft login to save tokens for the trusted owner session.',
      redirect_uri: redirectUri,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Outlook auth error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Auth failed' },
      { status: 500 }
    );
  }
}

// POST - Refresh token (persist only under trusted owner session)
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { refresh_token, userId: clientUserId } = body as {
      refresh_token: string;
      userId?: string;
    };

    if (!refresh_token) {
      return NextResponse.json(
        { error: 'refresh_token required' },
        { status: 400 }
      );
    }

    const tokens = await refreshAccessToken(refresh_token);

    // Saving to DB requires trusted owner — never bind using client userId alone.
    const trusted = requireTrustedOwner(req, {
      untrustedClientUserId: clientUserId,
    });
    if (clientUserId !== undefined || req.headers.get('x-user-id')) {
      if (!trusted.ok) {
        return ownerAuthErrorResponse(trusted);
      }
      const expiresAt = new Date(Date.now() + tokens.expires_in * 1000).toISOString();
      await saveOutlookTokens(
        trusted.ownerId,
        tokens.access_token,
        tokens.refresh_token || refresh_token,
        expiresAt
      );
      return NextResponse.json({
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_in: tokens.expires_in,
        status: 'refreshed_and_saved',
        ownerId: trusted.ownerId,
        identitySource: trusted.source,
      });
    }

    return NextResponse.json({
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
      status: 'refreshed',
      note: 'Token not saved; provide owner session to persist under trusted owner id.',
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Refresh failed' },
      { status: 500 }
    );
  }
}
