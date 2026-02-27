// DreAgent Cloud - Outlook Auth API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getAuthUrl, exchangeCodeForTokens, refreshAccessToken } from '@/lib/outlook/client';
import { saveOutlookTokens } from '@/lib/db/supabase';
import { resolveUserId } from '@/lib/outlook/tokens';

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
    const state = searchParams.get('state');
    const stateUserId = state && state.startsWith('user:') ? state.slice(5) : null;
    const userId = resolveUserId(req, stateUserId || undefined);
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

    // If code present, exchange for tokens
    if (code) {
      const tokens = await exchangeCodeForTokens(code, redirectUri);

      if (!userId) {
        return NextResponse.json(
          { error: 'userId required to store Outlook tokens' },
          { status: 400 }
        );
      }

      const expiresAt = new Date(Date.now() + tokens.expires_in * 1000).toISOString();
      await saveOutlookTokens(
        userId,
        tokens.access_token,
        tokens.refresh_token,
        expiresAt
      );

      return NextResponse.json({
        user_id: userId,
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_in: tokens.expires_in,
        status: 'connected',
      });
    }

    // No code — return the auth URL for the frontend to redirect the user to
    const authUrl = getAuthUrl(redirectUri);
    const withState = userId
      ? `${authUrl}&state=${encodeURIComponent(`user:${userId}`)}`
      : authUrl;

    return NextResponse.json({
      auth_url: withState,
      note: userId
        ? 'Complete login to save tokens for this userId.'
        : 'Pass userId to /api/outlook/auth?userId=... for persistent token storage.',
      redirect_uri: redirectUri,
    });
  } catch (error) {
    console.error('Outlook auth error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Auth failed' },
      { status: 500 }
    );
  }
}

// POST - Refresh token
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { refresh_token, userId } = body as {
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

    if (userId) {
      const expiresAt = new Date(Date.now() + tokens.expires_in * 1000).toISOString();
      await saveOutlookTokens(
        userId,
        tokens.access_token,
        tokens.refresh_token || refresh_token,
        expiresAt
      );
    }

    return NextResponse.json({
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
      ...(userId ? { status: 'refreshed_and_saved' } : { status: 'refreshed' }),
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Refresh failed' },
      { status: 500 }
    );
  }
}
