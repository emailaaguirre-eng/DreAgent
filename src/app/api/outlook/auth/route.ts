// DreAgent Cloud - Outlook Auth API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getAuthUrl, exchangeCodeForTokens, refreshAccessToken } from '@/lib/outlook/client';

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

    // Microsoft redirected back with an authorization code — exchange it for tokens
    if (code) {
      const tokens = await exchangeCodeForTokens(code, redirectUri);

      // Redirect back to the app with tokens as hash params so the frontend can store them
      const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://lea.codrex.com';
      const hashParams = new URLSearchParams({
        outlook_access_token: tokens.access_token,
        outlook_refresh_token: tokens.refresh_token,
        outlook_expires_in: String(tokens.expires_in),
      });
      return NextResponse.redirect(`${appUrl}#${hashParams.toString()}`);
    }

    // No code — return the auth URL for the frontend to redirect the user to
    const authUrl = getAuthUrl(redirectUri);
    return NextResponse.json({ auth_url: authUrl, redirect_uri: redirectUri });

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
    const { refresh_token } = body as { refresh_token: string };

    if (!refresh_token) {
      return NextResponse.json(
        { error: 'refresh_token required' },
        { status: 400 }
      );
    }

    const tokens = await refreshAccessToken(refresh_token);

    return NextResponse.json({
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Refresh failed' },
      { status: 500 }
    );
  }
}
