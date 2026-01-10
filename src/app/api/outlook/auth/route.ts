// DreAgent Cloud - Outlook Auth API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getAuthUrl, exchangeCodeForTokens, refreshAccessToken } from '@/lib/outlook/client';

export const runtime = 'nodejs';

// GET - Get auth URL or handle callback
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const code = searchParams.get('code');
    const redirectUri = searchParams.get('redirect_uri') || 
      process.env.NEXT_PUBLIC_OUTLOOK_REDIRECT_URI ||
      `${process.env.NEXT_PUBLIC_APP_URL}/api/outlook/auth`;

    // If code present, exchange for tokens
    if (code) {
      const tokens = await exchangeCodeForTokens(code, redirectUri);
      
      return NextResponse.json({
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_in: tokens.expires_in,
      });
    }

    // Return auth URL
    const authUrl = getAuthUrl(redirectUri);
    return NextResponse.json({ auth_url: authUrl });
    
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
