// DreAgent Cloud - Outlook Emails API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getEmails, sendEmail, GraphApiError } from '@/lib/outlook/client';
import { resolveOutlookAccessToken } from '@/lib/outlook/tokens';

export const runtime = 'nodejs';

// GET - List emails
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    // Client userId query is never trusted; DB tokens require owner session.
    const resolved = await resolveOutlookAccessToken({ req });
    const accessToken = resolved.accessToken;
    if (!accessToken) {
      return NextResponse.json(
        {
          error:
            'Outlook not connected. Provide Bearer token or establish owner session and connect via /api/outlook/auth. Client userId is not trusted.',
          identitySource: resolved.identitySource,
          identityReason: resolved.identityReason,
        },
        { status: 401 }
      );
    }

    const folder = searchParams.get('folder') || 'inbox';
    const limit = parseInt(searchParams.get('limit') || '50', 10);
    const unreadOnly = searchParams.get('unread_only') === 'true';
    const startDate = searchParams.get('start_date') || undefined;
    const endDate = searchParams.get('end_date') || undefined;

    const emails = await getEmails(accessToken, {
      folder,
      limit,
      unreadOnly,
      startDate,
      endDate,
    });

    return NextResponse.json({
      emails,
      count: emails.length,
      auth_source: resolved.source,
      filters: { folder, limit, unreadOnly, startDate, endDate },
    });
  } catch (error) {
    console.error('Emails GET error:', error);
    if (error instanceof GraphApiError) {
      return NextResponse.json(
        {
          error: error.message,
          hint:
            error.status === 401
              ? 'Outlook token expired or invalid. Reconnect at /api/outlook/auth with an owner session.'
              : undefined,
        },
        { status: error.status }
      );
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch' },
      { status: 500 }
    );
  }
}

// POST - Send email (Graph write path — not used by Smart LEA chat; still requires trusted identity for token lookup)
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { userId: clientUserId } = body as { userId?: string };
    const resolved = await resolveOutlookAccessToken({
      req,
      userId: clientUserId,
    });
    const accessToken = resolved.accessToken;
    if (!accessToken) {
      return NextResponse.json(
        {
          error:
            'Outlook not connected. Provide Bearer token or establish owner session and connect via /api/outlook/auth. Client userId is not trusted.',
          identitySource: resolved.identitySource,
          identityReason: resolved.identityReason,
        },
        { status: 401 }
      );
    }

    const { to, subject, body: emailBody, cc, isHtml } = body as {
      to: string | string[];
      subject: string;
      body: string;
      cc?: string | string[];
      isHtml?: boolean;
      userId?: string;
    };

    if (!to || !subject) {
      return NextResponse.json(
        { error: 'to and subject required' },
        { status: 400 }
      );
    }

    const toArray = Array.isArray(to) ? to : [to];
    const ccArray = cc ? (Array.isArray(cc) ? cc : [cc]) : undefined;

    await sendEmail(accessToken, toArray, subject, emailBody || '', {
      cc: ccArray,
      isHtml: isHtml ?? true,
    });

    return NextResponse.json({ status: 'sent', auth_source: resolved.source });
  } catch (error) {
    console.error('Emails POST error:', error);
    if (error instanceof GraphApiError) {
      return NextResponse.json(
        {
          error: error.message,
          hint:
            error.status === 401
              ? 'Outlook token expired or invalid. Reconnect at /api/outlook/auth with an owner session.'
              : undefined,
        },
        { status: error.status }
      );
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to send' },
      { status: 500 }
    );
  }
}
