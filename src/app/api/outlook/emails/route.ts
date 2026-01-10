// DreAgent Cloud - Outlook Emails API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getEmails, sendEmail } from '@/lib/outlook/client';

export const runtime = 'nodejs';

// GET - List emails
export async function GET(req: NextRequest) {
  try {
    const accessToken = getAccessToken(req);
    if (!accessToken) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const folder = searchParams.get('folder') || 'inbox';
    const limit = parseInt(searchParams.get('limit') || '50');
    const unreadOnly = searchParams.get('unread_only') === 'true';

    const emails = await getEmails(accessToken, { folder, limit, unreadOnly });

    return NextResponse.json({ emails, count: emails.length });
  } catch (error) {
    console.error('Emails GET error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch' },
      { status: 500 }
    );
  }
}

// POST - Send email
export async function POST(req: NextRequest) {
  try {
    const accessToken = getAccessToken(req);
    if (!accessToken) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { to, subject, body: emailBody, cc, isHtml } = body as {
      to: string | string[];
      subject: string;
      body: string;
      cc?: string | string[];
      isHtml?: boolean;
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

    return NextResponse.json({ status: 'sent' });
  } catch (error) {
    console.error('Emails POST error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to send' },
      { status: 500 }
    );
  }
}

function getAccessToken(req: NextRequest): string | null {
  const authHeader = req.headers.get('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    return authHeader.slice(7);
  }
  return null;
}
