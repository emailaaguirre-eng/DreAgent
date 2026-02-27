// DreAgent Cloud - Outlook Calendar API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getCalendarEvents, createCalendarEvent, GraphApiError } from '@/lib/outlook/client';
import { resolveOutlookAccessToken } from '@/lib/outlook/tokens';

export const runtime = 'nodejs';

// GET - List calendar events
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const userId = searchParams.get('userId') || undefined;
    const resolved = await resolveOutlookAccessToken({ req, userId });
    const accessToken = resolved.accessToken;
    if (!accessToken) {
      return NextResponse.json(
        {
          error: 'Outlook not connected. Provide Bearer token or connect via /api/outlook/auth?userId=...',
        },
        { status: 401 }
      );
    }

    const daysAhead = parseInt(searchParams.get('days_ahead') || '30', 10);
    const daysBehind = parseInt(searchParams.get('days_behind') || '0', 10);

    const events = await getCalendarEvents(accessToken, { daysAhead, daysBehind });

    return NextResponse.json({ events, count: events.length, auth_source: resolved.source });
  } catch (error) {
    console.error('Calendar GET error:', error);
    if (error instanceof GraphApiError) {
      return NextResponse.json(
        {
          error: error.message,
          hint:
            error.status === 401
              ? 'Outlook token expired or invalid. Reconnect at /api/outlook/auth?userId=...'
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

// POST - Create calendar event
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { userId } = body as { userId?: string };
    const resolved = await resolveOutlookAccessToken({ req, userId });
    const accessToken = resolved.accessToken;
    if (!accessToken) {
      return NextResponse.json(
        {
          error: 'Outlook not connected. Provide Bearer token or connect via /api/outlook/auth?userId=...',
        },
        { status: 401 }
      );
    }

    const { subject, start, end, location, body: eventBody, attendees } = body as {
      subject: string;
      start: string;
      end: string;
      location?: string;
      body?: string;
      attendees?: string[];
      userId?: string;
    };

    if (!subject || !start || !end) {
      return NextResponse.json(
        { error: 'subject, start, and end required' },
        { status: 400 }
      );
    }

    const event = await createCalendarEvent(accessToken, subject, start, end, {
      location,
      body: eventBody,
      attendees,
    });

    return NextResponse.json({ event, status: 'created', auth_source: resolved.source });
  } catch (error) {
    console.error('Calendar POST error:', error);
    if (error instanceof GraphApiError) {
      return NextResponse.json(
        {
          error: error.message,
          hint:
            error.status === 401
              ? 'Outlook token expired or invalid. Reconnect at /api/outlook/auth?userId=...'
              : undefined,
        },
        { status: error.status }
      );
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create' },
      { status: 500 }
    );
  }
}
