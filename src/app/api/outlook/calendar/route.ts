// DreAgent Cloud - Outlook Calendar API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getCalendarEvents, createCalendarEvent } from '@/lib/outlook/client';

export const runtime = 'nodejs';

// GET - List calendar events
export async function GET(req: NextRequest) {
  try {
    const accessToken = getAccessToken(req);
    if (!accessToken) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const daysAhead = parseInt(searchParams.get('days_ahead') || '30');
    const daysBehind = parseInt(searchParams.get('days_behind') || '0');

    const events = await getCalendarEvents(accessToken, { daysAhead, daysBehind });

    return NextResponse.json({ events, count: events.length });
  } catch (error) {
    console.error('Calendar GET error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch' },
      { status: 500 }
    );
  }
}

// POST - Create calendar event
export async function POST(req: NextRequest) {
  try {
    const accessToken = getAccessToken(req);
    if (!accessToken) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { subject, start, end, location, body: eventBody, attendees } = body as {
      subject: string;
      start: string;
      end: string;
      location?: string;
      body?: string;
      attendees?: string[];
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

    return NextResponse.json({ event, status: 'created' });
  } catch (error) {
    console.error('Calendar POST error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to create' },
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
