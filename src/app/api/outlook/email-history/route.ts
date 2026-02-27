// DreAgent Cloud - Outlook Email History CSV Export API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { getCalendarEvents, getEmails, GraphApiError } from '@/lib/outlook/client';
import { resolveOutlookAccessToken } from '@/lib/outlook/tokens';

export const runtime = 'nodejs';

function csvEscape(value: string | number | boolean): string {
  const text = String(value ?? '');
  return `"${text.replace(/"/g, '""')}"`;
}

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

    const folder = searchParams.get('folder') || 'inbox';
    const limit = parseInt(searchParams.get('limit') || '200', 10);
    const unreadOnly = searchParams.get('unread_only') === 'true';
    const includeCalendar = searchParams.get('include_calendar') === 'true';
    const daysAhead = parseInt(searchParams.get('days_ahead') || '30', 10);
    const daysBehind = parseInt(searchParams.get('days_behind') || '7', 10);
    const startDate = searchParams.get('start_date') || undefined;
    const endDate = searchParams.get('end_date') || undefined;

    const emails = await getEmails(accessToken, {
      folder,
      limit,
      unreadOnly,
      startDate,
      endDate,
    });

    const rows: string[] = [
      [
        'record_type',
        'id',
        'date_time',
        'subject',
        'from_name',
        'from_address',
        'preview_or_location',
        'is_read_or_all_day',
        'has_attachments',
      ].join(','),
    ];

    for (const email of emails) {
      rows.push(
        [
          csvEscape('email'),
          csvEscape(email.id),
          csvEscape(email.received),
          csvEscape(email.subject),
          csvEscape(email.fromName),
          csvEscape(email.from),
          csvEscape(email.preview),
          csvEscape(email.isRead),
          csvEscape(email.hasAttachments),
        ].join(',')
      );
    }

    if (includeCalendar) {
      const events = await getCalendarEvents(accessToken, { daysAhead, daysBehind });
      for (const event of events) {
        rows.push(
          [
            csvEscape('calendar'),
            csvEscape(event.id),
            csvEscape(event.start),
            csvEscape(event.subject),
            csvEscape(event.organizer),
            csvEscape(''),
            csvEscape(event.location),
            csvEscape(event.isAllDay),
            csvEscape(false),
          ].join(',')
        );
      }
    }

    const csv = `${rows.join('\n')}\n`;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = includeCalendar
      ? `outlook-history-${timestamp}.csv`
      : `email-history-${timestamp}.csv`;

    return new NextResponse(csv, {
      status: 200,
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': `attachment; filename="${filename}"`,
        'X-DreAgent-Auth-Source': resolved.source,
      },
    });
  } catch (error) {
    console.error('Email history CSV export error:', error);
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
      { error: error instanceof Error ? error.message : 'Export failed' },
      { status: 500 }
    );
  }
}
