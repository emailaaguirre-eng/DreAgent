// DreAgent - Owner session bootstrap API (single-owner M1 gate)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Deployment not included. Local/staging only when secrets are configured.

import { NextRequest, NextResponse } from 'next/server';
import {
  applyOwnerSessionCookie,
  clearOwnerSessionCookie,
  createOwnerSessionToken,
  getConfiguredOwnerId,
  isOwnerSessionConfigured,
  isValidOwnerBootstrapSecret,
  resolveTrustedOwnerId,
} from '@/lib/auth/owner-session';

export const runtime = 'nodejs';

// GET - Session status (does not leak secret material)
export async function GET(req: NextRequest) {
  const trusted = resolveTrustedOwnerId(req);
  return NextResponse.json({
    authenticated: trusted.ok,
    ownerId: trusted.ok ? trusted.ownerId : null,
    source: trusted.source,
    configured: isOwnerSessionConfigured(),
    configuredOwnerId: getConfiguredOwnerId(),
    note: 'Client localStorage/userId is not trusted owner identity. Use POST with bootstrap secret to mint HTTP-only session cookie.',
  });
}

// POST - Mint owner session cookie when bootstrap secret matches LEA_OWNER_SESSION_SECRET
export async function POST(req: NextRequest) {
  if (!isOwnerSessionConfigured()) {
    return NextResponse.json(
      {
        error: 'Owner session not configured',
        reason:
          'Set LEA_OWNER_SESSION_SECRET on the server (and optional LEA_OWNER_ID). Do not commit secrets.',
      },
      { status: 503 }
    );
  }

  let bodySecret: string | undefined;
  try {
    const body = (await req.json()) as { secret?: string; bootstrapSecret?: string };
    bodySecret = body.secret || body.bootstrapSecret;
  } catch {
    bodySecret = undefined;
  }

  const headerSecret =
    req.headers.get('x-lea-owner-secret') ||
    req.headers.get('x-owner-bootstrap-secret');

  const candidate = bodySecret || headerSecret || null;
  if (!isValidOwnerBootstrapSecret(candidate)) {
    return NextResponse.json(
      {
        error: 'Invalid owner bootstrap secret',
        reason: 'Bootstrap secret does not match LEA_OWNER_SESSION_SECRET',
      },
      { status: 401 }
    );
  }

  const ownerId = getConfiguredOwnerId();
  const token = createOwnerSessionToken(ownerId);
  const res = NextResponse.json({
    status: 'owner_session_established',
    ownerId,
    source: 'owner_session',
  });
  return applyOwnerSessionCookie(res, token);
}

// DELETE - Clear owner session cookie
export async function DELETE() {
  const res = NextResponse.json({ status: 'owner_session_cleared' });
  return clearOwnerSessionCookie(res);
}
