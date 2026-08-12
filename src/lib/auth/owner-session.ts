// DreAgent - Single-owner LEA session foundation (M1)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Client localStorage / body / x-user-id claims are NEVER authoritative.
// Server-verifiable HTTP-only owner session cookie is the trusted identity.

import { createHmac, timingSafeEqual } from 'node:crypto';
import type { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';

/** Fixed product owner key for single-owner LEA v2 (override via LEA_OWNER_ID). */
export const DEFAULT_OWNER_ID = 'lea-owner';

export const OWNER_SESSION_COOKIE = 'lea_owner_session';

/** Default session TTL: 7 days */
export const OWNER_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

export type OwnerSessionSource =
  | 'owner_session'
  | 'none'
  | 'untrusted_client_rejected'
  | 'invalid_session'
  | 'not_configured'
  | 'owner_mismatch';

export type TrustedOwnerResult =
  | {
      ok: true;
      ownerId: string;
      source: 'owner_session';
      reason?: undefined;
    }
  | {
      ok: false;
      ownerId: null;
      source: Exclude<OwnerSessionSource, 'owner_session'>;
      reason: string;
    };

export type OwnerSessionPayload = {
  ownerId: string;
  exp: number;
};

function getSecret(): string | null {
  const secret = process.env.LEA_OWNER_SESSION_SECRET?.trim();
  return secret || null;
}

/**
 * Configured durable owner id for single-owner LEA.
 * Not derived from client claims.
 */
export function getConfiguredOwnerId(): string {
  const fromEnv = process.env.LEA_OWNER_ID?.trim();
  return fromEnv || DEFAULT_OWNER_ID;
}

/** True when a signing secret is available for owner session cookies. */
export function isOwnerSessionConfigured(): boolean {
  return Boolean(getSecret());
}

function signPayload(payload: string, secret: string): string {
  return createHmac('sha256', secret).update(payload).digest('base64url');
}

function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

/**
 * Create a signed owner session token (not a multi-user JWT product).
 * Format: base64url(ownerId).exp.sig
 */
export function createOwnerSessionToken(
  ownerId: string = getConfiguredOwnerId(),
  maxAgeSeconds: number = OWNER_SESSION_MAX_AGE_SECONDS,
  nowMs: number = Date.now()
): string {
  const secret = getSecret();
  if (!secret) {
    throw new Error(
      'LEA_OWNER_SESSION_SECRET is not configured; cannot create owner session'
    );
  }
  const exp = Math.floor(nowMs / 1000) + maxAgeSeconds;
  const idPart = Buffer.from(ownerId, 'utf8').toString('base64url');
  const body = `${idPart}.${exp}`;
  const sig = signPayload(body, secret);
  return `${body}.${sig}`;
}

export function verifyOwnerSessionToken(
  token: string | null | undefined,
  nowMs: number = Date.now()
):
  | { ok: true; payload: OwnerSessionPayload }
  | { ok: false; reason: string } {
  if (!token?.trim()) {
    return { ok: false, reason: 'Missing owner session token' };
  }

  const secret = getSecret();
  if (!secret) {
    return {
      ok: false,
      reason: 'LEA_OWNER_SESSION_SECRET is not configured',
    };
  }

  const parts = token.trim().split('.');
  if (parts.length !== 3) {
    return { ok: false, reason: 'Malformed owner session token' };
  }

  const [idPart, expPart, sig] = parts;
  const body = `${idPart}.${expPart}`;
  const expected = signPayload(body, secret);
  if (!safeEqual(sig, expected)) {
    return { ok: false, reason: 'Invalid owner session signature' };
  }

  const exp = Number(expPart);
  if (!Number.isFinite(exp)) {
    return { ok: false, reason: 'Invalid owner session expiry' };
  }
  if (exp * 1000 <= nowMs) {
    return { ok: false, reason: 'Owner session expired' };
  }

  let ownerId: string;
  try {
    ownerId = Buffer.from(idPart, 'base64url').toString('utf8');
  } catch {
    return { ok: false, reason: 'Invalid owner id encoding' };
  }

  if (!ownerId.trim()) {
    return { ok: false, reason: 'Empty owner id in session' };
  }

  const configured = getConfiguredOwnerId();
  if (ownerId !== configured) {
    return {
      ok: false,
      reason: 'Session owner does not match configured single-owner id',
    };
  }

  return { ok: true, payload: { ownerId, exp } };
}

export function readOwnerSessionCookie(
  req: NextRequest | Request
): string | null {
  // NextRequest has cookies API; plain Request uses Cookie header.
  const anyReq = req as NextRequest;
  if (typeof anyReq.cookies?.get === 'function') {
    return anyReq.cookies.get(OWNER_SESSION_COOKIE)?.value ?? null;
  }

  const header = req.headers.get('cookie');
  if (!header) return null;
  const match = header
    .split(';')
    .map((p) => p.trim())
    .find((p) => p.startsWith(`${OWNER_SESSION_COOKIE}=`));
  if (!match) return null;
  return decodeURIComponent(match.slice(OWNER_SESSION_COOKIE.length + 1));
}

/**
 * Extract untrusted client identity claims for rejection/audit only.
 * NEVER use the return value as ownerId for data access.
 */
export function extractUntrustedClientUserId(
  req: NextRequest | Request,
  explicitUserId?: string | null
): string | null {
  if (explicitUserId?.trim()) return explicitUserId.trim();

  const headerUserId = req.headers.get('x-user-id');
  if (headerUserId?.trim()) return headerUserId.trim();

  try {
    const { searchParams } = new URL(req.url);
    const queryUserId = searchParams.get('userId');
    if (queryUserId?.trim()) return queryUserId.trim();
  } catch {
    // ignore bad URLs
  }

  return null;
}

/**
 * Resolve the single LEA owner from a server-verifiable session only.
 * Client body/header/query userId is never authoritative.
 */
export function resolveTrustedOwnerId(
  req: NextRequest | Request,
  options?: {
    /** Rejected if present without a matching trusted session */
    untrustedClientUserId?: string | null;
    /** When true (default), reject client claim that differs from configured owner */
    rejectMismatchedClientClaim?: boolean;
  }
): TrustedOwnerResult {
  const untrusted =
    options?.untrustedClientUserId !== undefined
      ? options.untrustedClientUserId?.trim() || null
      : extractUntrustedClientUserId(req);
  const rejectMismatch = options?.rejectMismatchedClientClaim !== false;

  if (!isOwnerSessionConfigured()) {
    return {
      ok: false,
      ownerId: null,
      source: 'not_configured',
      reason:
        'Owner session is not configured (set LEA_OWNER_SESSION_SECRET). Client userId is not authoritative.',
    };
  }

  const token = readOwnerSessionCookie(req);
  const verified = verifyOwnerSessionToken(token);
  if (!verified.ok) {
    if (untrusted) {
      return {
        ok: false,
        ownerId: null,
        source: 'untrusted_client_rejected',
        reason:
          'Client-provided userId is not authoritative; owner session required',
      };
    }
    return {
      ok: false,
      ownerId: null,
      source: token ? 'invalid_session' : 'none',
      reason: verified.reason,
    };
  }

  const ownerId = verified.payload.ownerId;

  if (
    rejectMismatch &&
    untrusted &&
    untrusted !== ownerId
  ) {
    return {
      ok: false,
      ownerId: null,
      source: 'owner_mismatch',
      reason:
        'Client-provided userId does not match server-trusted owner session (fail closed)',
    };
  }

  return {
    ok: true,
    ownerId,
    source: 'owner_session',
  };
}

/** Fail-closed helper for sensitive API routes. */
export function requireTrustedOwner(
  req: NextRequest | Request,
  options?: Parameters<typeof resolveTrustedOwnerId>[1]
): TrustedOwnerResult {
  return resolveTrustedOwnerId(req, options);
}

export function ownerAuthErrorResponse(
  result: Extract<TrustedOwnerResult, { ok: false }>,
  status: number = 401
): NextResponse {
  return NextResponse.json(
    {
      error: 'Owner session required',
      reason: result.reason,
      source: result.source,
      hint:
        result.source === 'not_configured'
          ? 'Configure LEA_OWNER_SESSION_SECRET (and optional LEA_OWNER_ID) on the server. Do not trust client userId.'
          : 'Establish an owner session via POST /api/auth/owner-session with the server bootstrap secret. Client localStorage userId is not trusted.',
    },
    { status }
  );
}

export function buildOwnerSessionCookieOptions(
  maxAgeSeconds: number = OWNER_SESSION_MAX_AGE_SECONDS
): {
  httpOnly: true;
  secure: boolean;
  sameSite: 'lax';
  path: string;
  maxAge: number;
} {
  const secure =
    process.env.NODE_ENV === 'production' ||
    process.env.LEA_OWNER_COOKIE_SECURE === 'true';

  return {
    httpOnly: true,
    secure,
    sameSite: 'lax',
    path: '/',
    maxAge: maxAgeSeconds,
  };
}

export function applyOwnerSessionCookie(
  res: NextResponse,
  token: string,
  maxAgeSeconds: number = OWNER_SESSION_MAX_AGE_SECONDS
): NextResponse {
  res.cookies.set(
    OWNER_SESSION_COOKIE,
    token,
    buildOwnerSessionCookieOptions(maxAgeSeconds)
  );
  return res;
}

export function clearOwnerSessionCookie(res: NextResponse): NextResponse {
  res.cookies.set(OWNER_SESSION_COOKIE, '', {
    ...buildOwnerSessionCookieOptions(0),
    maxAge: 0,
  });
  return res;
}

/**
 * Bootstrap check: request may present the same value as LEA_OWNER_SESSION_SECRET
 * (header or body) to mint a session cookie. Not multi-user login — single-owner gate.
 */
export function isValidOwnerBootstrapSecret(
  candidate: string | null | undefined
): boolean {
  const secret = getSecret();
  if (!secret || !candidate?.trim()) return false;
  return safeEqual(secret, candidate.trim());
}
