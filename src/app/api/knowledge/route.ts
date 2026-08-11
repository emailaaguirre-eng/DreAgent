// DreAgent Cloud - Knowledge API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import {
  ownerAuthErrorResponse,
  requireTrustedOwner,
} from '@/lib/auth/owner-session';
import { ingestDocument, listDocuments, deleteDocument } from '@/lib/rag/ingest';

export const runtime = 'nodejs';
export const maxDuration = 60;

// GET - List documents (owner session required; client userId not trusted)
export async function GET(req: NextRequest) {
  try {
    const trusted = requireTrustedOwner(req);
    if (!trusted.ok) {
      return ownerAuthErrorResponse(trusted);
    }

    const documents = await listDocuments(trusted.ownerId);
    return NextResponse.json({
      documents,
      ownerId: trusted.ownerId,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Knowledge GET error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to list' },
      { status: 500 }
    );
  }
}

// POST - Ingest document (bound to trusted owner only)
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      userId: clientUserId,
      title,
      content,
      metadata,
    } = body as {
      userId?: string;
      title: string;
      content: string;
      metadata?: Record<string, unknown>;
    };

    const trusted = requireTrustedOwner(req, {
      untrustedClientUserId: clientUserId,
    });
    if (!trusted.ok) {
      return ownerAuthErrorResponse(trusted);
    }

    if (!title || !content) {
      return NextResponse.json(
        { error: 'title and content required' },
        { status: 400 }
      );
    }

    const result = await ingestDocument(
      trusted.ownerId,
      title,
      content,
      metadata
    );

    if (!result.success) {
      return NextResponse.json(
        { error: result.error },
        { status: 400 }
      );
    }

    return NextResponse.json({
      status: 'ingested',
      documentId: result.documentId,
      chunksCreated: result.chunksCreated,
      ownerId: trusted.ownerId,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Knowledge POST error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to ingest' },
      { status: 500 }
    );
  }
}

// DELETE - Remove document (owner session required)
export async function DELETE(req: NextRequest) {
  try {
    const trusted = requireTrustedOwner(req);
    if (!trusted.ok) {
      return ownerAuthErrorResponse(trusted);
    }

    const body = await req.json();
    const { documentId } = body as { documentId: string };

    if (!documentId) {
      return NextResponse.json(
        { error: 'documentId required' },
        { status: 400 }
      );
    }

    await deleteDocument(documentId);
    return NextResponse.json({
      status: 'deleted',
      ownerId: trusted.ownerId,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Knowledge DELETE error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to delete' },
      { status: 500 }
    );
  }
}
