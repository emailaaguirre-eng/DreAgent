// DreAgent Cloud - Knowledge API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { ingestDocument, listDocuments, deleteDocument } from '@/lib/rag/ingest';

export const runtime = 'nodejs';
export const maxDuration = 60;

// GET - List documents
export async function GET(req: NextRequest) {
  try {
    const userId = req.headers.get('x-user-id');

    if (!userId) {
      return NextResponse.json(
        { error: 'userId required' },
        { status: 400 }
      );
    }

    const documents = await listDocuments(userId);
    return NextResponse.json({ documents });
  } catch (error) {
    console.error('Knowledge GET error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to list' },
      { status: 500 }
    );
  }
}

// POST - Ingest document
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      userId,
      title,
      content,
      metadata,
    } = body as {
      userId: string;
      title: string;
      content: string;
      metadata?: Record<string, unknown>;
    };

    if (!userId || !title || !content) {
      return NextResponse.json(
        { error: 'userId, title, and content required' },
        { status: 400 }
      );
    }

    const result = await ingestDocument(userId, title, content, metadata);

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
    });
  } catch (error) {
    console.error('Knowledge POST error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to ingest' },
      { status: 500 }
    );
  }
}

// DELETE - Remove document
export async function DELETE(req: NextRequest) {
  try {
    const body = await req.json();
    const { documentId } = body as { documentId: string };

    if (!documentId) {
      return NextResponse.json(
        { error: 'documentId required' },
        { status: 400 }
      );
    }

    await deleteDocument(documentId);
    return NextResponse.json({ status: 'deleted' });
  } catch (error) {
    console.error('Knowledge DELETE error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to delete' },
      { status: 500 }
    );
  }
}
